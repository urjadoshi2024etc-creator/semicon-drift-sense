"""
Drift-Sense Track 2 - V6 localization model.

Design goals learned from V2/V3/V4 failures:
- Never downsample the 1000x1000 reference pixels before local feature extraction.
- Use physically scale-aware, NON-SHARED branch encoders because one reference pixel
  is 1 nm while one search pixel is 10 nm. A shared encoder has different physical RFs.
- Reference: stride-4 local encoder (RF 15 nm), then learned 10x feature pooling.
- Search: stride-4 encoder with RF 7 search-px ~= 70 nm.
- Result: both feature grids have 4 search-pixels (40 nm) spacing.
- Use VALID correlation, not same-padding, so image borders cannot become landmarks.
- Use hard argmax + learned local offset rather than a global soft-argmax. This avoids
  averaging between periodic candidates.
- Train the candidate map with cross-entropy (single correct location) plus a local
  offset loss. This explicitly suppresses competing periodic peaks.

Coordinate convention: (x, y) in original 1000x1000 search-image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 1000
MAG_RATIO = 10
FEATURE_STRIDE = 4
REF_TEMPLATE_SIZE = 25
HEATMAP_SIZE = IMAGE_SIZE // FEATURE_STRIDE  # 250, conceptual search grid
CORR_SIZE = HEATMAP_SIZE - REF_TEMPLATE_SIZE + 1  # 226 valid positions
OUTPUT_STRIDE = float(FEATURE_STRIDE)
# Exact feature-grid alignment: two stride-2 k3/p1 convolutions have start=0;
# the 10-wide reference scale-alignment kernel has center at 4.5 feature cells.
# At 4 reference px/cell this is 18 nm = 1.8 search pixels.
REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX = 1.8


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int = 1, groups: int = 8):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout),
            nn.ReLU(inplace=True),
        )


class ReferenceEncoder(nn.Module):
    """Local reference encoder. Total stride=4, RF=15 reference pixels."""
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(1, 32, stride=2),
            ConvBNReLU(32, 48, stride=2),
            ConvBNReLU(48, out_channels, stride=1),
        )
        self.out_channels = out_channels
        self.stride = 4
        self.receptive_field = 15

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SearchEncoder(nn.Module):
    """Search encoder. Total stride=4, RF=7 search pixels (~70 nm)."""
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(1, 32, stride=2),
            ConvBNReLU(32, out_channels, stride=2),
        )
        self.out_channels = out_channels
        self.stride = 4
        self.receptive_field = 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReferenceScaleAlign(nn.Module):
    """Learned 10x physical-scale alignment on reference features.

    Input: 250x250 at 4 nm spacing.
    Kernel=10, stride=10 -> 25x25 at 40 nm spacing.
    RF becomes 15 + 9*4 = 51 nm approximately.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=10, stride=10, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OffsetHead(nn.Module):
    """Predicts sub-cell offsets in SEARCH pixels, limited to +/- 2 px."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(self.net(x))


@dataclass
class ModelOutput:
    candidate_logits: torch.Tensor       # (B,1,226,226), valid top-left positions
    probability_map: torch.Tensor        # (B,1,226,226)
    coords: torch.Tensor                 # final hard+offset (B,2), pixel space
    coords_argmax: torch.Tensor          # hard candidate center (B,2)
    offsets: torch.Tensor                # selected local offsets (B,2)
    offset_map: torch.Tensor             # (B,2,226,226)
    peak_probability: torch.Tensor       # (B,)
    peak_to_uniform: torch.Tensor        # (B,)


class DriftSenseV6(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.image_size = IMAGE_SIZE
        self.mag_ratio = MAG_RATIO
        self.feature_stride = FEATURE_STRIDE
        self.template_size = REF_TEMPLATE_SIZE
        self.heatmap_size = CORR_SIZE
        self.output_stride = OUTPUT_STRIDE

        self.ref_encoder = ReferenceEncoder(64)
        self.search_encoder = SearchEncoder(64)
        self.ref_scale_align = ReferenceScaleAlign(64)

        self.ref_proj = nn.Conv2d(64, embed_dim, 1, bias=False)
        self.search_proj = nn.Conv2d(64, embed_dim, 1, bias=False)

        # ------------------------------------------------------------------
        # V6 contextual candidate scorer
        # ------------------------------------------------------------------
        # V5 scored each candidate using only its own 32-channel correlation
        # vector (1x1 convolutions). That is a weak inductive bias for the
        # highly periodic DRAM search image: multiple candidates can have
        # similar local correlation signatures while differing in their
        # surrounding correlation pattern.
        #
        # V6 keeps the same correlation map, but lets the scorer see a local
        # spatial neighborhood around each candidate. Two 3x3 convolutions
        # give a 5x5 candidate-cell context (20x20 search pixels at the
        # 4-pixel candidate stride), followed by a 1x1 projection to one
        # candidate logit. Padding=1 preserves the required 226x226 map size.
        # GroupNorm is kept identical to the current V5 revision.
        self.score_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, 1),
        )

        # Offset head and everything before it are unchanged from V5 so the
        # only architectural variable in this experiment is candidate-score
        # context. The loss_v5.py interface remains fully compatible because
        # candidate_logits/probability_map/offset_map/coords are unchanged.
        self.offset_head = OffsetHead(embed_dim)

        # Logit-scale ("temperature") applied to the correlation-derived
        # score map before the softmax. NECESSARY because raw correlation-
        # to-logit spread starts out far too small for a 51,076-way
        # (226x226) softmax to concentrate probability mass without
        # explicit scaling -- confirmed empirically: without it, cls loss
        # sat at exactly log(226*226)=10.84, the uniform-distribution
        # cross-entropy, for 20+ epochs on the full dataset. (Note: ref/
        # search descriptors ARE L2-normalized per spatial position across
        # channels before correlation, but the depthwise correlation then
        # SUMS products over the full 25x25 reference window per channel --
        # so the resulting correlation values are not tightly bounded to
        # [-1,1] the way a single unit-vector dot product would be; the
        # scaling need is empirically confirmed, not derived from a strict
        # boundedness argument.)
        #
        # Parameterized in log-space (CLIP-style) so it trains stably and
        # can only grow within a bounded, numerically-safe range.
        self.log_logit_scale = nn.Parameter(torch.log(torch.tensor(8.0)))
        self.max_logit_scale = 20.0

    @staticmethod
    def _valid_depthwise_corr(search: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        b, c, hs, ws = search.shape
        _, _, hr, wr = ref.shape
        if hr > hs or wr > ws:
            raise ValueError(f"Reference template {hr}x{wr} larger than search {hs}x{ws}")
        search_g = search.reshape(1, b * c, hs, ws)
        kernel_g = ref.reshape(b * c, 1, hr, wr)
        corr = F.conv2d(search_g, kernel_g, padding=0, groups=b * c)
        return corr.reshape(b, c, corr.shape[-2], corr.shape[-1])

    def forward(self, ref_img: torch.Tensor, search_img: torch.Tensor) -> ModelOutput:
        # Full native resolution. No raw reference resize.
        ref = self.ref_encoder(ref_img)       # B,64,250,250
        search = self.search_encoder(search_img)  # B,64,250,250

        # Physical scale alignment after local feature extraction.
        ref = self.ref_scale_align(ref)       # B,64,25,25

        ref = self.ref_proj(ref)
        search = self.search_proj(search)

        # Unit-normalize descriptor vectors so correlation is about similarity,
        # not raw activation magnitude.
        ref = F.normalize(ref, dim=1, eps=1e-6)
        search = F.normalize(search, dim=1, eps=1e-6)

        corr = self._valid_depthwise_corr(search, ref)  # B,E,226,226
    
        raw_logits = self.score_head(corr)
        logit_scale = torch.clamp(self.log_logit_scale.exp(), max=self.max_logit_scale)
        # Forced to float32 regardless of autocast: this multiply is the most
        # likely fp16-overflow point under AMP (raw score magnitude varies a
        # lot across diverse DRAM pitch/quality samples, and can be scaled up
        # to 20x here) -- an overflow here causes GradScaler to silently skip
        # the optimizer step for that batch, which looks like "training that
        # never progresses" rather than an explicit error.
        logits = (raw_logits.float() * logit_scale.float())
        offsets = self.offset_head(corr)

        b, _, h, w = logits.shape
        probs = torch.softmax(logits.flatten(1), dim=1).view(b, 1, h, w)
        flat_idx = probs.flatten(1).argmax(dim=1)
        iy = flat_idx // w
        ix = flat_idx % w

        # Valid-correlation coordinate = top-left + template center.
        half = (REF_TEMPLATE_SIZE - 1) / 2.0
        center_x = (ix.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
        center_y = (iy.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
        coords_argmax = torch.stack([center_x, center_y], dim=1)

        off_flat = offsets.permute(0, 2, 3, 1).reshape(b, h * w, 2)
        selected_offsets = off_flat[torch.arange(b, device=logits.device), flat_idx]
        coords = coords_argmax + selected_offsets

        peak_probability = probs.flatten(1).max(dim=1).values
        uniform = 1.0 / float(h * w)
        peak_to_uniform = peak_probability / uniform

        return ModelOutput(
            candidate_logits=logits,
            probability_map=probs,
            coords=coords,
            coords_argmax=coords_argmax,
            offsets=selected_offsets,
            offset_map=offsets,
            peak_probability=peak_probability,
            peak_to_uniform=peak_to_uniform,
        )


def build_model(embed_dim: int = 32) -> DriftSenseV6:
    return DriftSenseV6(embed_dim=embed_dim)


# ---------------------------------------------------------------------------
# Geometry helpers used by loss/diagnostics
# ---------------------------------------------------------------------------
def gt_to_candidate(center_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert search pixel center (x,y) to valid-correlation cell + local offset."""
    half = (REF_TEMPLATE_SIZE - 1) / 2.0
    cell_x_float = (
    center_xy[:, 0] - REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
    ) / OUTPUT_STRIDE - half

    cell_y_float = (
    center_xy[:, 1] - REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
    ) / OUTPUT_STRIDE - half
    ix = torch.round(cell_x_float).long().clamp(0, CORR_SIZE - 1)
    iy = torch.round(cell_y_float).long().clamp(0, CORR_SIZE - 1)
    base_x = (ix.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
    base_y = (iy.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
    offset = torch.stack([center_xy[:, 0] - base_x, center_xy[:, 1] - base_y], dim=1)
    return torch.stack([iy, ix], dim=1), offset


def candidate_to_center(iy_ix: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
    half = (REF_TEMPLATE_SIZE - 1) / 2.0
    iy = iy_ix[:, 0].float()
    ix = iy_ix[:, 1].float()
    center = torch.stack([(ix + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX, (iy + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX], dim=1)
    if offset is not None:
        center = center + offset
    return center


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_model().eval()
    ref = torch.randn(2, 1, 1000, 1000)
    search = torch.randn(2, 1, 1000, 1000)
    with torch.no_grad():
        out = m(ref, search)
    print("V6 shape test")
    print("candidate_logits:", tuple(out.candidate_logits.shape))
    print("probability_map :", tuple(out.probability_map.shape))
    print("coords          :", tuple(out.coords.shape))
    print("offset_map      :", tuple(out.offset_map.shape))
    print("coords range    :", out.coords.min().item(), out.coords.max().item())
    assert out.candidate_logits.shape == (2, 1, 226, 226)
    assert out.probability_map.shape == (2, 1, 226, 226)
    assert out.coords.shape == (2, 2)
    print("PASS")
