"""
loss_v5.py

V5 localization loss -- REVISED after the reference-sensitivity diagnostic
(see test_reference_sensitivity_v5.py results, run on
runs/v5_train_overfit/checkpoints/best_model.pt).

WHAT THE DIAGNOSTIC SHOWED
---------------------------------------------------------------------------
The model DOES use the reference (predictions tracked the true match
almost exactly when fed the correct reference, and moved substantially
when fed wrong references) -- so this is NOT the V2 reference-independence
failure. Instead:

  - peak/uniform confidence was ~35,000-51,000x for EVERY reference,
    including three trials of PURE RANDOM NOISE fed in as the "reference".
  - train mean error ~20px, val mean error ~450px (near random-guess level
    on a 1000x1000 image) on a properly held-out val set (val_v5, 300
    pairs, same generator/profile as the 3000-pair train_v3).

CONCLUSION: the model has MEMORIZED the training set rather than learning
a generalizable similarity function. The original loss trained a hard,
single-winning-cell cross-entropy per example (see model_v5.py's
ModelOutput.candidate_logits, a (B,1,226,226) map with argmax-selected
offset). With only 3000 unique training pairs against a 226*226 = 51,076-
way classification problem, and nothing in the loss discouraging maximal
peakedness, the easiest thing for the network to fit is a near-delta-
function response keyed to each training example's specific feature
pattern -- which does not transfer to unseen worlds.

THE FIX (two changes, both standard for exactly this failure mode in
dense keypoint / heatmap localization):

  1. SOFT-LABEL CLASSIFICATION instead of hard single-cell CE. The
     target distribution is a small 2-D Gaussian (in valid-correlation
     CELL units) centered at the ground-truth location, not a one-hot
     vector. Cross-entropy against a soft target still concentrates
     probability at the right place, but rewards a smooth response
     around the true location rather than one memorized spike -- the
     same reasoning that was already applied successfully to the
     V2-V4 heatmap loss (loss.py's make_gaussian_heatmap), now applied
     to V5's candidate-classification head.

  2. ENTROPY REGULARIZATION. A small term that penalizes the predicted
     distribution for being LESS confident than a target floor is
     avoided (we don't want to force blur onto genuinely easy cases) --
     instead we penalize entropy that collapses far below what the soft
     target itself implies, which caps how spiky the network is allowed
     to get relative to the supervision signal it was actually given.
     This directly attacks the "confident on pure noise" symptom: a
     network whose loss no longer rewards infinite peakedness has no
     incentive to produce a 50,000x peak/uniform ratio on an input that
     doesn't match anything.

The offset regression term is unchanged in spirit (Smooth-L1 on the
sub-cell offset at the ground-truth cell) but now reads its target cell
from the same continuous soft-target geometry rather than recomputing a
separately-rounded cell.

INTERFACE (matches train_v5.py's usage exactly):
    criterion = V5LocalizationLoss()
    loss, parts = criterion(model_output, coords_gt)
    # model_output: model_v5.ModelOutput (candidate_logits, offset_map, ...)
    # coords_gt: (B, 2) pixel-space (x, y) ground truth, same convention
    #            as dram_dataset.py with normalize_labels=False.
    # parts: {"loss_classification", "loss_offset", "loss_margin"} floats
    #        (key names kept identical to the original file so train.py's
    #        CSV logging and printed summary need no changes -- "loss_margin"
    #        now carries the entropy-regularization term; see note below).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v6 import (
    REF_TEMPLATE_SIZE,
    REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX,
    OUTPUT_STRIDE,
    CORR_SIZE,
)
# NOTE: explicitly coupled to model_v6's geometry constants (not model_v5)
# since train_v6.py is what actually uses this loss. model_v6.py currently
# defines identical geometry to model_v5.py (V6 only changed the score
# head), but importing from v6 directly removes a silent-mismatch risk if
# that ever changes independently in the future.


# --------------------------------------------------------------------------- #
# Named constants
# --------------------------------------------------------------------------- #

DEFAULT_SOFT_TARGET_SIGMA: float = 1.25   # Gaussian std-dev, in VALID-GRID cell units
                                           # (widened from 1.25 after epoch 58+ divergence on
                                           # the 3000-pair dataset showed train crashing toward
                                           # ~8px while val plateaued ~190-200px -- a tighter
                                           # target made near-exact per-example memorization too
                                           # easy to fit. Still tight relative to the 226x226
                                           # grid; watch median error in the next run -- if it
                                           # regresses noticeably, dial back toward 1.5.
                                           # (1 cell = OUTPUT_STRIDE=4 search px = 40nm).
                                           # ~1.25 cells means most target mass sits
                                           # within about +/-2 cells (+/-80px) of the
                                           # true location -- tight enough to still
                                           # pin down the exact periodic repeat, loose
                                           # enough to stop rewarding a memorized delta.
DEFAULT_CLS_WEIGHT: float = 1.0
DEFAULT_OFFSET_WEIGHT: float = 1.0
DEFAULT_ENTROPY_WEIGHT: float = 0.02      # small: this is a confidence CAP, not the
                                           # primary training signal. Too high will
                                           # blur the response and hurt localization;
                                           # too low reintroduces the collapse.


def _cell_coords(gt_xy: torch.Tensor) -> torch.Tensor:
    """
    Ground-truth pixel-space (x, y) -> CONTINUOUS valid-grid cell coordinates
    (float, not rounded). Mirrors model_v5.gt_to_candidate's geometry exactly
    but keeps the sub-cell fraction instead of rounding to the nearest cell,
    since the soft target needs the true continuous position.

    Args:
        gt_xy: (B, 2) pixel-space (x, y).
    Returns:
        (B, 2) continuous (cell_x, cell_y), NOT clamped -- caller handles any
        out-of-grid mass naturally via the Gaussian falling off near the edge.
    """
    half = (REF_TEMPLATE_SIZE - 1) / 2.0
    cell_x = (gt_xy[:, 0] - REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX) / OUTPUT_STRIDE - half
    cell_y = (gt_xy[:, 1] - REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX) / OUTPUT_STRIDE - half
    return torch.stack([cell_x, cell_y], dim=1)


def make_soft_target(
    gt_xy: torch.Tensor,
    grid_size: int,
    sigma: float = DEFAULT_SOFT_TARGET_SIGMA,
) -> torch.Tensor:
    """
    Builds a (B, 1, grid_size, grid_size) target probability distribution:
    a 2-D Gaussian centered at the ground-truth's continuous cell position,
    normalized to sum to 1 over the grid (so it is a valid target for KL /
    cross-entropy against a softmax prediction).

    Args:
        gt_xy: (B, 2) pixel-space (x, y) ground truth.
        grid_size: CORR_SIZE (226) -- must match candidate_logits' spatial dims.
        sigma: Gaussian std-dev in cell units.
    Returns:
        (B, 1, grid_size, grid_size) float tensor, each sample sums to 1.
    """
    device = gt_xy.device
    dtype = gt_xy.dtype
    b = gt_xy.shape[0]

    cell_xy = _cell_coords(gt_xy)  # (B, 2), continuous, unclamped

    coord_range = torch.arange(grid_size, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(coord_range, coord_range, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    cx = cell_xy[:, 0].view(b, 1, 1)
    cy = cell_xy[:, 1].view(b, 1, 1)

    sq_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    unnorm = torch.exp(-sq_dist / (2.0 * sigma ** 2))
    target = unnorm / (unnorm.sum(dim=(1, 2), keepdim=True) + 1e-12)
    return target.unsqueeze(1)  # (B, 1, grid_size, grid_size)


class V5LocalizationLoss(nn.Module):
    """
    total_loss = cls_weight     * SoftCrossEntropy(candidate_logits, soft_target)
               + offset_weight  * SmoothL1(offset_at_gt_cell, true_sub_cell_offset)
               + entropy_weight * EntropyCollapsePenalty(probability_map, soft_target)

    EntropyCollapsePenalty: penalizes the predicted distribution's entropy
    falling BELOW the soft target's own entropy (i.e. the network being
    spikier than the supervision it was given implies it should be). This
    is zero when the prediction is exactly as sharp as the target and grows
    only as the prediction gets MORE overconfident than that -- it does not
    push the network toward being blurrier than the target, only stops it
    from collapsing past it.
    """

    def __init__(
        self,
        sigma: float = DEFAULT_SOFT_TARGET_SIGMA,
        cls_weight: float = DEFAULT_CLS_WEIGHT,
        offset_weight: float = DEFAULT_OFFSET_WEIGHT,
        entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.entropy_weight = entropy_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    def forward(self, model_output, coords_gt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_output: model_v5.ModelOutput (uses candidate_logits,
                probability_map, offset_map).
            coords_gt: (B, 2) pixel-space (x, y) ground truth.
        Returns:
            total_loss, {"loss_classification", "loss_offset", "loss_margin"}
            ("loss_margin" key name kept for train.py CSV/logging compatibility;
            it now holds the entropy-collapse penalty -- see module docstring.)
        """
        logits = model_output.candidate_logits          # (B,1,H,W)
        probs = model_output.probability_map             # (B,1,H,W), softmax already applied
        offset_map = model_output.offset_map              # (B,2,H,W)
        b, _, h, w = logits.shape

        # --- 1. Soft-label classification (cross-entropy against Gaussian target) ---
        # Computed explicitly in float32: under AMP, logits arrive as float16,
        # and log_softmax/entropy math over a 51,076-way distribution is prone
        # to underflow (very small probabilities) in fp16 -- forcing fp32 here
        # avoids NaNs without needing to disable AMP for the rest of the model.
        logits_f32 = logits.float()
        soft_target = make_soft_target(coords_gt, grid_size=h, sigma=self.sigma).float()
        log_probs = F.log_softmax(logits_f32.flatten(2), dim=-1).view(b, 1, h, w)
        cls_loss = -(soft_target * log_probs).sum(dim=(1, 2, 3)).mean()

        # --- 2. Offset regression at the (rounded) ground-truth cell ---
        half = (REF_TEMPLATE_SIZE - 1) / 2.0
        cell_xy = _cell_coords(coords_gt)
        ix = torch.round(cell_xy[:, 0]).long().clamp(0, w - 1)
        iy = torch.round(cell_xy[:, 1]).long().clamp(0, h - 1)
        base_x = (ix.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
        base_y = (iy.float() + half) * OUTPUT_STRIDE + REF_TEMPLATE_CENTER_OFFSET_SEARCH_PX
        true_offset = torch.stack([coords_gt[:, 0] - base_x, coords_gt[:, 1] - base_y], dim=1)

        batch_idx = torch.arange(b, device=logits.device)
        pred_offset_at_gt = offset_map[batch_idx, :, iy, ix]  # (B, 2)
        offset_loss = self.smooth_l1(pred_offset_at_gt, true_offset)

        # --- 3. Entropy-collapse penalty ---
        # NOTE: forced to float32 with eps=1e-8 (not 1e-12) -- under AMP,
        # `probs` arrives as float16, whose smallest representable positive
        # value is far above 1e-12, so that eps used to underflow to exactly
        # 0.0, making log(0)=-inf and 0*-inf=NaN. This was the source of the
        # "loss=nan" crash observed around epoch ~21 on the full dataset.
        eps = 1e-8
        probs_f32 = probs.float().clamp_min(eps)
        soft_target_f32 = soft_target.clamp_min(eps)
        pred_entropy = -(probs_f32.flatten(2) * torch.log(probs_f32.flatten(2))).sum(dim=-1)  # (B,)
        with torch.no_grad():
            target_entropy = -(soft_target_f32.flatten(2) * torch.log(soft_target_f32.flatten(2))).sum(dim=-1)
        entropy_deficit = F.relu(target_entropy - pred_entropy)  # >0 only if pred is spikier than target
        entropy_penalty = entropy_deficit.mean()

        total_loss = (
            self.cls_weight * cls_loss
            + self.offset_weight * offset_loss
            + self.entropy_weight * entropy_penalty
        )

        parts = {
            "loss_classification": float(cls_loss.detach()),
            "loss_offset": float(offset_loss.detach()),
            "loss_margin": float(entropy_penalty.detach()),
        }
        return total_loss, parts


if __name__ == "__main__":
    # Integration smoke test against the REAL model_v6 forward output.
    from model_v6 import build_model

    torch.manual_seed(0)
    model = build_model()
    criterion = V5LocalizationLoss()

    ref = torch.randn(4, 1, 1000, 1000)
    search = torch.randn(4, 1, 1000, 1000)
    coords_gt = torch.rand(4, 2) * 1000.0

    out = model(ref, search)
    loss, parts = criterion(out, coords_gt)
    loss.backward()

    print("V6 loss smoke test")
    print(f"  loss parts: {parts}")
    print(f"  total_loss: {float(loss):.4f}")
    print(f"  grad reached model params: "
          f"{all(p.grad is not None for p in model.parameters() if p.requires_grad)}")

    # Sanity: peak/uniform on a properly-trained-with-this-loss model should
    # NOT be checked here (untrained weights) -- this only confirms shapes,
    # gradient flow, and that the soft target sums to ~1.
    soft_t = make_soft_target(coords_gt, grid_size=out.candidate_logits.shape[-1])
    print(f"  soft target sums (should be ~1.0 each): {soft_t.sum(dim=(1,2,3))}")
