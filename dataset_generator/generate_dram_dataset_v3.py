"""
generate_dram_dataset_v3.py

VERSION 3 -- Synthetic DRAM SEM Dataset Generator for Wafer Navigation-Error
Recovery (Applied Materials "Drift-Sense" hackathon, Track 2).

DESIGN PHILOSOPHY
---------------------------------------------------------------------------
This is a ground-up redesign, not a merge, combining the strongest ideas
from two earlier generators:

  From "Generator A": modular multi-stage pipeline, clean docstrings,
  disciplined CSV/metadata logging, independent reference/search
  degradation.

  From "Generator B": the SINGLE VIRTUAL WORLD architecture -- one large
  physical layout is generated once; the reference and search images are
  both crops/renders of that SAME world at different scales. Nothing is
  ever pasted. Ground truth comes directly from the known crop
  coordinates, transformed through whatever geometric distortion is
  applied, so labels stay mathematically exact.

New in v3 (beyond both predecessors):
  - Domain randomization PROFILES (easy / medium / hard) for progressive
    training difficulty.
  - CORRELATED parameter sampling: a single latent "capture quality"
    value drives blur/noise/contrast together (poor capture quality
    pushes blur up AND noise up AND contrast down simultaneously), rather
    than sampling each independently -- matching how real degraded
    captures actually behave.
  - Fractal/multi-octave background illumination drift (cheap value-noise
    approximation, no external Perlin-noise dependency needed).
  - Scan-direction artifacts: row-wise drift and occasional
    stretched/skipped scan lines.
  - Multi-scale defect severity: common minor imperfections (line-width
    jitter, LER) are near-universal; rare-but-significant defects
    (broken lines, merged contacts, large contamination) occur at a much
    lower, independently-configured probability.
  - Perspective/lens-distortion tracked EXACTLY through a homography
    applied to both the image and the ground-truth point (elastic/local
    warps remain small, near-zero-mean, and are not point-tracked, same
    reasoning as prior generators: their effect on the labeled center is
    negligible next to the dominant tracked transform).
  - Parallel generation across pairs (ProcessPoolExecutor), resume
    support (skips already-generated indices), a progress bar, a JSON
    config dump, a text generation log, and a preview-grid image showing
    ground truth overlaid on a few sample pairs.

---------------------------------------------------------------------------
PHYSICAL SCALE (must match the official brief)
---------------------------------------------------------------------------
    - Both images: 1000 x 1000 px, grayscale
    - Reference:  1 nm/px  (100x magnification -> 1 um x 1 um FOV)
    - Search:    10 nm/px  (10x magnification  -> 10 um x 10 um FOV)
    - world units are chosen so that 1 world unit == 1 nm, making the
      REF_SCALE/SEARCH_SCALE ratio exactly 10, matching the brief's fixed
      10x magnification ratio between the two captures.

---------------------------------------------------------------------------
CITATIONS (for the 30%-weighted augmentation-realism score)
---------------------------------------------------------------------------
[C1] DRAM word-line/bit-line pitch: Micron "Inside 1-alpha DRAM" (2021);
     EDN "The 50-nm DRAM battle rages on" (2009) -- real historical
     half-pitches ~50-58nm -> full pitch ~50-120nm across generations.
[C2] Poisson (shot) + Gaussian (detector) noise as independent SEM noise
     sources: Villarrubia et al., JVST B 37(6) (2019); Bals et al., Adv.
     Intell. Syst. (2023) -- realistic Gaussian sigma ~0.01-0.1.
[C3] Edge brightening ("edge bloom"): US Patent 10,648,801 (closed-form
     linescan equation); JEOL Ltd. "edge effect" glossary.
[C4] Illumination gradient / vignetting / shading: Molecular Expressions
     Microscopy Primer, "Nonuniform Illumination" -- SEM shading from
     detector location and specimen tilt.
[C5] Line-edge roughness (LER) / line-width roughness (LWR): Bunday,
     Bishop, Villarrubia, Vladar, NIST/SPIE (2003); C. Mack, SPIE
     Optipedia -- LER ~5% of CD (3-sigma) typical for sub-100nm nodes.
[C6] Missing/broken contacts, voids, particles: US Patent 5,840,205
     (documented DRAM contact-open SEM photo); US Patents 6,989,583 and
     6,774,024 (via-void formation mechanisms).
[C7] Rotation/scale/drift as real navigation-error sources: US Patent
     9,619,727 (wafer coordinate misalignment, stage-accuracy limits).
[C8] Scan-line/raster acquisition artifacts: grouped with [C2]/[C4]
     general SEM acquisition-physics literature.

Usage:
    python generate_dram_dataset_v3.py --n_pairs 30 --out_dir ./eval_v3 \\
        --profile medium --seed 3

    python generate_dram_dataset_v3.py --n_pairs 3000 --out_dir ./train_v3 \\
        --profile medium --seed 1 --workers 6

Dependencies: numpy, opencv-python-headless, scipy, matplotlib (preview only)
    pip install numpy opencv-python-headless scipy matplotlib --break-system-packages
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import cv2

try:
    from scipy.ndimage import zoom as ndi_zoom
except ImportError:
    ndi_zoom = None


# ==========================================================================
# CONFIGURATION (dataclasses)
# ==========================================================================

@dataclass
class PhysicalScale:
    """Fixed physical scale, per the official brief. Do not randomize."""
    ref_size_px: int = 1000
    search_size_px: int = 1000
    ref_nm_per_px: float = 1.0
    search_nm_per_px: float = 10.0

    @property
    def mag_ratio(self) -> float:
        return self.search_nm_per_px / self.ref_nm_per_px  # = 10

    @property
    def ref_scale(self) -> float:
        """pixels per world-unit (world unit == 1 nm) at the reference capture."""
        return 1.0 / self.ref_nm_per_px

    @property
    def search_scale(self) -> float:
        """pixels per world-unit at the search capture."""
        return 1.0 / self.search_nm_per_px

    @property
    def world_size(self) -> float:
        """Full world extent (nm) -- the search image covers the whole world."""
        return self.search_size_px / self.search_scale

    @property
    def region_world_size(self) -> float:
        """World extent (nm) covered by the reference crop."""
        return self.ref_size_px / self.ref_scale


@dataclass
class ProfileParams:
    """
    One domain-randomization difficulty profile. Controls the RANGE each
    randomized parameter is drawn from. "hard" widens/shifts every range
    toward noisier/blurrier/more-defective conditions than "easy", so a
    model can be trained progressively (easy -> medium -> hard curriculum)
    or evaluated for robustness at a chosen difficulty.
    """
    name: str
    # structural (world-generation) ranges -- CITATION [C1]
    pitch_nm_range: tuple[float, float]
    line_width_nm_range: tuple[float, float]
    via_diameter_nm_range: tuple[float, float]
    # common (near-universal) minor-imperfection ranges -- CITATION [C5]
    ler_amp_fraction_range: tuple[float, float]
    via_diameter_jitter_range: tuple[float, float]
    # rare-but-significant defect probabilities -- CITATION [C6]
    missing_via_prob_range: tuple[float, float]
    broken_line_rate_range: tuple[float, float]      # breaks per 1000 world units
    merged_contact_prob_range: tuple[float, float]
    particle_rate_range: tuple[float, float]          # particles per 1000 world units
    # capture-quality latent (drives correlated blur/noise/contrast) -- CITATION [C2]
    quality_range_ref: tuple[float, float]    # 0 = pristine, 1 = worst
    quality_range_search: tuple[float, float]
    # geometric distortion ranges -- CITATION [C7]
    rotation_deg_range_ref: tuple[float, float]
    rotation_deg_range_search: tuple[float, float]
    scale_jitter_range_ref: tuple[float, float]
    scale_jitter_range_search: tuple[float, float]
    perspective_strength_range_search: tuple[float, float]
    elastic_alpha_range_search: tuple[float, float]
    # illumination / scan artifacts -- CITATION [C4]/[C8]
    vignette_strength_range: tuple[float, float]
    scanline_strength_range: tuple[float, float]
    fractal_bg_strength_range: tuple[float, float]


PROFILES: dict[str, ProfileParams] = {
    "easy": ProfileParams(
        name="easy",
        pitch_nm_range=(60, 100),
        line_width_nm_range=(8, 14),
        via_diameter_nm_range=(10, 18),
        ler_amp_fraction_range=(0.02, 0.05),
        via_diameter_jitter_range=(0.03, 0.10),
        missing_via_prob_range=(0.01, 0.04),
        broken_line_rate_range=(0.2, 0.6),
        merged_contact_prob_range=(0.0, 0.01),
        particle_rate_range=(0.2, 0.6),
        quality_range_ref=(0.0, 0.25),
        quality_range_search=(0.15, 0.45),
        rotation_deg_range_ref=(-0.5, 0.5),
        rotation_deg_range_search=(-1.5, 1.5),
        scale_jitter_range_ref=(-0.01, 0.01),
        scale_jitter_range_search=(-0.02, 0.02),
        perspective_strength_range_search=(0.0, 0.003),
        elastic_alpha_range_search=(0.5, 1.5),
        vignette_strength_range=(0.03, 0.10),
        scanline_strength_range=(0.005, 0.02),
        fractal_bg_strength_range=(0.02, 0.06),
    ),
    "medium": ProfileParams(
        name="medium",
        pitch_nm_range=(40, 120),
        line_width_nm_range=(6, 18),
        via_diameter_nm_range=(8, 22),
        ler_amp_fraction_range=(0.03, 0.08),
        via_diameter_jitter_range=(0.05, 0.20),
        missing_via_prob_range=(0.02, 0.08),
        broken_line_rate_range=(0.5, 1.5),
        merged_contact_prob_range=(0.005, 0.02),
        particle_rate_range=(0.5, 1.5),
        quality_range_ref=(0.0, 0.4),
        quality_range_search=(0.3, 0.7),
        rotation_deg_range_ref=(-1.0, 1.0),
        rotation_deg_range_search=(-3.0, 3.0),
        scale_jitter_range_ref=(-0.015, 0.015),
        scale_jitter_range_search=(-0.04, 0.04),
        perspective_strength_range_search=(0.0, 0.008),
        elastic_alpha_range_search=(1.0, 2.5),
        vignette_strength_range=(0.05, 0.18),
        scanline_strength_range=(0.01, 0.035),
        fractal_bg_strength_range=(0.04, 0.10),
    ),
    "hard": ProfileParams(
        name="hard",
        pitch_nm_range=(30, 140),
        line_width_nm_range=(5, 20),
        via_diameter_nm_range=(6, 24),
        ler_amp_fraction_range=(0.05, 0.12),
        via_diameter_jitter_range=(0.08, 0.30),
        missing_via_prob_range=(0.04, 0.14),
        broken_line_rate_range=(1.0, 2.5),
        merged_contact_prob_range=(0.01, 0.04),
        particle_rate_range=(1.0, 2.5),
        quality_range_ref=(0.0, 0.55),
        quality_range_search=(0.5, 1.0),
        rotation_deg_range_ref=(-1.5, 1.5),
        rotation_deg_range_search=(-5.0, 5.0),
        scale_jitter_range_ref=(-0.02, 0.02),
        scale_jitter_range_search=(-0.07, 0.07),
        perspective_strength_range_search=(0.0, 0.015),
        elastic_alpha_range_search=(1.5, 4.0),
        vignette_strength_range=(0.08, 0.28),
        scanline_strength_range=(0.02, 0.05),
        fractal_bg_strength_range=(0.06, 0.15),
    ),
}


# ==========================================================================
# Deterministic pseudo-random hashing (for jitter consistent across scales)
# ==========================================================================

def stable_hash01(i, j, salt):
    """
    Deterministic pseudo-random value in [0, 1), a function of (i, j, salt).
    Independent of pixel scale/crop origin -- the same via/line gets the
    same jitter whether rendered inside the reference crop or the full
    search field. Keeps local defects/irregularities physically consistent
    across the two magnifications (same principle as Generator B).
    """
    v = np.sin(i * 12.9898 + j * 78.233 + salt * 37.719) * 43758.5453
    return v - np.floor(v)


# ==========================================================================
# STAGE 1 + 2: Create World + Generate Physical Layout (spec only, no pixels)
# ==========================================================================

@dataclass
class LayoutSpec:
    pitch_x: float
    pitch_y: float
    line_width: float
    via_diameter: float
    phase_x: float
    phase_y: float
    missing_vias: set
    merged_contacts: set          # via coordinates rendered as an elongated blob
    broken_segments: list
    particles: list                # (x, y, radius, kind) kind in {"bright","dark"}
    ler_amp: float
    ler_freq: float
    via_jitter_amp: float
    salt_lw_v: float
    salt_lw_h: float
    salt_via: float


def create_world(rng: np.random.Generator, scale: PhysicalScale,
                  profile: ProfileParams) -> LayoutSpec:
    """
    STAGE 1+2: Randomly generate ONE DRAM layout specification covering the
    entire virtual world, in world units (== nm). All geometry is
    expressed independent of pixel scale, so the identical spec can be
    rendered at either magnification later (STAGE 5) -- this is what
    guarantees the reference and search images show the SAME physical
    structure, per Generator B's core idea.

    CITATION: pitch/line-width/via-diameter ranges [C1]; LER/via-jitter
    [C5]; missing-via/merged-contact/broken-line/particle rates [C6].
    """
    world = scale.world_size

    pitch_x = rng.uniform(*profile.pitch_nm_range)
    pitch_y = rng.uniform(*profile.pitch_nm_range)
    line_width = rng.uniform(*profile.line_width_nm_range)
    via_diameter = rng.uniform(*profile.via_diameter_nm_range)
    phase_x = rng.uniform(0, pitch_x)
    phase_y = rng.uniform(0, pitch_y)

    n_cols = int(world / pitch_x) + 4
    n_rows = int(world / pitch_y) + 4

    miss_prob = rng.uniform(*profile.missing_via_prob_range)
    merge_prob = rng.uniform(*profile.merged_contact_prob_range)
    missing_vias, merged_contacts = set(), set()
    for i in range(-2, n_cols):
        for j in range(-2, n_rows):
            roll = rng.random()
            if roll < miss_prob:
                missing_vias.add((i, j))
            elif roll < miss_prob + merge_prob:
                merged_contacts.add((i, j))

    n_breaks = max(0, int(world / 1000 * rng.uniform(*profile.broken_line_rate_range)))
    broken_segments = []
    for _ in range(n_breaks):
        orientation = rng.choice(["h", "v"])
        gap_len = rng.uniform(3, 10)
        along = rng.uniform(0, world)
        cross = rng.uniform(0, world)
        half = gap_len / 2
        if orientation == "h":
            rect = (along - half, cross - line_width, along + half, cross + line_width)
        else:
            rect = (cross - line_width, along - half, cross + line_width, along + half)
        broken_segments.append(rect)

    n_particles = max(0, int(world / 1000 * rng.uniform(*profile.particle_rate_range)))
    particles = []
    for _ in range(n_particles):
        dx, dy = rng.uniform(0, world), rng.uniform(0, world)
        # multi-scale severity: mostly small, occasionally large (rare-but-significant)
        radius = rng.uniform(1.5, 4.0) if rng.random() > 0.15 else rng.uniform(4.0, 9.0)
        kind = rng.choice(["bright", "dark"])
        particles.append((dx, dy, radius, kind))

    return LayoutSpec(
        pitch_x=pitch_x, pitch_y=pitch_y, line_width=line_width,
        via_diameter=via_diameter, phase_x=phase_x, phase_y=phase_y,
        missing_vias=missing_vias, merged_contacts=merged_contacts,
        broken_segments=broken_segments, particles=particles,
        ler_amp=rng.uniform(*profile.ler_amp_fraction_range),
        ler_freq=rng.uniform(0.08, 0.30),
        via_jitter_amp=rng.uniform(*profile.via_diameter_jitter_range),
        salt_lw_v=rng.uniform(0, 1000), salt_lw_h=rng.uniform(0, 1000),
        salt_via=rng.uniform(0, 1000),
    )


# ==========================================================================
# STAGE 3: Apply Manufacturing Variations (rasterize spec -> clean image)
# ==========================================================================

def render_layout(spec: LayoutSpec, px_per_nm: float, origin: tuple[float, float],
                   out_size: tuple[int, int]) -> np.ndarray:
    """
    STAGE 3: Rasterize a sub-region of the world spec into a clean
    (pre-SEM-imaging) grayscale image, including all manufacturing
    variation: line-edge roughness, per-line width jitter, missing/merged
    vias, broken segments, and particle defects.

    CITATION: LER sinusoidal width modulation [C5]; defect rendering [C6].
    """
    w, h = out_size
    ox, oy = origin
    pitch_x, pitch_y = spec.pitch_x, spec.pitch_y
    lw = spec.line_width

    ys, xs = np.indices((h, w), dtype=np.float32)
    world_x = ox + xs / px_per_nm
    world_y = oy + ys / px_per_nm

    mod_x = np.mod(world_x - spec.phase_x, pitch_x)
    mod_y = np.mod(world_y - spec.phase_y, pitch_y)

    line_idx_x = np.round((world_x - spec.phase_x) / pitch_x)
    phase_x = stable_hash01(line_idx_x, 0.0, spec.salt_lw_v) * 2 * np.pi
    width_v = lw * (1.0 + spec.ler_amp * np.sin(spec.ler_freq * world_y + phase_x))
    width_v = np.clip(width_v, 0.4 * lw, 1.6 * lw)

    line_idx_y = np.round((world_y - spec.phase_y) / pitch_y)
    phase_y = stable_hash01(0.0, line_idx_y, spec.salt_lw_h) * 2 * np.pi
    width_h = lw * (1.0 + spec.ler_amp * np.sin(spec.ler_freq * world_x + phase_y))
    width_h = np.clip(width_h, 0.4 * lw, 1.6 * lw)

    mask_v = mod_x < width_v
    mask_h = mod_y < width_h

    img = np.zeros((h, w), dtype=np.float32)
    img[mask_h] = 190.0
    img[mask_v] = 190.0

    for (x0, y0, x1, y1) in spec.broken_segments:
        px0, py0 = int(round((x0 - ox) * px_per_nm)), int(round((y0 - oy) * px_per_nm))
        px1, py1 = int(round((x1 - ox) * px_per_nm)), int(round((y1 - oy) * px_per_nm))
        if px1 < 0 or py1 < 0 or px0 > w or py0 > h:
            continue
        cv2.rectangle(img, (px0, py0), (px1, py1), 0.0, -1)

    i_min = int(np.floor((ox - spec.phase_x) / pitch_x)) - 1
    i_max = int(np.ceil((ox + w / px_per_nm - spec.phase_x) / pitch_x)) + 1
    j_min = int(np.floor((oy - spec.phase_y) / pitch_y)) - 1
    j_max = int(np.ceil((oy + h / px_per_nm - spec.phase_y) / pitch_y)) + 1

    base_r = spec.via_diameter / 2.0
    for i in range(i_min, i_max):
        wx = spec.phase_x + i * pitch_x
        px = int(round((wx - ox) * px_per_nm))
        if px < -int(base_r * px_per_nm) - 2 or px > w + int(base_r * px_per_nm) + 2:
            continue
        for j in range(j_min, j_max):
            if (i, j) in spec.missing_vias:
                continue
            wy = spec.phase_y + j * pitch_y
            py = int(round((wy - oy) * px_per_nm))
            if py < -int(base_r * px_per_nm) - 2 or py > h + int(base_r * px_per_nm) + 2:
                continue
            jitter = (stable_hash01(i, j, spec.salt_via) - 0.5) * 2.0 * spec.via_jitter_amp
            r_local = base_r * (1.0 + jitter)
            r_px = max(1, int(round(r_local * px_per_nm)))
            if (i, j) in spec.merged_contacts:
                # elongated blob simulating two contacts merged together
                cv2.ellipse(img, (px, py), (r_px * 2, r_px), 0, 0, 360, 235.0, -1)
            else:
                cv2.circle(img, (px, py), r_px, 235.0, -1)

    for (dx, dy, radius, kind) in spec.particles:
        px, py = int(round((dx - ox) * px_per_nm)), int(round((dy - oy) * px_per_nm))
        rr = max(1, int(round(radius * px_per_nm)))
        if px < -rr or px > w + rr or py < -rr or py > h + rr:
            continue
        color = 255.0 if kind == "bright" else 15.0
        cv2.circle(img, (px, py), rr, color, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


# ==========================================================================
# STAGE 4: Apply SEM Imaging Model (noise, blur, illumination, scan artifacts)
# ==========================================================================

def _fractal_field(h: int, w: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """
    Cheap multi-octave value-noise approximation to Perlin/fractal noise:
    sums several octaves of coarse random fields (each upsampled + blurred)
    at increasing frequency / decreasing amplitude. Used for slow SEM
    illumination drift across the field of view. No external Perlin-noise
    dependency required.
    """
    field = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for o in range(octaves):
        res = max(2, 3 * (2 ** o))
        low = rng.uniform(-1, 1, (res, res)).astype(np.float32)
        up = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), sigmaX=max(w, h) / (4.0 * (o + 1)))
        field += amplitude * up
        total_amp += amplitude
        amplitude *= 0.5
    field /= (total_amp + 1e-8)
    field /= (np.max(np.abs(field)) + 1e-8)
    return field


def sample_capture_quality_params(quality: float, rng: np.random.Generator) -> dict:
    """
    CORRELATED parameter sampling: a single latent `quality` in [0, 1]
    (0 = pristine capture, 1 = worst-case capture) drives blur, noise, and
    contrast TOGETHER, with a small amount of independent jitter layered
    on top -- matching how a real degraded SEM capture behaves (a noisy
    capture is usually also blurrier and lower-contrast, not independently
    randomized on each axis). CITATION: [C2] for the noise-model grounding.
    """
    jitter = lambda scale: rng.uniform(-scale, scale)
    blur_sigma = 0.3 + quality * 1.6 + jitter(0.15)
    gauss_sigma = 1.0 + quality * 9.5 + jitter(0.5)
    poisson_peak = 110.0 - quality * 90.0 + jitter(5.0)
    contrast = 1.05 - quality * 0.30 + jitter(0.03)
    brightness = jitter(6.0 + quality * 14.0)
    return {
        "blur_sigma": max(0.1, blur_sigma),
        "gauss_sigma": max(0.5, gauss_sigma),
        "poisson_peak": max(8.0, poisson_peak),
        "contrast": np.clip(contrast, 0.55, 1.15),
        "brightness": brightness,
    }


def apply_sem_imaging_model(img: np.ndarray, rng: np.random.Generator,
                             quality: float, vignette_strength: float,
                             scanline_strength: float,
                             fractal_bg_strength: float) -> tuple[np.ndarray, dict]:
    """
    STAGE 4: Apply the full SEM imaging model to a clean rendered image:
    correlated blur/noise/contrast (via capture-quality latent), edge
    brightening, vignetting/shading, fractal illumination drift, and
    scan-line (row-wise) intensity variation.

    CITATION: edge brightening [C3]; vignetting/shading [C4]; Poisson +
    Gaussian noise [C2]; scan-line artifacts [C8].
    """
    params = sample_capture_quality_params(quality, rng)
    img_f = img.astype(np.float32)

    # edge brightening [C3]
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6) * 255.0
    img_f = img_f + rng.uniform(0.05, 0.20) * grad

    # blur (correlated with quality)
    ksize = max(1, int(params["blur_sigma"] * 3) | 1)
    img_f = cv2.GaussianBlur(img_f, (ksize, ksize), params["blur_sigma"])

    # vignetting / shading [C4]
    h, w = img_f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= h
    xx /= w
    angle = rng.uniform(0, 2 * np.pi)
    grad_field = np.cos(angle) * (xx - 0.5) + np.sin(angle) * (yy - 0.5)
    grad_field /= (np.abs(grad_field).max() + 1e-8)
    shading = 1.0 - vignette_strength * (grad_field * 0.5 + 0.5)
    img_f = img_f * shading

    # fractal/multi-octave illumination drift
    fractal = _fractal_field(h, w, rng)
    img_f = img_f + fractal_bg_strength * 255.0 * fractal

    # scan-line (row-wise) intensity variation [C8]
    row_noise = rng.normal(0, 1.0, (h,)).astype(np.float32)
    row_noise = np.convolve(row_noise, [0.25, 0.5, 0.25], mode="same").reshape(h, 1)
    img_f = img_f * (1.0 + scanline_strength * row_noise)

    # occasional stretched/skipped scan line (rare, localized artifact)
    if rng.random() < 0.15:
        row = rng.integers(0, h)
        img_f[row, :] = img_f[max(0, row - 1), :]  # duplicate row = "skipped" line look

    # contrast/brightness (correlated with quality)
    img_f = img_f * params["contrast"] + params["brightness"]

    # detector noise: Poisson (shot) then Gaussian (readout) [C2]
    img_norm = np.clip(img_f, 0, 255) / 255.0
    img_norm = rng.poisson(img_norm * params["poisson_peak"]) / params["poisson_peak"]
    img_f = img_norm * 255.0 + rng.normal(0, params["gauss_sigma"], img_f.shape)

    return np.clip(img_f, 0, 255).astype(np.uint8), params


# ==========================================================================
# STAGE 5: Extract Reference and Search (geometric distortion, EXACT tracking)
# ==========================================================================

def apply_tracked_geometric_distortion(
    img: np.ndarray, rng: np.random.Generator, point: Optional[tuple[float, float]],
    rotation_range: tuple[float, float], scale_range: tuple[float, float],
    perspective_strength_range: tuple[float, float],
) -> tuple[np.ndarray, Optional[tuple[float, float]], dict]:
    """
    Applies rotation + scale + a small perspective warp as ONE combined
    homography about the image center. If `point` is given, it is
    transformed through the EXACT SAME homography, so the ground-truth
    label remains mathematically exact through this (the dominant)
    geometric distortion -- this is the key correctness property carried
    over from Generator B.

    CITATION: [C7] -- rotation/scale here directly model wafer coordinate
    misalignment and stage-accuracy limits as real navigation-error
    sources; the small perspective term approximates minor lens/viewing-
    angle distortion.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(*rotation_range)
    scale = 1.0 + rng.uniform(*scale_range) if isinstance(scale_range[0], float) and abs(scale_range[0]) < 1 \
        else rng.uniform(*scale_range)
    center = (w / 2.0, h / 2.0)

    M_affine = cv2.getRotationMatrix2D(center, angle, scale)
    M = np.vstack([M_affine, [0, 0, 1]]).astype(np.float64)

    persp_strength = rng.uniform(*perspective_strength_range)
    if persp_strength > 1e-9:
        # small random perspective term, applied about the same center
        P = np.eye(3)
        P[2, 0] = rng.uniform(-persp_strength, persp_strength) / w
        P[2, 1] = rng.uniform(-persp_strength, persp_strength) / h
        T1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
        T2 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
        M = T2 @ P @ T1 @ M

    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)

    new_point = None
    if point is not None:
        vec = np.array([point[0], point[1], 1.0])
        proj = M @ vec
        new_point = (float(proj[0] / proj[2]), float(proj[1] / proj[2]))

    return out, new_point, {"rotation_deg": angle, "scale_factor": scale,
                             "perspective_strength": persp_strength}


def apply_local_elastic_warp(img: np.ndarray, rng: np.random.Generator,
                              alpha: float, sigma: float = 15.0) -> np.ndarray:
    """
    Small, LOCAL, near-zero-mean elastic deformation simulating tiny
    stage/scan non-linearity and lens distortion residuals. Deliberately
    NOT applied to the tracked ground-truth point: its magnitude is small
    relative to the dominant tracked homography above, and its zero-mean,
    locally-varying nature means it does not systematically bias the
    labeled center (same reasoning used in the prior generator this
    design builds on).
    """
    h, w = img.shape[:2]
    low_res = max(4, int(min(h, w) / 20))
    dx = cv2.resize(rng.uniform(-1, 1, (low_res, low_res)).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.uniform(-1, 1, (low_res, low_res)).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_CUBIC)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma) * alpha

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = (gx + dx).astype(np.float32), (gy + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT101)


# ==========================================================================
# STAGE 6 + 7: Generate one pair (labels) and save (export)
# ==========================================================================

def generate_pair(idx: int, seed: int, scale: PhysicalScale,
                   profile: ProfileParams) -> tuple[np.ndarray, np.ndarray, dict]:
    """Builds one (reference, search, label) sample from a single virtual world."""
    rng = np.random.default_rng(seed)

    spec = create_world(rng, scale, profile)

    margin = scale.region_world_size * 0.3
    max_coord = scale.world_size - scale.region_world_size - margin
    x0 = rng.uniform(margin, max_coord)
    y0 = rng.uniform(margin, max_coord)
    true_center_world = (x0 + scale.region_world_size / 2.0, y0 + scale.region_world_size / 2.0)

    ref_clean = render_layout(spec, scale.ref_scale, (x0, y0),
                               (scale.ref_size_px, scale.ref_size_px))
    search_clean = render_layout(spec, scale.search_scale, (0.0, 0.0),
                                  (scale.search_size_px, scale.search_size_px))

    true_center_px = (true_center_world[0] * scale.search_scale,
                       true_center_world[1] * scale.search_scale)

    ref_quality = rng.uniform(*profile.quality_range_ref)
    search_quality = rng.uniform(*profile.quality_range_search)
    vignette = rng.uniform(*profile.vignette_strength_range)
    scanline = rng.uniform(*profile.scanline_strength_range)
    fractal_bg = rng.uniform(*profile.fractal_bg_strength_range)

    ref_img, ref_imaging_params = apply_sem_imaging_model(
        ref_clean, rng, ref_quality, vignette * 0.5, scanline * 0.5, fractal_bg * 0.5)
    search_img, search_imaging_params = apply_sem_imaging_model(
        search_clean, rng, search_quality, vignette, scanline, fractal_bg)

    ref_img, _, ref_geo = apply_tracked_geometric_distortion(
        ref_img, rng, None, profile.rotation_deg_range_ref,
        profile.scale_jitter_range_ref, (0.0, 0.0))

    search_img, gt_center, search_geo = apply_tracked_geometric_distortion(
        search_img, rng, true_center_px, profile.rotation_deg_range_search,
        profile.scale_jitter_range_search, profile.perspective_strength_range_search)

    elastic_alpha_search = rng.uniform(*profile.elastic_alpha_range_search)
    search_img = apply_local_elastic_warp(search_img, rng, elastic_alpha_search)

    label = {
        "pair_id": idx,
        "reference_file": f"ref_{idx:05d}.png",
        "search_file": f"search_{idx:05d}.png",
        "center_x": round(gt_center[0], 3),
        "center_y": round(gt_center[1], 3),
        "world_center_x": round(true_center_world[0], 3),
        "world_center_y": round(true_center_world[1], 3),
        "crop_x0": round(x0, 3),
        "crop_y0": round(y0, 3),
        "pitch_x_nm": round(spec.pitch_x, 3),
        "pitch_y_nm": round(spec.pitch_y, 3),
        "line_width_nm": round(spec.line_width, 3),
        "via_diameter_nm": round(spec.via_diameter, 3),
        "n_missing_vias": len(spec.missing_vias),
        "n_merged_contacts": len(spec.merged_contacts),
        "n_broken_segments": len(spec.broken_segments),
        "n_particles": len(spec.particles),
        "ref_quality": round(ref_quality, 4),
        "search_quality": round(search_quality, 4),
        "ref_rotation_deg": round(ref_geo["rotation_deg"], 4),
        "search_rotation_deg": round(search_geo["rotation_deg"], 4),
        "ref_scale_factor": round(ref_geo["scale_factor"], 5),
        "search_scale_factor": round(search_geo["scale_factor"], 5),
        "search_perspective_strength": round(search_geo["perspective_strength"], 6),
        "elastic_alpha_search": round(elastic_alpha_search, 4),
        "vignette_strength": round(vignette, 4),
        "scanline_strength": round(scanline, 4),
        "fractal_bg_strength": round(fractal_bg, 4),
        "profile": profile.name,
        "seed": seed,
    }
    return ref_img, search_img, label


def _generate_and_save(args):
    """Top-level worker function (must be importable/picklable for ProcessPoolExecutor)."""
    idx, seed, scale, profile, out_dir = args
    ref_img, search_img, label = generate_pair(idx, seed, scale, profile)
    ref_path = os.path.join(out_dir, "reference", label["reference_file"])
    search_path = os.path.join(out_dir, "search", label["search_file"])
    cv2.imwrite(ref_path, ref_img)
    cv2.imwrite(search_path, search_img)
    return label


# ==========================================================================
# Dataset export orchestration (parallel, resumable, logged)
# ==========================================================================

def save_dataset(n_pairs: int, out_dir: str, seed: int, profile_name: str,
                  workers: int = 1, pitch_override: tuple[float, float] | None = None) -> None:
    scale = PhysicalScale()
    profile = PROFILES[profile_name]
    if pitch_override is not None:
        # dataclasses.replace returns a NEW ProfileParams instance -- the
        # PROFILES dict preset is never mutated, so normal --profile runs
        # are unaffected. Only pitch_nm_range changes; every other medium-
        # profile parameter (quality, noise, defect rates, geometry, etc.)
        # stays identical to the train_v7 baseline generation, so pitch is
        # the only variable that differs between the two datasets.
        profile = replace(profile, pitch_nm_range=pitch_override)

    ref_dir = os.path.join(out_dir, "reference")
    search_dir = os.path.join(out_dir, "search")
    preview_dir = os.path.join(out_dir, "preview")
    for d in (ref_dir, search_dir, preview_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(out_dir, "generation_log.txt")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("dram_v3")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"physical_scale": asdict(scale), "profile": asdict(profile),
                    "n_pairs": n_pairs, "seed": seed}, f, indent=2)

    csv_path = os.path.join(out_dir, "labels.csv")

    # --- resume support: skip indices whose PNGs already exist ---
    done_indices = set()
    for i in range(n_pairs):
        rp = os.path.join(ref_dir, f"ref_{i:05d}.png")
        sp = os.path.join(search_dir, f"search_{i:05d}.png")
        if os.path.exists(rp) and os.path.exists(sp):
            done_indices.add(i)
    existing_rows = []
    if done_indices and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_rows = [r for r in existing_rows if int(r["pair_id"]) in done_indices]
    if done_indices:
        logger.info(f"Resuming: {len(done_indices)}/{n_pairs} pairs already exist, skipping them.")

    todo = [i for i in range(n_pairs) if i not in done_indices]
    tasks = [(i, seed * 1_000_003 + i, scale, profile, out_dir) for i in todo]

    new_rows = []
    interrupted = {"flag": False}

    def _handle_sigint(signum, frame):
        interrupted["flag"] = True
        logger.warning("Interrupt received -- finishing in-flight pairs, then stopping "
                        "gracefully. Re-run the same command to resume.")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    t_start = time.time()
    try:
        if workers <= 1:
            for n, task in enumerate(tasks):
                label = _generate_and_save(task)
                new_rows.append(label)
                _progress(n + 1, len(tasks), t_start, logger)
                if interrupted["flag"]:
                    break
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_generate_and_save, task): task for task in tasks}
                n = 0
                for fut in as_completed(futures):
                    new_rows.append(fut.result())
                    n += 1
                    _progress(n, len(tasks), t_start, logger)
                    if interrupted["flag"]:
                        for f2 in futures:
                            f2.cancel()
                        break
    finally:
        signal.signal(signal.SIGINT, old_handler)

    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda r: int(r["pair_id"]))
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    logger.info(f"Done. {len(all_rows)}/{n_pairs} pairs present in '{out_dir}'.")
    if len(all_rows) >= 3:
        _save_preview_grid(out_dir, all_rows[:3], scale)
        logger.info(f"Preview grid saved to {os.path.join(out_dir, 'preview')}")


def _progress(n, total, t_start, logger, every=25):
    if n % every == 0 or n == total:
        elapsed = time.time() - t_start
        rate = n / elapsed if elapsed > 0 else 0
        eta = (total - n) / rate if rate > 0 else float("nan")
        logger.info(f"[{n}/{total}] {rate:.2f} pairs/sec, ETA {eta/60:.1f} min")


def _save_preview_grid(out_dir: str, rows: list[dict], scale: PhysicalScale) -> None:
    """STAGE 7 (validation): save reference/search pairs with ground-truth
    center + bounding box overlaid, for a quick visual sanity check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for row, (ax_ref, ax_search) in zip(rows, axes):
        ref_img = cv2.imread(os.path.join(out_dir, "reference", row["reference_file"]),
                              cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join(out_dir, "search", row["search_file"]),
                                 cv2.IMREAD_GRAYSCALE)
        ax_ref.imshow(ref_img, cmap="gray")
        ax_ref.set_title(f"Reference (pair {row['pair_id']})")
        ax_ref.axis("off")

        ax_search.imshow(search_img, cmap="gray")
        cx, cy = float(row["center_x"]), float(row["center_y"])
        half = scale.region_world_size * scale.search_scale / 2.0
        ax_search.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                                           fill=False, edgecolor="lime", linewidth=1.5))
        ax_search.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
        ax_search.set_title(f"Search (ground truth: {cx:.1f}, {cy:.1f})")
        ax_search.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preview", "preview_grid.png"), dpi=110)
    plt.close(fig)


# ==========================================================================
# Entry point
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V3 DRAM synthetic dataset generator (single virtual "
                     "world architecture, domain-randomization profiles).")
    parser.add_argument("--n_pairs", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="./dram_dataset_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=str, default="medium",
                         choices=list(PROFILES.keys()),
                         help="Domain-randomization difficulty profile.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel worker processes (default: 1, sequential).")
    parser.add_argument("--pitch_min", type=float, default=None,
                         help="Override the profile's pitch_nm_range minimum (nm). "
                              "Must be used together with --pitch_max. Every other "
                              "profile parameter (quality/noise/defects/geometry) is "
                              "left unchanged, so only pitch differs from a normal run.")
    parser.add_argument("--pitch_max", type=float, default=None,
                         help="Override the profile's pitch_nm_range maximum (nm). "
                              "Must be used together with --pitch_min.")
    parser.add_argument("--style", type=str, default="dram", choices=["dram", "finfet"],
                         help="Die architecture style. Only 'dram' is currently implemented "
                              "(the chosen architecture per the brief's 'participant's choice, "
                              "judged equally either way' clause). Passing 'finfet' raises a "
                              "clear NotImplementedError rather than silently generating DRAM "
                              "layouts under a mismatched label.")
    args = parser.parse_args()

    if (args.pitch_min is None) != (args.pitch_max is None):
        parser.error("--pitch_min and --pitch_max must be provided together.")

    pitch_override = None
    if args.pitch_min is not None:
        if args.pitch_min >= args.pitch_max:
            parser.error(f"--pitch_min ({args.pitch_min}) must be < --pitch_max ({args.pitch_max}).")
        pitch_override = (args.pitch_min, args.pitch_max)
        print(f"Pitch range OVERRIDDEN to ({args.pitch_min}, {args.pitch_max}) nm -- "
              f"all other '{args.profile}' profile parameters unchanged.")

    if args.style == "finfet":
        parser.error(
            "--style finfet is not implemented. This project's dataset generator "
            "implements the DRAM-style architecture only, per the brief's explicit "
            "'participant's choice, judged equally either way' clause (Problem "
            "Statement, slide 6). Use --style dram (the default)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    save_dataset(args.n_pairs, args.out_dir, args.seed, args.profile, args.workers, pitch_override)


if __name__ == "__main__":
    main()
