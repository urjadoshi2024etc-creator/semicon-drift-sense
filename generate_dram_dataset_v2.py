"""
Synthetic DRAM SEM Dataset Generator for Wafer Navigation-Error Recovery
=========================================================================
(Applied Materials "Drift-Sense" hackathon style dataset)

SINGLE-WORLD PIPELINE
----------------------
Earlier versions of this generator built a reference layout and a
*separate* background layout, then pasted a downsampled copy of the
reference into that background. That made the reference region trivially
visible (a different local statistic pasted on top of unrelated
statistics), which is not how a real wafer-navigation problem behaves.

This version generates a SINGLE, large, continuous DRAM world:

    1. One layout spec (pitch, phase, missing vias, broken lines,
       bright/dark defects, line-width jitter, via-diameter jitter) is
       drawn for the ENTIRE world extent.
    2. A random physical sub-region of that world is chosen.
    3. That sub-region is rendered at 100x magnification -> REFERENCE.
    4. The ENTIRE world is rendered at 10x magnification -> SEARCH.
    5. The ground-truth center is simply the chosen sub-region's center,
       expressed directly in world/search pixel coordinates.

Nothing is pasted. The reference physically, naturally exists inside the
search image because both are different renderings of the same world.
The only way to relocate it is to match the local pattern of defects and
structural irregularities -- exactly like the real problem.

Output layout:
    train/reference/ref_XXX.png
    train/search/search_XXX.png
    labels.csv

Run:
    python generate_dram_dataset.py --n_pairs 30 --out_dir train --seed 0
"""

import os
import csv
import argparse

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REF_SIZE = 1000            # reference image size (px), 100x magnification
SEARCH_SIZE = 1000         # search image size (px), 10x magnification
MAG_RATIO = 10             # 100x / 10x
REF_SCALE = 10.0           # pixels-per-world-unit at 100x
SEARCH_SCALE = REF_SCALE / MAG_RATIO   # pixels-per-world-unit at 10x -> 1.0

# The search image covers the whole world at SEARCH_SCALE, so the world
# extent (in world units) equals the search image size in pixels.
WORLD_SIZE = SEARCH_SIZE / SEARCH_SCALE          # = 1000 world units

# Physical extent (in world units) covered by the reference crop.
REGION_WORLD_SIZE = REF_SIZE / REF_SCALE          # = 100 world units

# Keep the chosen reference region away from the world edges so that
# rotation/scale/elastic augmentation never needs to sample outside the
# rendered area.
REGION_MARGIN = 25.0


# --------------------------------------------------------------------------- #
# Deterministic pseudo-random hashing (for jitter that must stay consistent
# across different render scales / crops of the SAME world)
# --------------------------------------------------------------------------- #

def stable_hash01(i, j, salt):
    """
    Deterministic pseudo-random value in [0, 1), a function of integer/float
    grid coordinates (i, j) and a salt. Works on scalars or numpy arrays.
    Because it depends only on (i, j, salt) -- not on pixel scale or crop
    origin -- the same via / line gets the same jitter whether it is
    rendered inside the reference crop (100x) or inside the full search
    field (10x). This is what keeps local defects/irregularities consistent
    across magnifications.
    """
    v = np.sin(i * 12.9898 + j * 78.233 + salt * 37.719) * 43758.5453
    return v - np.floor(v)


# --------------------------------------------------------------------------- #
# 1. Structure generation
# --------------------------------------------------------------------------- #

def make_layout_spec(rng, world_size):
    """
    Randomly generate ONE DRAM cell-array layout specification that covers
    the entire world. The spec describes a periodic grid of horizontal word
    lines and vertical bit lines with contact vias at every intersection,
    plus randomized defects: missing vias, broken line segments, local blob
    defects (particles/voids), slight per-via diameter jitter and slight
    per-line width jitter (subtle line-edge roughness). All geometry is
    expressed in "world units" so the exact same spec can be rendered at
    any pixel scale / crop -- this is what keeps the reference and the
    matching region of the search image physically identical.

    Parameters
    ----------
    rng : numpy.random.Generator
    world_size : float
        Full extent (in world units) of the DRAM world.

    Returns
    -------
    dict
    """
    pitch_x = rng.uniform(8.0, 16.0)
    pitch_y = rng.uniform(8.0, 16.0)
    line_width = rng.uniform(2.0, 4.0)
    via_diameter = rng.uniform(3.0, 6.0)
    phase_x = rng.uniform(0, pitch_x)
    phase_y = rng.uniform(0, pitch_y)

    n_cols = int(world_size / pitch_x) + 4
    n_rows = int(world_size / pitch_y) + 4

    # --- missing vias (random dropout across the intersection grid) ---
    miss_frac = rng.uniform(0.03, 0.12)
    missing_vias = set()
    for i in range(-2, n_cols):
        for j in range(-2, n_rows):
            if rng.random() < miss_frac:
                missing_vias.add((i, j))

    # --- broken line segments: small rectangular gaps cut into lines ---
    n_breaks = max(1, int(world_size / 40 * rng.uniform(0.5, 1.5)))
    broken_segments = []
    for _ in range(n_breaks):
        orientation = rng.choice(["h", "v"])
        gap_len = rng.uniform(3, 8)
        along = rng.uniform(0, world_size)
        cross = rng.uniform(0, world_size)
        half = gap_len / 2
        if orientation == "h":
            rect = (along - half, cross - line_width, along + half, cross + line_width)
        else:
            rect = (cross - line_width, along - half, cross + line_width, along + half)
        broken_segments.append(rect)

    # --- local blob defects (particles / voids) ---
    n_defects = max(1, int(world_size / 35 * rng.uniform(0.5, 1.5)))
    defects = []
    for _ in range(n_defects):
        dx = rng.uniform(0, world_size)
        dy = rng.uniform(0, world_size)
        radius = rng.uniform(1.5, 4.0)
        kind = rng.choice(["bright", "dark"])
        defects.append((dx, dy, radius, kind))

    # --- subtle per-line width jitter & per-via diameter jitter params ---
    line_width_jitter_amp = rng.uniform(0.05, 0.15)
    line_width_jitter_freq = rng.uniform(0.10, 0.30)
    via_diameter_jitter_amp = rng.uniform(0.05, 0.20)
    salt_lw_v = rng.uniform(0, 1000)
    salt_lw_h = rng.uniform(0, 1000)
    salt_via = rng.uniform(0, 1000)

    return {
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "line_width": line_width,
        "via_diameter": via_diameter,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "missing_vias": missing_vias,
        "broken_segments": broken_segments,
        "defects": defects,
        "line_width_jitter_amp": line_width_jitter_amp,
        "line_width_jitter_freq": line_width_jitter_freq,
        "via_diameter_jitter_amp": via_diameter_jitter_amp,
        "salt_lw_v": salt_lw_v,
        "salt_lw_h": salt_lw_h,
        "salt_via": salt_via,
    }


def render_layout(spec, scale, origin, out_size):
    """
    Rasterize a (sub-region of a) layout spec into a grayscale image.

    Parameters
    ----------
    spec : dict from make_layout_spec
    scale : float
        Pixels per world-unit (i.e. magnification-equivalent resolution).
    origin : (float, float)
        World-space (x, y) coordinate of the image's top-left pixel.
    out_size : (int, int)
        (width, height) of the rendered image in pixels.

    Returns
    -------
    np.ndarray, dtype uint8, shape (H, W)
    """
    w, h = out_size
    ox, oy = origin
    pitch_x, pitch_y = spec["pitch_x"], spec["pitch_y"]
    lw = spec["line_width"]
    phase_x, phase_y = spec["phase_x"], spec["phase_y"]
    lw_amp = spec["line_width_jitter_amp"]
    lw_freq = spec["line_width_jitter_freq"]

    ys, xs = np.indices((h, w), dtype=np.float32)
    world_x = ox + xs / scale
    world_y = oy + ys / scale

    mod_x = np.mod(world_x - phase_x, pitch_x)
    mod_y = np.mod(world_y - phase_y, pitch_y)

    # --- subtle per-line width jitter (line-edge roughness) ---
    line_index_x = np.round((world_x - phase_x) / pitch_x)
    phase_line_x = stable_hash01(line_index_x, 0.0, spec["salt_lw_v"]) * 2 * np.pi
    width_local_v = lw * (1.0 + lw_amp * np.sin(lw_freq * world_y + phase_line_x))
    width_local_v = np.clip(width_local_v, 0.4 * lw, 1.6 * lw)

    line_index_y = np.round((world_y - phase_y) / pitch_y)
    phase_line_y = stable_hash01(0.0, line_index_y, spec["salt_lw_h"]) * 2 * np.pi
    width_local_h = lw * (1.0 + lw_amp * np.sin(lw_freq * world_x + phase_line_y))
    width_local_h = np.clip(width_local_h, 0.4 * lw, 1.6 * lw)

    mask_v = mod_x < width_local_v   # vertical bit lines
    mask_h = mod_y < width_local_h   # horizontal word lines

    img = np.zeros((h, w), dtype=np.float32)
    img[mask_h] = 190.0
    img[mask_v] = 190.0

    # --- carve out broken-line gaps ---
    for (x0, y0, x1, y1) in spec["broken_segments"]:
        px0 = int(round((x0 - ox) * scale))
        py0 = int(round((y0 - oy) * scale))
        px1 = int(round((x1 - ox) * scale))
        py1 = int(round((y1 - oy) * scale))
        if px1 < 0 or py1 < 0 or px0 > w or py0 > h:
            continue
        cv2.rectangle(img, (px0, py0), (px1, py1), 0.0, -1)

    # --- contact vias at every surviving grid intersection ---
    i_min = int(np.floor((ox - phase_x) / pitch_x)) - 1
    i_max = int(np.ceil((ox + w / scale - phase_x) / pitch_x)) + 1
    j_min = int(np.floor((oy - phase_y) / pitch_y)) - 1
    j_max = int(np.ceil((oy + h / scale - phase_y) / pitch_y)) + 1

    base_r = spec["via_diameter"] / 2.0
    via_jitter_amp = spec["via_diameter_jitter_amp"]
    salt_via = spec["salt_via"]

    for i in range(i_min, i_max):
        wx = phase_x + i * pitch_x
        px = int(round((wx - ox) * scale))
        if px < -int(base_r * scale) - 2 or px > w + int(base_r * scale) + 2:
            continue
        for j in range(j_min, j_max):
            if (i, j) in spec["missing_vias"]:
                continue
            wy = phase_y + j * pitch_y
            py = int(round((wy - oy) * scale))
            if py < -int(base_r * scale) - 2 or py > h + int(base_r * scale) + 2:
                continue
            jitter = (stable_hash01(i, j, salt_via) - 0.5) * 2.0 * via_jitter_amp
            r_local = base_r * (1.0 + jitter)
            r_px = max(1, int(round(r_local * scale)))
            cv2.circle(img, (px, py), r_px, 235.0, -1)

    # --- local blob defects (drawn last, on top) ---
    for (dx, dy, radius, kind) in spec["defects"]:
        px = int(round((dx - ox) * scale))
        py = int(round((dy - oy) * scale))
        rr = max(1, int(round(radius * scale)))
        if px < -rr or px > w + rr or py < -rr or py > h + rr:
            continue
        color = 255.0 if kind == "bright" else 20.0
        cv2.circle(img, (px, py), rr, color, -1)

    return np.clip(img, 0, 255).astype(np.uint8)


def choose_reference_region(rng, world_size, region_size, margin):
    """
    Pick a random world-coordinate top-left corner (x0, y0) for the
    reference sub-region, keeping a margin away from the world edges.
    """
    max_coord = world_size - region_size - margin
    x0 = rng.uniform(margin, max_coord)
    y0 = rng.uniform(margin, max_coord)
    return x0, y0


# --------------------------------------------------------------------------- #
# 2. Noise
# --------------------------------------------------------------------------- #

def add_gaussian_noise(img, rng, sigma):
    """Additive Gaussian (read/thermal/amplifier) noise."""
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_poisson_noise(img, rng, peak):
    """
    Poisson (shot) noise, modeling SEM secondary-electron detection
    statistics. `peak` controls the effective electron-count scale --
    lower peak => noisier image.
    """
    img_f = img.astype(np.float32) / 255.0
    noisy = rng.poisson(img_f * peak) / peak
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. Augmentation
# --------------------------------------------------------------------------- #

def gaussian_blur(img, ksize, sigma):
    ksize = max(1, int(ksize) | 1)  # force odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def adjust_brightness_contrast(img, alpha, beta):
    """alpha: contrast gain, beta: brightness offset."""
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_brighten(img, strength):
    """
    SEM images show characteristic edge brightening (secondary-electron
    yield increases at topographic edges). Approximate this by adding a
    fraction of the gradient magnitude back onto the image.
    """
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6) * 255.0
    out = img_f + strength * grad
    return np.clip(out, 0, 255).astype(np.uint8)


def add_illumination_gradient(img, rng, strength):
    """
    Low-frequency illumination gradient / vignette, as seen in real SEM
    imaging due to uneven detector collection efficiency across the field
    of view. Built from a coarse random field, upsampled and blurred so it
    varies smoothly across the whole image.
    """
    h, w = img.shape[:2]
    low = rng.uniform(-1.0, 1.0, (4, 4)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(w, h) / 6.0)
    max_abs = np.max(np.abs(field)) + 1e-6
    field = field / max_abs
    out = img.astype(np.float32) + strength * 255.0 * field
    return np.clip(out, 0, 255).astype(np.uint8)


def add_scan_line_noise(img, rng, strength):
    """
    Small row-to-row intensity variation, mimicking raster scan-line
    fluctuations common in SEM acquisition (slightly uneven dwell time /
    detector gain per scan line).
    """
    h, w = img.shape[:2]
    row_noise = rng.normal(0.0, 1.0, (h,)).astype(np.float32)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    row_noise = np.convolve(row_noise, kernel, mode="same")
    row_noise = row_noise.reshape(h, 1)
    out = img.astype(np.float32) * (1.0 + strength * row_noise)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_local_contrast_variation(img, rng, strength):
    """
    Smooth, low-frequency local-contrast modulation, simulating slight
    charging / topography-dependent contrast shifts across the field.
    """
    h, w = img.shape[:2]
    low = rng.uniform(-1.0, 1.0, (6, 6)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(w, h) / 8.0)
    max_abs = np.max(np.abs(field)) + 1e-6
    field = field / max_abs
    mean_val = float(img.mean())
    out = mean_val + (img.astype(np.float32) - mean_val) * (1.0 + strength * field)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_elastic_distortion(img, rng, alpha, sigma):
    """
    Mild elastic distortion, simulating tiny local stage/scan non-linearity.
    A coarse random displacement field is upsampled and heavily smoothed so
    displacements vary gently across the image, then scaled to `alpha`
    pixels of maximum displacement and applied with cv2.remap.
    Kept deliberately small: this is a local, near-zero-mean warp and does
    not meaningfully shift the labeled center (the large-scale rotation /
    scale / stage-drift translation applied in `affine_augment` is the
    transform that is tracked exactly for the ground-truth point).
    """
    h, w = img.shape[:2]
    low_res = max(4, int(min(h, w) / 20))
    dx_low = rng.uniform(-1.0, 1.0, (low_res, low_res)).astype(np.float32)
    dy_low = rng.uniform(-1.0, 1.0, (low_res, low_res)).astype(np.float32)
    dx = cv2.resize(dx_low, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_low, (w, h), interpolation=cv2.INTER_CUBIC)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma)
    dx *= alpha
    dy *= alpha

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT101)


def affine_augment(img, rng, point=None, max_angle=3.0, scale_range=(0.97, 1.03), max_drift=1.0):
    """
    Apply rotation + scale + a small stage-drift translation, all combined
    into ONE affine matrix about the image center. If `point` (x, y) is
    given (e.g. the search image's ground-truth center), it is transformed
    by the exact same matrix, so the label stays exact after this
    (the dominant) geometric augmentation.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    scale = rng.uniform(*scale_range)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    tx = rng.uniform(-max_drift, max_drift)
    ty = rng.uniform(-max_drift, max_drift)
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)

    new_point = None
    if point is not None:
        vec = np.array([point[0], point[1], 1.0])
        nx, ny = M @ vec
        new_point = (float(nx), float(ny))

    return out, new_point, angle, scale, (tx, ty)


# --- degradation parameter presets: reference is higher quality, ------------ #
# --- search is always more degraded, as required. --------------------------- #

REFERENCE_PARAMS = {
    "max_angle": 1.5,
    "scale_range": (0.98, 1.02),
    "max_drift": 1.0,
    "elastic_alpha": 1.0,
    "elastic_sigma": 20.0,
    "blur_ksize": 3,
    "blur_sigma_range": (0.3, 0.6),
    "illum_strength": 0.03,
    "scanline_strength": 0.01,
    "local_contrast_strength": 0.05,
    "contrast_range": (0.95, 1.05),
    "brightness_range": (-8.0, 8.0),
    "edge_strength_range": (0.05, 0.15),
    "poisson_peak_range": (60.0, 100.0),
    "gauss_sigma_range": (1.0, 3.0),
}

SEARCH_PARAMS = {
    "max_angle": 4.0,
    "scale_range": (0.93, 1.07),
    "max_drift": 3.0,
    "elastic_alpha": 2.5,
    "elastic_sigma": 15.0,
    "blur_ksize": 5,
    "blur_sigma_range": (0.8, 1.8),
    "illum_strength": 0.08,
    "scanline_strength": 0.03,
    "local_contrast_strength": 0.12,
    "contrast_range": (0.75, 1.10),
    "brightness_range": (-20.0, 20.0),
    "edge_strength_range": (0.05, 0.20),
    "poisson_peak_range": (15.0, 35.0),
    "gauss_sigma_range": (4.0, 10.0),
}


def sem_style_augment(img, rng, point, params):
    """
    Bundle of SEM-style augmentations applied in a physically-motivated
    order: geometric warps first (rotation/scale/drift, then a small
    elastic distortion), then illumination / contrast field effects, then
    brightness-contrast + edge brightening, and finally detector noise
    (Poisson then Gaussian) last.

    `params` selects the degradation strength (REFERENCE_PARAMS for a
    high-quality capture, SEARCH_PARAMS for a lower-quality one).
    """
    img, new_point, angle, scale, drift = affine_augment(
        img, rng, point=point,
        max_angle=params["max_angle"],
        scale_range=params["scale_range"],
        max_drift=params["max_drift"],
    )
    img = apply_elastic_distortion(img, rng, params["elastic_alpha"], params["elastic_sigma"])
    img = gaussian_blur(img, ksize=params["blur_ksize"], sigma=rng.uniform(*params["blur_sigma_range"]))
    img = add_illumination_gradient(img, rng, params["illum_strength"])
    img = add_scan_line_noise(img, rng, params["scanline_strength"])
    img = add_local_contrast_variation(img, rng, params["local_contrast_strength"])
    img = adjust_brightness_contrast(img, alpha=rng.uniform(*params["contrast_range"]),
                                      beta=rng.uniform(*params["brightness_range"]))
    img = edge_brighten(img, strength=rng.uniform(*params["edge_strength_range"]))
    img = add_poisson_noise(img, rng, peak=rng.uniform(*params["poisson_peak_range"]))
    img = add_gaussian_noise(img, rng, sigma=rng.uniform(*params["gauss_sigma_range"]))
    return img, new_point, angle, scale, drift


# --------------------------------------------------------------------------- #
# 4. Pair generation & saving
# --------------------------------------------------------------------------- #

def generate_pair(idx, rng):
    """
    Build one (reference, search, label) sample from a SINGLE DRAM world.

    Returns
    -------
    ref_img, search_img : np.ndarray (uint8)
    label : dict
    """
    # one layout spec for the entire world -- no separate background spec
    spec = make_layout_spec(rng, world_size=WORLD_SIZE)

    # pick where inside that world the reference region physically sits
    x0, y0 = choose_reference_region(rng, WORLD_SIZE, REGION_WORLD_SIZE, REGION_MARGIN)
    true_center_world = (x0 + REGION_WORLD_SIZE / 2.0, y0 + REGION_WORLD_SIZE / 2.0)

    # render that sub-region at 100x -> reference (high-res crop)
    ref_clean = render_layout(spec, scale=REF_SCALE, origin=(x0, y0),
                               out_size=(REF_SIZE, REF_SIZE))

    # render the ENTIRE world at 10x -> search (the reference region is
    # already naturally present inside it -- nothing is pasted)
    search_clean = render_layout(spec, scale=SEARCH_SCALE, origin=(0.0, 0.0),
                                  out_size=(SEARCH_SIZE, SEARCH_SIZE))

    # world coords -> search pixel coords (origin=0, scale=SEARCH_SCALE)
    true_center_px = (true_center_world[0] * SEARCH_SCALE, true_center_world[1] * SEARCH_SCALE)

    # independent, asymmetric SEM-style degradation
    ref_img, _, ref_angle, ref_scale, ref_drift = sem_style_augment(
        ref_clean, rng, point=None, params=REFERENCE_PARAMS)
    search_img, gt_center, s_angle, s_scale, s_drift = sem_style_augment(
        search_clean, rng, point=true_center_px, params=SEARCH_PARAMS)

    label = {
        "pair_id": idx,
        "reference_file": f"ref_{idx:03d}.png",
        "search_file": f"search_{idx:03d}.png",
        "center_x": round(gt_center[0], 2),
        "center_y": round(gt_center[1], 2),
        "patch_size": REGION_WORLD_SIZE * SEARCH_SCALE,
        "ref_size_px": REF_SIZE,
        "search_size_px": SEARCH_SIZE,
        "pitch_x": round(spec["pitch_x"], 3),
        "pitch_y": round(spec["pitch_y"], 3),
        "line_width": round(spec["line_width"], 3),
        "via_diameter": round(spec["via_diameter"], 3),
        "reference_rotation_deg": round(ref_angle, 3),
        "search_rotation_deg": round(s_angle, 3),
        "reference_scale_factor": round(ref_scale, 4),
        "search_scale_factor": round(s_scale, 4),
        "reference_drift_x": round(ref_drift[0], 3),
        "reference_drift_y": round(ref_drift[1], 3),
        "search_drift_x": round(s_drift[0], 3),
        "search_drift_y": round(s_drift[1], 3),
    }
    return ref_img, search_img, label


def save_dataset(n_pairs, out_dir, seed):
    """Generate `n_pairs` samples and save reference/search images + labels.csv."""
    ref_dir = os.path.join(out_dir, "reference")
    search_dir = os.path.join(out_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    labels = []

    for idx in range(n_pairs):
        ref_img, search_img, label = generate_pair(idx, rng)

        ref_path = os.path.join(ref_dir, label["reference_file"])
        search_path = os.path.join(search_dir, label["search_file"])
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        labels.append(label)
        print(f"[{idx + 1}/{n_pairs}] saved {label['reference_file']} / "
              f"{label['search_file']}  center=({label['center_x']}, {label['center_y']})")

    csv_path = os.path.join(out_dir, "..", "labels.csv") if os.path.basename(out_dir) == "train" \
        else os.path.join(out_dir, "labels.csv")
    csv_path = os.path.normpath(csv_path)
    fieldnames = list(labels[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nSaved {n_pairs} pairs.")
    print(f"  reference images -> {ref_dir}")
    print(f"  search images    -> {search_dir}")
    print(f"  labels           -> {csv_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic DRAM SEM navigation dataset.")
    parser.add_argument("--n_pairs", type=int, default=30, help="Number of reference/search pairs to generate.")
    parser.add_argument("--out_dir", type=str, default="train", help="Output directory (will contain reference/ and search/ subfolders).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    save_dataset(args.n_pairs, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
