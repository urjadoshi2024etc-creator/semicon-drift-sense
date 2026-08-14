"""
analyze_failure_causes.py

Follows up on analyze_v6_failures.py's finding that periodic pitch lock-on
does NOT explain the V6/9K checkpoint's catastrophic failures (match rate
was at-or-below what a random error vector would hit by chance against a
dense pitch lattice). This script asks the next question directly: which
GENERATOR PARAMETERS (quality, rotation, defect density, etc. -- all
already logged per-pair in labels.csv by generate_dram_dataset_v3.py)
distinguish the catastrophic failures from the clean successes?

METHOD
---------------------------------------------------------------------------
1. Run the checkpoint over every pair in --data_dir, computing per-pair
   Euclidean error (same logic as analyze_v6_failures.py).
2. Split pairs into:
     FAILURE group: error_px > --fail_threshold   (default 100px)
     SUCCESS group: error_px < --success_threshold (default 10px)
   (Pairs strictly between the two thresholds are excluded from the
   comparison -- they're "medium" cases, not the sharp contrast we want.)
3. For every generator parameter column present in labels.csv, compute
   mean/std/min/max in each group, plus Cohen's d (standardized mean
   difference) as a simple, comparable ranking of "how much does this
   parameter separate failures from successes" across very different
   units (degrees vs nm vs counts vs strength-fractions).
4. Print a ranked table (largest |Cohen's d| first) so the most likely
   causal factor(s) are immediately visible, rather than having to eyeball
   19 separate distributions.

INTERPRETATION GUIDE (printed at the end too):
  |d| < 0.2   : negligible separation, not a meaningful factor
  |d| 0.2-0.5 : small effect, worth noting but probably not sufficient alone
  |d| 0.5-0.8 : moderate effect, a real candidate explanation
  |d| > 0.8   : large effect, strong candidate -- this is what we're hunting for

Usage:
    python analyze_failure_causes.py \
        --checkpoint ./runs/v6_train_v7/checkpoints/best_model.pt \
        --data_dir ./eval_v5 --fail_threshold 100 --success_threshold 10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dram_dataset import DramPairDataset
from model_v6 import build_model


# Generator parameter columns worth comparing (numeric, present in
# generate_dram_dataset_v3.py's labels.csv). Deliberately excludes
# bookkeeping columns (pair_id, filenames, coordinates, profile, seed).
PARAM_COLUMNS = [
    "pitch_x_nm", "pitch_y_nm", "line_width_nm", "via_diameter_nm",
    "n_missing_vias", "n_merged_contacts", "n_broken_segments", "n_particles",
    "ref_quality", "search_quality",
    "ref_rotation_deg", "search_rotation_deg",
    "ref_scale_factor", "search_scale_factor",
    "search_perspective_strength", "elastic_alpha_search",
    "vignette_strength", "scanline_strength", "fractal_bg_strength",
]


def load_checkpoint_into(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {path}  (epoch={ckpt.get('epoch')}, "
          f"best_metric={ckpt.get('best_metric')})")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between two samples (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                          / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--fail_threshold", type=float, default=100.0,
                     help="error_px above this = FAILURE group")
    ap.add_argument("--success_threshold", type=float, default=10.0,
                     help="error_px below this = SUCCESS group")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    load_checkpoint_into(args.checkpoint, model, device)
    model.eval()

    ds = DramPairDataset(args.data_dir, normalize_labels=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(ds)} pairs from {args.data_dir}\n")

    errors = {}
    with torch.no_grad():
        for reference, search, gt, pair_id in loader:
            reference = reference.to(device)
            search = search.to(device)
            out = model(reference, search)
            pred = out.coords.cpu().numpy()
            true = gt.numpy()
            for i in range(pred.shape[0]):
                pid = int(pair_id[i])
                err = float(np.hypot(pred[i, 0] - true[i, 0], pred[i, 1] - true[i, 1]))
                errors[pid] = err

    # Build failure/success groups, pulling metadata straight from the
    # dataset's already-loaded labels.csv rows (ds.rows[pid]).
    available_cols = [c for c in PARAM_COLUMNS if c in ds.rows[0]]
    missing_cols = [c for c in PARAM_COLUMNS if c not in ds.rows[0]]
    if missing_cols:
        print(f"(Note: columns not found in this labels.csv, skipping: {missing_cols})\n")

    failure_rows, success_rows = [], []
    for pid, err in errors.items():
        row = ds.rows[pid]
        if err > args.fail_threshold:
            failure_rows.append(row)
        elif err < args.success_threshold:
            success_rows.append(row)

    print(f"FAILURE group (error > {args.fail_threshold}px): {len(failure_rows)} samples")
    print(f"SUCCESS group (error < {args.success_threshold}px): {len(success_rows)} samples")
    print(f"(excluded {len(errors) - len(failure_rows) - len(success_rows)} samples in between)\n")

    if len(failure_rows) < 3 or len(success_rows) < 3:
        print("Too few samples in one group for a meaningful comparison "
              "(need at least ~3 each). Try adjusting --fail_threshold / --success_threshold.")
        return

    results = []
    for col in available_cols:
        fail_vals = np.array([float(r[col]) for r in failure_rows])
        succ_vals = np.array([float(r[col]) for r in success_rows])
        d = cohens_d(fail_vals, succ_vals)
        results.append({
            "col": col, "d": d,
            "fail_mean": fail_vals.mean(), "fail_std": fail_vals.std(),
            "fail_min": fail_vals.min(), "fail_max": fail_vals.max(),
            "succ_mean": succ_vals.mean(), "succ_std": succ_vals.std(),
            "succ_min": succ_vals.min(), "succ_max": succ_vals.max(),
        })

    results.sort(key=lambda r: abs(r["d"]) if not np.isnan(r["d"]) else -1, reverse=True)

    print("=" * 100)
    print("GENERATOR PARAMETER COMPARISON: FAILURE vs SUCCESS  (ranked by |Cohen's d|)")
    print("=" * 100)
    print(f"{'parameter':>28} | {'d':>6} | {'FAIL mean±std':>18} {'[min,max]':>16} | "
          f"{'SUCC mean±std':>18} {'[min,max]':>16}")
    print("-" * 100)
    for r in results:
        d_str = f"{r['d']:+.2f}" if not np.isnan(r["d"]) else "  n/a"
        fail_str = f"{r['fail_mean']:.3f}±{r['fail_std']:.3f}"
        fail_range = f"[{r['fail_min']:.2f},{r['fail_max']:.2f}]"
        succ_str = f"{r['succ_mean']:.3f}±{r['succ_std']:.3f}"
        succ_range = f"[{r['succ_min']:.2f},{r['succ_max']:.2f}]"
        print(f"{r['col']:>28} | {d_str:>6} | {fail_str:>18} {fail_range:>16} | "
              f"{succ_str:>18} {succ_range:>16}")
    print("-" * 100)

    print("\nINTERPRETATION GUIDE:")
    print("  |d| < 0.2   : negligible -- not a meaningful factor")
    print("  |d| 0.2-0.5 : small effect -- worth noting, probably not sufficient alone")
    print("  |d| 0.5-0.8 : moderate effect -- a real candidate explanation")
    print("  |d| > 0.8   : large effect -- strong candidate, look here first")

    top = results[0]
    if not np.isnan(top["d"]) and abs(top["d"]) > 0.5:
        print(f"\n  Strongest separating factor: '{top['col']}' (d={top['d']:+.2f}). "
              f"FAILURE group {'higher' if top['d']>0 else 'lower'} on average than SUCCESS group.")
        print("  Recommended next step: target this specific regime -- either generate more")
        print("  training examples concentrated in that parameter range, or add augmentation")
        print("  that emphasizes it, rather than a generic architecture/loss change.")
    else:
        print("\n  No single parameter shows a large, clean separation. Failures may be driven")
        print("  by a COMBINATION of factors, or by something not captured in these columns")
        print("  (e.g. specific spatial defect layout near the true center). Consider visually")
        print("  inspecting a handful of failure cases directly (plot_heatmap_overlay from")
        print("  utils.py) as the next step instead of another metadata pass.")
    print("=" * 100)


if __name__ == "__main__":
    main()
