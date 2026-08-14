"""
analyze_v6_failures.py

Diagnostic for the V6 checkpoint's mean/median gap (val median ~0.6-1.0px,
val mean ~97-120px). The gap implies most predictions are near-perfect and
a smaller set of outliers drag the mean up. This script tests the specific
hypothesis this project has been built around: that those outliers are
cases where the model locked onto the WRONG periodic repeat -- i.e. the
predicted location sits close to (true location + k * pitch) for some
small nonzero integer k, in x and/or y, rather than being an unstructured
large miss.

This distinction matters for what to do next:
  - If errors cluster near integer pitch multiples -> confirms periodic
    ambiguity is the dominant remaining failure mode -> the right fix is
    a targeted hard-negative/margin loss against pitch-neighbor candidate
    cells (a structural fix), not more data or generic regularization.
  - If errors do NOT cluster near pitch multiples -> the outliers are
    something else (e.g. specific defect patterns, extreme quality/noise
    samples, edge-of-image cases) -> needs a different diagnostic before
    picking a fix.

Usage:
    python analyze_v6_failures.py \
        --checkpoint ./runs/v6_train_v8/checkpoints/best_model.pt \
        --data_dir ./eval_v5 --n_worst 20

(v6_train_v8/checkpoints/best_model.pt is the submitted checkpoint,
identical to submission_model/driftsense_final.pt.)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dram_dataset import DramPairDataset
from model_v6 import build_model


def load_checkpoint_into(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {path}  (epoch={ckpt.get('epoch')}, "
          f"best_metric={ckpt.get('best_metric')})")


def pitch_multiple_distance(dx: float, dy: float, pitch_x: float, pitch_y: float,
                             max_k: int = 100, tol_frac: float = 0.25) -> tuple[bool, int, int, float]:
    """
    Checks whether the (dx, dy) error vector is close to (k_x * pitch_x,
    k_y * pitch_y) for small integers k_x, k_y (searched independently in
    [-max_k, max_k], not required to be equal -- pitch_x and pitch_y can
    differ). "Close" means within tol_frac of one pitch cell in whichever
    axis has the larger nominal pitch step.

    Returns: (is_periodic_match, best_kx, best_ky, residual_px)
    """
    best_residual = float("inf")
    best_kx, best_ky = 0, 0
    for kx in range(-max_k, max_k + 1):
        for ky in range(-max_k, max_k + 1):
            if kx == 0 and ky == 0:
                continue
            target_x = kx * pitch_x
            target_y = ky * pitch_y
            residual = np.hypot(dx - target_x, dy - target_y)
            if residual < best_residual:
                best_residual = residual
                best_kx, best_ky = kx, ky
    tol_px = tol_frac * min(pitch_x, pitch_y)
    is_match = best_residual <= tol_px
    return is_match, best_kx, best_ky, best_residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_worst", type=int, default=20)
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

    records = []
    with torch.no_grad():
        idx = 0
        for reference, search, gt, pair_id in loader:
            reference = reference.to(device)
            search = search.to(device)
            out = model(reference, search)
            pred = out.coords.cpu().numpy()
            true = gt.numpy()
            for i in range(pred.shape[0]):
                pid = int(pair_id[i])
                row = ds.rows[pid] if pid < len(ds.rows) else None
                # labels store pitch in nanometers
                # search image scale = 10 nm per pixel
                pitch_x_nm = float(row["pitch_x_nm"]) if row and "pitch_x_nm" in row else None
                pitch_y_nm = float(row["pitch_y_nm"]) if row and "pitch_y_nm" in row else None
                
                pitch_x = pitch_x_nm / 10.0 if pitch_x_nm is not None else None
                pitch_y = pitch_y_nm / 10.0 if pitch_y_nm is not None else None
                dx = pred[i, 0] - true[i, 0]
                dy = pred[i, 1] - true[i, 1]
                error = float(np.hypot(dx, dy))
                records.append({
                    "pair_id": pid, "error": error, "dx": dx, "dy": dy,
                    "pitch_x": pitch_x, "pitch_y": pitch_y,
                    "pred": pred[i].tolist(), "true": true[i].tolist(),
                })
            idx += 1

    errors = np.array([r["error"] for r in records])
    print("=" * 78)
    print("OVERALL VAL SET ERROR SUMMARY")
    print("=" * 78)
    print(f"  n={len(errors)}  mean={errors.mean():.2f}px  median={np.median(errors):.2f}px  "
          f"max={errors.max():.2f}px  std={errors.std():.2f}px")
    print(f"  acc@5px:  {100*np.mean(errors<=5):.1f}%")
    print(f"  acc@10px: {100*np.mean(errors<=10):.1f}%")
    print(f"  acc@50px: {100*np.mean(errors<=50):.1f}%")

    # --- worst-case periodic-lock-on test ---
    worst = sorted(records, key=lambda r: r["error"], reverse=True)[:args.n_worst]

    print("\n" + "=" * 78)
    print(f"TOP {args.n_worst} WORST CASES -- periodic pitch-multiple test")
    print("=" * 78)
    print(f"{'pair_id':>8} | {'error_px':>9} | {'pitch_x':>10} {'pitch_y':>10} | "
          f"{'periodic?':>10} {'k_x':>4} {'k_y':>4} {'residual':>9}")
    print("-" * 78)

    n_periodic = 0
    for r in worst:
        if r["pitch_x"] is None:
            print(f"{r['pair_id']:>8} | {r['error']:>9.2f} | (no pitch info in labels.csv)")
            continue
        is_match, kx, ky, residual = pitch_multiple_distance(
            r["dx"], r["dy"], r["pitch_x"], r["pitch_y"]
        )
        n_periodic += int(is_match)
        flag = "YES" if is_match else "no"
        print(f"{r['pair_id']:>8} | {r['error']:>9.2f} | {r['pitch_x']:>10.2f} {r['pitch_y']:>10.2f} | "
              f"{flag:>10} {kx:>4} {ky:>4} {residual:>9.2f}")

    print("-" * 78)
    print(f"\n{n_periodic}/{len(worst)} of the worst-{args.n_worst} cases are within one pitch-cell "
          f"tolerance of an integer pitch multiple.")
    print("\nINTERPRETATION:")
    if n_periodic >= 0.5 * len(worst):
        print("  MAJORITY of worst cases match the periodic-lock-on pattern (predicted location")
        print("  sits near true_location + k*pitch for small integer k). This CONFIRMS periodic")
        print("  ambiguity as the dominant remaining failure mode. Recommended next step: add a")
        print("  hard-negative / margin loss term that explicitly penalizes the model when a")
        print("  pitch-shifted candidate cell scores too close to the true cell.")
    elif n_periodic > 0:
        print("  SOME worst cases match periodic lock-on, but it's not the dominant pattern.")
        print("  Worth combining hard-negative mining with inspecting the non-periodic outliers")
        print("  directly (e.g. via plot_heatmap_overlay-style visualization) to see if they")
        print("  share another common cause (extreme noise/quality, edge-of-image crops, etc).")
    else:
        print("  Periodic lock-on does NOT explain the worst cases. Do not implement hard-negative")
        print("  mining yet -- inspect these specific pair_ids' images directly first.")
    print("=" * 78)


if __name__ == "__main__":
    main()
