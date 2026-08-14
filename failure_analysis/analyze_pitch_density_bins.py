"""
analyze_pitch_density_bins.py

Follow-up to analyze_failure_causes.py. That script found n_missing_vias
as the strongest failure-separating factor (d~-0.87), but with failures
having FEWER missing vias -- the opposite of what "more defects clutter
the match" would predict. This script tests a specific hypothesis for why:
n_missing_vias is likely a confounded proxy for pitch, not an independent
cause, because of how generate_dram_dataset_v3.py builds the via grid.

THE CONFOUND
---------------------------------------------------------------------------
create_world() lays out via sites across the ENTIRE world
(scale.world_size x scale.world_size, ~10000nm x 10000nm), with
n_cols ~ world_size/pitch_x and n_rows ~ world_size/pitch_y. Each site is
independently missing with probability miss_prob. So the RAW COUNT
n_missing_vias scales approximately as:

    n_missing_vias  ~  (world_size / pitch_x) * (world_size / pitch_y) * miss_prob

i.e. it is mechanically, strongly anti-correlated with pitch_x * pitch_y --
finer pitch means more via sites fit in the same fixed world area, so more
missing vias in raw count, REGARDLESS of any real difficulty difference.
This script checks that correlation directly.

THE MORE PHYSICALLY MEANINGFUL VARIABLE
---------------------------------------------------------------------------
What might actually matter for MATCHING difficulty is not the world-wide
via count, but how much distinguishing local structure fits inside the
FIXED-SIZE reference crop (1000nm x 1000nm at 1nm/px, per
generate_dram_dataset_v3.py's PhysicalScale). Coarser pitch means fewer
via sites -- and therefore fewer potential defects -- inside that fixed
window, i.e. less local information for the model to lock onto. We compute:

    ref_via_density_estimate = (1000 / pitch_x_nm) * (1000 / pitch_y_nm)

as an approximate count of via sites that fit inside one reference crop
(ignoring phase offset, which only shifts this by at most one row/column).

WHAT THIS SCRIPT DOES
---------------------------------------------------------------------------
1. Confound check: correlation between n_missing_vias and
   1/(pitch_x_nm * pitch_y_nm) across the whole dataset (no model needed).
2. Runs the checkpoint to get per-pair error (same logic as
   analyze_failure_causes.py).
3. Bins samples by pitch_x_nm (quantile bins) and reports, per bin:
   n, mean error, median error, failure rate (>100px), success rate (<10px).
4. Same binning by ref_via_density_estimate, for direct comparison.
5. Same binning by n_missing_vias, explicitly labeled as CONFOUNDED, so
   it's visible whether its apparent effect survives once pitch is
   accounted for (it shouldn't, if the confound hypothesis is right).

Usage:
    python analyze_pitch_density_bins.py \
        --checkpoint ./runs/v6_train_v7/checkpoints/best_model.pt \
        --data_dir ./eval_v5 --n_bins 5
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


def get_errors(model, ds, device, batch_size=8) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
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
                errors[pid] = float(np.hypot(pred[i, 0] - true[i, 0], pred[i, 1] - true[i, 1]))
    return errors


def confound_check(rows: list[dict]) -> None:
    print("=" * 90)
    print("STEP 1: CONFOUND CHECK -- is n_missing_vias just a proxy for pitch?")
    print("=" * 90)
    n_missing = np.array([float(r["n_missing_vias"]) for r in rows])
    pitch_x = np.array([float(r["pitch_x_nm"]) for r in rows])
    pitch_y = np.array([float(r["pitch_y_nm"]) for r in rows])
    inv_area = 1.0 / (pitch_x * pitch_y)

    corr = np.corrcoef(n_missing, inv_area)[0, 1]
    print(f"\nCorrelation(n_missing_vias, 1/(pitch_x*pitch_y)): r = {corr:.3f}")
    if abs(corr) > 0.5:
        print("  -> STRONG correlation confirmed. n_missing_vias is substantially explained")
        print("  by pitch alone (finer pitch = more via sites in the fixed world area = more")
        print("  missing vias in raw count, independent of any real 'difficulty' difference).")
        print("  Treat n_missing_vias's earlier Cohen's d result as LARGELY REDUNDANT with")
        print("  the pitch signal, not an independent cause.")
    else:
        print("  -> Correlation is weaker than expected. The confound hypothesis is NOT")
        print("  strongly supported -- n_missing_vias may carry some independent signal.")
    print()


def bin_analysis(rows: list[dict], errors: dict, key_fn, label: str, n_bins: int,
                  fail_thresh: float, succ_thresh: float) -> None:
    print("-" * 90)
    print(f"BINNED ANALYSIS: {label}")
    print("-" * 90)

    values = []
    errs = []
    for r in rows:
        pid = int(r["pair_id"])
        if pid not in errors:
            continue
        values.append(key_fn(r))
        errs.append(errors[pid])
    values = np.array(values)
    errs = np.array(errs)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] -= 1e-6  # ensure the minimum value falls inside the first bin

    print(f"{'bin range':>22} | {'n':>4} | {'mean_err':>9} | {'median_err':>10} | "
          f"{'fail%(>'+str(int(fail_thresh))+'px)':>16} | {'succ%(<'+str(int(succ_thresh))+'px)':>16}")
    print("-" * 90)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (values > lo) & (values <= hi)
        if mask.sum() == 0:
            continue
        bin_errs = errs[mask]
        fail_rate = 100.0 * np.mean(bin_errs > fail_thresh)
        succ_rate = 100.0 * np.mean(bin_errs < succ_thresh)
        range_str = f"({lo:.2f}, {hi:.2f}]"
        print(f"{range_str:>22} | {mask.sum():>4} | {bin_errs.mean():>9.2f} | "
              f"{np.median(bin_errs):>10.2f} | {fail_rate:>15.1f}% | {succ_rate:>15.1f}%")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--fail_threshold", type=float, default=100.0)
    ap.add_argument("--success_threshold", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    load_checkpoint_into(args.checkpoint, model, device)
    model.eval()

    ds = DramPairDataset(args.data_dir, normalize_labels=False)
    print(f"Loaded {len(ds)} pairs from {args.data_dir}\n")

    # --- Step 1: confound check (no model needed, pure metadata) ---
    confound_check(ds.rows)

    # --- Step 2: run model to get per-pair errors ---
    print("Running model over dataset to compute per-pair errors...\n")
    errors = get_errors(model, ds, device, args.batch_size)

    all_errs = np.array(list(errors.values()))
    print(f"Overall: n={len(all_errs)}  mean={all_errs.mean():.2f}px  "
          f"median={np.median(all_errs):.2f}px  "
          f"fail_rate(>{args.fail_threshold}px)={100*np.mean(all_errs>args.fail_threshold):.1f}%  "
          f"succ_rate(<{args.success_threshold}px)={100*np.mean(all_errs<args.success_threshold):.1f}%\n")

    print("=" * 90)
    print("STEP 2: BINNED FAILURE-RATE ANALYSIS")
    print("=" * 90)
    print("(Real signal should show a clear MONOTONIC trend in fail%/succ% across bins.")
    print(" A flat or noisy pattern across bins means that variable isn't really driving it.)\n")

    bin_analysis(
        ds.rows, errors,
        key_fn=lambda r: float(r["pitch_x_nm"]),
        label="pitch_x_nm  (the physically meaningful candidate)",
        n_bins=args.n_bins, fail_thresh=args.fail_threshold, succ_thresh=args.success_threshold,
    )

    bin_analysis(
        ds.rows, errors,
        key_fn=lambda r: float(r["pitch_y_nm"]),
        label="pitch_y_nm  (checked for symmetry with pitch_x -- generator samples x/y "
              "pitch independently, so this confirms the y-axis carries the same signal "
              "on its own, not just through the combined density estimate below)",
        n_bins=args.n_bins, fail_thresh=args.fail_threshold, succ_thresh=args.success_threshold,
    )

    bin_analysis(
        ds.rows, errors,
        key_fn=lambda r: (1000.0 / float(r["pitch_x_nm"])) * (1000.0 / float(r["pitch_y_nm"])),
        label="ref_via_density_estimate = (1000/pitch_x)*(1000/pitch_y)  "
              "(estimated via sites per reference crop)",
        n_bins=args.n_bins, fail_thresh=args.fail_threshold, succ_thresh=args.success_threshold,
    )

    bin_analysis(
        ds.rows, errors,
        key_fn=lambda r: float(r["n_missing_vias"]),
        label="n_missing_vias  [CONFOUNDED -- see Step 1; likely redundant with pitch above]",
        n_bins=args.n_bins, fail_thresh=args.fail_threshold, succ_thresh=args.success_threshold,
    )

    print("=" * 90)
    print("HOW TO READ THIS:")
    print("  - If pitch_x_nm and ref_via_density_estimate show similar, clean monotonic")
    print("    trends (fail% rising as pitch increases / density falls), that's strong,")
    print("    non-confounded evidence for the 'coarse pitch = less local reference detail")
    print("    = harder match' hypothesis.")
    print("  - If n_missing_vias's trend looks similar to pitch's trend (just mirrored),")
    print("    that confirms it's redundant, not an independent factor.")
    print("  - If none of the three show a clean monotonic trend, the failure cause is")
    print("    likely NOT a single generator parameter -- worth visually inspecting")
    print("    individual failure cases next instead of more metadata slicing.")
    print("=" * 90)


if __name__ == "__main__":
    main()
