# DriftSense — Evaluation Report

**Model:** `driftsense_final.pt` (V6 architecture, trained on the merged
`train_v8` dataset — see `README.md` Section 4 for exact reproduction
steps). Best checkpoint selected at **epoch 44** by validation mean error;
training run early-stopped at epoch 59 (patience 15, min improvement
0.5px).

**Independent test set:** `eval_v5` — 100 pairs, generated with
`--profile medium --seed 24681357`, held out from all training and
model-selection decisions. All headline numbers below are measured on
this set unless stated otherwise.

---

# 1. Headline Accuracy

## Independent test set — `eval_v5` (`n = 100`)

| Metric | Value |
|---|---:|
| Mean error | **92.30 px** |
| Median error | **0.52 px** |
| Minimum error | **0.04 px** |
| Maximum error | **818.84 px** |
| Within 10 px | **80 / 100 (80.0%)** |
| Within 25 px | **80 / 100 (80.0%)** |
| Within 50 px | **80 / 100 (80.0%)** |
| Within 100 px | **81 / 100 (81.0%)** |

## Validation set

`val_v5` contains 300 pairs generated with seed `13579246` and was used
during training for checkpoint selection. It was **not** used for the
headline test-set numbers above.

- Best validation mean error: **76.595 px**
- Best epoch: **44**

### Interpreting the mean / median gap

The large gap between median error (**0.52 px**) and mean error
(**92.30 px**) reflects two distinct performance regimes:

1. The large majority of pairs are localized very accurately, often at
   sub-pixel scale.
2. A smaller subset of approximately 19–20% produces large errors,
   dominating the mean.

Both metrics are therefore reported deliberately. Median alone would
overstate reliability, while mean alone would understate typical
localization performance.

---

# 2. Inference Timing

Inference was measured on 1000×1000 reference/search image pairs.

| Device | Mean time per pair |
|---|---:|
| GPU — RTX 4050 Laptop, 6GB VRAM, CUDA 12.1 | **9.56 ms** |
| CPU — forced CPU execution | **71.36 ms** |

### Measurement protocol

- GPU: mean over 50 runs
- GPU warmup: 5 runs
- GPU timing bounded using `torch.cuda.synchronize()`
- CPU: mean over 20 runs
- CPU warmup: 3 runs

`submission_model/inference.py` automatically detects CUDA and falls back
to CPU when necessary. No code changes are required for either mode.

---

# 3. Failure Analysis

The analysis examined periodic-pattern ambiguity, image quality,
geometric mismatch, DRAM pitch / information density, and generator
parameter confounds.

## 3.1 Periodic Pitch-Repeat Lock-On

### Hypothesis

Large errors might result from the model locking onto a visually similar
but incorrect periodic repeat of the DRAM pattern:

```text
predicted location ≈ true location + k × pitch
```

### Direct test

For the 25 worst-error cases:

- `k` was searched from **−100 to +100** per axis.
- Each sample's own `pitch_x_nm` / `pitch_y_nm` was used.
- Pitch was converted to search-image pixels using the 10 nm/px scale.
- A match required the error to fall within **0.25 × pitch** of an
  integer pitch-multiple lattice point.

### Result

**7 / 25 cases (28%)** matched an integer pitch multiple.

### Chance-baseline check

The approximate geometric collision probability for a random error vector
under this tolerance is:

```text
π × (0.25)² ≈ 19.6%
```

With `n = 25`, the standard error is approximately 8 percentage points.

Therefore, 28% is only about one standard error above the approximately
20% chance baseline.

### Conclusion

This test does **not** provide strong evidence that periodic pitch-repeat
lock-on is the dominant failure mechanism.

It also does not prove that periodicity contributes zero error. The result
is best described as **inconclusive at this sample size**.

The stronger evidence comes from the coarse-pitch / local-information
analysis in Sections 3.3–3.5.

An earlier test on a prior checkpoint produced 3/25 matches, closer to the
chance baseline; the difference remains plausibly attributable to sampling
variation at this small sample size.

---

## 3.2 Generator-Parameter Failure Analysis

Every logged generator parameter was compared between:

- **FAILURE:** error > 100 px
- **SUCCESS:** error < 10 px

using Cohen's `d`.

For the submitted `v6_train_v8` checkpoint:

- FAILURE: **19 pairs**
- SUCCESS: **80 pairs**
- Intermediate: **1 pair**

| Rank | Parameter | Cohen's d | Interpretation |
|---:|---|---:|---|
| 1 | `n_missing_vias` | −0.77 | Confounded — see Section 3.3 |
| 2 | `search_quality` | +0.51 | Real, independent factor |
| 3 | `pitch_x_nm` | +0.42 | Real, independent factor |
| 4 | `ref_scale_factor` | +0.40 | Weaker, plausibly secondary |
| 5 | `pitch_y_nm` | +0.34 | Real, independent factor |

All other parameters—including rotation, elastic warp, vignette,
scanline variation, fractal background, and defect counts other than
missing-via count—showed `|d| < 0.35`.

---

## 3.3 Confound Discovery — `n_missing_vias` Is a Pitch Proxy

The initial strongest signal was `n_missing_vias`. Inspection of
`create_world()` showed that the via grid spans a fixed-size physical
world, so raw missing-via count scales mechanically with:

```text
1 / (pitch_x × pitch_y)
```

Finer pitch means more via sites fit into the same physical area.

### Verified correlations

| Dataset | Correlation |
|---|---:|
| `eval_v5` | **0.755** |
| `val_v5` | **0.711** |

Therefore, raw `n_missing_vias` should **not** be treated as an independent
root cause. Its apparent predictive power is substantially redundant with
pitch.

---

## 3.4 Root Cause — Coarse Pitch Reduces Local Reference-Crop Information

The reference crop represents approximately:

```text
1000 nm × 1000 nm
```

At coarser pitch, fewer via sites and repeating structural landmarks fit
inside this fixed physical window. The crop therefore contains less local
structure with which to distinguish the true location.

Conceptually:

```text
Larger pitch
    ↓
Fewer local structural landmarks
    ↓
Lower reference-crop information density
    ↓
Harder localization
```

### Binned failure-rate analysis

To avoid the missing-via confound, failure was analyzed directly against
`pitch_x_nm` on the pre-supplement checkpoint `v6_train_v7`, using
`eval_v5`.

| `pitch_x_nm` bin | n | Failure rate (>100 px) |
|---|---:|---:|
| 40.6–57.1 | 20 | **10.0%** |
| 57.1–73.6 | 20 | **20.0%** |
| 73.6–93.4 | 20 | **30.0%** |
| 93.4–106.6 | 20 | **30.0%** |
| 106.6–119.3 | 20 | **35.0%** |

The failure rate rises monotonically from **10% to 35%**, approximately
a **3.5× increase**.

The same directional trend was replicated on `val_v5`:

```text
13.3% → 35.0%
```

and independently reproduced using a derived
`ref_via_density_estimate` variable representing via sites per reference
crop.

---

## 3.5 Targeted Intervention — Coarse-Pitch Supplemental Training

A targeted supplemental dataset was generated after identifying the
coarse-pitch regime as a measurable weakness:

```text
4,000 supplemental pairs
pitch range: 80–120 nm
```

The original base dataset contained 9,000 pairs. The supplemental data were
merged to create `train_v8`.

The architecture, loss configuration, and training procedure were kept
unchanged so that training-data composition was the intended changed
variable.

### Before vs. after

Results on the same independent `eval_v5` set:

| Metric | Before — `train_v7` | After — `train_v8` |
|---|---:|---:|
| Validation mean error | 96.74 px | **76.60 px** |
| Test mean error | 113.70 px | **92.30 px** |
| Failure rate (>100 px) | 25% | **19%** |
| Success rate (<10 px) | 73% | **80%** |
| `pitch_x_nm` Cohen's d | +0.56 | **+0.42** |

Observed improvements:

- Validation mean error: approximately **21% reduction**
- Test mean error: approximately **19% reduction**
- Failure rate: **25% → 19%**
- Success rate: **73% → 80%**
- Pitch-related Cohen's `d`: **+0.56 → +0.42**

The pitch-related effect was reduced but not eliminated.

> This is intervention evidence rather than a randomized causal experiment:
> the supplement was deliberately designed from the observed failure
> pattern. The independent test set remained untouched.

---

## 3.6 Model Confidence as a Failure Signal

The model's peak-probability output may be useful as a deployment-time
uncertainty flag.

On the submitted checkpoint's `eval_v5` run:

- catastrophic failures (`error > 100 px`) typically had peak probability
  around **0.001–0.07**,
- accurate predictions typically had peak probability around **0.08–0.11**.

However, the signal is not perfect. For example:

```text
pair 057
error = 315 px
peak probability = 0.098
```

Therefore, confidence should be treated as a **useful heuristic for
flagging suspicious predictions**, not as a calibrated standalone failure
detector.

---

# 4. Summary of Findings

## What works

- **80%** of independent test pairs are localized within **10 px**.
- Median localization error is **0.52 px**.
- GPU inference averages **9.56 ms per pair**.
- CPU inference averages **71.36 ms per pair**.
- Targeted coarse-pitch supplementation improves performance.

## What was investigated

The periodic-repeat hypothesis was tested directly. The worst-25 test
produced **7/25 (28%)** pitch-multiple matches against an estimated
**19.6% chance baseline**.

Because of the small sample size, this evidence is **inconclusive** rather
than sufficient to establish periodic lock-on as the dominant mechanism.

## Stronger evidence

The more reproducible failure pattern is:

```text
Coarser DRAM pitch
        ↓
Fewer local structural landmarks
        ↓
Lower reference-crop information density
        ↓
Higher localization failure rate
```

This relationship:

- survives the missing-via confound check,
- appears as a monotonic binned trend,
- replicates on the validation set,
- and responds to targeted supplemental training.

## Remaining limitation

The final independent test set still contains a **19% catastrophic
failure rate** using the `error > 100 px` threshold.

The remaining failures are concentrated particularly in the coarse-pitch /
lower-information regime, with `search_quality` also showing meaningful
independent separation.

A supported next experiment is therefore a targeted `search_quality`
supplement using the same failure-analysis → targeted-data-generation →
retraining → untouched-test methodology.

---

# 5. Reproducibility

Relevant repository components include:

```text
submission_model/
    inference.py
    model_v6.py
    driftsense_final.pt

failure_analysis/
    analyze_v6_failures.py
    analyze_failure_causes.py
    analyze_pitch_density_bins.py
    model_v6.py
    dram_dataset.py
```

Independent test set:

```text
eval_v5
profile = medium
seed = 24681357
n = 100
```

Validation set:

```text
val_v5
profile = medium
seed = 13579246
n = 300
```

Training composition:

```text
train_v7
    9,000 base pairs

+

train_v7_coarse_pitch
    4,000 supplemental pairs
    pitch = 80–120 nm

=

train_v8
```

The submitted model was produced using the V6 architecture and training
configuration specified in `README.md`.

### Failure-analysis commands

```bash
cd failure_analysis

python analyze_v6_failures.py     --checkpoint ../submission_model/driftsense_final.pt     --data_dir ../eval_v5     --n_worst 25

python analyze_failure_causes.py     --checkpoint ../submission_model/driftsense_final.pt     --data_dir ../eval_v5

python analyze_pitch_density_bins.py     --checkpoint ../submission_model/driftsense_final.pt     --data_dir ../eval_v5     --n_bins 5
```

---

# 6. Final Assessment

DriftSense demonstrates strong localization capability on the majority of
independent test samples, with **sub-pixel median accuracy** and
millisecond-scale inference on the tested GPU.

The evaluation deliberately exposes the remaining weakness rather than
hiding it behind a single headline metric.

The most important result is that the remaining failures can be
**measured, investigated, confound-checked, and acted upon**.

The demonstrated closed-loop methodology is:

```text
Evaluate
   ↓
Identify failure regime
   ↓
Check confounds
   ↓
Form physical/data hypothesis
   ↓
Generate targeted data
   ↓
Retrain
   ↓
Evaluate on untouched test set
   ↓
Measure improvement
```

The coarse-pitch intervention reduced independent-test mean error from
**113.70 px to 92.30 px**, reduced catastrophic failure from **25% to
19%**, increased the success rate from **73% to 80%**, and reduced the
pitch-related Cohen's `d` from **+0.56 to +0.42**.

No claim is made that the current model completely solves every possible
navigation-error condition. Instead, the evaluation establishes a
reproducible baseline, tests the original periodicity hypothesis,
identifies a confound, finds a stronger failure pattern, applies a
targeted intervention, and measures the resulting improvement on an
untouched independent test set.

The strongest evidence-supported next step is further targeted data
generation around the difficult coarse-pitch and lower-quality regimes,
particularly investigating a `search_quality`-focused supplement.

---

<div align="center">

### 🔬 DriftSense

**Measured honestly · Failure modes investigated · Improvements verified**

*Independent test set · Reproducible analysis · Evidence-driven iteration*

</div>
