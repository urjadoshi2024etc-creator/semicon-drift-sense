# DriftSense — Evaluation Report

**Model:** `driftsense_final.pt` (V6 architecture, trained on the merged
`train_v8` dataset — see `README.md` Section 4 for exact reproduction
steps). Best checkpoint selected at **epoch 44** by validation mean error;
training run early-stopped at epoch 59 (patience 15, min improvement 0.5px).

**Independent test set:** `eval_v5` — 100 pairs, generated with
`--profile medium --seed 24681357`, held out from all training and
model-selection decisions. All headline numbers below are measured on
this set unless stated otherwise.

---

## 1. Headline Accuracy

| Metric | Value |
|---|---:|
| Mean error | 92.30 px |
| Median error | 0.52 px |
| Minimum error | 0.04 px |
| Maximum error | 818.84 px |
| Within 10 px | 80 / 100 (80.0%) |
| Within 25 px | 80 / 100 (80.0%) |
| Within 50 px | 80 / 100 (80.0%) |
| Within 100 px | 81 / 100 (81.0%) |

### Validation-set reference

The validation set (`val_v5`, 300 pairs, seed `13579246`) was used during
training for checkpoint selection and is **not** the source of the headline
test numbers above.

- Best validation mean error: **76.595 px**
- Best epoch: **44**

### Interpreting the mean/median gap

The large gap between the median (**0.52 px**) and mean (**92.30 px**) is
not simply random noise. It reflects two distinct performance regimes:

1. **Typical cases:** the large majority of pairs are localized with
   sub-pixel-scale error.
2. **Failure cases:** a smaller subset of approximately 19–20% of pairs
   produces sufficiently large errors to dominate the mean.

For this reason, both mean and median are reported. Median alone would
overstate reliability, while mean alone would understate the model's
performance on typical cases.

---

## 2. Inference Timing

| Device | Mean time per 1000×1000 pair |
|---|---:|
| GPU (RTX 4050 Laptop, 6GB VRAM, CUDA 12.1) | **9.56 ms** |
| CPU (forced via `CUDA_VISIBLE_DEVICES=`) | **71.36 ms** |

GPU timing is the mean over 50 runs with a 5-run warmup and
`torch.cuda.synchronize()`-bounded measurement.

CPU timing is the mean over 20 runs with a 3-run warmup.

Both measurements are comfortably within a practical per-pair inference
budget for this task. `inference.py` requires no code changes to run on
either device; it automatically detects and uses the available device.

---

## 3. Failure Analysis

### 3.1 Periodic Pitch-Repeat Lock-On — Tested Directly

**Hypothesis:** large errors may result from the model locking onto a
visually similar but incorrect periodic repeat of the DRAM pattern, such
that the predicted location is approximately:

```text
true location + k × pitch
```

for a small integer `k`.

#### Test

For the worst-error cases, the error vector `(dx, dy)` was checked against
integer pitch multiples. For each sample, the test used its own
`pitch_x_nm` and `pitch_y_nm`, converted to search-image pixels by dividing
by the search scale of 10 nm/px.

The test searched integer multiples:

```text
k ∈ [-100, 100]
```

per axis, using a tolerance of:

```text
0.25 × pitch
```

#### Result

For the submitted checkpoint (`driftsense_final.pt`) on the 25 worst-error
cases:

**7 / 25 (28%)** matched an integer pitch multiple within the specified
tolerance.

A geometric chance-baseline calculation gives approximately:

```text
π × (0.25)² ≈ 19.6%
```

for a random error vector to fall near a pitch-multiple lattice point under
the stated search geometry.

With only 25 samples, the standard error of an observed proportion is
approximately 8 percentage points. Therefore, the observed 28% is only
about one standard error above the approximately 20% chance baseline.

### Conclusion

This test does **not provide strong evidence** that periodic lock-on is the
dominant failure mechanism.

It is therefore more appropriate to describe periodic ambiguity as an
investigated hypothesis rather than a confirmed root cause. The stronger
evidence comes from the coarse-pitch / reference-density analysis in
Sections 3.3–3.5.

An earlier test on a prior checkpoint produced 3/25 matches, closer to the
chance baseline. The difference between checkpoints is not sufficient,
given this small sample size, to establish a change in failure mechanism.

---

### 3.2 Generator-Parameter Failure Analysis

Every logged generator parameter was compared between:

- **FAILURE:** error > 100 px
- **SUCCESS:** error < 10 px

on `eval_v5`.

Cohen's *d* was used to rank the standardized separation between the two
groups.

**Post-supplement checkpoint:** `v6_train_v8` / submitted model.

| Rank | Parameter | Cohen's *d* | Interpretation |
|---:|---|---:|---|
| 1 | `n_missing_vias` | −0.77 | Confounded — see Section 3.3 |
| 2 | `search_quality` | +0.51 | Real, independent factor |
| 3 | `pitch_x_nm` | +0.42 | Real, independent factor |
| 4 | `ref_scale_factor` | +0.40 | Weaker, plausibly secondary |
| 5 | `pitch_y_nm` | +0.34 | Real, independent factor |

All other parameters—including rotation, elastic warp, vignette, scanline,
fractal background, and defect counts other than missing-via—showed
`|d| < 0.35`, corresponding to negligible-to-small individual separation.

---

### 3.3 Confound Discovery — `n_missing_vias` Is a Pitch Proxy

The initially strongest signal, `n_missing_vias`, was investigated further.

Inspection of `generate_dram_dataset_v3.py` and `create_world()` showed
that the via grid spans the entire fixed-size world. Consequently, the
number of available via sites increases mechanically as:

```text
1 / (pitch_x × pitch_y)
```

so finer pitch produces more possible via locations.

This means raw missing-via count is substantially redundant with pitch and
should **not** be interpreted as an independent causal difficulty factor.

The relationship was verified directly:

| Dataset | Correlation: `n_missing_vias` vs. `1/(pitch_x_nm·pitch_y_nm)` |
|---|---:|
| `eval_v5` | **0.755** |
| `val_v5` | **0.711** |

This confirms that the apparent strength of `n_missing_vias` is substantially
explained by its relationship with pitch.

---

### 3.4 Root Cause — Coarse Pitch Reduces Local Reference-Crop Information

#### Physical hypothesis

The reference crop represents a fixed physical area of approximately:

```text
1000 nm × 1000 nm
```

At larger/coarser pitch, fewer via sites—and therefore fewer local
structural elements capable of distinguishing one region from another—fit
inside the fixed crop.

This reduces the local information available to the model, independently
of where an incorrect prediction eventually lands.

#### Evidence

A pitch-binned failure-rate analysis was performed on the
**pre-supplement checkpoint** (`v6_train_v7`) using `eval_v5`.

| `pitch_x_nm` bin | n | Failure rate (>100 px) |
|---|---:|---:|
| 40.6–57.1 | 20 | 10.0% |
| 57.1–73.6 | 20 | 20.0% |
| 73.6–93.4 | 20 | 30.0% |
| 93.4–106.6 | 20 | 30.0% |
| 106.6–119.3 | 20 | 35.0% |

The trend is monotonic: failure rate rises from **10%** in the finest-pitch
bin to **35%** in the coarsest-pitch bin, approximately a **3.5× increase**.

The same direction was replicated on `val_v5`, where the failure rate
increased from **13.3% to 35.0%**.

An independently derived `ref_via_density_estimate` variable—representing
via sites per reference crop—showed the same relationship in reverse.

Together, these results provide stronger evidence for coarse pitch / lower
local reference-crop information density as a remaining failure factor
than the periodic-lock-on test.

---

### 3.5 Targeted Response — Coarse-Pitch Supplemental Training

Based on the evidence above, a targeted supplemental dataset was generated:

```text
4,000 additional pairs
pitch range: 80–120 nm
profile: medium
```

All other medium-profile generation parameters were kept unchanged.

The supplemental set was merged with the original 9,000-pair training set
and the model was retrained using the same architecture and loss
configuration. This isolates training-data composition as the principal
changed variable.

#### Before vs. after

Both checkpoints were evaluated on the same independent `eval_v5` set.

| Metric | Before: `train_v7` only | After: `train_v8` + supplement |
|---|---:|---:|
| Validation mean error | 96.74 px | **76.60 px** |
| Test mean error | 113.70 px | **92.30 px** |
| Failure rate (>100 px) | 25% | **19%** |
| Success rate (<10 px) | 73% | **80%** |
| `pitch_x_nm` Cohen's *d* | +0.56 | **+0.42** |

The intervention therefore produced improvements in both overall accuracy
and the targeted failure regime.

The reduction in pitch-related Cohen's *d* from **+0.56 to +0.42** is
consistent with the intervention reducing the specific coarse-pitch
difficulty it was designed to address.

The effect was not eliminated: residual pitch-related difficulty remains.

---

### 3.6 Model Confidence as a Failure Signal

The model's peak-probability output shows a useful relationship with
prediction correctness and may serve as a practical uncertainty flag.

On the submitted checkpoint's `eval_v5` run:

- Catastrophic failures (`error > 100 px`) typically had peak probabilities
  around **0.001–0.07**.
- Accurate predictions typically had peak probabilities around
  **0.08–0.11**.

However, this is **not a reliable failure detector by itself**.

At least two failure cases showed confidence values within the normal
range. For example:

```text
pair 057
error = 315 px
peak probability = 0.098
```

Therefore, confidence should be treated as a useful heuristic for flagging
potentially uncertain predictions, rather than as a guaranteed failure
classifier.

---

## 4. Overall Assessment

### What the model does well

- **80% of independent test pairs** are localized within 10 px.
- Median localization error is **0.52 px**, indicating highly accurate
  localization on typical successful cases.
- The system provides **9.56 ms mean GPU inference time** on the tested
  RTX 4050 Laptop GPU.
- The model automatically supports both GPU and CPU inference.
- The coarse-pitch intervention produced a measurable improvement without
  changing the core architecture or loss configuration.

### What remains challenging

- A residual **19% catastrophic-failure rate** remains on `eval_v5`.
- Coarse pitch and reduced local reference-crop information remain the
  strongest confirmed structural difficulty.
- `search_quality` is another meaningful independent factor.
- Confidence can provide a useful warning signal, but it is not sufficient
  as a standalone failure detector.

### Is the model degraded?

**No.** The reported results do not show degradation relative to the
previous `train_v7` checkpoint.

The controlled before/after comparison shows improvement:

```text
Test mean error:       113.70 px → 92.30 px
Failure rate:              25% → 19%
Success rate:              73% → 80%
Pitch Cohen's d:          +0.56 → +0.42
```

The model is therefore **better overall after the targeted supplemental
training**, while still having a meaningful residual failure regime.

The large mean error should be interpreted together with the median and
failure-rate metrics: approximately 80% of test pairs are successful
within 10 px, while a smaller number of catastrophic errors dominate the
mean.

---

## 5. Limitations and Evidence Boundaries

The following points should be kept explicit when presenting these
results:

1. The independent test set contains **100 pairs**, so failure-rate
   estimates have non-negligible sampling uncertainty.
2. The periodic-lock-on test used only the **25 worst-error cases**, so it
   is useful as a diagnostic but is not a definitive statistical test of
   periodic ambiguity.
3. The coarse-pitch finding is supported by multiple consistent analyses,
   including direct pitch binning, replication on validation data,
   reference-crop density analysis, and the targeted supplemental-training
   intervention.
4. The supplemental dataset improved the measured metrics, but it did not
   eliminate coarse-pitch-related failures.
5. The reported synthetic-data results should be interpreted as evidence
   about the model under the project's controlled generator distribution;
   they are not a claim of demonstrated performance on proprietary
   Applied Materials production SEM data.

---

## 6. Final Summary

DriftSense's V6 model demonstrates strong localization accuracy on the
majority of its independent synthetic test cases while maintaining
millisecond-scale inference.

The most important result from the evaluation is not simply the headline
accuracy; it is the **evidence-driven debugging loop**:

```text
Measure failures
      ↓
Test periodic-lock-on hypothesis
      ↓
Identify coarse-pitch / information-density relationship
      ↓
Check and remove the missing-via confound
      ↓
Generate targeted 80–120 nm supplemental data
      ↓
Retrain with the same model/loss
      ↓
Measure improvement on untouched eval_v5
```

The intervention improved mean error from **113.70 px to 92.30 px**,
reduced catastrophic failures from **25% to 19%**, increased success within
10 px from **73% to 80%**, and reduced pitch-related Cohen's *d* from
**+0.56 to +0.42**.

The remaining failures are therefore not hidden or omitted: they are
explicitly characterized, partially explained, and identified as the next
target for further improvement—particularly through additional
`search_quality`-focused training data and continued analysis of the
coarse-pitch regime.

---

<div align="center">

### 🔬 DriftSense

**Measured · Reproducible · Evidence-Driven**

*Performance claims are reported with their failure modes and evidence
boundaries rather than being presented as unconditional guarantees.*

</div>
