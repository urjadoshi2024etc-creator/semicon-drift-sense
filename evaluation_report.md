<div align="center">

# 📊 DriftSense — Evaluation Report

### Independent evaluation, inference performance, and failure analysis

**Model:** `driftsense_final.pt`  
**Architecture:** V6  
**Training Dataset:** merged `train_v8`  
**Independent Test Set:** `eval_v5` — 100 pairs

</div>

---

## 🧭 Executive Summary

DriftSense was evaluated on an **independent 100-pair test set** that was held out from all training and model-selection decisions.

The submitted checkpoint:

```text
driftsense_final.pt
```

was selected at:

```text
Best checkpoint: Epoch 44
Early stopping: Epoch 59
Patience: 15 epochs
Minimum improvement: 0.5 px
```

### Headline result

> **80% of independent test pairs were localized within 10 pixels, with a median localization error of only 0.52 px.**

The large difference between median and mean error is important: most examples are localized extremely accurately, while a smaller catastrophic-failure subset produces very large errors.

The failure-analysis pipeline investigated multiple hypotheses and identified **coarse pitch / low local reference-crop information density** as the strongest confirmed structural difficulty. A targeted supplemental training set was then used to address that regime, producing a measurable improvement.

---

# 1. 🧪 Evaluation Setup

## 1.1 Submitted Model

| Property | Configuration |
|---|---|
| Model | `driftsense_final.pt` |
| Architecture | V6 |
| Training Dataset | merged `train_v8` |
| Best Checkpoint | Epoch 44 |
| Early Stopping | Epoch 59 |
| Patience | 15 epochs |
| Minimum Improvement | 0.5 px |

The exact reproduction procedure is documented in:

```text
README.md
```

See the training/reproduction section for the complete dataset-generation, merge, and training commands.

---

## 1.2 Independent Test Set

All headline evaluation numbers in this report are measured on:

```text
eval_v5
```

Configuration:

```text
Number of pairs: 100
Profile: medium
Seed: 24681357
```

The test set was:

- Held out from training
- Held out from checkpoint selection
- Held out from model-selection decisions
- Used only for final reported evaluation

This separation is important because it prevents the headline test numbers from being directly optimized during model development.

---

# 2. 🎯 Headline Accuracy

## `eval_v5` — n = 100

| Metric | Result |
|---|---:|
| **Mean error** | **92.30 px** |
| **Median error** | **0.52 px** |
| **Minimum error** | **0.04 px** |
| **Maximum error** | **818.84 px** |
| **Within 10 px** | **80 / 100 (80.0%)** |
| **Within 25 px** | **80 / 100 (80.0%)** |
| **Within 50 px** | **80 / 100 (80.0%)** |
| **Within 100 px** | **81 / 100 (81.0%)** |

---

## 2.1 Validation Set

The validation set was:

```text
Dataset: val_v5
Pairs: 300
Seed: 13579246
```

It was used during training for checkpoint selection and **was not used for the independent-test headline numbers above**.

Best validation performance:

```text
Best mean error = 76.595 px
Epoch = 44
```

---

# 3. 📐 Understanding the Mean / Median Gap

The large difference between:

```text
Median error = 0.52 px
Mean error   = 92.30 px
```

is not simply random noise.

It reflects two genuinely different performance regimes:

```text
                    100 Test Pairs
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
       Majority of pairs         Smaller subset
       highly accurate           catastrophic failures
             │                         │
             ▼                         ▼
       sub-pixel scale             very large error
             │                         │
             └────────────┬────────────┘
                          ▼
                Mean strongly affected
                by catastrophic cases
```

The majority of pairs are localized at or near sub-pixel accuracy, while approximately **19–20%** of cases fail badly enough to dominate the mean.

Therefore:

- **Median alone** would overstate overall reliability.
- **Mean alone** would understate typical-case performance.
- Both are reported deliberately.

This is an important characteristic of the current model and should not be hidden by reporting only one statistic.

---

# 4. ⚡ Inference Timing

Inference was measured using 1000 × 1000 image pairs.

| Device | Mean Time / Pair |
|---|---:|
| **GPU — RTX 4050 Laptop, 6 GB, CUDA 12.1** | **9.56 ms** |
| **CPU — forced CPU execution** | **71.36 ms** |

### GPU measurement

```text
Runs: 50
Warmup: 5 runs
Synchronization: torch.cuda.synchronize()
```

### CPU measurement

```text
Runs: 20
Warmup: 3 runs
```

Both execution modes are comfortably fast for per-pair inference.

`inference.py` automatically detects the available execution device:

```text
CUDA available
      ↓
GPU inference

CUDA unavailable
      ↓
CPU inference
```

No source-code modification is required.

---

# 5. 🔍 Failure Analysis

The failure-analysis workflow was designed to move beyond simply reporting errors.

The investigation considered:

1. Periodic-pitch-repeat lock-on
2. Generator-parameter differences between successful and failed pairs
3. Potential confounding between defect counts and pitch
4. Pitch-dependent failure rates
5. Reference-crop information density
6. Targeted supplemental training
7. Model confidence as a potential failure signal

The analysis is implemented in:

```text
failure_analysis/
│
├── analyze_v6_failures.py
├── analyze_failure_causes.py
└── analyze_pitch_density_bins.py
```

---

# 6. 🔁 Periodic Pitch-Repeat Lock-On

## 6.1 Hypothesis

The initial hypothesis was:

> Large localization errors may occur because the model locks onto a visually similar but physically incorrect periodic repeat of the DRAM pattern.

The expected error structure would approximately be:

```text
predicted location
      ≈
true location + k × pitch
```

for some integer `k`.

This is a particularly plausible failure mechanism for highly periodic DRAM structures.

---

## 6.2 Direct Test

For the worst-error cases, the analysis checked whether the error vector:

```text
(dx, dy)
```

landed within one pitch-cell tolerance of an integer pitch multiple.

Parameters:

```text
Tolerance:
0.25 × pitch

Pitch:
sample-specific pitch_x_nm / pitch_y_nm

Search-image conversion:
divide by 10
```

Integer multiples searched:

```text
k ∈ [-100, 100]
```

per axis.

---

## 6.3 Result

On the submitted checkpoint:

```text
Model:
driftsense_final.pt

Dataset:
eval_v5

Worst cases:
25
```

the result was:

```text
7 / 25 = 28%
```

of cases matching an integer pitch multiple within the defined tolerance.

---

## 6.4 Chance-Baseline Check

The analysis also compared this against the geometric chance probability.

Given:

```text
Tolerance radius = 0.25 × pitch
```

the approximate geometric collision probability is:

```text
π × (0.25)²
≈ 19.6%
```

for a random error vector to land near some pitch-multiple lattice point simply due to lattice density.

With:

```text
n = 25
```

the standard error of the observed proportion is approximately:

```text
8 percentage points
```

Therefore:

```text
Observed: 28%
Chance baseline: ~20%
```

The observed value is only about one standard error above the baseline.

---

## 6.5 Conclusion

The test does **not** provide strong evidence either for or against periodic lock-on as a meaningful failure contributor on the submitted checkpoint.

This result is intentionally reported conservatively.

An earlier test on a prior checkpoint showed:

```text
3 / 25
```

matching cases, which was closer to the chance baseline.

The difference between the earlier and current values can plausibly be explained by sampling noise at this sample size rather than a confirmed change in failure mechanism.

> **Important:** periodic lock-on should therefore not be presented as the confirmed dominant failure mode.

The stronger evidence comes from the coarse-pitch / reference-density analysis described below.

---

# 7. 📊 Generator-Parameter Failure Analysis

The next analysis compared every logged generator parameter between:

```text
FAILURE:
error > 100 px

SUCCESS:
error < 10 px
```

on `eval_v5`.

One intermediate case was excluded.

The analysis used **Cohen's d** to rank the separation between failure and success groups.

For the submitted model:

```text
Checkpoint:
v6_train_v8

Failure group:
n = 19

Success group:
n = 80

Intermediate / excluded:
n = 1
```

---

## 7.1 Ranked Parameter Separation

| Rank | Parameter | Cohen's d | Interpretation |
|---:|---|---:|---|
| **1** | `n_missing_vias` | **−0.77** | Confounded — see Section 8 |
| **2** | `search_quality` | **+0.51** | Real, independent factor |
| **3** | `pitch_x_nm` | **+0.42** | Real, independent factor |
| **4** | `ref_scale_factor` | **+0.40** | Weaker, plausibly secondary |
| **5** | `pitch_y_nm` | **+0.34** | Real, independent factor |

All other parameters showed:

```text
|d| < 0.35
```

including:

- Rotation
- Elastic warp
- Vignette
- Scan-line variation
- Fractal background
- Defect counts other than missing-via

These showed negligible-to-small individual separation in this analysis.

---

# 8. ⚠️ Confound Discovery: Missing-Via Count

The strongest raw parameter signal was:

```text
n_missing_vias
```

with:

```text
Cohen's d = -0.77
```

Initially, this could appear to suggest that missing-via defects were the primary cause of failures.

Further investigation showed that this interpretation would be misleading.

---

## 8.1 Why It Is Confounded

The generator lays the via grid across a fixed-size world.

Therefore:

```text
Finer pitch
     ↓
More via sites fit into the same physical area
     ↓
More opportunities for missing vias
     ↓
Higher raw missing-via count
```

Mathematically, raw missing-via count scales mechanically with approximately:

```text
1 / (pitch_x × pitch_y)
```

Therefore the count is partly a proxy for pitch.

---

## 8.2 Direct Verification

Measured correlation between:

```text
n_missing_vias
```

and:

```text
1 / (pitch_x_nm × pitch_y_nm)
```

was:

| Dataset | Correlation |
|---|---:|
| `eval_v5` | **0.755** |
| `val_v5` | **0.711** |

This confirms that the apparent predictive power of `n_missing_vias` is substantially redundant with the pitch signal.

> **Conclusion:** missing-via count should not be interpreted as an independent causal explanation for the failure pattern.

---

# 9. 📐 Root Cause: Coarse Pitch Reduces Local Information

The stronger physical hypothesis is:

> A coarse DRAM pitch reduces the amount of distinguishing local structure contained inside the fixed-size reference crop.

The reference crop represents a fixed physical region:

```text
1000 nm × 1000 nm
```

At coarse pitch:

```text
Larger pitch
      ↓
Fewer via sites / repeated structures
      ↓
Less local structural information
      ↓
Fewer distinguishing features available to the model
      ↓
Higher localization difficulty
```

This explanation is independent of where the model eventually makes its wrong prediction.

---

# 10. 📈 Non-Confounded Binned Pitch Analysis

The hypothesis was tested by directly binning samples by pitch rather than by the confounded via-count proxy.

The analysis was performed on:

```text
Checkpoint:
v6_train_v7

Dataset:
eval_v5
```

---

## 10.1 Failure Rate by Pitch

| `pitch_x_nm` Bin | n | Failure Rate >100 px |
|---|---:|---:|
| 40.6–57.1 | 20 | **10.0%** |
| 57.1–73.6 | 20 | **20.0%** |
| 73.6–93.4 | 20 | **30.0%** |
| 93.4–106.6 | 20 | **30.0%** |
| 106.6–119.3 | 20 | **35.0%** |

This shows a clean monotonic increase:

```text
Fine pitch
   │
   │ 10%
   │
   │       20%
   │
   │             30%
   │
   │                   30%
   │
   │                         35%
   ▼
Coarse pitch
```

The failure rate rises approximately:

```text
3.5×
```

from the finest to the coarsest pitch bin.

---

## 10.2 Replication

The same direction of trend was replicated on:

```text
val_v5
```

with failure rates ranging from:

```text
13.3% → 35.0%
```

The trend was also independently reproduced using:

```text
ref_via_density_estimate
```

defined as the estimated number of via sites per reference crop.

The via-density variable showed the same relationship in the opposite direction:

```text
Higher via density
      ↓
Lower failure rate

Lower via density
      ↓
Higher failure rate
```

This provides mutually consistent evidence for the local-information-density explanation.

---

# 11. 🛠️ Targeted Intervention

Once coarse pitch was identified as a strong non-confounded failure factor, a targeted dataset intervention was performed.

A supplemental dataset of:

```text
4,000 pairs
```

was generated specifically in:

```text
80–120 nm pitch
```

using:

```bash
--pitch_min 80
--pitch_max 120
```

All other parameters remained on the `medium` profile.

The supplemental set was merged with the original:

```text
9,000-pair base dataset
```

giving:

```text
13,000 total training pairs
```

The model architecture and loss configuration were kept unchanged.

This was intended to isolate **training-data composition** as the changed variable.

---

# 12. 📊 Before vs. After Supplemental Training

Evaluation was performed on the same:

```text
eval_v5
```

test set.

| Metric | Before — `train_v7` | After — + Coarse-Pitch Supplement |
|---|---:|---:|
| Validation mean error | 96.74 px | **76.60 px** |
| Test mean error | 113.70 px | **92.30 px** |
| Failure rate >100 px | 25% | **19%** |
| Success rate <10 px | 73% | **80%** |
| `pitch_x_nm` Cohen's d | +0.56 | **+0.42** |

---

## 12.1 Interpretation

The intervention produced improvements in multiple related measurements:

```text
Validation mean error
96.74 → 76.60 px

Test mean error
113.70 → 92.30 px

Failure rate
25% → 19%

Success rate
73% → 80%

Pitch Cohen's d
+0.56 → +0.42
```

The shrinkage in pitch-related Cohen's d is especially informative.

The targeted intervention reduced the measured separation associated with the regime it was designed to address.

This is directionally consistent with the original hypothesis.

However:

> The effect was **not eliminated**.

Residual pitch-related difficulty remains.

The report therefore treats the intervention as evidence supporting the diagnosis, rather than claiming that the problem has been completely solved.

---

# 13. 🎯 Model Confidence as a Potential Failure Signal

The model's own peak-probability output was also examined as a possible deployment-time uncertainty signal.

On the submitted checkpoint's `eval_v5` run:

### Catastrophic failures

```text
Error > 100 px
```

typically showed:

```text
Peak probability:
~0.001–0.07
```

### Accurate predictions

typically showed:

```text
Peak probability:
~0.08–0.11
```

This suggests that the model's confidence may provide a useful practical signal for flagging uncertain predictions.

---

## 13.1 Important Limitation

The confidence signal is **not a reliable failure detector by itself**.

At least two failure cases showed confidence within the normal range.

Example:

```text
Pair 057
Error = 315 px
Peak probability = 0.098
```

Therefore:

```text
Confidence
    ↓
Useful heuristic
    ≠
Guaranteed failure detector
```

A production deployment should treat confidence as an additional uncertainty signal rather than a definitive decision rule.

---

# 14. 🧠 Evidence Chain

The evaluation process can be summarized as an evidence-driven loop:

```text
Initial Model
     │
     ▼
Independent Evaluation
     │
     ▼
Large Failure Subset Identified
     │
     ▼
Hypothesis:
Periodic Repeat Lock-On
     │
     ▼
Direct Test
     │
     ▼
Evidence Inconclusive
     │
     ▼
Analyze Generator Parameters
     │
     ▼
Missing-Via Count Appears Strong
     │
     ▼
Confound Check
     │
     ▼
Missing-Via Count ↔ Pitch
     │
     ▼
Confound Identified
     │
     ▼
Direct Pitch Binning
     │
     ▼
Monotonic Failure-Rate Trend
     │
     ▼
Coarse-Pitch / Low-Density Hypothesis
     │
     ▼
Targeted 4,000-Pair Supplement
     │
     ▼
Retrain
     │
     ▼
Independent Evaluation
     │
     ▼
Measured Improvement
```

This is the central experimental reasoning behind the current V6 training dataset.

---

# 15. 🧪 Reproducibility

The failure-analysis scripts used to produce these findings are included in the repository:

```text
failure_analysis/
├── analyze_v6_failures.py
├── analyze_failure_causes.py
└── analyze_pitch_density_bins.py
```

They can be rerun against the submitted checkpoint.

### Periodic-repeat analysis

```bash
cd failure_analysis

python analyze_v6_failures.py \
    --checkpoint ../submission_model/driftsense_final.pt \
    --data_dir ../eval_v5 \
    --n_worst 25
```

### Failure-cause comparison

```bash
python analyze_failure_causes.py \
    --checkpoint ../submission_model/driftsense_final.pt \
    --data_dir ../eval_v5
```

### Pitch / density analysis

```bash
python analyze_pitch_density_bins.py \
    --checkpoint ../submission_model/driftsense_final.pt \
    --data_dir ../eval_v5 \
    --n_bins 5
```

---

# 16. 📋 Final Results Summary

## Core Performance

- **80 / 100 (80.0%)** of independent test pairs localized within **10 px**.
- **81 / 100 (81.0%)** localized within **100 px**.
- Median localization error was **0.52 px**.
- Best individual error was **0.04 px**.
- Worst individual error was **818.84 px**.
- Mean error was **92.30 px**.

## Runtime

- **9.56 ms/pair** on RTX 4050 Laptop GPU.
- **71.36 ms/pair** on CPU.

## Failure Analysis

- Periodic-pitch-repeat lock-on was **tested directly**.
- The result was **not strong enough to establish periodic lock-on as the dominant mechanism**.
- `n_missing_vias` initially appeared strongly associated with failures, but was shown to be **confounded with pitch**.
- Coarse pitch showed a clear, monotonic relationship with failure rate.
- Lower estimated via density inside the fixed reference crop showed the same relationship.
- A targeted coarse-pitch supplemental dataset produced measurable improvement.

---

# 17. ⚠️ Known Limitations

The current model has an important residual limitation:

```text
~19% catastrophic-failure rate
```

on the reported `eval_v5` set when catastrophic failure is defined as:

```text
error > 100 px
```

The remaining failures are concentrated particularly in:

```text
Coarse-pitch regimes
+
Lower-quality search images
```

The evidence-supported next step is therefore **not** to randomly enlarge the dataset, but to continue the same targeted methodology.

A logical next intervention would be:

```text
Search-quality-focused supplemental dataset
```

followed by the same:

```text
Generate
  ↓
Train
  ↓
Evaluate
  ↓
Analyze
  ↓
Intervene
  ↓
Re-evaluate
```

workflow.

---

# 18. 🏁 Conclusion

DriftSense demonstrates a strong typical-case localization capability:

> **Median error = 0.52 px**

while achieving:

> **80% success within 10 px on an untouched 100-pair test set.**

At the same time, the evaluation deliberately exposes the model's current weakness rather than hiding it behind a favorable statistic.

The large mean/median gap reveals a distinct catastrophic-failure regime.

The investigation then followed a falsifiable, evidence-driven process:

```text
Hypothesis
   ↓
Direct test
   ↓
Confound analysis
   ↓
Binned analysis
   ↓
Targeted intervention
   ↓
Independent verification
```

The strongest confirmed remaining difficulty is associated with:

```text
coarse DRAM pitch
+
low local reference-crop information density
```

and a targeted supplemental training set produced a measurable improvement:

```text
Failure rate:
25% → 19%

Success rate (<10 px):
73% → 80%

Test mean error:
113.70 → 92.30 px
```

The remaining limitation is explicitly acknowledged, and the report provides a reproducible path for further improvement.

---

<div align="center">

### 🔬 DriftSense

**Measure → Analyze → Diagnose → Intervene → Verify**

*Evaluation is treated as an engineering feedback loop, not just a final score.*

</div>
