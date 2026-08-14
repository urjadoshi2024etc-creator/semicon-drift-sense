<div align="center">

# 🧭 DriftSense

### AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**SEMICON India Hackathon 2026 · Problem Statement 02**

**Physically informed synthetic DRAM SEM data · Deep-learning localization · Reproducible evaluation**

</div>

---

## 📌 Project Overview

Modern wafer-inspection tools repeatedly revisit the same locations on a wafer. At high magnification, even very small stage/navigation errors can shift the observed location away from the intended site.

**DriftSense** addresses this problem as an image-localization task:

```bash
High-quality reference image
            +
Low-dose / distorted search image
            │
            ▼
       DriftSense Model
            │
            ▼
 Predicted (x, y) location
 in search-image coordinates
```

The submitted system is designed around **DRAM periodic structure**, synthetic SEM acquisition effects, geometric drift, process variation, and hard defects.

### Core objective

Recover the position of a reference pattern inside a degraded search image despite:

- Translation / navigation drift
- Rotation
- Scale variation
- Local elastic distortion
- SEM noise
- Blur
- Illumination variation
- Raster / scan-line artifacts
- DRAM process variation
- Local defects

The model outputs a single:

```bash
(x, y)
```

coordinate in search-image pixel space.

---

# 🏗️ Repository Structure

```bash
DriftSense/
│
├── dataset_generator/
│   └── generate_dram_dataset_v3.py
│       └── Standalone synthetic DRAM SEM dataset generator
│
├── submission_model/
│   ├── inference.py
│   ├── model_v6.py
│   └── driftsense_final.pt
│
├── training/
│   ├── train_v6.py
│   ├── loss_v5.py
│   ├── dram_dataset.py
│   ├── model_v6.py
│   ├── merge_train_v8.py
│   └── utils_v5.py
│
├── failure_analysis/
│   ├── analyze_v6_failures.py
│   ├── analyze_failure_causes.py
│   ├── analyze_pitch_density_bins.py
│   ├── model_v6.py
│   └── dram_dataset.py
│
├── evaluation_report.md
├── citations.md
├── requirements.txt
└── README.md
```

### Documentation map

| File | Purpose |
|---|---|
| `README.md` | Installation, usage, reproduction, and project navigation |
| `citations.md` | Scientific / patent justification for augmentation and SEM modeling choices |
| `evaluation_report.md` | Accuracy, timing, failure analysis, intervention, and limitations |

> Detailed scientific justification and evaluation results are intentionally kept in their dedicated Markdown files so that this README remains focused on **using and reproducing the system**.

---

# ⚙️ 1. Environment Setup

## Clone the repository
```bash
git clone <this-repo-url>
cd DriftSense
```

## Create a virtual environment
```bash
python -m venv venv_drift
```

### Windows
```bash
venv_drift\Scripts\activate
```

### Linux / macOS
```bash
source venv_drift/bin/activate
```

---

## Install PyTorch
Install the version appropriate for your machine.

### Option A — NVIDIA GPU / CUDA 12.1
This is the environment used during development:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### Option B — CPU-only
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

For another CUDA version, use the corresponding PyTorch installation command.
Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

### Tested environment

```bash
Python       3.11
PyTorch      2.5.1+cu121
GPU          NVIDIA RTX 4050 Laptop GPU
VRAM         6 GB
```

The project also runs on CPU. `inference.py` automatically detects CUDA and falls back to CPU when unavailable.

---

# 🚀 2. Quick Start — Run Inference
The fastest way to verify the complete pipeline is to generate one sample pair and run the submitted model.
From the repository root:

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 1 \
    --out_dir ./sample \
    --profile medium \
    --seed 1
```

Then:
```bash
cd submission_model

python inference.py \
    ../sample/reference/ref_00000.png \
    ../sample/search/search_00000.png
```
Expected output:

```bash
(x.xx, y.xx)
```

This is the predicted center coordinate in **search-image pixel space**.

The corresponding ground-truth values are available in:

```bash
sample/labels.csv
```

---

# 🧪 3. Dataset Generator

The generator creates physically informed synthetic DRAM SEM image pairs.

## Basic command

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs <N> \
    --out_dir <DIR> \
    --profile {easy,medium,hard} \
    --seed <SEED> \
    [--workers <N>] \
    [--style dram] \
    [--pitch_min <NM> --pitch_max <NM>]
```

### Generator options

| Argument | Description |
|---|---|
| `--n_pairs` | Number of image pairs |
| `--out_dir` | Output directory |
| `--profile` | Difficulty profile: `easy`, `medium`, or `hard` |
| `--seed` | Reproducibility seed |
| `--workers` | Optional parallel workers |
| `--style` | DRAM style; currently the only implemented option |
| `--pitch_min` | Optional minimum pitch override |
| `--pitch_max` | Optional maximum pitch override |

`--pitch_min` and `--pitch_max` must be supplied together.

Passing an unsupported style such as:

```bash
--style finfet
```

raises an explicit error rather than silently producing a mismatched dataset.

---

## Dataset outputs

Each run creates:

```bash
<DIR>/
├── reference/
├── search/
├── labels.csv
└── config.json
```

### Image specification

```bash
Reference image: 1000 × 1000 grayscale PNG
Search image:    1000 × 1000 grayscale PNG

Reference scale: 1 nm / pixel
Search scale:   10 nm / pixel

Magnification ratio: 10×
```

### `labels.csv`

Contains ground-truth localization information and generation metadata, including:

- Reference/search paths
- Center coordinates
- Pitch
- Image quality
- Rotation
- Defect information
- Other generation parameters

### `config.json`

Stores the exact configuration used for a generation run, including the random seed.

---

# 🧬 4. DRAM-Centered Synthetic Data

The dataset generator models DRAM because its repeating structure provides a useful controlled environment for studying navigation-error recovery.

The synthetic world includes:

```bash
DRAM grid structure
      │
      ├── Word-line / bit-line geometry
      ├── Contact / via sites
      ├── Line-width variation
      ├── Via-size variation
      └── Periodic local structure
```

Image acquisition then introduces controlled degradation:

```bash
Ideal DRAM structure
        │
        ├── SEM noise
        ├── Blur
        ├── Contrast / illumination variation
        ├── Scan-line artifacts
        ├── Translation
        ├── Rotation
        ├── Scale variation
        ├── Elastic distortion
        └── Defects
        │
        ▼
Synthetic reference/search pair
```

The detailed scientific rationale for these modeling choices is documented separately in:

```bash
citations.md
```

---

# 🔄 5. Reproduce the Submitted Training Dataset

The submitted checkpoint was trained using:

```bash
9,000-pair base dataset
+
4,000-pair coarse-pitch supplemental dataset
=
13,000 training pairs
```

The supplemental dataset targets the:

```bash
80–120 nm
```

pitch regime identified during failure analysis.

A separate:

```bash
300-pair validation set
```

was used during training.

The final:

```bash
100-pair eval_v5
```

test set was held out from training and model-selection decisions.

> The detailed accuracy numbers, failure analysis, statistical checks, and before/after intervention results are documented in `evaluation_report.md`.

---

## Step 1 — Generate base dataset

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 9000 \
    --out_dir ./train_v7 \
    --profile medium \
    --seed 7 \
    --workers 6
```

---

## Step 2 — Generate coarse-pitch supplemental dataset

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 4000 \
    --out_dir ./train_v7_coarse_pitch \
    --profile medium \
    --seed 17 \
    --workers 6 \
    --pitch_min 80 \
    --pitch_max 120
```

---

## Step 3 — Generate validation set

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 300 \
    --out_dir ./val_v5 \
    --profile medium \
    --seed 13579246
```

---

## Step 4 — Merge training datasets

Run from the directory containing both:

```bash
train_v7/
train_v7_coarse_pitch/
```

Then:

```bash
python training/merge_train_v8.py
```

The merge script validates the input datasets and refuses to proceed when the required row counts do not match or when the destination already exists.

---

# 🏋️ 6. Train From Scratch

After generating and merging the training data:

```bash
python training/train_v6.py \
    --train_dir ./train_v8 \
    --val_dir ./val_v5 \
    --output_dir ./runs/v6_train_v8 \
    --epochs 100 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 0.0003 \
    --weight_decay 1e-5 \
    --seed 42 \
    --lr_patience 20 \
    --lr_factor 0.5 \
    --grad_clip 5.0 \
    --save_every 10 \
    --early_stop_patience 15 \
    --min_delta 0.5
```

The best checkpoint is written to:

```bash
runs/v6_train_v8/checkpoints/best_model.pt
```

The submitted:

```bash
submission_model/driftsense_final.pt
```

is the corresponding copied checkpoint from the training run.

The submitted training run selected its best checkpoint at **epoch 44** and stopped at **epoch 59** through early stopping.

For the complete evaluation and interpretation of this training run, see:

```bash
evaluation_report.md
```

---

# 🎯 7. Submission Inference

This is the interface intended for evaluation/deployment.

```bash
python submission_model/inference.py \
    <reference_image_path> \
    <search_image_path>
```

### Input requirements

Both images must be:

```bash
1000 × 1000
grayscale PNG
```

### Output

The script prints one line:

```bash
(x, y)
```

where `(x, y)` is the predicted reference-pattern center in **search-image pixel coordinates**.

### Automatic model loading

`inference.py` automatically:

- Loads `driftsense_final.pt`
- Resolves the checkpoint relative to the script location
- Loads the architecture from `submission_model/model_v6.py`
- Detects CUDA automatically
- Falls back to CPU automatically

No code modification or additional runtime arguments are required.

---

# 🔬 8. Evaluation & Failure Analysis

The repository includes reproducible analysis scripts:

```bash
failure_analysis/
├── analyze_v6_failures.py
├── analyze_failure_causes.py
└── analyze_pitch_density_bins.py
```

Before running them, generate or provide a compatible held-out dataset at:

```bash
eval_v5/
```

The independent evaluation set used for the submitted results contains:

```bash
100 pairs
profile = medium
seed = 24681357
```

### Per-pair error / periodicity analysis

```bash
cd failure_analysis

python analyze_v6_failures.py \
    --checkpoint ../submission_model/driftsense_final.pt \
    --data_dir ../eval_v5 \
    --n_worst 25
```

### Generator-parameter failure comparison

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

# 📊 9. Results

The complete evaluation is intentionally documented in:

```bash
evaluation_report.md
```

That document contains:

- Headline accuracy
- Mean / median error interpretation
- Accuracy thresholds
- GPU / CPU inference timing
- Periodic-repeat hypothesis testing
- Generator-parameter comparison
- Confound analysis
- Pitch-density analysis
- Coarse-pitch intervention
- Before/after results
- Confidence analysis
- Known limitations
- Reproducible analysis commands

For a quick reference, the submitted model achieved on `eval_v5`:

| Metric | Result |
|---|---:|
| Median error | **0.52 px** |
| Mean error | **92.30 px** |
| Within 10 px | **80.0%** |
| Within 100 px | **81.0%** |
| GPU inference | **9.56 ms/pair** |
| CPU inference | **71.36 ms/pair** |

> These values should be interpreted together with the full failure analysis in `evaluation_report.md`, particularly because the mean/median gap reveals a smaller catastrophic-failure regime.

---

# 📚 10. Scientific References

The detailed scientific and patent references behind the synthetic SEM/DRAM modeling choices are maintained separately in:

```bash
citations.md
```

It documents the rationale for:

- DRAM pitch ranges
- Poisson + Gaussian noise
- Edge brightening
- Illumination/shading
- LER/LWR
- Via/contact defects
- Rotation / scale / drift
- Scan-line artifacts
- Periodic matching considerations
- Localization architecture context
- Engineering parameters that are not directly literature-derived

This separation keeps the README focused on reproducibility while preserving full scientific traceability.

---

# 🗺️ 11. End-to-End Workflow

```bash
                 ┌─────────────────────────┐
                 │  DRAM World Generator   │
                 │ generate_dram_dataset   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Synthetic SEM Pairs     │
                 │ Reference + Search      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Training Dataset        │
                 │ Base + Coarse-Pitch     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ V6 Model Training       │
                 │ train_v6.py             │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ driftsense_final.pt     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ submission_model/       │
                 │ inference.py            │
                 └────────────┬────────────┘
                              │
                              ▼
                     Predicted (x, y)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Independent Evaluation  │
                 │ + Failure Analysis      │
                 └─────────────────────────┘
```

---

# 📁 12. Key Files

| File | Role |
|---|---|
| `dataset_generator/generate_dram_dataset_v3.py` | Generates synthetic DRAM SEM pairs |
| `training/train_v6.py` | Trains the V6 model |
| `training/loss_v5.py` | Custom training loss |
| `training/dram_dataset.py` | PyTorch dataset wrapper |
| `training/merge_train_v8.py` | Merges training datasets |
| `training/utils_v5.py` | Metrics, checkpoints, reproducibility utilities |
| `submission_model/inference.py` | Final inference entry point |
| `submission_model/model_v6.py` | Submitted model architecture |
| `submission_model/driftsense_final.pt` | Submitted trained weights |
| `failure_analysis/analyze_v6_failures.py` | Per-pair errors and periodicity test |
| `failure_analysis/analyze_failure_causes.py` | Failure-vs-success parameter analysis |
| `failure_analysis/analyze_pitch_density_bins.py` | Pitch/density confound and bin analysis |
| `evaluation_report.md` | Full evaluation and failure analysis |
| `citations.md` | Scientific and patent justification |
| `requirements.txt` | Python dependencies |

---

# 🔁 13. Minimal Reproduction Checklist

For a clean reproduction:

```bash
[ ] Clone repository
[ ] Create Python 3.11 virtual environment
[ ] Install PyTorch 2.5.1
[ ] Install requirements.txt

[ ] Generate train_v7
[ ] Generate train_v7_coarse_pitch
[ ] Generate val_v5
[ ] Merge into train_v8

[ ] Run train_v6.py
[ ] Obtain best_model.pt
[ ] Copy/use as driftsense_final.pt

[ ] Generate held-out eval_v5
[ ] Run inference
[ ] Run failure-analysis scripts
[ ] Review evaluation_report.md
```

---

# 🧩 14. Important Design Principles

### Reproducibility

Generation and training commands explicitly expose seeds and configuration.

### No proprietary dataset dependency

The system is designed around a synthetic, physically informed dataset rather than requiring proprietary wafer-inspection imagery.

### Separation of concerns

```bash
README.md
    → How to install, run, reproduce

citations.md
    → Why the synthetic imaging choices are justified

evaluation_report.md
    → What the model achieved and what failed
```

### Honest evaluation

The evaluation pipeline reports both strong typical-case performance and the remaining catastrophic failures rather than relying on a single favorable metric.

---

# 🏁 Final Takeaway

DriftSense is an end-to-end prototype for recovering wafer-inspection localization errors from paired reference/search SEM imagery.

Its reproducible pipeline is:

```bash
Physically informed DRAM generation
            ↓
Synthetic SEM degradation
            ↓
Training on diverse navigation conditions
            ↓
V6 localization model
            ↓
(x, y) prediction
            ↓
Independent evaluation
            ↓
Failure diagnosis
            ↓
Targeted dataset improvement
```

The repository is intentionally structured so that:

- **README.md** tells you how to use and reproduce the project.
- **citations.md** explains the scientific basis of the synthetic data.
- **evaluation_report.md** documents the evidence, results, failure modes, and limitations.

---

<div align="center">

### 🧭 DriftSense

**Generate → Train → Localize → Evaluate → Diagnose → Improve**

</div>
