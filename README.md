<div align="center">

# 🔬 DriftSense

### **AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**

**SEMICON India Hackathon 2026 — Track 2 (Applied Materials, PS-02)**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## 🧭 Overview

**DriftSense** is an AI-powered navigation-error recovery system designed for semiconductor wafer inspection tools.

The system localizes a small SEM reference-image crop inside a larger SEM search image of the **same physical DRAM region**, while accounting for the challenges introduced by scale differences, periodic structures, imaging variation, and navigation uncertainty.

### Core Task

DriftSense takes:

- A small **SEM reference-image crop** at **1 nm/px**
- A larger **SEM search image** at **10 nm/px**

Both images represent the **same physical DRAM region**.

The model outputs the **sub-pixel `(x, y)` center** of the reference pattern in search-image pixel coordinates.

> 🎯 **Primary challenge:**  
> Disambiguating the true physical match from visually similar periodic repeats in the DRAM structure.

---

# 🧠 1. Problem Statement

Wafer inspection tools repeatedly navigate to precise physical locations on semiconductor dies.

Even very small navigation errors can accumulate due to factors such as:

- Stage translation errors
- Mechanical drift
- Rotation
- Magnification variation
- Imaging noise
- Local pattern variation
- Periodic semiconductor structures
- Spatial distortion

For periodic structures such as DRAM, the challenge is especially difficult because many locations can look visually similar.

Therefore, DriftSense treats the problem as a **learned spatial-localization task**:

```bash
Reference SEM Crop
        │
        ▼
Feature Extraction
        │
        ▼
Search SEM Image
        │
        ▼
Spatial Localization
        │
        ▼
Sub-pixel (x, y)
```

The goal is not merely to determine whether two images look similar.

The goal is to determine:

> **Where exactly does the reference region occur inside the search image?**

---

# 🏗️ 2. Architecture Choice

DriftSense uses a **DRAM-style semiconductor layout** consisting of periodic memory-cell, via, and line structures.

This architecture was selected according to the problem statement's explicit:

> **"participant's choice, judged equally either way"**

clause.

### Why DRAM?

DRAM provides:

- Highly periodic structures
- Repeating memory-cell patterns
- Line structures
- Contact-via structures
- Local variations in via density
- Pitch-dependent visual ambiguity

These characteristics provide a challenging environment for evaluating navigation-error recovery under periodic-pattern ambiguity.

> **Note:** FinFET-style layouts are not implemented in the current version.

---

# 🎯 3. Key Objective

The system is designed to solve:

> **Given a small high-resolution SEM reference image and a larger lower-resolution SEM search image of the same DRAM region, determine the exact location of the reference region in the search image with sub-pixel accuracy.**

### Input

```bash
Reference Image
1000 × 1000
1 nm/px
High-resolution SEM crop
```

and:

```bash
Search Image
1000 × 1000
10 nm/px
Larger-field SEM image
```

### Output

```bash
(x, y)
```

where:

- `x` = predicted horizontal center
- `y` = predicted vertical center
- Coordinates are expressed in **search-image pixel space**
- Prediction is **sub-pixel**

---

# 📁 4. Repository Structure

```bash
DriftSense/
│
├── dataset_generator/
│   └── generate_dram_dataset_v3.py
│       └── Standalone synthetic dataset generator
│
├── submission_model/
│   ├── inference.py
│   │   └── Standalone inference script (run this)
│   ├── model_v6.py
│   │   └── Model architecture
│   └── driftsense_final.pt
│       └── Trained weights
│
├── training/
│   ├── train_v6.py
│   │   └── Training script
│   ├── loss_v5.py
│   │   └── Custom loss function
│   ├── dram_dataset.py
│   │   └── PyTorch Dataset wrapper
│   ├── model_v6.py
│   │   └── Architecture copy (self-contained training)
│   ├── merge_train_v8.py
│   │   └── Merges the two training datasets
│   └── utils_v5.py
│       └── Metrics, checkpoint I/O, and seeding
│
├── failure_analysis/
│   ├── analyze_v6_failures.py
│   │   └── Per-pair error report + periodic-lock-on test
│   ├── analyze_failure_causes.py
│   │   └── Failure-vs-success generator-parameter comparison
│   ├── analyze_pitch_density_bins.py
│   │   └── Confound check + binned pitch/density analysis
│   ├── model_v6.py
│   │   └── Architecture copy
│   └── dram_dataset.py
│       └── Dataset wrapper copy
│
├── evaluation_report.md
│   └── Accuracy, timing, and failure analysis
│
├── citations.md
│   └── Augmentation/noise-model references
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

# 🛠️ 5. Installation & Environment Setup

> **All commands in this README are designed to be copy-pasted and executed from the repository root directory.**

## 5.1 Clone the Repository

```bash
git clone https://github.com/urjadoshi2024etc-creator/semicon-drift-sense.git
cd semicon-drift-sense
```

## 5.2 Create and Activate a Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv_drift
source venv_drift/bin/activate
```

### Windows — PowerShell

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

Then:

```powershell
python -m venv venv_drift
.\venv_drift\Scripts\Activate.ps1
```

### Windows — Command Prompt (CMD)

```cmd
python -m venv venv_drift
venv_drift\Scripts\activate.bat
```

After activation, your terminal should show something similar to:

```bash
(venv_drift)
```

## 5.3 Install PyTorch

Install PyTorch **before** the remaining dependencies. Choose **one** option.

### Option A — NVIDIA GPU / CUDA 12.1

This is the configuration used during project development:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Option B — CPU-only

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> `inference.py` automatically detects CUDA and falls back to CPU when CUDA is unavailable. No code change is required.

### Tested Development Environment

| Component | Version / Configuration |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| CPU Mode | Supported |

---

# 🚀 6. Quick Start — Verify the Submitted Model

This is the fastest way to verify that the repository and pretrained submission work correctly. **No training is required.**

> **Important:** Run every command below from the **repository root**. You do not need to `cd submission_model`.

## 6.1 Generate One Sample Pair

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 1 \
    --out_dir ./sample \
    --profile medium \
    --seed 1
```

This creates:

```bash
sample/
├── reference/
│   └── ref_00000.png
├── search/
│   └── search_00000.png
├── labels.csv
└── config.json
```

## 6.2 Run Inference

```bash
python submission_model/inference.py \
    ./sample/reference/ref_00000.png \
    ./sample/search/search_00000.png
```

## 6.3 Expected Output

The script outputs one line in this format:

```bash
(x.xx, y.xx)
```

This is the predicted center coordinate in **search-image pixel space**.

Compare the prediction against the ground-truth `center_x` and `center_y` values in:

```bash
sample/labels.csv
```

> The exact prediction depends on the generated sample. Do **not** document a fixed output such as `(501.23, 498.87)` unless it was actually produced by the exact sample being distributed.

---

# 🧪 7. Dataset Generator — Full Usage

The standalone DRAM dataset generator can be executed with:

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

## 7.1 Generator Options

### `--n_pairs`

Number of reference/search pairs to generate.

Example:

```bash
--n_pairs 9000
```

### `--out_dir`

Directory in which the generated dataset is stored.

Example:

```bash
--out_dir ./train_v7
```

### `--profile`

Controls the dataset generation profile:

```bash
easy
medium
hard
```

Example:

```bash
--profile medium
```

### `--seed`

Controls reproducibility.

Example:

```bash
--seed 7
```

### `--workers`

Optional number of workers used for generation.

Example:

```bash
--workers 6
```

### `--style`

Specifies the layout style.

The current implementation supports:

```bash
dram
```

`dram` is the default and only implemented option.

Passing:

```bash
--style finfet
```

raises a clear error rather than silently generating a mismatched dataset.

### `--pitch_min` / `--pitch_max`

Optional overrides for the profile's pitch range.

Both values must be supplied together.

Example:

```bash
--pitch_min 80 --pitch_max 120
```

These options are used for the **coarse-pitch supplemental dataset** described later.

---

# 📦 8. Dataset Outputs

Each generator run writes:

```bash
<DIR>/
│
├── reference/
│   ├── ref_00000.png
│   ├── ref_00001.png
│   └── ...
│
├── search/
│   ├── search_00000.png
│   ├── search_00001.png
│   └── ...
│
├── labels.csv
│
└── config.json
```

## 8.1 `labels.csv`

`labels.csv` contains:

- Ground-truth center coordinates
- Reference/search paths
- Pitch
- Quality
- Rotation
- Defect counts
- Other generation parameters used for each pair

Conceptually:

```bash
pair_id
reference_path
search_path
center_x
center_y
pitch
rotation
defect_count
...
```

## 8.2 `config.json`

`config.json` stores the exact settings used for the generation run, including the seed.

This makes dataset generation reproducible.

---

# 🖼️ 9. Image Specifications

Generated images are:

| Property | Reference | Search |
|---|---:|---:|
| Format | Grayscale PNG | Grayscale PNG |
| Resolution | 1000 × 1000 | 1000 × 1000 |
| Scale | 1 nm/px | 10 nm/px |
| Relative Scale | 1× | 10× |
| Field of View | Smaller | Larger |

The fixed 10× magnification ratio follows the problem-statement setup used by the project.

---

# 🔬 10. DRAM Representation

The dataset is specifically generated around a **DRAM-style periodic structure**.

The synthetic scene represents repeating semiconductor patterns containing:

```bash
DRAM Layout
│
├── Periodic memory-cell structure
│
├── Line structures
│
├── Contact / via structures
│
└── Local pattern variation
```

The periodic nature of DRAM introduces a central localization challenge:

```bash
Many regions look similar
          ↓
Visual matching becomes ambiguous
          ↓
Model must learn spatially discriminative features
          ↓
Correct physical location must be recovered
```

---

# 🔄 11. Multi-Scale Imaging Concept

DriftSense works with two views of the same physical region:

```bash
                    SAME PHYSICAL DRAM REGION
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       Reference Image              Search Image
       ───────────────              ─────────────
       1 nm/px                      10 nm/px
       High resolution              Larger field of view
       Small crop                   Broad search region
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     DriftSense Model
                              │
                              ▼
                       (x, y) Location
```

---

# 🏋️ 12. Reproducing the Training Data and Trained Model

The submitted checkpoint:

```bash
submission_model/driftsense_final.pt
```

was trained using a merged dataset consisting of:

```bash
Base dataset:
9,000 pairs

Supplemental dataset:
4,000 pairs

Total:
13,000 training pairs
```

The supplemental dataset is concentrated in the:

```bash
80–120 nm pitch range
```

This dataset was added after failure analysis showed reduced accuracy in that regime.

The exact reproduction workflow is described below.

---

## Step 1 — Base Training Set

Generate **9,000 pairs**:

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 9000 \
    --out_dir ./train_v7 \
    --profile medium \
    --seed 7 \
    --workers 6
```

---

## Step 2 — Coarse-Pitch Supplemental Set

Generate **4,000 pairs** concentrated in the **80–120 nm pitch range**:

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

## Step 3 — Validation Set

Generate **300 pairs**:

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 300 \
    --out_dir ./val_v5 \
    --profile medium \
    --seed 13579246
```

This validation set is used during model training.

---

## Step 4 — Merge Training Datasets

Run:

```bash
python training/merge_train_v8.py
```

This must be executed from the directory containing:

```bash
train_v7/
train_v7_coarse_pitch/
```

The merge script:

- Combines the base and supplemental training datasets.
- Verifies that row counts match exactly.
- Refuses to proceed if the expected row counts do not match.
- Refuses to overwrite an existing `train_v8/`.

The resulting dataset is:

```bash
train_v8/
```

---

## Step 5 — Train the Model

Run:

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

---

# ⚙️ 13. Training Configuration

| Parameter | Value |
|---|---:|
| Base Training Pairs | 9,000 |
| Supplemental Pairs | 4,000 |
| Total Training Pairs | 13,000 |
| Validation Pairs | 300 |
| Epochs | 100 |
| Batch Size | 4 |
| Workers | 4 |
| Learning Rate | 0.0003 |
| Weight Decay | 1e-5 |
| Seed | 42 |
| LR Patience | 20 |
| LR Factor | 0.5 |
| Gradient Clipping | 5.0 |
| Checkpoint Interval | 10 epochs |
| Early Stop Patience | 15 |
| Minimum Improvement | 0.5 px |

---

# 🛑 14. Early Stopping and Checkpoints

The training configuration uses:

```bash
Early stopping patience:
15 epochs

Minimum improvement:
0.5 px
```

For the run that produced the submitted checkpoint:

```bash
Best validation performance:
Epoch 44

Early stopping:
Epoch 59
```

The best checkpoint is written to:

```bash
runs/v6_train_v8/checkpoints/best_model.pt
```

The submitted weights are:

```bash
submission_model/driftsense_final.pt
```

The latter is the checkpoint copied over after training.

---

# 🧪 15. Independent Test Set

An independent **100-pair** test set was held out from:

- Training
- Validation
- Model-selection decisions

Configuration:

```bash
Dataset:
eval_v5

Number of pairs:
100

Seed:
24681357
```

The held-out set is used only for final reported evaluation numbers in:

```bash
evaluation_report.md
```

---

# 🚀 16. Inference — Submission Entry Point

The inference command is:

```bash
python submission_model/inference.py \
    <reference_image_path> \
    <search_image_path>
```

This is the script intended to be run for the final submission/inference workflow.

---

# 📥 17. Inference Input Requirements

Both input images must be:

```bash
1000 × 1000
Grayscale PNG
```

Expected scale:

```bash
Reference:
1 nm/px

Search:
10 nm/px
```

---

# 📤 18. Inference Output

The script outputs a single line:

```bash
(x, y)
```

Example:

```bash
(421.37, 583.92)
```

This represents the predicted center of the reference pattern within the search image in **search-image pixel coordinates**.

---

# 🤖 19. Automatic Inference Behavior

`inference.py` automatically:

1. Detects whether CUDA is available.
2. Uses the GPU if available.
3. Falls back to CPU otherwise.
4. Loads `driftsense_final.pt`.
5. Resolves the checkpoint relative to the script's own location.
6. Loads the model architecture from `model_v6.py` in the same folder.

No manual model-path changes are required.

No additional inference arguments are needed beyond:

```bash
<reference_image_path>
<search_image_path>
```

---

# 🔍 20. Failure Analysis

Failure analysis is an important component of the DriftSense development workflow.

The repository includes:

```bash
failure_analysis/
│
├── analyze_v6_failures.py
├── analyze_failure_causes.py
└── analyze_pitch_density_bins.py
```

These scripts were used to investigate model failures and identify difficult generation regimes.

---

# 📊 21. Failure Analysis Tools

## 21.1 Per-Pair Error Report + Periodic Test

Run:

```bash
python failure_analysis/analyze_v6_failures.py \
    --checkpoint ./submission_model/driftsense_final.pt \
    --data_dir ./eval_v5 \
    --n_worst 25
```

This performs:

- Per-pair error reporting
- Worst-case analysis
- Periodic pitch-multiple testing

---

## 21.2 Failure-vs-Success Parameter Comparison

Run:

```bash
python failure_analysis/analyze_failure_causes.py \
    --checkpoint ./submission_model/driftsense_final.pt \
    --data_dir ./eval_v5
```

This compares generator parameters across successful and failed predictions.

---

## 21.3 Pitch / Density Binned Analysis

Run:

```bash
python failure_analysis/analyze_pitch_density_bins.py \
    --checkpoint ./submission_model/driftsense_final.pt \
    --data_dir ./eval_v5 \
    --n_bins 5
```

This performs:

- Confound checking
- Pitch-binned analysis
- Via-density-binned analysis
- Failure-rate analysis

---

# 🧩 22. Failure Analysis Findings

The evaluation workflow documented in `evaluation_report.md` reports that:

- Periodic-pitch-repeat confusion was explicitly tested.
- Periodic-repeat confusion was not identified as the dominant remaining failure mode.
- The dominant remaining failure pattern was traced to a confirmed, non-confounded correlation with **coarse pitch** and **low local reference-crop via density**.
- The supplemental 80–120 nm pitch dataset was generated specifically to address this difficult regime.

This resulted in a failure-driven refinement loop:

```bash
Initial Training
      │
      ▼
Evaluate Model
      │
      ▼
Failure Analysis
      │
      ▼
Identify Difficult Pitch / Density Regime
      │
      ▼
Generate Supplemental Dataset
      │
      ▼
Merge Training Data
      │
      ▼
Retrain Model
      │
      ▼
Evaluate on Held-Out Data
```

---

# 📈 23. Results and Evaluation

Detailed results are documented in:

```bash
evaluation_report.md
```

The report includes:

- Per-pair timing
- Accuracy at multiple pixel thresholds
- Failure analysis
- Periodic-repeat testing
- Pitch analysis
- Via-density analysis
- Evidence supporting the identified failure regime

The repository's failure-analysis scripts can be rerun against the submitted checkpoint.

> **Requirement:** `eval_v5` must be generated first, as described in the training/evaluation workflow, or an equivalent held-out dataset must be generated using the same generator/profile.

---

# 🧭 24. Navigation Recovery Pipeline

The complete conceptual pipeline is:

```bash
        SEM Inspection Tool
                │
                ▼
       Reference SEM Crop
                │
                │
                ├──────────────┐
                │              │
                ▼              ▼
         High-resolution   Search SEM Image
            Reference        Larger FOV
                │              │
                └──────┬───────┘
                       ▼
                DriftSense Model
                       │
                       ▼
              Spatial Localization
                       │
                       ▼
                 Sub-pixel (x,y)
                       │
                       ▼
              Navigation Recovery
```

---

# 🔬 25. Why Periodic DRAM Is Challenging

DRAM contains strongly repeating structures.

Conceptually:

```bash
┌───┬───┬───┬───┬───┐
│ A │ B │ A │ B │ A │
├───┼───┼───┼───┼───┤
│ B │ A │ B │ A │ B │
├───┼───┼───┼───┼───┤
│ A │ B │ A │ B │ A │
└───┴───┴───┴───┴───┘
```

Multiple regions can contain visually similar patterns.

Therefore, a model that relies only on local appearance may incorrectly lock onto a neighboring repeat.

DriftSense is designed to learn spatially discriminative features that help distinguish the correct physical location.

---

# 🧪 26. Complete Reproduction Pipeline

The complete reproduction workflow is:

```bash
Step 1
│
├── Generate 9,000-pair base dataset
│
▼
Step 2
│
├── Generate 4,000-pair coarse-pitch supplemental dataset
│
▼
Step 3
│
├── Generate 300-pair validation dataset
│
▼
Step 4
│
├── Merge training datasets
│
▼
Step 5
│
├── Train model
│
▼
Step 6
│
├── Select best checkpoint
│
▼
Step 7
│
├── Evaluate on independent 100-pair test set
│
▼
Step 8
│
└── Run failure analysis
```

---

# 📦 27. Complete Reproduction Commands

## Base Training Dataset

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 9000 \
    --out_dir ./train_v7 \
    --profile medium \
    --seed 7 \
    --workers 6
```

## Coarse-Pitch Supplemental Dataset

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

## Validation Dataset

```bash
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 300 \
    --out_dir ./val_v5 \
    --profile medium \
    --seed 13579246
```

## Merge

```bash
python training/merge_train_v8.py
```

## Training

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

## Inference

```bash
python submission_model/inference.py \
    <reference_image_path> \
    <search_image_path>
```

---

# 🧪 28. Quick Validation Workflow

If you only want to verify that the repository works:

```bash
# Generate one sample pair
python dataset_generator/generate_dram_dataset_v3.py \
    --n_pairs 1 \
    --out_dir ./sample \
    --profile medium \
    --seed 1

# Run inference
python submission_model/inference.py \
    ./sample/reference/ref_00000.png \
    ./sample/search/search_00000.png
```

Expected output:

```bash
(x.xx, y.xx)
```

Then compare the prediction against:

```bash
sample/labels.csv
```

using:

```bash
center_x
center_y
```

---

# 🛠️ 29. Troubleshooting & Edge Cases

If something fails while following the README, use the solutions below before modifying project code.

| Issue / Error | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: submission_model/driftsense_final.pt` | Command was run from the wrong directory or the path is incorrect | Run commands from the repository root and use `./submission_model/driftsense_final.pt` |
| `ps1 cannot be loaded because running scripts is disabled` | Windows PowerShell execution policy blocks virtual-environment activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`, then activate `venv_drift` again |
| `ModuleNotFoundError: No module named 'torch'` | Virtual environment is not active or PyTorch was not installed inside it | Activate `venv_drift`, install the correct PyTorch build, then run `pip install -r requirements.txt` |
| `CUDA out of memory` | GPU VRAM is insufficient for the selected workload | Reduce dataset `--workers`, reduce training `--batch_size` (for example to `2`), or run inference on CPU |
| `NotImplementedError: finfet style not implemented` | Unsupported generator style was selected | Use `--style dram` or omit `--style`; DRAM is the implemented style |
| `FileNotFoundError` for sample images | Dataset generation did not finish or the inference path is wrong | Re-run the generator and verify `sample/reference/` and `sample/search/` exist |
| Wrong PyTorch/CUDA installation | The selected install command does not match the machine | Reinstall PyTorch using either the CUDA 12.1 or CPU command in Section 5.3 |
| Inference is slow on CPU | CUDA is unavailable | This is expected. `inference.py` automatically falls back to CPU; use an NVIDIA GPU for faster inference |
| Training is very slow | CPU training or unsuitable worker/batch settings | Prefer a supported NVIDIA GPU and tune `--num_workers` / `--batch_size` for the available hardware |
| `train_v8` already exists | A previous merged dataset is present | Remove or rename `train_v8/` only if you intentionally want to regenerate it, then rerun `merge_train_v8.py` |
| Merge reports mismatched row counts | Training datasets were not generated with the expected structure/counts | Regenerate `train_v7` and `train_v7_coarse_pitch` using the exact commands in Section 27, then rerun the merge |

## 🔎 General Diagnostic Checklist

When an error occurs, check these in order:

```bash
1. Are you in the repository root?
       ↓
2. Is (venv_drift) active?
       ↓
3. Does PyTorch import successfully?
       ↓
4. Do the required files/directories exist?
       ↓
5. Are paths written relative to the repository root?
       ↓
6. Does the GPU/CPU installation match the machine?
       ↓
7. Only then adjust workers/batch size or regenerate data
```

### Check Python and PyTorch

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For the tested CUDA environment, CUDA should be available. For CPU-only installation, `CUDA available: False` is expected.

### Check that the submission checkpoint exists

```bash
python -c "from pathlib import Path; p=Path('submission_model/driftsense_final.pt'); print('Checkpoint exists:', p.exists())"
```

> **Do not edit `submission_model/inference.py` just to switch between GPU and CPU.** Device selection is already handled automatically.

---

# 📚 30. Documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, setup, usage, reproduction, and inference |
| `evaluation_report.md` | Accuracy, timing, and failure analysis |
| `citations.md` | Augmentation and SEM noise-model references |
| `requirements.txt` | Python dependencies |

---

# 📖 31. Citations & Scientific Justification

Augmentation and SEM noise-model choices are justified against public literature in:

```bash
citations.md
```

The references are cross-referenced with the idea-submission PDF.

The repository therefore separates:

```bash
Implementation
      +
Scientific / Literature Justification
      +
Experimental Evaluation
```

---

# 🧾 32. Project Summary

| Category | DriftSense |
|---|---|
| Hackathon | SEMICON India Hackathon 2026 |
| Track | Track 2 |
| Problem Statement | PS-02 |
| Domain | Semiconductor Inspection |
| Layout | DRAM |
| Input | Reference + Search SEM images |
| Reference Resolution | 1000 × 1000 |
| Search Resolution | 1000 × 1000 |
| Reference Scale | 1 nm/px |
| Search Scale | 10 nm/px |
| Magnification Ratio | 10× |
| Output | Sub-pixel `(x, y)` |
| Dataset Type | Synthetic |
| Base Training Set | 9,000 pairs |
| Supplemental Set | 4,000 pairs |
| Total Training Set | 13,000 pairs |
| Validation Set | 300 pairs |
| Independent Test Set | 100 pairs |
| Model | `model_v6.py` |
| Final Weights | `driftsense_final.pt` |
| Framework | PyTorch 2.5.1 |
| Python | 3.11 |
| CUDA | 12.1 |
| GPU Tested | NVIDIA RTX 4050 Laptop GPU |
| GPU VRAM | 6 GB |
| CPU Support | Yes |
| Dataset Generator | `generate_dram_dataset_v3.py` |
| Training Script | `train_v6.py` |
| Inference Script | `submission_model/inference.py` |

---

# 🏁 33. Final Takeaway

DriftSense approaches wafer-navigation recovery as a **learned spatial localization problem** over SEM images.

The system combines:

```bash
DRAM-specific synthetic data
          +
Multi-scale SEM image generation
          +
Ground-truth spatial coordinates
          +
Deep-learning localization
          +
Failure-driven dataset refinement
          +
Independent evaluation
```

The central objective is not simply to identify whether two SEM images look similar.

Instead, the model must determine:

> **Where exactly does the reference region occur inside the search image?**

This distinction is critical for navigation-error recovery in periodic semiconductor structures, where multiple visually similar regions can coexist.

---

## 🔬 DriftSense

### **AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**

```bash
Reference SEM Image
        +
Search SEM Image
        │
        ▼
   DriftSense AI
        │
        ▼
Sub-pixel (x, y)
        │
        ▼
Accurate Site Recovery
```

---

**SEMICON India Hackathon 2026 — Track 2 — Problem Statement 02**
