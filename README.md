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
## 🔬 NanoNav
### **Navigate. Locate. Verify.**

> * **Karan Choudhary** ([@K478-tech](https://github.com/K478-tech))
> * **Harsh** ([@Harsh689956](https://github.com/Harsh689956))
> * **Urja Doshi** ([@urjadoshi2024etc-creator](https://github.com/urjadoshi2024etc-creator))
> * **Priyadarshini** ([@PriyadarshiniK003](https://github.com/PriyadarshiniK003))

*Developed for SEMICON India Hackathon 2026 (Problem Statement 02)*

---

# 🧭 Overview

**DriftSense** is an AI-powered navigation-error recovery system designed for semiconductor wafer inspection tools.

The system localizes a small SEM reference-image crop inside a larger SEM search image of the **same physical DRAM region**, while handling scale differences, periodic structures, imaging variation, and navigation uncertainty.

### Core Task

- Reference SEM crop: **1000 × 1000**, **1 nm/px**
- Search SEM image: **1000 × 1000**, **10 nm/px**
- Both images represent the same physical DRAM region.
- Output: sub-pixel **`(x, y)`** center in search-image pixel coordinates.

> 🎯 **Primary challenge:** Disambiguating the true physical match from visually similar periodic repeats in DRAM structures.

---

# 🧠 1. Problem Statement

Wafer inspection tools repeatedly navigate to precise physical locations on semiconductor dies. Small navigation errors can result from:

- Stage translation errors
- Mechanical drift
- Rotation
- Magnification variation
- Imaging noise
- Local pattern variation
- Periodic semiconductor structures
- Spatial distortion

For periodic structures such as DRAM, many locations can look visually similar. DriftSense therefore treats the task as a **learned spatial-localization problem**:

```text
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

The goal is not simply to determine whether two images look similar, but to determine:

> **Where exactly does the reference region occur inside the search image?**

---

# 🏗️ 2. Architecture Choice

DriftSense uses a **DRAM-style semiconductor layout** consisting of periodic memory-cell, via, and line structures.

This architecture was selected according to the problem statement's:

> **"participant's choice, judged equally either way"**

clause.

### Why DRAM?

DRAM provides:

- Highly periodic structures
- Repeating memory-cell patterns
- Line structures
- Contact/via structures
- Local variations in via density
- Pitch-dependent visual ambiguity

> **Note:** FinFET-style layouts are not implemented in the current version.

---

# 🎯 3. Objective

Given a small high-resolution SEM reference image and a larger lower-resolution SEM search image of the same DRAM region, determine the exact location of the reference region in the search image with **sub-pixel accuracy**.

### Input

| Image | Resolution | Scale | Description |
|---|---:|---:|---|
| Reference | 1000 × 1000 | 1 nm/px | High-resolution SEM crop |
| Search | 1000 × 1000 | 10 nm/px | Larger-field SEM image |

### Output

```text
(x, y)
```

- `x` = predicted horizontal center
- `y` = predicted vertical center
- Coordinates are in search-image pixel space
- Prediction is sub-pixel

---

# 📁 4. Repository Structure

```text
DriftSense/
│
├── dataset_generator/
│   └── generate_dram_dataset_v3.py
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

---

# 🛠️ 5. Installation & Environment Setup

> **All commands below are intended to be run from the repository root directory.**

## 5.1 Clone the Repository

```bash
git clone https://github.com/urjadoshi2024etc-creator/semicon-drift-sense.git
cd semicon-drift-sense
```

## 5.2 Python 3.11 — IMPORTANT

> ⚠️ **This project was developed and tested with Python 3.11.**
>
> The pinned dependencies in `requirements.txt` are intended for Python 3.11. Some dependencies, such as `contourpy==1.3.3`, require Python 3.11 or newer.
>
> **Do not continue with dependency installation until `python --version` reports Python 3.11.x.**
>
> If multiple Python versions are installed, make sure the virtual environment is created specifically with Python 3.11.

### Check Python

```bash
python --version
python -c "import sys; print(sys.executable); print(sys.version)"
```

Expected:

```text
Python 3.11.x
```

### Windows — Install Python 3.11 if Needed

```cmd
winget install --id Python.Python.3.11 -e --source winget
```

Close and reopen Command Prompt / PowerShell, then verify:

```cmd
py -3.11 --version
```

If `winget` is unavailable, install Python 3.11 from the official Python website:

https://www.python.org/downloads/

### Windows — Create the Virtual Environment

**Command Prompt:**

```cmd
py -3.11 -m venv venv_drift
venv_drift\Scriptsctivate.bat
python --version
```

**PowerShell:**

```powershell
py -3.11 -m venv venv_drift
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.env_drift\Scripts\Activate.ps1
python --version
```

### Linux / macOS

```bash
python3.11 -m venv venv_drift
source venv_drift/bin/activate
python --version
```

After activation, the terminal should show:

```text
(venv_drift)
```

Verify:

```bash
python --version
```

It must report:

```text
Python 3.11.x
```

> **Important:** Do not use `python -m venv venv_drift` blindly when multiple Python versions are installed. On Windows, prefer `py -3.11`; on Linux/macOS, prefer `python3.11`.

---

# 📦 6. Install Dependencies

Install PyTorch first, then the remaining project dependencies.

## Option A — NVIDIA GPU / CUDA 12.1

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

## Option B — CPU-only

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

> **Choose only one PyTorch installation option. Do not install both.**

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`inference.py` automatically detects CUDA and falls back to CPU when CUDA is unavailable.

### Tested Development Environment

| Component | Version / Configuration |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| CPU Mode | Supported |

---

# 🚀 7. Quick Start — Verify the Submitted Model

No training is required for this verification.

> **Run the following commands from the repository root.**

## 7.1 Generate One Sample Pair

```bash
python dataset_generator\generate_dram_dataset_v3.py --n_pairs 1 --out_dir .\sample --profile medium --seed 1
```

This creates:

```text
sample/
├── reference/
│   └── ref_00000.png
├── search/
│   └── search_00000.png
├── labels.csv
└── config.json
```

## 7.2 Run Inference

```bash
python submission_model\inference.py .\sample\reference\ref_00000.png .\sample\search\search_00000.png
```

Expected format:

```text
(x.xx, y.xx)
```

The exact values depend on the generated sample.

Compare the prediction with the ground-truth `center_x` and `center_y` values in:

```text
sample/labels.csv
```

---

# 🧪 8. Dataset Generator

The standalone DRAM dataset generator supports:

```text
--n_pairs <N>
--out_dir <DIR>
--profile {easy,medium,hard}
--seed <SEED>
[--workers <N>]
[--style dram]
[--pitch_min <NM> --pitch_max <NM>]
```

Example:

```bash
python dataset_generator\generate_dram_dataset_v3.py --n_pairs 1 --out_dir .\sample --profile medium --seed 1
```

### Generator Options

| Option | Purpose |
|---|---|
| `--n_pairs` | Number of reference/search pairs |
| `--out_dir` | Output directory |
| `--profile` | `easy`, `medium`, or `hard` |
| `--seed` | Reproducibility |
| `--workers` | Number of generation workers |
| `--style` | Current implementation supports `dram` |
| `--pitch_min` / `--pitch_max` | Optional pitch-range override |

The default style is `dram`. `finfet` is not implemented.

### Dataset Output

```text
<DIR>/
├── reference/
│   ├── ref_00000.png
│   ├── ref_00001.png
│   └── ...
├── search/
│   ├── search_00000.png
│   ├── search_00001.png
│   └── ...
├── labels.csv
├── config.json
└── generation_log.txt
```

`labels.csv` contains ground-truth coordinates and generation parameters such as:

- Reference/search paths
- `center_x`
- `center_y`
- Pitch
- Quality
- Rotation
- Defect count
- Other generation parameters

`config.json` stores the exact generation settings, including the seed.

---

# 🖼️ 9. Image Specifications

| Property | Reference | Search |
|---|---:|---:|
| Format | Grayscale PNG | Grayscale PNG |
| Resolution | 1000 × 1000 | 1000 × 1000 |
| Scale | 1 nm/px | 10 nm/px |
| Relative Scale | 1× | 10× |
| Field of View | Smaller | Larger |

---

# 🔬 10. DRAM Representation

The dataset represents repeating semiconductor structures containing:

```text
DRAM Layout
│
├── Periodic memory-cell structure
├── Line structures
├── Contact / via structures
└── Local pattern variation
```

The periodic nature creates the central localization challenge:

```text
Many regions look similar
        ↓
Visual matching becomes ambiguous
        ↓
Model learns spatially discriminative features
        ↓
Correct physical location is recovered
```

---

# 🏋️ 11. Training Data & Model Reproduction

The submitted checkpoint is:

```text
submission_model/driftsense_final.pt
```

It was trained using:

- Base dataset: **9,000 pairs**
- Supplemental dataset: **4,000 pairs**
- Total training pairs: **13,000**
- Validation set: **300 pairs**
- Independent test set: **100 pairs**

The supplemental dataset focuses on the **80–120 nm pitch range**, which was added after failure analysis identified reduced accuracy in that regime.

## 11.1 Generate Base Training Set

```bash
python dataset_generator\generate_dram_dataset_v3.py --n_pairs 9000 --out_dir .\train_v7 --profile medium --seed 7 --workers 6
```

## 11.2 Generate Supplemental Dataset

```bash
python dataset_generator\generate_dram_dataset_v3.py --n_pairs 4000 --out_dir .\train_v7_coarse_pitch --profile medium --seed 17 --workers 6 --pitch_min 80 --pitch_max 120
```

## 11.3 Generate Validation Set

```bash
python dataset_generator\generate_dram_dataset_v3.py --n_pairs 300 --out_dir .\val_v5 --profile medium --seed 13579246
```

## 11.4 Merge Training Datasets

```bash
python training\merge_train_v8.py
```

This creates:

```text
train_v8/
```

The merge script checks the expected dataset structure and row counts and refuses to overwrite an existing `train_v8/`.

## 11.5 Train the Model

```bash
python training\train_v6.py --train_dir .\train_v8 --val_dir .\val_v5 --output_dir .\runs\v6_train_v8 --epochs 100 --batch_size 4 --num_workers 4 --lr 0.0003 --weight_decay 1e-5 --seed 42 --lr_patience 20 --lr_factor 0.5 --grad_clip 5.0 --save_every 10 --early_stop_patience 15 --min_delta 0.5
```

### Training Configuration

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

For the documented training run:

- Best validation performance: **Epoch 44**
- Early stopping: **Epoch 59**
- Best checkpoint: `runs/v6_train_v8/checkpoints/best_model.pt`
- Submitted weights: `submission_model/driftsense_final.pt`

---

# 🧪 12. Independent Test Set

An independent **100-pair** test set was held out from:

- Training
- Validation
- Model-selection decisions

Configuration:

```text
Dataset: eval_v5
Pairs: 100
Seed: 24681357
```

The held-out results are documented in:

```text
evaluation_report.md
```

---

# 🚀 13. Inference

The final submission entry point is:

```bash
python submission_model\inference.py <reference_image_path> <search_image_path>
```

Example:

```bash
python submission_model\inference.py .\sample\reference\ref_00000.png .\sample\search\search_00000.png
```

Input requirements:

```text
Reference: 1000 × 1000 grayscale PNG, 1 nm/px
Search:    1000 × 1000 grayscale PNG, 10 nm/px
```

Output:

```text
(x, y)
```

`inference.py` automatically:

1. Detects CUDA availability.
2. Uses the GPU when available.
3. Falls back to CPU otherwise.
4. Loads `driftsense_final.pt`.
5. Resolves the checkpoint relative to the inference script.
6. Loads `model_v6.py` from the submission-model directory.

No manual model-path or device changes are required.

---

# 🔍 14. Failure Analysis

The repository contains:

```text
failure_analysis/
├── analyze_v6_failures.py
├── analyze_failure_causes.py
└── analyze_pitch_density_bins.py
```

### Per-Pair Error Report

```bash
python failure_analysis\analyze_v6_failures.py --checkpoint .\submission_model\driftsense_final.pt --data_dir .\eval_v5 --n_worst 25
```

### Failure-vs-Success Parameter Comparison

```bash
python failure_analysis\analyze_failure_causes.py --checkpoint .\submission_model\driftsense_final.pt --data_dir .\eval_v5
```

### Pitch / Density Binned Analysis

```bash
python failure_analysis\analyze_pitch_density_bins.py --checkpoint .\submission_model\driftsense_final.pt --data_dir .\eval_v5 --n_bins 5
```

These tools provide:

- Per-pair error reporting
- Worst-case analysis
- Periodic-pitch-repeat testing
- Generator-parameter comparison
- Pitch-binned analysis
- Via-density-binned analysis
- Failure-rate analysis

### Key Failure-Analysis Finding

The documented evaluation reports that:

- Periodic-pitch-repeat confusion was explicitly tested.
- Periodic-repeat confusion was not the dominant remaining failure mode.
- The dominant remaining failure pattern was associated with **coarse pitch** and **low local reference-crop via density**.
- The supplemental **80–120 nm pitch** dataset was generated to address this difficult regime.

---

# 📊 15. Results & Evaluation

Detailed results are available in:

```text
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

For failure-analysis reruns, `eval_v5` must be generated first, or an equivalent held-out dataset must be generated using the same generator/profile.

---

# 🧭 16. Navigation Recovery Pipeline

```text
        SEM Inspection Tool
                │
                ▼
        Reference SEM Crop
                │
                ├──────────────┐
                │              │
                ▼              ▼
        High-resolution   Search SEM Image
           Reference          Larger FOV
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

# 📈 17. Complete Reproduction Pipeline

```text
Generate 9,000-pair base dataset
              ↓
Generate 4,000-pair coarse-pitch dataset
              ↓
Generate 300-pair validation dataset
              ↓
Merge training datasets
              ↓
Train model
              ↓
Select best checkpoint
              ↓
Evaluate on independent 100-pair test set
              ↓
Run failure analysis
```

---

# 🛠️ 18. Troubleshooting

| Issue / Error | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: submission_model/driftsense_final.pt` | Wrong working directory or path | Run commands from the repository root |
| PowerShell activation is blocked | Execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` |
| `ModuleNotFoundError: No module named 'torch'` | Environment is inactive or PyTorch is missing | Activate `venv_drift` and install the correct PyTorch build |
| `CUDA out of memory` | Insufficient GPU VRAM | Reduce `--workers` / `--batch_size`, or use CPU |
| `NotImplementedError: finfet style not implemented` | Unsupported generator style | Use `dram` or omit `--style` |
| Sample images are missing | Generation failed or path is incorrect | Regenerate the sample and verify `sample/reference/` and `sample/search/` |
| Wrong PyTorch/CUDA installation | Incorrect wheel selected | Install either the CUDA 12.1 or CPU build, not both |
| CPU inference is slow | CUDA unavailable | Expected behavior; use an NVIDIA GPU for faster inference |
| Training is very slow | CPU training or unsuitable settings | Prefer a supported NVIDIA GPU and tune workers/batch size |
| `train_v8` already exists | Previous merged dataset exists | Remove/rename it only if you intentionally want to regenerate it |
| Merge reports mismatched row counts | Training datasets are incorrect | Regenerate both training datasets using the exact commands above |

### General Diagnostic Checklist

```text
1. Are you in the repository root?
       ↓
2. Is (venv_drift) active?
       ↓
3. Is Python 3.11.x being used?
       ↓
4. Does PyTorch import successfully?
       ↓
5. Do the required files/directories exist?
       ↓
6. Does the PyTorch installation match the machine?
       ↓
7. Only then adjust workers/batch size or regenerate data
```

### Check Python and PyTorch

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For the CUDA environment, CUDA should be available. For CPU-only installation, `CUDA available: False` is expected.

### Check the Submission Checkpoint

```bash
python -c "from pathlib import Path; p=Path('submission_model/driftsense_final.pt'); print('Checkpoint exists:', p.exists())"
```

> **Do not edit `submission_model/inference.py` just to switch between GPU and CPU.** Device selection is already handled automatically.

---

# 📚 19. Documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, setup, usage, reproduction, and inference |
| `evaluation_report.md` | Accuracy, timing, and failure analysis |
| `citations.md` | Augmentation and SEM noise-model references |
| `requirements.txt` | Python dependencies |

Augmentation and SEM noise-model choices are justified against public literature in `citations.md`.

---

# 🧾 20. Project Summary

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

---

# 🏁 21. Final Takeaway

DriftSense approaches wafer-navigation recovery as a **learned spatial-localization problem** over SEM images.

The system combines:

```text
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

The central objective is:

> **Where exactly does the reference region occur inside the search image?**

This is critical for navigation-error recovery in periodic semiconductor structures, where multiple visually similar regions can coexist.

---

## 🔬 DriftSense

### **AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**

```text
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

**SEMICON India Hackathon 2026 — Track 2 — Problem Statement 02**
