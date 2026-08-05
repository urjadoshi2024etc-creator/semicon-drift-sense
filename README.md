<div align="center">

# 🔬 Drift-Sense DRAM SEM Generator (v2)

### *Physically Grounded Synthetic SEM Image Pair Generator for Semiconductor Metrology*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*A high-precision synthetic data generator simulating multi-magnification SEM acquisition pairs ($100\times$ high-dose reference vs. $10\times$ low-dose search scans) for advanced node DRAM wafer review, defect inspection, and drift-tolerant pattern matching.*

[Key Features](#-key-features) • [Local Quick Start](#%EF%B8%8F-local-quick-start) • [Repository Layout](#-repository-layout) • [Pipeline Architecture](#%EF%B8%8F-pipeline-architecture) • [Dataset Schema](#-dataset-schema) • [Physics Grounding](#-physics--process-grounding)

---

</div>

## 📌 Executive Summary

Wafer review and inspection tools in advanced semiconductor fabs rely on matching low-magnification search scans against high-magnification reference libraries to navigate to precise coordinates ($\le 5\text{ nm}$ accuracy). However, real wafer SEM images are strictly proprietary and scarce due to intellectual property restrictions.

**Drift-Sense DRAM SEM Generator** bridges this gap by generating photorealistic, physically grounded image pairs of nanometer-scale DRAM cell arrays. Rather than using simple image filters, it simulates the underlying wafer layout in continuous world coordinates, models electron-beam physics, applies process-induced structural variation (LER/LWR), and simulates realistic SEM tool distortions.

> 💡 **Zero Proprietary Data Used:** All parameter ranges and contrast mechanisms are derived strictly from publicly available patents, peer-reviewed SEM physics literature, and open metrology papers. See [`JUSTIFICATION.md`](./JUSTIFICATION.md) for full citations.

---

## ✨ Key Features

* **📐 Authentic $6F^2/8F^2$ DRAM Topography:** Generates orthogonal word-line / bit-line grids with contact vias at intersections on a single global pitch ($1F$ line width, $1F$ gap).
* **⚛️ Dual-Dose Noise Modeling:** Simulates signal-dependent Poisson (shot) noise combined with additive Gaussian (detector read) noise. High-dose reference scans feature minimal noise, while fast search scans capture low-dose shot noise.
* **🔬 Contrast & Optical Artifacts:** Edge brightening (secondary electron yield elevation), low-frequency shading/vignetting, scan-line raster jitter, and dielectric charging (local contrast modulation).
* **🌀 Geometric Aberrations:** Stage drift translation, rotation, magnification scale drift, and non-affine elastic spatial warping.
* **🧬 Process-Level Variations:** Continuous line-edge roughness (LER), line-width roughness (LWR), and per-via diameter jitter baked into the world-coordinate layout.
* **💥 Hard Defect Injection:** Introduces realistic hard wafer defects: contact opens (missing vias), line breaks (open circuits), and particle contamination.

---

## 🛠️ Local Quick Start

### 1. Environment Setup

Ensure you are working inside your project root folder (`~/semicon-drift-sense`):

```bash
# Navigate to project directory
cd ~/semicon-drift-sense

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

<details>
<summary>📋 Click to view dependencies in <code>requirements.txt</code></summary>

```text
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.4.0
tqdm>=4.60.0
```
</details>

---

### 2. Run the Generator : To generate or refresh synthetic image pairs and update labels.csv:Bashpython3 generate_dram_dataset_v2.py
This will generate matched frames under dram_sem_dataset_v2/train/reference/ and dram_sem_dataset_v2/train/search/, and populate dram_sem_dataset_v2/labels.csv.
```
📂 Repository LayoutPlaintextsemicon-drift-sense/
├── ⚙️ generate_dram_dataset_v2.py  # Primary dataset generator script (v2)
├── 📄 JUSTIFICATION.md             # Detailed 300+ line academic & patent justification
├── 📄 LICENSE                      # MIT License file
├── 📄 README.md                    # Project documentation & guide
├── 📄 requirements.txt             # Required Python libraries
├── 📦 dram_sem_dataset_v2.zip      # Compressed archive of generated dataset
└── 📁 dram_sem_dataset_v2/         # Main dataset directory
    ├── 📊 labels.csv               # Unified annotations, transformations & defect labels
    └── 📁 train/
        ├── 📁 reference/           # High-dose 100x reference images (ref_000.png to ref_029.png)
        └── 📁 search/              # Low-dose 10x search images (search_000.png to search_029.png)
```
---


## 🏗️ Pipeline Architecture : The image generation process separates structural world-space layout definition from tool-specific imaging physics:
```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. WORLD SPECIFICATION                                │
│   • Define 6F² / 8F² DRAM Grid Pitch (x, y) & Line Widths                        │
│   • Bake Line-Edge Roughness (LER/LWR) & Via Diameter Jitter in World Coordinates│
│   • Inject Hard Defects (Missing Vias, Broken Lines, Particles)                │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      2. MULTI-MAGNIFICATION RENDERING                           │
│   ┌────────────────────────────────────┐   ┌────────────────────────────────┐   │
│   │   Reference Frame (100x Mag)       │   │    Search Frame (10x Mag)      │   │
│   │   • High resolution grid rendering │   │    • Broad FOV grid rendering  │   │
│   │   • Crisp feature boundaries       │   │    • Coarser sampling grid     │   │
│   └─────────────────┬──────────────────┘   └────────────────┬───────────────┘   │
└─────────────────────┼───────────────────────────────────────┼───────────────────┘
                      │                                       │
                      ▼                                       ▼
┌────────────────────────────────────────┐   ┌────────────────────────────────────┐
│      3. HIGH-DOSE SEM STYLING          │   │      4. LOW-DOSE SEM STYLING       │
│  • High Poisson Peak (Low Shot Noise)  │   │  • Low Poisson Peak (High Noise)   │
│  • Minimal Gaussian Read Blur (k=3)    │   │  • Strong Gaussian Blur (k=5)      │
│  • Mild Stage Translation Drift        │   │  • Heavy Stage Drift & Elastic Warp│
│  • Sharp Edge Brightening (Sobel)      │   │  • Shading Field & Scan-line Noise │
└─────────────────────┬──────────────────┘   └────────────────┬───────────────────┘
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             5. OUTPUT & LABELS                                  │
│   • Reference Frames: train/reference/ref_XXX.png                               │
│   • Search Frames:    train/search/search_XXX.png                               │
│   • Unified Metadata: labels.csv (Includes transforms, parameters, & defects)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# 📊 Dataset Schema & Physics Grounding

All ground-truth transformation matrices, noise parameters, and defect annotations are logged centrally in `dram_sem_dataset_v2/labels.csv`.

## 📋 Column Definitions

| Column | Data Type | Description |
| :--- | :--- | :--- |
| **`pair_id`** | string | Matching sample ID (e.g., `000` corresponds to `ref_000.png` & `search_000.png`) |
| **`ref_path`** | string | Relative path to reference image (`train/reference/ref_000.png`) |
| **`search_path`** | string | Relative path to search image (`train/search/search_000.png`) |
| **`dx_px`** | float | X-axis translation drift between search and reference frames |
| **`dy_px`** | float | Y-axis translation drift between search and reference frames |
| **`rotation_deg`** | float | Relative angular stage rotation misalignment |
| **`scale_factor`** | float | Relative magnification scale difference |
| **`defect_type`** | string | Defect classification (`none`, `missing_via`, `broken_line`, `particle`) |
| **`defect_coords_world`** | string | Ground-truth continuous world coordinates $(x, y)$ of injected defect |

---

## 🔬 Physics & Process Grounding

The physical parameters and noise distributions applied during dataset generation are derived from semiconductor metrology literature and electron optics patents:

| Physical Effect | Generator Implementation | Reference Setting | Search Setting | Theoretical Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Noise Model** | Mixed Poisson (Shot) + Gaussian (Read) | Peak: $60\text{--}100$, $\sigma$: $1\text{--}3$ | Peak: $15\text{--}35$, $\sigma$: $4\text{--}10$ | Secondary electron emission counting kinetics |
| **Beam Broadening** | Gaussian Point Spread Blur | $k=3, \sigma \in [0.3, 0.6]$ | $k=5, \sigma \in [0.8, 1.8]$ | Interaction volume vs. pixel sampling resolution |
| **Edge Brightening** | Sobel Gradient Field Addition | Factor: $0.05\text{--}0.15$ | Factor: $0.05\text{--}0.20$ | Elevated secondary electron yield at sidewalls |
| **Wafer Charging** | Local Contrast Field Rescaling | Factor: $0.05$ | Factor: $0.12$ | Electron accumulation in inter-line oxide regions |
| **Stage & Tool Drift** | Rigid Affine + Non-Affine Elastic Warp | Drift $\le 1\text{ px}, \alpha=1.0$ | Drift $\le 3\text{ px}, \alpha=2.5$ | Thermal drift, piezo creep, & ambient vibration |
| **Line-Edge Roughness** | Sinusoidal/Random World-Space Jitter | Baked into Layout | Baked into Layout | Litho/etch stochastics & resist EUV photon noise |

> 📖 For full reference citations, equations, and literature justifications, view `JUSTIFICATION.md`.

---

## 📜 License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.
