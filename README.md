<div align="center">

# 🔬 Drift-Sense DRAM SEM Generator (v2)

### *Physically Grounded Synthetic SEM Image Pair Generator for Semiconductor Metrology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A high-precision synthetic data generator simulating multi-magnification SEM acquisition pairs (100x high-dose reference vs. 10x low-dose search scans) for advanced node DRAM wafer review, defect inspection, and drift-tolerant pattern matching.*

[Key Features](#-key-features) • [Local Quick Start](#-local-quick-start) • [Repository Layout](#-repository-layout) • [Pipeline Architecture](#-pipeline-architecture) • [Dataset Schema](#-dataset-schema) • [Physics Grounding](#-physics--process-grounding)

---

</div>

## 📌 Executive Summary

Wafer review and inspection tools in advanced semiconductor fabs rely on matching low-magnification search scans against high-magnification reference libraries to navigate to precise coordinates ($<5\text{ nm}$ accuracy). However, real wafer SEM images are strictly proprietary and scarce.

**Drift-Sense DRAM SEM Generator** bridges this gap by generating photorealistic, physically grounded image pairs of nanometer-scale DRAM cell arrays. Rather than using simple image filters, it simulates the underlying wafer layout in continuous world coordinates, models electron-beam physics, applies process-induced structural variation (LER/LWR), and simulates realistic SEM tool distortions.

> 💡 **Zero Proprietary Data Used:** All parameter ranges and contrast mechanisms are derived strictly from publicly available patents, peer-reviewed SEM physics literature, and open metrology papers. See [`JUSTIFICATION.md`](./JUSTIFICATION.md) for full citations.

---

## ✨ Key Features

- **📐 Authentic $6F^2/8F^2$ DRAM Topography:** Generates orthogonal word-line / bit-line grids with contact vias at intersections on a single global pitch ($1F$ line width, $1F$ gap).
- **⚛️ Dual-Dose Noise Modeling:** Simulates signal-dependent Poisson (shot) noise combined with additive Gaussian (detector read) noise. High-dose reference scans feature minimal noise, while fast search scans capture low-dose shot noise.
- **🔬 Contrast & Optical Artifacts:** Edge brightening (secondary electron yield elevation), low-frequency shading/vignetting, scan-line raster jitter, and dielectric charging (local contrast modulation).
- **🌀 Geometric Aberrations:** Stage drift translation, rotation, magnification scale drift, and non-affine elastic spatial warping.
- **🧬 Process-Level Variations:** Continuous line-edge roughness (LER), line-width roughness (LWR), and per-via diameter jitter baked into the world-coordinate layout.
- **💥 Hard Defect Injection:** Introduces realistic hard wafer defects: contact opens (missing vias), line breaks (open circuits), and particle contamination.

---

## 📂 Repository Layout

```text
semicon-drift-sense/
├── ⚙️ generate_dram_dataset_v2.py  # Main dataset generator script (v2)
├── 📄 JUSTIFICATION.md             # Detailed 300+ line academic & patent justification
├── 📄 README.md                    # Project documentation & guide
├── 📄 requirements.txt             # Python dependencies
├── 📦 dram_sem_dataset_v2.zip      # Archived generated dataset
└── 📁 dram_sem_dataset_v2/         # Generated synthetic dataset directory
    ├── 📊 labels.csv               # Ground-truth transformation & defect annotations
    └── 📁 train/
        ├── 📁 reference/           # High-dose 100x reference images (ref_000.png - ref_029.png)
        └── 📁 search/              # Low-dose 10x search images (search_000.png - search_029.png)
