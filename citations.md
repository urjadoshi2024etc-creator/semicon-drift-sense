<div align="center">

# 📚 DriftSense — Citations & Scientific Justification

### Public references supporting the synthetic DRAM SEM generator, imaging augmentations, navigation-error model, and localization approach

**SEMICON India Hackathon 2026 · Problem Statement 02**

</div>

---

## 🎯 Purpose

This document provides the **scientific and engineering justification** for the augmentation and SEM noise-model choices implemented in:

```text
dataset_generator/generate_dram_dataset_v3.py
```

The citation tags **[C1]–[C8]** correspond to comments embedded directly in the generator's docstring, allowing each implementation choice to be traced to its supporting rationale without ambiguity.

> **Important transparency note:**  
> A citation supporting the *existence or qualitative behavior* of an effect does **not** necessarily mean that every numeric parameter used by the generator was directly derived from that source. Where a parameter is an engineering choice rather than literature-derived, this document explicitly says so.

---

## 🗂️ Citation Map

| Tag | Generator / System Choice | Primary Justification |
|:---:|---|---|
| **[C1]** | DRAM pitch range | Historical DRAM process dimensions |
| **[C2]** | Poisson + Gaussian SEM noise | SEM imaging / detector-noise literature |
| **[C3]** | Edge brightening / edge bloom | Secondary-electron edge effects |
| **[C4]** | Illumination gradient / shading | SEM illumination and detector-position effects |
| **[C5]** | LER / LWR | CD-SEM / semiconductor roughness literature |
| **[C6]** | Missing vias, broken contacts, voids, particles | Semiconductor defect / DRAM patent precedents |
| **[C7]** | Rotation, scale, drift | Wafer inspection and navigation-error literature |
| **[C8]** | Scan-line / raster artifacts | SEM acquisition physics |
| — | Periodic matching ambiguity | Semiconductor template-matching patents |
| — | Center-nearest tie-break | Inspection pattern-matching precedent |
| — | Siamese / correlation architecture | General template-matching literature |

---

# 🔬 [C1] DRAM Pitch Range

### Choice being justified

The generator's:

```text
pitch_nm_range
```

across the difficulty profiles, together with the targeted **80–120 nm supplemental dataset** introduced after failure analysis.

The generator uses broader ranges across profiles to deliberately create progressively harder conditions.

| Profile | Pitch Range |
|---|---:|
| Easy | 60–100 nm |
| Medium | 40–120 nm |
| Hard | 30–140 nm |

The hard profile intentionally extends beyond the literature-typical range as a **stress-test regime**, rather than claiming that the entire range represents typical contemporary DRAM dimensions.

### Supporting sources

1. **Micron Technology**, *Inside 1-alpha DRAM* (2021).
2. **EDN**, *The 50-nm DRAM battle rages on* (2009).

### Rationale

These sources document historical DRAM half-pitches of approximately **50–58 nm**, corresponding to full pitches in the approximate **50–120 nm** range across relevant process generations.

This provides the physical basis for the generator's central pitch regime.

The additional 80–120 nm supplemental dataset was introduced specifically after failure analysis identified reduced model accuracy in that regime.

---

# 🧪 [C2] SEM Noise Model — Poisson + Gaussian

### Choice being justified

The generator's two-stage imaging-noise pipeline:

```python
rng.poisson(...)
```

followed by:

```python
rng.normal(0, gauss_sigma, ...)
```

where `gauss_sigma` is correlated with the latent capture-quality variable.

Conceptually:

```text
Ideal SEM Image
       │
       ▼
Poisson / Shot Noise
       │
       ▼
Gaussian Detector / Readout Noise
       │
       ▼
Synthetic SEM Image
```

### Supporting source 1

**Villarrubia, J. S. et al.**

> *Determining the ultimate resolution of scanning electron microscope images*

**Journal of Vacuum Science & Technology B**, 37(6), 2019.

**DOI:** `10.1116/1.5122758`

This work describes a synthetic SEM pipeline that separates edge/positional effects from detector/shot noise as distinct sources with different statistical behavior.

This directly motivates treating the noise components as separate stages rather than collapsing them into one blended noise term.

### Supporting source 2

**Bals, S. et al.**

> *Artificial Scanning Electron Microscopy Images Created by Generative Adversarial Networks from Simulated Particle Assemblies*

**Advanced Intelligent Systems**, 2023.

**DOI:** `10.1002/aisy.202300004`

The cited work reports Gaussian noise with approximately **0.01–0.1 sigma** on a normalized `[0,1]` intensity scale and notes that values above approximately 0.1 can become visibly unrealistic.

The generator's actual `gauss_sigma` range is approximately:

```text
1.0–10.5 on 0–255 intensity scale
≈ 0.004–0.041 on normalized [0,1] scale
```

Thus, this reference is used as **directional physical justification** for including Gaussian detector/readout noise.

> **Important:** These references justify the *noise-model structure and direction*, not an exact derivation of every numeric parameter used by the generator.

---

# ✨ [C3] Edge Brightening / "Edge Bloom"

### Choice being justified

The generator applies a simplified gradient-based edge-brightening term:

```python
img_f + rng.uniform(0.05, 0.20) * grad
```

where `grad` is the normalized Sobel-gradient magnitude.

This is intentionally a **simplified heuristic**, rather than an implementation of a full SEM secondary-electron transport model.

### Supporting source 1

**US Patent 10,648,801 B2**

> *System and method for generating and analyzing roughness measurements and their use for process monitoring and control*

The patent provides a closed-form line-scan model:

$$\frac{\text{SE}(x)}{\text{SE}(\infty)} = 1 + \alpha_e e^{-x/\sigma_e} - \alpha_v e^{-x/\sigma_v}$$


This represents edge brightness as a combination of enhanced secondary-electron escape and reduced interaction-volume effects.

The DriftSense implementation does **not** reproduce this exact biexponential equation. Instead, its gradient-proportional approximation captures the same qualitative behavior:

```text
Higher gradient
      ↓
Brighter edge
      ↓
Brightness decreases away from edge
```

### Supporting source 2

**JEOL Ltd.**

> *edge effect* — SEM Terms Glossary

The glossary explains the physical mechanism: secondary electrons generated near an edge have a shorter escape path, producing a brighter rim.

It also notes that edge-effect strength depends on accelerating voltage.

### Additional supporting source

> *Beam Cross Sections Create Mixtures: Improving Feature Localization in Secondary Electron Imaging* (2025 arXiv preprint)

This work further formalizes elevated secondary-electron yield near edges relative to flat regions through a mixture/convolution perspective.

---

# 💡 [C4] Illumination Gradient / Vignetting / Shading

### Choice being justified

The generator models nonuniform illumination through:

```text
shading
```

as a directional multiplicative gradient field, together with a multi-octave background illumination-drift field:

```text
_fractal_field()
```

### Supporting source

**Molecular Expressions Microscopy Primer**

> *Nonuniform Illumination*

This source documents SEM shading artifacts associated with detector position relative to the specimen and specimen tilt.

This supports modeling shading as a **directional gradient** rather than assuming a perfectly symmetric vignette.

> The multi-octave/fractal background is an engineering approximation used to produce spatially smooth low-frequency variation; its exact numeric strength is not claimed to be directly literature-derived.

---

# 📐 [C5] Line-Edge Roughness (LER) / Line-Width Roughness (LWR)

### Choice being justified

`render_layout()` introduces line-width variation using a sinusoidal per-line modulation:

```text
ler_amp
```

with a stable, hash-seeded phase.

The layout also includes via-diameter jitter.

### Supporting sources

**Bunday, B.; Bishop, M.; Villarrubia, J.; Vladar, A.**

> *Determination of Optimal Parameters for CD-SEM Measurement of Line-Edge Roughness*

**NIST/SPIE Proceedings**, 2003.

**Mack, C.**

> *Line Edge Roughness*

**SPIE Optipedia**

### Rationale

These sources establish LER magnitudes of approximately **5% of critical dimension (3σ)** as representative of sub-100 nm process nodes.

The generator uses:

| Profile | `ler_amp_fraction_range` |
|---|---:|
| Easy | 2–5% |
| Medium | 3–8% |
| Hard | 5–12% |

The easy/medium ranges are centered around the cited order of magnitude.

The hard profile deliberately extends beyond the literature-typical value to create a stronger stress-test regime.

> **The 12% hard-profile upper range should therefore not be interpreted as a claim that 12% LER is typical of real-world manufacturing.**

---

# 🧩 [C6] Missing/Broken Contacts, Voids & Particle Defects

### Choice being justified

`create_world()` generates defect classes including:

```text
missing_vias
merged_contacts
broken_segments
particles
```

### Supporting sources

**US Patent 5,840,205**

Documents a DRAM contact-open SEM photograph, providing direct visual precedent for the missing-via/contact-open defect class.

**US Patent 6,989,583**

Describes mechanisms associated with via void formation.

**US Patent 6,774,024**

Provides additional support for via/contact defect mechanisms.

### Rationale

These sources support the physical plausibility of defect classes involving:

- Missing contacts
- Contact/via voids
- Merged contacts
- Broken structures

The generator therefore includes these as hard local perturbations capable of altering otherwise periodic DRAM patterns.

---

# 🧭 [C7] Rotation, Scale & Drift as Navigation-Error Sources

### Choice being justified

`apply_tracked_geometric_distortion()` applies:

```text
Rotation
+
Scale variation
+
Small perspective homography
```

independently to reference and search captures, with different magnitude ranges.

### Primary supporting source

**US Patent 9,619,727 B2**

> *Matching process device, matching process method, and inspection device employing same*

**Assignee / lineage:** Hitachi High-Technologies

The patent explains that stage-positioning accuracy alone is insufficient at high SEM magnification and that wafer-coordinate-system misalignment, including rotation, can compound navigation error.

This provides direct conceptual support for including:

- Rotation
- Scale jitter
- Navigation-dependent geometric mismatch

and for allowing the search image to experience a wider rotation range than the reference.

### Additional supporting sources

**US Patent 9,892,885 B2**

KLA-Tencor lineage; describes drift compensation using stage-interferometer synchronization.

This illustrates that hardware-level drift compensation can reduce—but does not necessarily eliminate—navigation error, motivating an additional software/AI recovery layer.

**US Patent 5,315,123**

Describes the fundamental throughput-versus-accuracy trade-off associated with periodic drift correction, supporting the idea that navigation error can accumulate between inspection visits.

---

# 🖥️ [C8] Scan-Line / Raster Acquisition Artifacts

### Choice being justified

The synthetic SEM imaging model introduces:

```text
Row-wise scan-line intensity variation
+
Occasional duplicated/skipped scan lines
```

The row variation is smoothed to resemble low-frequency raster-acquisition variation.

### Rationale

This augmentation is grouped with the SEM acquisition-physics literature supporting [C2] and [C4].

Raster-scan acquisition can introduce row-dependent intensity variation and occasional line artifacts through detector and scanning behavior.

> The **category of artifact** is literature-supported; the exact probability assigned to a particular artifact is an engineering choice and is disclosed explicitly below.

---

# 🎯 Multi-Candidate Tie-Break Rule

### Choice being justified

When multiple candidate locations have nearly equal matching confidence, the evaluation baseline uses:

> **Choose the candidate closest to the center of the search image.**

### Supporting source

**US Patent 10,255,519 B2**

> *Inspection apparatus and method using pattern matching*

**Assignee / lineage:** Hitachi High-Technologies

The patent describes selecting the candidate closest to the image center when multiple candidate positions have similarly high matching likelihood.

The rationale is that the smaller assumed positional shift from the tool's previous/expected location is physically more plausible.

This provides a close precedent for the same tie-breaking principle used in the DriftSense evaluation baseline.

---

# 🔁 Core Problem Framing — Periodicity Breaks Classical Template Matching

Periodicity is a central challenge in semiconductor localization.

The project originally hypothesized that repeated DRAM structures could cause classical template matching to produce multiple plausible locations.

This hypothesis was explicitly tested during failure analysis.

> **Final-model finding:** periodic-pitch-repeat confusion was tested but was **not identified as the dominant remaining failure mode**. The final failure analysis instead identified a confirmed, non-confounded relationship with coarse pitch and low local reference-crop via density.

See:

```text
evaluation_report.md
```

for the experimental evidence.

### Supporting source 1

**US Patent 8,139,868 B2**

> *Image processing method for determining matching position between template and search image*

**Assignee / lineage:** Hitachi High-Technologies

Documents periodic-pattern ambiguity using hole-array wafer images and shows that standard template matching can produce multiple equally valid matches under periodicity.

### Supporting source 2

**US Patent 7,925,095 B2**

Describes periodic-structure-specific pattern matching for SEM / inspection field-of-view alignment.

---

# 🧠 Architectural Approach

> **Context only — not a generator augmentation citation.**

The correlation-based Siamese approach implemented in:

```text
model_v6.py
```

uses concepts including:

- Shared / scale-aware feature encoders
- Cross-correlation
- Candidate scoring
- Heatmap-style localization
- Center-point estimation

The overall architecture belongs to a broader family of Siamese and correlation-based template-matching methods.

### Supporting source 1

**Wu, Xian, Su, Ren**

> *A Siamese Template Matching Method for SAR and Optical Image*

**IEEE Geoscience and Remote Sensing Letters**, 2022.

### Supporting source 2

**MDPI Remote Sensing**

> *An Accurate and Robust Multimodal Template Matching Method Based on Center-Point Localization in Remote Sensing Imagery*

2024.

The center-point localization framing is particularly relevant because DriftSense is required to output:

```text
(x, y)
```

rather than a bounding box.

---

# ⚠️ Parameters Without Direct Citation Backing

For full scientific transparency, not every numeric generator parameter is claimed to be literature-derived.

The following choices are **physically plausible engineering decisions**, but their exact numeric ranges are not directly derived from a specific citation.

This distinction is important:

```text
Literature-supported effect
          ≠
Literature-derived numeric parameter
```

---

## 🟡 Partially Justified
### Effect is cited; exact magnitude is an engineering choice

### 1. Rotation ranges

```text
rotation_deg_range_ref
rotation_deg_range_search
```

[C7] supports including rotation as a navigation-error source.

However, specific ranges such as:

```text
±0.5° easy reference
up to ±5.0° hard search
```

are engineering stress-test choices, not directly literature-derived magnitudes.

---

### 2. Scale-jitter ranges

```text
scale_jitter_range_ref
scale_jitter_range_search
```

[C7] supports the conceptual inclusion of scale variation due to stage/navigation limitations.

The exact percentage ranges are engineering choices.

---

### 3. Capture-quality ranges

```text
quality_range_ref
quality_range_search
```

These latent variables jointly drive:

```text
blur_sigma
poisson_peak
contrast
brightness
```

through:

```text
sample_capture_quality_params()
```

The correlated degradation principle—allowing blur, noise, contrast, and brightness to vary together—is a reasonable modeling choice.

It is not itself derived from one specific source.

The Gaussian-noise component has a loose connection to [C2].

---

# 🔴 Fully Uncited — No Source Claimed

The following are intentionally treated as **engineering parameters**, not literature-derived quantities:

```text
line_width_nm_range
via_diameter_nm_range
via_diameter_jitter_range

blur_sigma
poisson_peak
contrast
brightness
```

and the formulas controlling these quantities inside:

```text
sample_capture_quality_params()
```

Additional engineering parameters include:

```text
perspective_strength_range_search

elastic_alpha_range_search
```

where the latter controls local elastic-warp magnitude.

Also:

```text
fractal_bg_strength_range
```

The multi-octave value-noise illumination-drift technique is explicitly documented in the generator as:

```text
"a cheap value-noise approximation,
no external Perlin-noise dependency needed"
```

It is therefore an intentional engineering approximation, not a literature-derived implementation.

Finally:

```text
15% per-image probability
```

for an occasional duplicated/skipped scan line is an engineering probability.

The **artifact category** is grouped under [C8], but this exact probability is not derived from a cited source.

---

# 🧾 Transparency Statement

None of the uncited choices above are presented as experimentally measured manufacturing parameters.

They are used to create a **controlled, reproducible, physically informed synthetic stress-test environment**.

The distinction is:

| Classification | Meaning |
|---|---|
| 🟢 Literature-derived / supported | The source directly supports the effect and/or approximate scale |
| 🟡 Partially justified | The effect is supported, but the exact numeric magnitude is an engineering choice |
| 🔴 Engineering judgment | No specific source is claimed for the exact parameter |

This is intentional and should be described as **informed engineering judgment**, rather than literature-derived calibration, when discussing these parameters.

---

# 📌 Source Traceability

The project maintains three complementary layers of documentation:

```text
Generator Code
     │
     │  citation tags [C1]–[C8]
     ▼
citations.md
     │
     │  scientific / patent justification
     ▼
Research Dossier
     │
     │  broader technical evidence
     ▼
Idea-Submission PDF
```

The broader supporting material is available in the project's research dossier:

```text
Drift_Sense_DRAM_Research_Dossier_Expanded.xlsx
```

and is cross-referenced with the citation slide in the idea-submission PDF.

---

# 📚 Reference Index

| ID | Reference |
|---|---|
| **[C1]** | Micron Technology — *Inside 1-alpha DRAM* (2021) |
| **[C1]** | EDN — *The 50-nm DRAM battle rages on* (2009) |
| **[C2]** | Villarrubia et al. — *Determining the ultimate resolution of scanning electron microscope images* — JVST B 37(6), 2019 — DOI: `10.1116/1.5122758` |
| **[C2]** | Bals et al. — *Artificial Scanning Electron Microscopy Images Created by Generative Adversarial Networks from Simulated Particle Assemblies* — Advanced Intelligent Systems, 2023 — DOI: `10.1002/aisy.202300004` |
| **[C3]** | US Patent 10,648,801 B2 — *System and method for generating and analyzing roughness measurements and their use for process monitoring and control* |
| **[C3]** | JEOL Ltd. — *edge effect* — SEM Terms Glossary |
| **[C3]** | *Beam Cross Sections Create Mixtures: Improving Feature Localization in Secondary Electron Imaging* — arXiv preprint, 2025 |
| **[C4]** | Molecular Expressions Microscopy Primer — *Nonuniform Illumination* |
| **[C5]** | Bunday, Bishop, Villarrubia, Vladar — *Determination of Optimal Parameters for CD-SEM Measurement of Line-Edge Roughness* — NIST/SPIE, 2003 |
| **[C5]** | Mack — *Line Edge Roughness* — SPIE Optipedia |
| **[C6]** | US Patent 5,840,205 — DRAM contact-open SEM precedent |
| **[C6]** | US Patent 6,989,583 — via-void formation mechanisms |
| **[C6]** | US Patent 6,774,024 — via/contact defect mechanisms |
| **[C7]** | US Patent 9,619,727 B2 — *Matching process device, matching process method, and inspection device employing same* |
| **C7 supporting** | US Patent 9,892,885 B2 — stage/interferometer drift compensation |
| **C7 supporting** | US Patent 5,315,123 — throughput/accuracy trade-off in periodic drift correction |
| **[C8]** | General SEM acquisition-physics literature grouped under [C2] and [C4] |
| **Tie-break** | US Patent 10,255,519 B2 — *Inspection apparatus and method using pattern matching* |
| **Periodicity** | US Patent 8,139,868 B2 — *Image processing method for determining matching position between template and search image* |
| **Periodicity** | US Patent 7,925,095 B2 — periodic-structure-specific pattern matching |
| **Architecture** | Wu, Xian, Su, Ren — *A Siamese Template Matching Method for SAR and Optical Image* — IEEE GRSL, 2022 |
| **Architecture** | MDPI Remote Sensing — *An Accurate and Robust Multimodal Template Matching Method Based on Center-Point Localization in Remote Sensing Imagery*, 2024 |

---

<div align="center">

### 🔬 DriftSense

**Physically informed · Reproducible · Explicitly traceable**

*Every cited effect is documented. Every engineering assumption is disclosed.*

</div>
