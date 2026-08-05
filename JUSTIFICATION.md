# Justification of Dataset-Generation Choices — Drift-Sense Synthetic DRAM SEM Generator

Every structural, noise, and augmentation choice in `generate_dram_dataset.py` is grounded in publicly available literature — peer-reviewed papers, textbooks, and issued/published patents on SEM imaging physics and semiconductor device structure. No proprietary Applied Materials or fab-internal data was used; all numbers below come from open sources. Where the generator's parameter ranges are approximations (there is no single "correct" value — every SEM/tool/node combination differs), the sources establish the *direction and physical mechanism* of the effect, which is what the generator needs to reproduce.

---

## 1. Structural Choice: DRAM Periodic Word-Line / Bit-Line / Contact-Via Array

### What the Code Does
`make_layout_spec()` / `render_layout()` build an orthogonal grid of horizontal word lines and vertical bit lines on a single pitch, with a contact via at every surviving intersection.

### Why This is the Right Structure to Imitate
DRAM cell arrays are literally built this way. Multiple issued patents describe the canonical $6F^2$ / $8F^2$ DRAM cell as word lines and bit lines laid out orthogonally on a single minimum-feature-size grid, with line width and inter-line spacing both equal to "1F," where F is the process's minimum pitch feature **[1][2][3]**. This is exactly the "line width == inter-line gap, single global pitch" rule the generator's `pitch_x`/`pitch_y` and `line_width` parameters encode. The contact/via at every word-line–bit-line crossing corresponds to the bit-line contact / storage-node contact described at each active-area intersection in these same patents **[1][4]**.

> **Note on FinFET Alternative:**  
> The prompt permits a FinFET-style die instead: public teardown data and patents describe FinFET arrays as parallel fins on a fixed fin pitch — reported around 30 nm — under parallel gate lines on a separate, coarser contacted-poly pitch — reported around 50 nm **[5][6]**. The generator's periodic line/pitch machinery would apply equally well to that geometry; DRAM was chosen for this submission because contact vias give a second, independently-defectable structural class beyond just lines.

### Sources
* **[1]** **WO2013002884A1**, *6F2 DRAM cell* — bit lines/word lines on an F-based half-pitch grid, $6F^2$ cell area.
* **[2]** **US6339241B1**, *Structure and process for 6F2 trench capacitor DRAM cell...* — explicit 2F word-line pitch / 3F bit-line pitch.
* **[3]** **US7349232B2**, *6F2 DRAM cell design with 3F-pitch folded digitline sense amplifier* — word line / digitline (bit line) orthogonal grid, folded array architecture.
* **[4]** **US11315928 / US11647623**, *Semiconductor structure with buried power line and buried signal line* — explicit statement that word-line width, bit-line width, and inter-line spacing are all "1F" in a $6F^2$ layout.
* **[5]** **ASIC North**, *FinFET Technology and Layout — Part 1* — published fin-grid pitch (~30 nm) vs. contacted-poly/gate pitch (~50 nm) from device teardowns.
* **[6]** **US9793271 (and family)**, *Semiconductor device with different fin pitches* — parallel-fin geometry with fixed fin-to-fin pitch as the FinFET structural primitive.

---

## 2. Noise Model: Mixed Poisson (Shot) + Gaussian (Read) Noise

### What the Code Does
`add_poisson_noise()` is applied before `add_gaussian_noise()` in `sem_style_augment()`, with the reference image using a higher Poisson "peak" (effectively higher electron dose $\Rightarrow$ less noise) than the search image.

### Why
SEM signal originates from counting discrete secondary/backscattered electrons, so the dominant noise source is Poisson (shot) noise, with additive Gaussian noise from detector/amplifier electronics layered on top — this "Poisson-Gaussian" model is the standard description used in the SEM-denoising literature **[7][8][9]**. One of these sources explicitly notes that Gaussian noise becomes a good approximation of the Poisson distribution once the mean electron count is large (faster scan / lower dose $\Rightarrow$ further from that regime, i.e. more visibly Poisson-shaped) **[9]**, which is the physical basis for the reference image (slower, higher-dose "review" capture) using a higher `poisson_peak_range` than the faster, lower-dose search scan.

### Sources
* **[7]** **Sim, Nia, Tso**, *Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring for Micro-Nanomanipulation* — states Poisson shot noise is the primary noise type in electron microscopy images.
* **[8]** **M-Denoiser: Unsupervised image denoising for real-world optical and electron microscopy data**, *ScienceDirect* — signal-dependent Poisson shot noise plus signal-independent Gaussian read noise is the standard model for microscopy data.
* **[9]** **Timischl, Date & Nemoto (2012)**, *A statistical model of signal-noise in scanning electron microscopy*, Scanning 34(3):137–144 — identifies five physical SEM noise sources (primary emission, secondary emission, scintillator, photocathode, photomultiplier), all Poisson in origin, and notes the Gaussian approximation only holds at high mean counts.

---

## 3. Gaussian Blur (Optics / Beam-Interaction-Volume Broadening)

### What the Code Does
`gaussian_blur()` uses a larger kernel/sigma for the search image (`blur_ksize=5`, $\sigma \in [0.8, 1.8]$) than the reference (`blur_ksize=3`, $\sigma \in [0.3, 0.6]$).

### Why
SEM spatial resolution is fundamentally limited by the electron beam's interaction volume in the sample, not just probe diameter — the same physical interaction volume corresponds to more sample area (and therefore more image pixels) at lower magnification, i.e. more effective blur in pixel units at 10x than at 100x for a fixed detector/beam configuration **[10][11]**. This is the same interaction-volume mechanism cited for edge brightening below, applied to resolution rather than brightness.

### Sources
* **[10]** **GWU AMC Workshop tutorial**, *Scanning Electron Microscopy and Focused Ion Beams* (citing Goldstein et al., *Scanning Electron Microscopy and X-Ray Microanalysis*, Plenum Press) — interaction-volume-dependent resolution and signal effects.
* **[11]** **JEOL Ltd. Glossary**, *edge effect* — explicitly ties interaction/scattering volume size to accelerating voltage and to how features of a given physical size appear in the resulting image.

---

## 4. SEM-Characteristic Edge Brightening

### What the Code Does
`edge_brighten()` adds a fraction of the Sobel gradient magnitude back onto the image, brightening line/via boundaries.

### Why
This is one of the best-documented SEM contrast mechanisms. Secondary-electron yield rises sharply at edges and topographic steps because the beam's interaction volume intersects the surface from an additional lateral direction there, producing measurably brighter edges than flat regions — described consistently as the "edge effect" or "edge brightening effect" across a university SEM lab's teaching materials, a vendor's technical glossary, and a workshop tutorial referencing the standard SEM textbook **[10][11][12]**.

### Sources
* **[10]** **GWU AMC Workshop tutorial** (Goldstein et al.) — "Edge Brightening Effect on Contrast... 'Excess' SEs generated when interaction volume intersects an edge."
* **[11]** **JEOL Ltd. Glossary**, *edge effect* — bright emission at protrusion tips/steps due to elevated secondary-electron emission, strengthening with accelerating voltage.
* **[12]** **CMMP (Cambridge) SEM teaching blog, 2011** — bright-edge artifacts explicitly attributed to enhanced secondary-electron *collection* at feature edges, independent of underlying topography.

---

## 5. Low-Frequency Illumination Gradient (Vignetting / Shading)

### What the Code Does
`add_illumination_gradient()` adds a smooth, low-frequency 2-D field across the whole frame, stronger for the search image.

### Why
Field-dependent shading — brighter center, darker edges, or an arbitrary smooth gradient — is a well-documented artifact of detector angular collection efficiency and non-ideal optics that varies across a field of view, and is corrected in practice via "flat-fielding" **[13]**. While that particular study is optical microscopy, the underlying mechanism it describes (non-uniform angular sensitivity of detectors, radial fall-off of collection efficiency) applies directly to SEM's off-axis Everhart-Thornley-type secondary-electron detectors, whose collection efficiency is inherently angle- and position-dependent — the same detector-geometry dependence that produces the edge-brightening and detector-position effects discussed in the SEM tutorial material above **[10]**.

### Sources
* **[13]** **Kask et al. (2016)**, *Flat field correction for high-throughput imaging of fluorescent samples*, Journal of Microscopy (Wiley) — non-uniform angular detector sensitivity and radial collection-efficiency fall-off as the physical origin of field-dependent shading.
* **[10]** **GWU AMC Workshop tutorial** (Goldstein et al.) — detector position/geometry dependence of collected SE signal.

---

## 6. Scan-Line Intensity Noise

### What the Code Does
`add_scan_line_noise()` perturbs each row's overall brightness by a smooth, row-correlated random amount, stronger in the search image.

### Why
SEM images are built by raster-scanning a single probe across the sample with a fixed dwell time per pixel/line; the number of detected particles per dwell is itself a random variable, and independent papers describe both the pixel-count randomness of raster acquisition **[14]** and, separately, scanline-to-scanline positioning/intensity artifacts from the serial, time-delayed nature of raster scanning **[15]**. A lower-quality/faster search-mode scan (shorter dwell time) has proportionally larger relative fluctuation per line, which is why `scanline_strength` is set higher for the search image than the reference.

### Sources
* **[14]** *Source Shot Noise Mitigation in Focused Ion Beam Microscopy by Time-Resolved Measurement* (arXiv) — raster scanning with fixed dwell time per pixel, with randomness in the number of detected particles per dwell.
* **[15]** *Correcting nonlinear drift distortion of scanning probe microscopy from image pairs with orthogonal scan directions* (arXiv 1507.00320) — describes scanline-level artifacts including "random jitter of each scanline's origin position" arising from the serial nature of raster acquisition.

---

## 7. Local Contrast Variation (Charging)

### What the Code Does
`add_local_contrast_variation()` locally rescales contrast around the image mean using a smooth random field, stronger in the search image (`local_contrast_strength` 0.05 vs. 0.12).

### Why
DRAM structures include dielectric (oxide) regions between conductive lines, and SEM imaging of insulating regions is well known to accumulate surface charge under the electron beam, locally shifting brightness/contrast and in extreme cases washing out the image in that region — documented explicitly for insulating materials in SEM secondary-electron imaging **[16]**. Because this charging is dose/time dependent, a faster, lower-dose search-mode scan is modeled with a smaller but present effect, while the reference (slower/higher quality capture) still shows some but relatively less variation — consistent with the parameter asymmetry chosen.

### Sources
* **[16]** **Nanoscience Instruments**, *Secondary Electrons in SEM: Unlocking Surface Insights at the Nanoscale* — describes charging of insulating materials under electron-beam imaging causing localized, unwanted brightness/contrast artifacts ("washed-out" regions).

---

## 8. Mild Elastic Distortion + Tiny Stage-Drift Translation

### What the Code Does
`apply_elastic_distortion()` (small, smooth non-affine warp) is layered on top of a stage-drift translation folded into `affine_augment()`'s matrix; both are larger for the search image.

### Why
Multiple independent studies of SEM/SPM instrumentation identify drift from ambient vibration (high-frequency, essentially random within a single image), and thermal expansion/piezo-stage creep/magnetic drift (slower, longer-term trends that manifest as a whole-field translation or a gentle nonlinear warp) as the dominant sources of image-to-image and within-image geometric distortion **[17][18][19]**. This is exactly the two-tier model implemented: a single rigid translation (`affine_augment`'s drift term, i.e. the long-term trend) plus a small, smooth, spatially varying elastic field (`apply_elastic_distortion`, i.e. the higher-frequency within-image component).

### Sources
* **[17]** *Correction of image drift and distortion in a scanning electron microscopy* (ResearchGate) — identifies ambient vibration (high-frequency, within-image random drift) vs. thermal expansion/stage creep/magnetic drift (long-term, whole-field-translation trend) as the two distinct drift regimes in SEM.
* **[18]** **unDrift: A versatile software for fast offline SPM image drift correction** (PMC) — describes thermal drift accumulating over the seconds-to-minutes timescale of a typical scan due to instrument temperature fluctuation during acquisition.
* **[19]** *Correcting nonlinear drift distortion of scanning probe microscopy from image pairs with orthogonal scan directions* (arXiv 1507.00320) — models linear (shear/expansion/contraction) and nonlinear (scanline-jitter) drift distortion components separately, matching the code's affine-plus-elastic split.

---

## 9. Rotation + Scale Variation

### What the Code Does
`affine_augment()` applies independent random rotation and scale about the image center for the reference ($\pm 1.5^\circ$, $0.98\text{--}1.02\times$) and search ($\pm 4^\circ$, $0.93\text{--}1.07\times$) images, with the search range always wider.

### Why
A U.S. patent on correcting systematic SEM measurement errors explicitly enumerates rotation error, and X/Y scaling error, alongside positioning error and perpendicularity/tilt/linearity aberrations, as characteristic, recurring error sources across SEM tool setup and sample positioning — i.e. exactly the two augmentation axes implemented here **[20]**. A separate metrology paper on calibrating high magnification for CD-SEM/AFM shows that measured linewidths (and by extension apparent scale) drift and require periodic recalibration against certified reference materials — meaning two images of "the same" structure captured in different sessions (as reference and search are, in this dataset's premise) can differ in effective magnification even on a single, well-maintained tool **[21]**. A third paper on edge-placement-error metrology states directly that the quality of matching a design/reference pattern to a wafer SEM image depends on the accuracy of SEM scan rotation and magnification — the precise navigation-matching problem this dataset is built to train against **[22]**.

### Sources
* **[20]** **US7930654**, *System and method of correcting errors in SEM-measurements* — explicitly lists "scaling error in X and Y, positioning error in X and Y, rotation error or aberrations like perpendicularity, tilt, linearity" as systematic SEM tool/sample errors.
* **[21]** **Kwak et al.**, *Calibration of high magnification in the measurement of critical dimension by AFM and SEM*, ScienceDirect — magnification calibration drift and the need for certified reference materials to keep CD/scale measurements consistent across sessions.
* **[22]** *Fine pixel CD-SEM for measurements of two-dimensional patterns* (ResearchGate) — states that edge-placement-error measurement quality "depends on both the accuracy of the SEM image scan rotation and magnification... [and] the accuracy of pattern matching between the design layout pattern and the realized pattern (wafer)."

---

## 10. Per-Line Width Jitter and Per-Via Diameter Jitter (Baked into the World Spec)

### What the Code Does
`render_layout()` perturbs each bit-line/word-line's local width with a smooth per-line sinusoidal jitter, and each via's diameter with a deterministic per-via random jitter (`stable_hash01`), both defined once in world coordinates so they render identically at 100x and 10x.

### Why These Are Structural, Not Imaging, Effects
Line-edge roughness (LER) and line-width roughness (LWR) are well-documented, unavoidable products of the lithography/etch process itself — random pattern-edge variation on the order of a few nanometers, driven by resist chemistry, photon shot noise in the exposure, and etch stochastics, independent of how the pattern is later imaged **[23][24][25]**. Because LER/LWR is a property of the *fabricated structure*, not of the SEM tool, it must be encoded in the shared world spec (so it looks identical, just at different sampling resolution, in both the reference and search renders) rather than as a post-hoc image filter — which is exactly how the generator implements it.

### Sources
* **[23]** **Bonam et al. (IBM Research)**, *Comprehensive analysis of line-edge and line-width roughness for EUV lithography*, SPIE Advanced Lithography 2017 — LER/LWR as an intrinsic, process-driven pattern-transfer artifact requiring dedicated roughness metrology.
* **[24]** **ScienceDirect Topics**, *Edge Roughness* overview — LER as intrinsic random variability from subwavelength lithography and etch, producing nonuniform structure dimensions.
* **[25]** **US11402742**, *Undercut EUV absorber reflective contrast enhancement* — quantifies LER as 1.5–6 nm random pattern-edge variation (5–20%+ of final CD), attributed to resist chemistry, photon generation/absorption statistics, and shot noise.

---

## 11. Missing Vias, Broken Line Segments, Bright/Dark Blob Defects

### What the Code Does
`make_layout_spec()` randomly removes some via intersections, cuts small rectangular gaps into lines, and scatters bright/dark circular blobs.

### Why These Specific Defect Classes
These map directly onto the standard hard-defect taxonomy used in wafer inspection: contact/via opens (missing via $\Rightarrow$ "open contact failure," explicitly described and automatically classified in a charged-particle-beam wafer inspection patent **[26]**), line opens/bridges (broken line segment), and particle contamination (bright/dark blob), all listed as the core hard-defect categories that SEM-based review tools are built to detect and classify **[27][28]**. A recent paper on generative/synthetic SEM dataset construction for defect-inspection model training independently validates the overall approach of injecting exactly these defect classes into a simulated periodic pattern rather than relying solely on scarce real defect images **[29]**.

### Sources
* **[26]** **US6700122**, *Wafer inspection system and wafer inspection process using charged particle beam* — automated detection/classification of "holes with open contact failure" on contact-hole patterns, directly analogous to the generator's missing-via defect.
* **[27]** **TSI**, *Surface Defect Inspection Tools* — standard hard-defect taxonomy (opens, shorts, voids, cracks) vs. soft defects (particle contamination), identified via SEM/optical/X-ray inspection.
* **[28]** **Averroes.ai**, *Wafer Defect Detection Guide* — defect categories used industry-wide: particle contamination, and pattern defects (bridging, opens, shorts) from lithography/etch anomalies.
* **[29]** *Addressing Class Imbalance and Data Limitations in Advanced Node Semiconductor Defect Inspection: A Generative Approach for SEM Images* (arXiv 2407.10348) — precedent for building synthetic/simulated SEM datasets with injected defect classes (rather than scarce real defect images) to train inspection models, and for combining simulated and real data with augmentation for best performance.

---

## Summary Table

| Generator Effect | Function(s) | Reference Setting | Search Setting | Key Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Periodic word-line/bit-line/via grid** | `make_layout_spec`, `render_layout` | — | — | **[1]–[4]** |
| **Poisson + Gaussian noise** | `add_poisson_noise`, `add_gaussian_noise` | Peak 60–100, $\sigma \text{ 1--3}$ | Peak 15–35, $\sigma \text{ 4--10}$ | **[7]–[9]** |
| **Gaussian blur** | `gaussian_blur` | $k=3$, $\sigma \in [0.3, 0.6]$ | $k=5$, $\sigma \in [0.8, 1.8]$ | **[10][11]** |
| **Edge brightening** | `edge_brighten` | 0.05–0.15 | 0.05–0.20 | **[10]–[12]** |
| **Illumination gradient** | `add_illumination_gradient` | 0.03 | 0.08 | **[13][10]** |
| **Scan-line noise** | `add_scan_line_noise` | 0.01 | 0.03 | **[14][15]** |
| **Local contrast variation (charging)** | `add_local_contrast_variation` | 0.05 | 0.12 | **[16]** |
| **Elastic distortion + stage drift** | `apply_elastic_distortion`, `affine_augment` (translation) | $\alpha=1.0$, Drift $\le 1\text{ px}$ | $\alpha=2.5$, Drift $\le 3\text{ px}$ | **[17]–[19]** |
| **Rotation + scale** | `affine_augment` | $\pm 1.5^\circ$, $0.98\text{--}1.02\times$ | $\pm 4^\circ$, $0.93\text{--}1.07\times$ | **[20]–[22]** |
| **Line-width / via-diameter jitter (in spec)** | `render_layout`, `stable_hash01` | Baked into world (identical at both mags) | Baked into world (identical at both mags) | **[23]–[25]** |
| **Missing vias / broken lines / blob defects** | `make_layout_spec` | Baked into world | Baked into world | **[26]–[29]** |

---

> ### Note on Methodology
> All facts above are paraphrased from the cited public sources; no proprietary Applied Materials data, internal recipes, or fab-specific process parameters were used anywhere in this generator or its justification.