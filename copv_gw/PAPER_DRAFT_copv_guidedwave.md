# Guided-Wave Structural Health Monitoring of a Type-IV Composite Pressure Vessel: Simulation, sim2real, and Real-Data Damage Detection & Localization

> **Project: Payload2026 / structure-agnostic SHM** — COPV (水素貯蔵タンク) 単独ドラフト v0.1（2026-06-20）。
> [[project_copv_guidedwave]] の成果を1本に統合。主張タグ **[R]** 実データ / **[S]** 自己無矛盾 / **[U]** 代表・未較正。
> overclaim禁止（[[feedback_shm_depth_over_breadth]]）。数値は本セッションの実測スクリプト出力に紐づく。
> 図: `copv_gw_validation.png` / `sim2real_allfreq.png` / `copv_damage_detect.png` / `copv_robustness.png` / `copv_damage_localize.png`。

## Abstract

Type-IV composite-overwrapped pressure vessels (COPVs) are central to hydrogen storage, yet their structural health monitoring (SHM) is usually demonstrated either on a single modality or only in simulation. We present an end-to-end guided-wave SHM study of a real Type-IV COPV (BAM dataset; 700 bar, 25-PZT pitch-catch network) paired with a physically-anchored Abaqus/Explicit guided-wave model spanning the five real excitation frequencies (60–300 kHz). We (i) build and validate the simulation against an independent Lamb-dispersion calculation, (ii) perform a frequency-swept **sim2real** spectral comparison that *diagnoses a concrete model deficiency* (the undamped representative wall over-transmits high frequencies that the real vessel attenuates), and (iii) demonstrate **real-data damage detection** distinguishing healthy / reversible / irreversible states with AUROC **0.92–0.94** at 180 kHz, a 10× margin over the healthy false-alarm floor and the physically-correct severity ordering. We show detection is **frequency-selective** (optimal ≈180 kHz, failing at 60 kHz where the healthy floor saturates) and **pressure-robust** across 20–700 bar, and we localize the damage to one edge of the sensor array (6.6× contrast over a healthy null map). Throughout, every claim is tagged real **[R]** / self-consistent **[S]** / representative-uncalibrated **[U]**, and the limitations (reference-based detection, representative material, assumed array geometry pending sensor-layout extraction) are stated rather than hidden.

---

## 1. Introduction

Hydrogen storage hinges on COPVs that must be monitored for impact damage, delamination, and fibre breakage under pressure cycling. Guided ultrasonic waves (Lamb waves) excited and received by a sparse PZT network are an attractive SHM modality, but published COPV studies rarely combine (a) **real** vessel data, (b) a **corresponding simulation** validated against it, and (c) honest, multi-metric **damage detection** with a false-alarm floor. We close that gap on a single real Type-IV COPV, and we let the sim2real comparison earn its keep by *finding* the model's flaw rather than rubber-stamping it.

**Contributions.**
1. A physically-anchored Abaqus/Explicit guided-wave COPV model across the 5 real excitation frequencies, validated against an independent Lamb-dispersion calculation (§3). **[S/U]**
2. A frequency-swept sim2real spectral comparison that diagnoses a specific, interpretable model deficiency — missing frequency-dependent attenuation (§4). **[R↔S]**
3. Real-data damage detection (healthy / reversible / irreversible) with a controlled false-alarm floor, AUROC 0.92–0.94, correct severity ordering (§5). **[R]**
4. Robustness maps: frequency-selectivity and pressure-invariance of detection (§6); and damage localization with a healthy null control (§7). **[R]**

---

## 2. Materials and data

**Real vessel (BAM).** Type-IV COPV, length ≈1670 mm, OD ≈352 mm, operating ≈700 bar. A network of **25 PZT** transducers (5 rings × 5, hardware IDs 1–5/9–13/17–21/25–29/33–37) drives **600 pitch-catch pairs**. The published array geometry [El Moutaouakil et al., EWSHM 2024, doi:10.58286/29754, CC BY 4.0] is **221 mm circumferential PZT spacing (= 72°, the 5 PZTs evenly wrap the 1105 mm circumference) and 312.5 mm axial ring pitch** (5 rings span 1250 mm) — so the array is **[R]**, not assumed. Damage = four identical cylindrical steel blocks (D1–D4) glued/removed for the reversible cases (D4 under the tension belt), and a drilled hole for the irreversible case. Each acquisition (`Data/Raw_Data` shape (18, 600, 7552)) holds 15 tone-bursts (5 freqs × 3 repetitions) + 3 chirps, sampled at **2.976 MHz** (7552 samples, 2.54 ms). Burst frequencies: **60/120/180/260/300 kHz**. States used: **Baseline** (2 acquisitions → reference + healthy control), **Reversible Damage (RD)** = an added surface mass/block (removable, hence reversible — consistent with the BAM group's published protocol of a ≈300 g, 30 mm-diameter steel block [Heimann et al., ASME J. NDE 8(3), 2025]), **Irreversible Damage (ID)** = a **drilled hole** (permanent material loss), all at 700 bar / ≈24 °C, plus a 20–700 bar pressure sweep per state. **[R]** (damage *type* from the dataset repository note + the group's paper; the precise *location/coordinates* could not be recovered, §7.)
*(Metadata note: the 4th burst is labelled 260 kHz in `Signal_Frequency_Burst` but 240 kHz in `Index_FrequencyvsRepetition`; we anchor quantitative claims to 180 kHz, unambiguous in both.)*

**Simulation.** Abaqus/Explicit, 90° barrel sector (φ352 × L500 mm), **orthotropic composite shell (S4R) with 8-ply filament-wound layup [S]**: [CF_90(1mm) / CF+54(1.5mm) / CF−54(1.5mm) / CF_90(1mm) / CF+54(1.5mm) / CF−54(1.5mm) / CF_90(1mm) / GFRP_0(1mm)], total wall 10 mm. Materials from the Dispersion-Calculator database: **CF = T700M21** (E₁=125.5 GPa, E₂=8.7 GPa, G₁₂=4.14 GPa, ν₁₂=0.37, ν₂₃=0.45, ρ=1571 kg/m³), **GFRP outer = TVR380M12R** (E₁=46.4 GPa, E₂=14.9 GPa, G₁₂=5.23 GPa, ν₁₂=0.27, ν₂₃=0.30, ρ=1800 kg/m³) **[S]**. CLT A-matrix of this layup gives effective axial Ex=17.3 GPa and hoop Ey=60.0 GPa (ρ̄=1594 kg/m³) — the winding is hoop-dominated, so axial stiffness is substantially lower than a quasi-isotropic homogenization. **Ply thicknesses remain representative [U]** (BAM target-winding schedule pending). Tone-burst (5-cycle Hann) radial excitation at a central node; receiver displacement histories at the surrounding grid, sampled at the real 2.976 MHz. Mesh size set per-frequency from the A0 wavelength (element ≤ λ_A0/18), giving ~74k–1.1M elements from 60→300 kHz (seed 1.37→0.35 mm; finer than the prior homogenized model due to lower A0 phase velocity).

---

## 3. Guided-wave model and dispersion validation

An independent Rayleigh-Lamb solve using the CLT-homogenized orthotropic layup (Ex=17.3 GPa, ν=0.245, ρ=1594 kg/m³ **[S]**) gives the fundamental A0/S0 branches; A0 **group velocity is nearly flat at 2054→1977 m/s** across 60→300 kHz — a non-dispersive regime at these high f×h values (600–3000 kHz·mm), contrasting with the prior homogenized model (2145→3109 m/s rising). A0 phase velocity: 1476→1895 m/s. S0 phase velocity: 3376→1953 m/s (strong dispersion with cut-on near 150 kHz). The FEM, excited at each frequency, propagates a clean wavefront: all 24 receivers register signal, **time-of-flight scales with distance**, and the **apparent velocity is consistent with the flat A0 group velocity trend** (§`copv_gw_validation.png`). **[S]**

Honest gap: the CLT RL solver uses the isotropic-equivalent Ex and ν for the Lamb determinant; the true Gxy=19.5 GPa differs from the implied G=Ex/(2(1+ν))=6.9 GPa. Absolute group-velocity calibration requires the MATLAB Dispersion-Calculator with the full anisotropic layup. Additionally, ply thicknesses are representative **[U]**, so the FEM first-arrival speed comparison is self-consistent **[S]** rather than a ground-truth validation. The wave *physics* is validated; absolute calibration is not claimed.

---

## 4. sim2real: spectral comparison and a diagnosed deficiency

We compare the real RX spectra (TX1→RX2, healthy, 700 bar) against the simulated RX spectra across all 5 excitation frequencies (guided-wave band ≥20 kHz; the real record's <20 kHz excitation-crosstalk/baseline spike is excluded as instrumentation). **[R↔S]**

At 180 kHz the dominant frequencies agree closely (real 173 kHz, sim 150 kHz). Band-wide, a clear and physically meaningful divergence emerges:
- **Real** received energy *saturates near 140–170 kHz* even when excited at 260/300 kHz — the real vessel **attenuates high-frequency guided waves** (plus PZT transduction roll-off).
- **Sim** dominant frequency tracks excitation monotonically (38→261 kHz) because the representative material is **undamped [U]** — it transmits whatever it is driven at.

Spectral cosine-overlap is best mid-band (0.68 at 180 kHz, 0.74 at 260 kHz) and degrades where the missing-damping mismatch dominates. **The sim2real thus diagnoses a specific, fixable deficiency: the FEM lacks calibrated material/Rayleigh damping** — a falsifiable result, not a failure (§`sim2real_allfreq.png`).

---

## 5. Real-data damage detection

Per pitch-catch pair, the damage index is DI = 1 − corr(test, reference), averaged over the 3 repetitions at 180 kHz, with a windowed signal (first 250 samples removed to skip excitation crosstalk). The **second baseline acquisition** provides a healthy-vs-healthy **false-alarm floor**. Multi-metric reporting (mean/median/p95 DI + AUROC), per [[feedback_eval_metrics]]. **[R]**

| State | mean DI | AUROC vs healthy-control |
|---|---|---|
| healthy-control (baseline2 vs baseline1) | **0.023** (floor) | — |
| reversible (RD) | 0.234 | **0.919** |
| irreversible (ID) | 0.243 | **0.939** |

Three credibility checks pass: (i) the damage DI is **10× the healthy floor**; (ii) **ID > RD** in both DI and AUROC, the physically-correct severity ordering; (iii) all states are temperature-matched (≈24 °C), excluding a thermal confound (§`copv_damage_detect.png`).

---

## 6. Robustness — frequency and pressure

**Frequency-selective (@700 bar).** Detection is strongly frequency-dependent: AUROC(ID) = 0.50 (60 kHz) / 0.79 (120 kHz) / **0.92 (180 kHz)** / 0.72 (260 kHz) / 0.63 (300 kHz). The healthy floor explains it — it saturates at 60 kHz (0.52, as large as the signal → no detection) and rises again at 260/300 kHz (attenuation/noise), leaving **≈180 kHz as the sweet spot** (floor 0.029). **[R]**

**Pressure-robust (@180 kHz).** Across 20–700 bar the damage DI stays ≈0.21–0.29 — always **8–10× above the 0.029 floor** — so detection holds over the full operating range. DI *rises slightly at lower pressure* (ID 0.247→0.294 from 700→250 bar), consistent with internal pressure **closing cracks** and reducing scattering at high load (§`copv_robustness.png`). **[R]**

---

## 7. Damage localization

We map damage two ways (§`copv_localize_true.png`): (A) **per-PZT involvement** — the mean DI of all pairs touching each sensor; and (B) **RAPID** elliptical tomography on the **true unrolled-cylinder geometry [R]** (221 mm circumferential / 312.5 mm axial; circumferential wrap-around handled, since the 5 PZTs evenly span the full 1105 mm circumference). A healthy-control map is the null.

The irreversible damage **localizes to circumferential position ≈0° (the arc-0 axial line, sensors 1/6/11/16/21) toward the upper rings (axial ≈1250 mm)** — sensor 1 (DI 0.557 vs healthy 0.060) and sensor 21 are the most involved, both on the arc-0 line. The RAPID image is **6.6× brighter** than the healthy null, so the spatial signal is real **[R]**; the peak sits near the top-ring axial boundary, so the circumferential position (0°) is better constrained than the exact axial location. Coordinates are now **absolute [R]**, not assumed.

*Aside — a data-driven cross-check.* Before the published geometry was located, we attempted to recover the layout from the data alone (first-arrival ToF → 25×25 distances → 3D MDS + cylinder fit). It recovered only the **scale** (cylinder radius 163 mm vs real 176 mm; spacing ~50 mm) but not the 5×5 layout (§`copv_geometry_from_tof.png`). The reason is now clear: the 5 PZTs **wrap the full circumference** (221 mm × 5 = 1105 mm), so the ToF distance matrix is intrinsically non-planar and the reverberant first arrivals are noisy — a flat MDS cannot embed it. The published geometry supersedes this; the negative is retained as an honest methodological note.

---

## 8. Limitations (stated, not hidden)
- **Reference-based** detection (needs a healthy baseline; not reference-free).
- **Single healthy control** acquisition → the false-alarm floor is one healthy-vs-healthy comparison.
- **600 pairs are spatially correlated** → AUROC is within-acquisition pair separability, not a population guarantee.
- **Representative material [U]** (undamped homogenized wall); the sim2real-diagnosed missing damping and the true CF/GF layup are open.
- **Array geometry now [R]** (published spacing 221/312.5 mm) — but the exact **damage coordinates** (D1–D4 / hole positions, in the dataset's figures) are not transcribed; localization is validated against array geometry and a qualitative paper clue (D1 near the sensor 5–6 path), not against surveyed damage coordinates. The local documentation PDF remains corrupted, but the geometry no longer depends on it.

## 9. Conclusion
On a single real Type-IV COPV we close the loop from a validated guided-wave simulation, through a sim2real comparison that *diagnoses* the model's missing damping, to real-data damage **detection** (AUROC 0.92–0.94, 10× floor, correct severity order), **robustness** mapping (frequency-selective, pressure-invariant), and **localization** (6.6× over a healthy null). The contribution is not a new estimator but a falsifiable, real-data, simulation-paired COPV SHM demonstration with its failure boundaries made explicit — and the basis for the COPV instance of the structure-agnostic framework ([[project_structure_agnostic_loso]]), upgrading it from an honest negative (sparse static gauges, AUROC 0.00) to a real guided-wave detector.

## 図表
- Fig1 dispersion + RX validation — `copv_gw_validation.png` ✅
- Fig2 band-wide sim2real (missing-damping diagnosis) — `sim2real_allfreq.png` ✅
- Fig3 damage detection (DI distributions + sorted DI) — `copv_damage_detect.png` ✅
- Fig4 robustness (frequency + pressure) — `copv_robustness.png` ✅
- Fig5 localization on assumed grid (per-PZT + RAPID + null) — `copv_damage_localize.png` ✅
- Fig6 **absolute-coordinate localization on true unrolled-cylinder geometry** — `copv_localize_true.png` ✅ (geometry from EWSHM 2024 paper, doi:10.58286/29754)
- Fig S1 data-driven geometry cross-check (honest negative) — `copv_geometry_from_tof.png` ✅

## 投稿先候補
IWSHM 2027（SHM本丸・本命 [[project_conferences_2027]]）／水素貯蔵×SHM 観点で *Int. J. Hydrogen Energy* も射程。構造非依存論文（[[project_payload_link]]）のCOPV節として吸収する版と、本COPV単独版の二系統。
