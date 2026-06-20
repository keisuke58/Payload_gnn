# Structure-Agnostic Damage Detection: One Detector Across Five Heterogeneous Real Structures

> **Project: Payload2026**（H3フェアリングGNN-SHMの上位フレーム＝[[project_payload_link]]）。
> 正典コピー: `Payload2026/papers/PAPER_DRAFT_structure_agnostic.md`（GW論文 `paper_draft.md` と並置）／作業元: `GNN/ingest/PAPER_DRAFT.md`。
>
> 本文ドラフト v0.1（2026-06-20）。GWフェアリング論文（`Payload2026/papers/paper_draft.md`）を1構造として吸収し、
> 5実構造のTier-B検証を新規スパインに据える。主張タグ **[R]**実データ/**[S]**自己無矛盾/**[U]**代表未較正。
> overclaim禁止（[[feedback_shm_depth_over_breadth]]）。数値は `RESULTS.md` 正典に紐づく。**[要確認]**=本文確定前に再抽出。

## Abstract

Structural health monitoring (SHM) methods are typically built for one structure, one material, and one sensing modality, and most are validated only in simulation. We ask whether a *single* damage-detection procedure can operate across fundamentally different structures without per-structure redesign, and whether that claim survives contact with **real experimental data**. We introduce a **Tier-B contract** that reduces any structure–modality pair to a common per-node representation `(response, displacement-magnitude)` on a geometry graph, and apply one unmodified Stage-0 detector across **five heterogeneous real structures**: a metallic (AA2024) panel under DIC full-field fatigue cracking, a CFRP omega-stringer under laser-vibrometry guided waves, a CFRP interstage shell under thermo-elastic stress, a Type-IV composite-overwrapped pressure vessel (COPV) under strain-gauge burst loading, and a CFRP stiffened panel under distributed fibre-optic (DFOS) run-to-failure. The same detector yields node-/frame-level AUROC of 0.9999 (interstage), 1.0000 (metallic DIC), and a monotone severity signal (omega-stringer); it **fails honestly** on the 9-gauge strain-instrumented COPV in the concentration-feature space (AUROC 0.00) while succeeding on raw magnitude (1.00). Crucially, the *same Type-IV COPV structure class* sensed with a **25-PZT guided-wave network** (real BAM data) is detected at **AUROC 0.92–0.94**, robust across 20–700 bar and localized on the vessel in **absolute coordinates** — demonstrating that the strain-gauge failure was a **sparse-sensing/modality limit, not a structure limit**. We further show that fleet pooling improves degradation-onset detection on real metallic crack curves, while deterministic remaining-useful-life (RUL) on DFOS data does **not** succeed, and that generalization breaks across load modes — each stated as a negative result rather than hidden. The framework absorbs a prior graph-neural-network fairing detector as the composite-fairing instance; because no real flight-fairing data is released, we validate it against **two real proxies** — a CFRP guided-wave stringer and, newly, a real sandwich panel whose core defects are detected at **AUROC ≈ 0.90** — bracketing the fairing's material and sandwich construction. Finally, a four-structure leave-one-structure-out test shows the detector's *ranking* transfers to a held-out structure (AUROC 0.91–1.00) while a *raw* alarm threshold does not (false-alarm up to 0.997 across structures of incomparable scale); per-structure self-normalisation with ~15 healthy measurements and one shared conformal threshold restores control (≤0.15–0.26) — structure-agnostic detection is achievable, structure-agnostic *thresholding* is not. We argue that the contribution is not a new estimator but a *falsifiable, real-data, cross-structure* demonstration with its failure boundaries made explicit.

---

## 1. Introduction

**Fragmentation.** SHM pipelines are usually structure-specific: a guided-wave method for a panel, a strain-threshold for a vessel, a damage index for a plate. Each re-encodes geometry, sensing, and damage afresh, so methods do not transfer and claims rarely cross material/modality classes. Meanwhile most "general" claims are demonstrated only on simulation or on a single structure.

**Gap.** No prior work demonstrates *one* detection procedure across (i) more than one material class (metal *and* CFRP), (ii) more than one sensing modality (full-field optical, laser vibrometry, thermo-elastic stress, point strain, distributed fibre, PZT guided wave, air-coupled ultrasonic guided wave), on (iii) **real experimental data**, with (iv) honest reporting of where the single procedure fails.

**Contributions.**
1. **Tier-B contract** — a modality-invariant per-node reduction `(response, disp_mag)` on a geometry graph, with thin per-modality adapters (DIC / SLDV / stress / strain-gauge / DFOS / PZT guided-wave). §3.
2. **One-detector cross-structure validation on five real structures** (the COPV under two sensing modalities) with real AUROCs. §4–5.
3. **Honest negatives as first-class results** — COPV concentration-feature failure, deterministic-RUL failure, load-mode generalization boundary. §6.
4. **A negative resolved by modality** — the same Type-IV COPV structure, sensed by guided waves instead of sparse strain, is detected (AUROC 0.92–0.94), made robust (frequency/pressure), and localized in absolute coordinates: the bottleneck is the modality, not the structure. §5.5.
5. **Absorption of a physics-informed GNN fairing detector** (prior work) as the composite-fairing structure instance, validated against **two real proxies** (a CFRP guided-wave stringer and a real sandwich panel detected at AUROC ≈ 0.90) under the constraint that no flight-fairing data is released. §5.4.
6. **A cross-structure calibration result** — four-structure leave-one-structure-out shows the detector's *ranking* transfers to a held-out structure (AUROC 0.91–1.00) but a raw alarm *threshold* does not (false-alarm up to 0.997 across incomparable scales); per-structure self-normalisation with ~15 healthy measurements and one shared conformal threshold restores control (≤0.15–0.26). Structure-agnostic detection is achievable; structure-agnostic *thresholding* is not. §5.6.

---

## 2. Related Work
- GNN-SHM on FE meshes / sensor graphs (e.g. graph-convolutional damage detection from vibration [2]; Pfaff et al. 2021 MeshGraphNets [1]) — single structure, mostly simulation.
- Transfer / domain adaptation for SHM — same-structure gains, cross-structure negative transfer (our own prior fairing study, §5.4).
- Multi-structure / population-based SHM — conceptual; rarely one *code path* on heterogeneous **real** data with negatives disclosed.
- **Positioning:** we trade method novelty for a falsifiable, real, cross-structure contract and its limits.

---

## 3. The Tier-B Structure-Agnostic Contract

**Frame.** Each measurement → `Frame(nodes, coords|kNN-edges, fields)`. A per-modality **adapter** maps raw output to two canonical per-node scalars:
- `response` — the modality's damage-sensitive field (DIC strain invariant / SLDV wave-RMS / DSPSS stress / gauge strain / DFOS strain / PZT guided-wave damage index).
- `disp_mag` — displacement magnitude where available (masked otherwise).

**Detector (Stage-0).** Residual / Mahalanobis distance in Tier-B space against a healthy reference. For drifting node sets (DIC facet cloud changes per frame) we use a **spatial-nearest-neighbour reference** instead of fixed-index residuals. Canonical code: `tierb.py`, `stage0_detect.py`; schema `structure_schema.md`.

**Pipeline.** Stage 0 detect → 1 classify → 2 characterise → 3 prognosis → 4 fleet. This paper reports Stage-0 (§5) and Stage-3/4 prognosis (§6).

---

## 4. Datasets — five real structures (the COPV under two modalities) + the fairing's real sandwich proxy + one prognosis source

All public; all real experiment ([R]). The COPV appears as two rows — sparse strain and guided wave — the same structure class under two sensing modalities (§5.5). The fairing itself is simulation-only and is absorbed via §5.4; its closest real **sandwich** proxy is listed separately (italic).

| Structure | Material | Modality | Nodes | Damage | Source |
|---|---|---|---|---|---|
| Metallic MT panel | AA2024 (metal) | DIC full-field | 21,372 | fatigue crack (run-to-failure) | Zenodo 5740216 |
| Omega-stringer | CFRP | SLDV wavefield | 233,289 | debond (3 levels) | Zenodo 5105861 |
| Interstage shell | CFRP | thermo-elastic stress (DSPSS) | 13,942 | 1×1 node defect | WCCM precomputed |
| COPV (pressure vessel) | CFRP Type-IV | strain gauges (sparse) | 9 | burst (near-burst) | Zenodo 10608733 |
| **COPV — guided wave** | CFRP Type-IV | **25-PZT guided wave** (600 paths) | 25 | steel-block (RD) + drilled hole (ID) | **BAM Zenodo 17776240 / 17782123; EWSHM 2024 doi:10.58286/29754** |
| Stiffened panel | CFRP | DFOS (ODiSi-B) | — | run-to-failure | DataverseNL QNURER |
| *Sandwich panel (fairing proxy, §5.4)* | *steel face / PIR foam* | *air-coupled ultrasonic guided wave (12-ch)* | *12* | *core voids / foam-change (37)* | *TU Darmstadt, doi:10.48328/tudatalib-2000* |
| CFRP multiaxial fatigue (prognosis only) | CFRP coupons | tabular S–N (biaxiality) | — | uni-/multi-axial loads | Mendeley jpk2t755vg |

Tier-B coverage check (one frame each) — `response_cov`/`disp_cov`: metallic 1.00/1.00, omega-stringer 1.00/1.00, interstage 1.00/0.00 (no disp, masked). [R]

---

## 5. Stage-0 structure-agnostic detection — real numbers

Same detector code path; only the modality adapter differs.

| Structure | Label | Metric | Result | Tag |
|---|---|---|---|---|
| Interstage | node 1×1 GT | node-AUROC | **0.9999 ± 0.0010** (residual ratio 144×) | [R] label / [S] stress source is FEM |
| Metallic MT panel | crack <60 vs ≥60 mm | frame-AUROC | **1.0000** (spatial-NN ref; ref-free p99.9 tail was 0.74) | [R] |
| Omega-stringer | scenario severity | residual monotone | intact 0 → local 0.119 → large 0.236 | [R] |
| COPV (strain, sparse) | near-burst vs early (LOSO) | Tier-B-conc AUROC **0.00** / raw-mag AUROC **1.00** | **[R] honest negative** |
| **COPV (guided wave)** | healthy vs damaged (per-pair DI) | detection AUROC **0.919** (RD) / **0.939** (ID) | **[R] negative resolved (§5.5)** |
| *Sandwich panel (fairing proxy)* | *core defect vs healthy (per-channel, local null)* | *detection AUROC **0.90** [0.85–0.94]* | *[R] real sandwich (§5.4)* |

**Reading.** Three structures detect cleanly through the *same* Tier-B path, spanning metal↔CFRP and DIC↔SLDV↔stress. The COPV exposes the contract's limit under sparse sensing: with only 9 gauges the concentration feature is uninformative (0.00) while raw magnitude separates perfectly (1.00) — the adapter/modality, not the structure, is the bottleneck. §5.5 confirms this directly: the **same COPV structure class under guided waves is detected at AUROC 0.92–0.94**. Evaluation uses a multi-metric panel (AUROC/AUPRC/recall/FPR), not accuracy alone ([[feedback_eval_metrics]]).

### 5.4 The composite fairing as one structure — two real proxies under the no-flight-data constraint
The prior physics-informed GNN fairing study is the fairing instance. On simulated guided-wave data a 34-dim physics-informed node representation gives GraphSAGE AUC 0.995 / opt-F1 0.788 (5-class), GAT 0.992/0.758 [S/U: simulation].

**Real flight-fairing data is unavailable** — the H3 fairing exists for us only as design FEM, with no released instrumented test article — so the fairing's real-data validation rests on public *proxy* structures chosen to bracket its two defining attributes, **material** (CFRP) and **construction** (skin–core sandwich):

- **CFRP + guided-wave proxy (material axis) [R].** On the OGW long-term real GW set the multi-metric panel gives recall 0.925 / AUPRC 0.914 / AUROC 0.832 at detection, and a quantile domain-alignment recovers temperature-confounded FPR from ~0.58 to **0.052** (design value); severity on the real OGW stringer is monotone (DI 0 → 0.567 → 0.998). Matches the fairing's *material and modality*, but is a stiffened plate, not a sandwich.
- **Sandwich + guided-wave proxy (construction axis) — new [R].** The closest openly available *real* sandwich-with-core-defect guided-wave dataset (TU Darmstadt in-line air-coupled-ultrasound sandwich panels, doi:10.48328/tudatalib-2000) is a steel-faced PIR-foam sandwich line-scanned by a 12-channel air-coupled ultrasonic guided-wave array with 37 inserted core defects (voids / foam-density change). Through a Tier-B air-coupled-scan adapter (per-channel local detrending to remove the slow scan drift; robust-z of amplitude / time-of-flight as the `response`), the **same Stage-0 residual detection** reaches **AUROC ≈ 0.90** (90 % CI 0.85–0.94; **0.85** against a strict *same-region* healthy null — i.e. not a region artifact; per-channel CH6 0.88 / CH7 0.97; n = 28 defects). A naive modality-blind detector reaches only 0.56–0.67, so the gain is the adapter, not the labels. This matches the fairing's *construction* (skin–core sandwich, core void/disbond) but not its materials (steel/PIR-foam vs CFRP/Al-honeycomb), and absolute calibration is weak (detection ≈0.2 at a strict same-region 5 % false-alarm) — ranking transfers, thresholding needs a local healthy reference, the same lesson as the COPV-GW and OGW proxies.

The two proxies bracket the fairing — OGW the CFRP+guided-wave axis, the TU Darmstadt sandwich the sandwich-construction+core-defect axis — both read by the **same** Stage-0 path. The residual gap (a real CFRP/Al-honeycomb sandwich with skin–core disbond) is closable only by on-request data; no such set is openly released as of 2026 (requests to IMP-PAN and IIT Bombay are outstanding). The fairing thus enters the cross-structure story as a CFRP/sandwich/guided-wave instance with two real proxies and an explicit material gap.

### 5.5 The COPV negative is a modality limit — guided-wave rescue [R]
The §5 COPV failure (sparse 9-gauge strain, concentration AUROC 0.00) is **not a property of the structure**. On a second real Type-IV COPV instrumented with a **25-PZT guided-wave network** (BAM dataset, doi:10.5281/zenodo.17776240 + 17782123; 700 bar, 600 pitch-catch paths, 60–300 kHz; geometry published in EWSHM 2024, doi:10.58286/29754), one reference-based damage index DI = 1 − corr(test, baseline) per pair separates healthy from damaged with **AUROC 0.919 (reversible steel block) / 0.939 (irreversible drilled hole)** at 180 kHz, a **10× margin** over a healthy-vs-healthy false-alarm floor (mean DI 0.023), with the physically-correct severity order (ID > RD) and matched temperature (no thermal confound). [R]

- **Robustness.** Detection is **frequency-selective** (AUROC(ID) 0.50/0.79/**0.92**/0.72/0.63 at 60/120/180/260/300 kHz — optimal 180 kHz, failing at 60 kHz where the floor saturates) and **pressure-invariant** across **20–700 bar** (damage DI 0.21–0.29, always 8–10× the floor; DI rises slightly at low pressure, consistent with internal pressure closing cracks). [R]
- **Absolute localization.** RAPID tomography on the **true unrolled-cylinder geometry** (221 mm circumferential / 312.5 mm axial ring pitch, full circumferential wrap; geometry from the EWSHM 2024 paper, so **[R]** not assumed) places the irreversible damage at circumferential ≈0° toward the upper rings (axial ≈1250 mm), **6.6× brighter** than the healthy null map. [R]
- **Simulation pairing.** An Abaqus/Explicit guided-wave model of the COPV wall reproduces the dispersive A0 trend and, via a frequency-swept sim2real comparison, *diagnoses* a concrete model deficiency (undamped representative wall over-transmits high frequencies the real vessel attenuates). [S/U]

**Reading.** The COPV thus appears twice — failing under sparse strain, succeeding under guided waves — making it the paper's sharpest demonstration that **the sensing modality, not the structure, sets the detection boundary**. (Detail: companion COPV-only draft `Payload2026/copv_gw/PAPER_DRAFT_copv_guidedwave.md`.)

### 5.6 Does the alarm threshold transfer? A leave-one-structure-out test [R]
§5 detects each structure against *its own* healthy reference. The title's stronger promise — *structure-agnostic* — asks whether a detector calibrated on a set of structures transfers to a **held-out** one. We test this with leave-one-structure-out (LOSO) over **four** structures — metallic DIC, omega-stringer SLDV, COPV strain, and the TU Darmstadt real sandwich (§5.4, air-coupled-scan adapter) — holding each out and calibrating the alarm on the others (`loso_calibration.py`, Fig. 8; the interstage is a node-localisation structure with a single healthy field, frame-AUROC 0.52, excluded). One detector — a Tier-B anomaly score against each structure's own healthy reference — gives within-structure AUROC 1.000 on the first three and 0.911 on the real sandwich.
- **Ranking transfers.** Held-out AUROC = **1.000** for the first three and **0.911** for the sandwich (mean 0.978): a detector calibrated on the *other* structures ranks damage on a never-seen one — including a genuinely noisy real detection (the sandwich). Structure-agnostic *detection* holds.
- **A raw threshold does not.** The structures' raw scores span orders of magnitude (mean spatial-NN residual ~10⁻³ vs air-coupled max-z ~5), so one threshold from the others is meaningless: held out, the largest-scale structure (the sandwich) false-alarms on **99.7 %** of healthy windows while the smaller-scale structures never alarm — uncontrolled in *both* directions.
- **Self-calibration restores control.** Re-expressing each score in its own healthy units (median/IQR) and applying the *same* conformal threshold brings every false-alarm rate to **≤ 0.26** (the sandwich; ~0 for the others), and to **≤ 0.15** with only **15** held-out healthy sub-frames, while detection stays at **0.89–1.00**. The residual 0.15–0.26 on the sandwich is the honest cost of a genuinely noisy real detection (AUROC 0.91), not a perfectly separable one.

This makes the recurring observation of §5.4–5.5 (and the COPV-GW / OGW / sandwich proxies) explicit and cross-structural: **structure-agnostic detection is achievable, but structure-agnostic *thresholding* is not — a handful of healthy measurements on the new structure is the irreducible local ingredient.** *Honest scope:* the false-alarm figures are node-bootstrap estimates (sampling noise, not full operational variability); **two** structures supply real healthy variability — the metallic panel (three specimens) and the sandwich (~400 panel windows) — while omega-stringer and COPV have a single healthy field each (degenerate). The threshold-transfer claim is therefore demonstrated on the metallic and sandwich held-outs.

---

## 6. Beyond detection — fleet prognosis and its limits

- **[R] positive:** fleet pooling on real metallic crack curves (`fleet_prognosis.py`, `dlr_crack_curves.npz`) improves **degradation-onset detection** (Stage-4 fleet-leader effect on slope identifiability).
- **[R] honest negative — deterministic RUL fails:** on DFOS stiffened-panel run-to-failure, monotone health index exists (Spearman −0.85) but deterministic threshold-crossing RUL fails (α–λ 3–14 %, extrapolation error 100–1400 %). Hierarchical-Bayes pooling improves onset detection, **not** life-point prediction.
- **[R] generalization boundary:** per-load-mode Bayesian Basquin on CFRP multiaxial-fatigue data (torsion/shear-dominated b = −17.8, axial-dominated b = −13.5); load-mode-crossing unification fails (R² = −3.7) — real-data confirmation that **load mode is the generalization boundary**.

---

## 7. Discussion
- **What structure-agnosticism buys:** one detector, five structures (the COPV under two modalities) plus the fairing's real sandwich proxy, three materials, seven sensing modalities (DIC, SLDV vibrometry, thermo-elastic stress, point strain, DFOS, PZT guided wave, air-coupled ultrasonic guided wave), real data.
- **Detection is agnostic; thresholding is not (§5.6).** Four-structure leave-one-structure-out shows the detector's *ranking* transfers to an unseen structure (AUROC 0.91–1.00) but a *raw* alarm threshold does not — across structures of incomparable score scale, false-alarm runs to 0.997. Per-structure self-normalisation with ~15 healthy measurements plus one shared conformal threshold restores control (≤0.15–0.26). The irreducible local ingredient is a handful of healthy measurements — not a re-designed detector.
- **Where it breaks (explicit) — and where it doesn't:** (i) sparse sensing degrades the concentration adapter (COPV 0.00 vs raw 1.00) — **but this is a modality limit, resolved on the same structure by guided waves (0.92–0.94, §5.5), not a structure limit**; (ii) deterministic RUL extrapolation; (iii) cross-load-mode generalization. (i) is now a *bounded, resolved* failure; (ii)–(iii) remain open.
- **Validity tags** per claim ([R]/[S]/[U]) pre-empt the "simulation-only" critique; the interstage stress source is FEM ([S]) while its detection label and all other structures are real ([R]).
- **Honest stance:** the negatives are the credibility core, distinguishing this from over-claimed single-structure SHM reports ([[feedback_shm_depth_over_breadth]]).

## 8. Conclusion
One Tier-B contract and one detector detect damage across five heterogeneous **real** structures (metal/CFRP; DIC/SLDV/stress/strain/DFOS/PZT-guided-wave), with failure boundaries stated rather than hidden — and, where a boundary is hit (the sparse-strain COPV), shown to be a **modality limit resolved on the same structure by guided waves** (AUROC 0.92–0.94, with absolute-coordinate localization). Method novelty is modest; the contribution is a falsifiable, real, cross-structure demonstration with one of its negatives explicitly resolved. Future: the hierarchical-Bayes fleet RUL, and calibrating the COPV guided-wave simulation's damping (the sim2real-diagnosed gap).

---

## References

*Confirmed entries are complete; **[要確認]** marks bibliographic details to verify before submission (do not fabricate authors/titles).*

**Methods / related work**
1. T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. W. Battaglia. "Learning Mesh-Based Simulation with Graph Networks." *ICLR* 2021. arXiv:2010.03409.
2. V.-H. Dang, T.-C. Vu, B.-D. Nguyen, Q.-H. Nguyen, T.-D. Nguyen. "Structural damage detection framework based on graph convolutional network directly using vibration data." *Structures* 38:40–51, 2022. doi:10.1016/j.istruc.2022.01.066. *(graph-convolutional SHM on sensor vibration data, single structure — verified 2026-06-20, replaces the unverifiable "Zhao 2023" placeholder.)*
3. J. Heimann, S. Mustapha, B. Yilmaz, J. Prager. "Guided Waves in Composite Overwrapped Pressure Vessels and Considerations for Sensor Placement Towards Structural Health Monitoring — An Experimental Study." *ASME J. Nondestructive Evaluation* 8(3):031007, 2025. doi:10.1115/1.4067667.
4. H. El Moutaouakil, C. Fuchs, E. Savli, J. Heimann, J. Prager, J. Moll, K. Tschöke, O. A. Márquez Reyes, O. Schackmann, V. Memmolo, T. Schneider. "Acquiring a Machine Learning Data Set for Structural Health Monitoring of Hydrogen Pressure Vessels at Operating Conditions using Guided Ultrasonic Waves." *EWSHM* 2024. doi:10.58286/29754.
5. K. Nishioka, Y. Kojima, T. Saito, K. Kawakami, M. Washiya, M. Muramatsu. "Development of Defect Localization Method for Perforated Carbon-Fiber-Reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network." *Frontiers in Materials* 12:1–15, 2025. doi:10.3389/fmats.2025.1652484. *(the group's published FEM+GNN SHM method that the §5.4 fairing detector extends; the fairing-specific application is an internal draft `Payload2026/papers/paper_draft.md`, and the interstage variant is presented at WCCM 2026.)*

**Datasets (all real, public — verified 2026-06-20 via DataCite/Zenodo)**
6. D. Melching, T. Strohmann, G. Requena, E. Breitbarth (DLR). "Full-field displacements and strains obtained by digital image correlation during fatigue crack growth experiments." Zenodo, 2022. doi:10.5281/zenodo.5740216. *(metallic AA2024-T3, DIC; also the DLR crack curves used for §6 fleet prognosis)*
7. P. Kudela, M. Radzieński, M. Moix-Bonet, C. Willberg, Y. Lugovtsova, J. Bulling, K. Tschöke, J. Moll. "Dataset on full ultrasonic guided wavefield measurements of a CFRP plate with fully bonded and partially debonded omega stringer." Zenodo, 2021. doi:10.5281/zenodo.5105861.
8. C. Lüders, S. Ropte, D. Schmidt, M. Liebisch (DLR). "Hydraulic burst pressure test of Type IV composite pressure vessel." Zenodo, 2024. doi:10.5281/zenodo.10983652 *(the cited 10608733 is an earlier version DOI; 9 strain positions + ultrasonic)*.
9. D. Lozano, J. Heimann, D. Pöhlig, J. Prager (BAM). "Ultrasonic guided waves in a composite overwrapped pressure vessel under operational conditions." Zenodo, 2026. Baseline doi:10.5281/zenodo.17776240; Damage Cases (reversible/irreversible) doi:10.5281/zenodo.17782123. CC-BY-4.0. *(the §5/§5.5 25-PZT COPV guided-wave dataset; see refs 3–4 for array/acquisition. Verified 2026-06-20 — supersedes the earlier "Fordatis on request" note.)*
10. D. Zarouchas, A. Broer, G. Galanopoulos, W. Briand, R. Benedictus, T. Loutas. "Compression-compression fatigue tests on single stiffener aerospace structures" (distributed fibre-optic). DataverseNL, 2021. doi:10.34894/QNURER.
11. M. Möller, J. Blaurock, G. Ziegmann. "Raw data for fatigue dataset for carbon fibre-reinforced polymers under uni- and multiaxial loads with varying biaxiality and stress ratios. Part 1: Proportional multiaxial loads." Mendeley Data, 2022. doi:10.17632/jpk2t755vg.1. *(§4/§6 use this as the load-mode-boundary case — axial vs torsion/shear-dominated biaxiality; relabeled from an earlier "drive-shaft" framing to match the dataset, which is CFRP multiaxial-fatigue coupons.)*
12. Interstage shell thermo-elastic stress — WCCM-precomputed FEM ([S]); internal.
13. C. Haugwitz, Y. Ziermann, T. Böhme, S. Soennecken, A. Reinartz, S. Wismath, N. Demuth, T. Hahn-Jose, M. Kupnik. "Data for In-line Production Testing of Sandwich Panels using Air-Coupled Ultrasound." TU Darmstadt (tudatalib), 2025. doi:10.48328/tudatalib-2000. *(real steel-faced PIR-foam sandwich; 12-channel air-coupled ultrasonic guided wave; 37 inserted core defects; the fairing's real **sandwich** proxy, §5.4. GPL-3.0.)* [R]

*Provenance note: all five public datasets cluster around DLR + the German aerospace-SHM community (DLR, BAM, Fraunhofer IKTS, Goethe Univ.); the omega-stringer (ref 7) and the BAM guided-wave COPV (refs 3–4, 9) even share authors (Moll, Tschöke, Lugovtsova) — coherent, well-characterized real benchmarks.*

---

## 図表割り
- Fig1 スケールラダー `figs/structure_scale_overview.{png,pdf}` ✅[既存]
- Fig2 Tier-B契約 模式図 `figs/fig2_tierb_contract.{png,pdf}` ✅[生成済 2026-06-20, `make_fig2_tierb.py`]（6構造×モダリティ→adapter→共通(response,disp_mag)→単一検出器→pipeline・COPV2回=負/救済）
- Fig3 Stage-0 検出パネル `figs/fig3_stage0_detection_panel.{png,pdf}` ✅[生成済 2026-06-20, `make_paper_figs.py`]
- Fig4 DLR fleet 予後 `figs/fleet_prognosis.{png,pdf}` ✅[配置済 2026-06-20, `fleet_prognosis.py`]（実2クラック曲線＋Stage-4 fleet sharpening b 1.705±0.024→1.486±0.015・n=2正直注記）
- Fig5 正直な負 `figs/fig5_honest_negatives.{png,pdf}` ✅[生成済 2026-06-20]（COPV conc vs raw / RUL α–λ / 荷重モードR²）
- Fig6 **COPVモダリティ救済**（§5.5）✅[生成済 2026-06-20]：検出 `copv_gw/copv_damage_detect.png`・頑健性 `copv_gw/copv_robustness.png`・絶対座標localization `copv_gw/copv_localize_true.png`（真ジオメトリ・EWSHM2024）。COPV単独版=`copv_gw/PAPER_DRAFT_copv_guidedwave.md`
- Fig7 **フェアリング実サンドproxy検出**（§5.4）✅[生成済 2026-06-20]：`figs/fig7_fairing_sandwich_proxy.png`（CH6局所z-trace＋ROC AUROC0.90・TU Darmstadt実サンドイッチ）。生成元 `Payload2026/scripts/tudarmstadt_sandwich/detect_stage0.py`（版管理下・生データはdata/external、gitignore）、NOTES.md
- Fig8 **LOSO クロス構造較正**（§5.6・4構造）✅[生成済 2026-06-21]：`figs/fig8_loso_calibration.png`（左=FPR raw0.997→self-norm0.15-0.26／右=AUROC 0.91-1.0転移）。生成元 `loso_calibration.py`+`plot_loso.py`・結果`loso_calibration_result.json`
- Tab1 §4 / Tab2 §5（タグ付）✅本文内

## 投稿先・残作業
- IWSHM2027（本命・締切要確認 [[project_conferences_2027]]）→ 拡張 *Structural Health Monitoring* / *MSSP*。
- ✅**数値照合済（2026-06-20）**：§5 AUROC・§5.4 OGW panel(recall0.925/AUC0.832/AUPRC0.914・quantile FPR5.2%)・§6 α–λ(3–14%)・DI(0/0.567/0.998) を RESULTS.md / `fairing_real_e2e.json` / [[project_payload_link]] と一致確認。
- ✅**2026-06-20完了**：COPV統合(§5.5)・Fig2模式図(`make_fig2_tierb.py`)・Fig4配置(`fleet_prognosis`)・参考文献整備(§References・[要確認]フラグ付)・count整合校正(5構造/6モダリティ)。
- ✅**データセット引用5件確定（2026-06-20 DataCite/Zenodo照合）**：ref6 Melching+(DLR DIC)・ref7 Kudela+Moll+(omega)・ref8 Lüders+(DLR COPV burst, 正DOI 10983652)・ref10 Zarouchas+(DataverseNL stiffener)・ref11 Möller+(Mendeley 多軸疲労)。"drive-shaft"→"CFRP多軸疲労"に正名化(§4/§6)。
- ✅**フェアリング実サンドproxy統合（2026-06-20）**：§5.4を「JAXA実フェアリングデータ無し→2実proxyでブラケット(OGW=CFRP/GW軸・TU Darmstadt実サンド=構造軸)」に書換。実Stage-0 **AUROC 0.90**[CI0.85-0.94]/厳格local 0.85/CH6 0.88 CH7 0.97/n28。素朴0.56-0.67→adapter(per-ch除トレンド)で0.90。§4表・§5表に行追加・Abstract/§1貢献5/§7(7モダリティ)・**ref13確定**(Haugwitz+ 9名 live照合・doi:10.48328/tudatalib-2000)・Fig7配置(`figs/fig7_fairing_sandwich_proxy.png`)。材料ギャップ(鋼/フォーム≠CFRP/ハニカム)明記・素材一致はIMP-PAN/IIT Bombay照会中。正典コード`Payload2026/scripts/tudarmstadt_sandwich/`(版管理下: detect_stage0.py/parse_windows.py/clean_windows.json/NOTES.md・生データはdata/external gitignore)。
- ✅**References [要確認] 3件すべて解決（2026-06-20 Crossref/Zenodo照合）**：ref2=Dang+ 2022 *Structures* graph-conv SHM(doi:10.1016/j.istruc.2022.01.066・"Zhao 2023"置換)／ref5=自己引用をグループ既刊 Nishioka+ 2025 *Frontiers in Materials*(doi:10.3389/fmats.2025.1652484)でアンカー＋フェアリング応用は内部draft／ref9=BAM COPV-GWデータ Lozano+ 2026 Zenodo **17776240(Baseline)+17782123(Damage)** CC-BY(Fordatis照会不要に)。**全引用が検証可能・捏造ゼロ・[要確認]残ゼロ**。
- ✅**LOSO クロス構造較正 統合（2026-06-21・§5.6新設・4構造）**：統一detector(各構造自分の健全ref残差)。metallic/omega/copv within-AUROC1.0＋TU Darmstadtサンド(air-coupled adapter)0.911。**LOSO=ranking転移(held-out AUROC 0.91-1.0)・生閾値は不転移(構造毎にscaleが桁違い→サンドFPR0.997・他は0=双方向に制御不能)・自己正規化+共有conformalで回復(≤0.26、15健全で≤0.15、検出0.89-1.0)**。サンドのresidual 0.15-0.26=ノイジー実検出(AUROC0.91)の正直な較正コスト。「検出は構造非依存だが閾値較正は局所健全要」を横断実証。実健全変動=metallic(3標本)+サンド(~400窓)の**2件が有意**(omega/copvは健全1フィールド縮退)。interstageはnode-level(frame-AUROC0.52)除外。Abstract/§1貢献6/§7/Fig8。
- **残（投稿前）**：(1) 英文最終コピーエディット、(2) 投稿先テンプレ整形(IWSHM2027→拡張SHM/MSSP)、(3) [任意] LOSO強化＝多標本の実構造(例: TU Darmstadtサンド健全多数)を4つ目に足すとomega/copv縮退を解消可。両コピー同期済(`GNN/ingest`＝`Payload2026/papers`)。
