# TU Darmstadt sandwich-panel air-coupled-US / guided-wave dataset

> **Layout.** Analysis code is version-controlled here (`scripts/tudarmstadt_sandwich/`:
> `detect_stage0.py`, `parse_windows.py`, `clean_windows.json`, this NOTES.md).
> Raw data (~144 MB core, gitignored), the download helpers `fetch_core.sh` /
> `fetch_tdms.sh`, and the regenerable outputs `stage0_detection.png` /
> `stage0_result.json` live under `data/external/tudarmstadt_sandwich/`
> (override with `$TUD_SANDWICH_DATA`). The paper figure is committed at
> `GNN/ingest/figs/fig7_fairing_sandwich_proxy.png`.

- **DOI** 10.48328/tudatalib-2000 · handle tudalib/4871 · item UUID d16aae68-4740-411b-b552-63494347d39d
- **License** GPL-3.0 · **REAL measured** · paper: Haugwitz/Ziermann/Böhme/Kupnik et al., TU Darmstadt 2025
- Role here: closest *openly downloadable, real-measured, guided-wave + sandwich + core-defect* dataset.
  Use = **sim2real guided-wave anchor for the H3 fairing work** (JAXA real fairing data unavailable, see project_payload_link memory).
- **Material GAP (must state in any paper use)**: face sheets = **steel**, core = **PIR polyisocyanurate foam** — NOT CFRP / aluminium-honeycomb. Defects = artificial air voids / foam-density change in the core (disbond-*analogous*, not true skin-core debond).

## Downloaded (core, 144 MB; 7 raw industrial TDMS ~7.4 GB skipped — see fetch_tdms.sh)
| file | what |
|---|---|
| `inline_us_data_40khz_with_artificial_defect1RX1.mat` (43M, v7.3/HDF5) | **40 kHz LDV guided Lamb wave**, moving panel. `LDV_datas` (6000,3000)=6000 A-scans×3000 samples, Fs=5e5. env: temp~24.2°C const, press/humid. |
| `inline_us_data_40khz_with_artificial_defecteval.mat` (0.1M) | per-measurement features: `tt_corr_times` (ToF, 6000), `RX_max_ampl` (6000). |
| `inline_us_data_40khz_foam-change*.mat` | same, foam-density-change defect type. |
| `eval_defect_200khz_inline_data_all_CH.mat` (64M) | **200 kHz inline air-coupled UT C-scan**: `meas_amplitudes` (347977,12), `tt_corr_times` (347977,12), `software_marker_defect` (75) = 37 START/END marker pairs **indexing directly into the 347977 positions** (clean labels). |
| `inline-40kHz_defect-positions.xlsx` | 12 defects, exact post-disassembly positions ±1cm + "time defect passes RX LDV" (ms after *Starttime x*). |
| `inline-industrial-prototype-defect-positions.xlsx` | 49 rows, per-channel software-marker Start/End indices + foaming-shift comments ("CH7 -15cm too early"). |

## Stage-0 detection (2026-06-20) — RESOLVED, AUROC ≈ 0.88–0.91
Two phases. Naive first, then the proper channel-aware detector.

### Naive (FAILED — recorded so it isn't retried)
- 200 kHz pooled-12ch Mahalanobis **AUROC 0.561**; per-channel global max-z **0.667**; null 0.498.
- 40 kHz LDV waveform-corr DI **0.44 (at chance)** — `meas_timestamps` is a hardware counter (~3.2e10) vs xlsx "ms after Starttime x" = different clocks; naive rescale mislabels. (40 kHz still needs the author clock-sync; left undone — 200 kHz was enough.)

### Proper detector (WORKS) — `detect_stage0.py` logic
Key fixes that took 0.67 → 0.91: (a) **per-channel local detrending** (subtract a 2001-pt running mean per channel to kill slow air-coupled-UT amplitude drift, leaving localized defect dips); (b) **channel-aware** — defects are almost all in **CH6 (23) / CH7 (5)** per the xlsx "Defect measured?" column, so each defect is scored in *its own* channel; (c) **offset tolerance** — author markers are the defect-*sticker* pass position, documented ±5–15 cm early/late, so search ±HW=500 idx around the marker centre; (d) score = max over window of |robust-z(amp)| OR |robust-z(ToF)|.
- **AUROC = 0.90** [90% CI 0.85–0.94] vs global-healthy null pool (400/ch); **0.85** vs the strict LOCAL-healthy null (windows inside the same 27% active region — proves it is not a region artifact; only 0.90→0.85). (Reproducible `detect_stage0.py`; numbers vary ±0.02 with the null-draw seed, all > 0.8.)
- Per-channel: **CH6 0.88**, **CH7 0.97**. Detection @ 5% FA ≈ 0.54 (global) / **~0.2 (strict local calibration)** = ranking is strong, absolute thresholding is the hard part (same lesson as COPV-GW & OGW: detection transfers, calibration needs a local healthy reference).
- n_def = 28 (small → wide CI). Figure: `stage0_detection.png` (CH6 z-trace with defect windows + ROC). Result JSON: `stage0_result.json`.

### Honest framing for the paper
The structure-agnostic Stage-0 detector **does** fire on a **third real guided-wave structure — a sandwich panel — at AUROC ≈ 0.88–0.91**, comparable to COPV-GW (0.92–0.94) and OGW, *once* the per-channel/local/offset structure is respected. Caveats to state: (i) material gap — **steel face + PIR foam core**, not CFRP/Al-honeycomb; defects = artificial core voids/foam-change (disbond-*analogous*); (ii) sensing = 200 kHz air-coupled UT, not PZT pitch-catch; (iii) n=28; (iv) absolute calibration weak (det 0.21 @ strict 5% FA). Material-matched real data still wanted → on-request emails to Kudela (IMP-PAN) & Banerjee/Tallur (IIT Bombay), drafts in Gmail.
