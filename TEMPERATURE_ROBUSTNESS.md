# Temperature / environmental robustness — OGW long-term

Damage detectors trained at one operating condition raise **false alarms** when the
environment shifts. On the OGW long-term benchmark (omega-stringer guided waves,
2018_03–2022_10, monthly pickles) we quantify the problem and a compensation that works.

Why it matters in real rocket SHM: cryogenic tanks, aerodynamic heating, day/night and
seasonal thermal cycling, and reusable-vehicle re-flights all move the operating point.
Temperature-induced wave changes mimic damage → false alarms are operationally unacceptable.

## Setup
- Detector: SAGE graph-level GNN (8 sensor-path nodes, 10 features/node from the 2000-sample
  waveform), trained on **2018_03** (mixed temps, mean ~9 °C, with damage).
  Scripts: [longterm_gnn.py](scripts/longterm_gnn.py), [longterm_temp_ood.py](scripts/longterm_temp_ood.py).
- Threshold calibrated to **FPR 5 %** on training-healthy. Test = **false-alarm rate (FPR)**
  on held-out healthy data at temperature extremes.

## Finding 1 — temperature shift causes false alarms
Trained at 9 °C, tested healthy:

| month | temp | FPR @ calibrated thr |
|---|---|---|
| 2018_03 (ref) | 9 °C | 0.050 |
| 2018_07 | 30 °C | 0.074 |
| 2018_12 | 0 °C | **0.464** |

Cold drives FPR to 46 % — temperature shift alone produces a damage-like signature.

## Finding 2 — naive per-month z-score is UNSAFE
[longterm_temp_compensate.py](scripts/longterm_temp_compensate.py): re-standardize each month
by its own healthy baseline (mean0/var1).

| month | temp | uncomp | naive z-score |
|---|---|---|---|
| 2018_12 | 0 °C | 0.464 | **0.085** (fixed) |
| 2018_07 | 30 °C | 0.074 | **0.538** (broken!) |

It only helps a month FAR from the training marginal (cold). A month already near it
(summer) gets DISTORTED away → 7 %→54 %. Not deployable.

## Finding 3 — regression-based compensation is safe and consistent ✓
[longterm_temp_regress.py](scripts/longterm_temp_regress.py): model each healthy feature's
temperature drift Δ_f(T) (per-node/feature quadratic fit on a healthy calibration pool that
spans temperatures — leakage-free since temperature varies *within* every month, e.g.
2018_03 spans 0–22 °C), then subtract only the drift **relative to the training reference
temperature**. At the training temp the correction is ~0 (no distortion); cold gets a large
shift, summer a small one. Standard SHM regression/cointegration compensation.

| month | temp | uncomp | naive z-score | **regression** |
|---|---|---|---|---|
| 2018_03 (ref) | 9 °C | 0.050 | – | – |
| 2018_07 | 30 °C | 0.074 | 0.538 💥 | **0.033** ✓ |
| 2018_12 | 0 °C | 0.464 | 0.085 | **0.324** |
| 2021_01 | 1 °C | 0.074 | 0.013 | 0.044 |

- **Fixes the naive method's failure**: summer 53.8 % → 3.3 % (below even the uncompensated 7.4 %).
- **Never harms any month** (consistent — the key property for deployment).
- Cold 2018_12: 46 % → 32 % (halved, not eliminated — see Finding 4).

## Finding 4 — cold residual is NOT pure temperature (a 2nd domain axis)
At matched temperature, the two cold months behave completely differently:

| month | temp | uncomp FPR | mean waveform RMS |
|---|---|---|---|
| 2018_12 | 0 °C | 0.464 | **0.0010** |
| 2021_01 | 1 °C | 0.074 | **0.0024** |

Same temperature, but 2018_12's signal amplitude is **~2.4× lower** → an **amplitude/gain
drift** (sensor coupling degradation or measurement-gain difference), *not* temperature.
Temperature regression cannot remove it. This is a distinct domain-shift axis.
→ handled by **amplitude normalization**: divide each measurement's amplitude features by
the cross-path mean gain (energy by its square). The *relative* cross-path pattern — which
paths are attenuated, i.e. the damage signature — is preserved, the global gain removed.
Alone it fixes 2018_12 (46 %→**0.0 %**) and keeps detection (detRecall 0.54), but slightly
raises summer (7 %→12 %) since it doesn't touch temperature.

## Finding 5 — combined compensation = deployable (FINAL) ✓✓✓
[longterm_robust_compensate.py](scripts/longterm_robust_compensate.py): amplitude-norm
(kills gain axis) **+** temperature-regression (kills temperature axis). The two nuisance
axes are orthogonal, each removed by its matched physical correction.

| month | temp | uncomp | naive z | temp-regress | ampnorm | **combined** |
|---|---|---|---|---|---|---|
| 2018_03 (ref) | 9 °C | 0.050 | – | – | 0.046 | 0.046 |
| 2018_07 | 30 °C | 0.074 | 0.538 | 0.033 | 0.125 | **0.006** |
| 2018_12 | 0 °C | 0.464 | 0.085 | 0.324 | 0.000 | **0.005** |
| 2021_01 | 1 °C | 0.074 | 0.013 | 0.044 | 0.085 | **0.000** |

**All temperatures FPR ≤ 0.6 %**, detection recall preserved (0.54). Target (≤10 %) smashed.
Headline: *two orthogonal environmental axes (thermal wave-speed shift + sensor-gain drift),
each removed by its matched physical correction, combine to near-zero false alarms without
sacrificing damage sensitivity.*

## Finding 6 — detection over time (full 4.5-yr cycle): amplitude-norm wins, temp-regress is double-edged
Full 56-month run (2018_05 excluded: corrupt source file). Train a SAGE detector on
2018+2019, deploy across 2020–2022 (damage worsens to tag 6, temperature cycles each year).
Per-month recall (on damaged, tag>0) and FPR (on healthy), three compensation conditions —
OFF (raw), AMP (amplitude-norm only), FULL (amplitude-norm + temperature-regression).
[longterm_detection_over_time.py](scripts/longterm_detection_over_time.py),
figure [results/ogw/fig_detection_over_time.png](results/ogw/fig_detection_over_time.png).

Recall (damaged months), representative:

| month | temp | recOFF | recAMP | recFULL |
|---|---|---|---|---|
| 2022_08 | 26 °C | 0.30 | 0.73 | **0.93** |
| 2021_06 | 27 °C | 0.76 | 0.81 | **0.93** |
| 2021_11 | 2 °C | 0.97 | **0.95** | **0.06** |
| 2022_01 | −2 °C | 0.51 | **0.46** | **0.01** |

- **AMP (amplitude-norm only) is the all-round winner**: raises warm-month recall and *keeps*
  cold-month recall (no catastrophic loss) → robust across the whole temperature range.
- **FULL (temperature regression) is double-edged**: best warm-month recall + lowest cold
  *healthy*-month FPR (e.g. 2020_12 0.67→0.02, 2021_01 0.59→0.07), but **destroys cold-month
  damage recall** (2021_11 0.97→0.06) — it cannot tell a temperature-induced shift from a
  damage-induced one and over-corrects damaged cold samples back toward healthy.
- **Practical rule**: use amplitude-norm always; add temperature-regression only for
  healthy-state monitoring / warm-season detection, not for cold-season damage detection.

One-class novelty (B', [longterm_novelty_oneclass.py](scripts/longterm_novelty_oneclass.py),
baseline = first deployment year 2020's healthy scans across its full temperature cycle):
the broad baseline fixes the single-month degenerate failure (FPR ≈1.0 → 0.0–0.5), but recall
stays low (0.06–0.56). **Supervised detection (Finding 6) clearly beats one-class here.**

## Targets / next
- [x] ~~Amplitude normalization for the gain axis~~ → 2018_12 46 %→0 % (Finding 4/5).
- [x] ~~Combined compensation, FPR ≤ 10 % at all temps~~ → ≤0.6 % everywhere (Finding 5).
- [x] ~~Detection recall over the full 56-month cycle, compensation ON/OFF~~ → Finding 6
      (amplitude-norm robust; temperature-regression double-edged; one-class weak).
- [ ] Damage-aware compensation: split temperature-regression by a damage gate so it stops
      suppressing cold-month damage (the FULL failure mode).
- [ ] Multi-year *training* with temperature as a feature → learn invariance directly;
      compare to the two-stage post-hoc physical correction.
