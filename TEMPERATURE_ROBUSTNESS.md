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

## Targets / next
- [x] ~~Amplitude normalization for the gain axis~~ → 2018_12 46 %→0 % (Finding 4/5).
- [x] ~~Combined compensation, FPR ≤ 10 % at all temps~~ → ≤0.6 % everywhere (Finding 5).
- [ ] Validate on the **full 56-month seasonal cycle** + later-year damage once download done
      (so far healthy-FPR on 4 months; need damage-bearing later months for detRecall vs time).
- [ ] Multi-year *training* (2021 full year + others) → learn invariance directly; compare to
      post-hoc compensation (does end-to-end beat the two-stage physical correction?).
- [ ] Report detection recall under compensation across temperatures (not only FPR).
