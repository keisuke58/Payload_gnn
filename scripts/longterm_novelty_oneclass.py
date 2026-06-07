#!/usr/bin/env python3
"""Experiment B': one-class (healthy-only) novelty detection over time, compensation ON/OFF.

The textbook SHM framing: never train on damage (its distribution shifts over years — see the
recall collapse in the supervised variant). Instead characterise the HEALTHY distribution and
flag deviations as damage. Environmental compensation (amplitude-norm + temperature-
regression, TEMPERATURE_ROBUSTNESS.md) is then the central lever: it removes the environmental
deviation so that only the DAMAGE deviation remains -> low false alarms, recall preserved,
and no damage-distribution-shift confound.

Method: per-graph feature vector (8 paths x 10 feats = 80-D). Standardise by healthy
mean/std, fit a shrinkage covariance on healthy -> Mahalanobis distance = novelty score.
Threshold = 95th percentile of training-healthy distances (FPR 5%). Compare OFF (raw) vs
ON (ampnorm + temp-regress fit on healthy across the training years).

Train healthy : tag==0 samples from B_TRAIN_YEARS (default 2018,2019).
Test          : all months of B_TEST_YEARS (default 2020,2021,2022); recall on tag>0, FPR on tag==0.

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$HOME/miniconda3/lib python3 scripts/longterm_novelty_oneclass.py
"""
import os, sys, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from longterm_detection_over_time import (LT, cache_features, ampnorm, fit_drift, regress,
                                          TRAIN_YEARS, TEST_YEARS)
rng = np.random.default_rng(0)
SHRINK = 1e-2


def flat(f):  # (N,8,10) -> (N,80)
    return f.reshape(f.shape[0], -1)


def fit_oneclass(Zh):
    mu = Zh.mean(0); sd = Zh.std(0); sd[sd < 1e-8] = 1
    Z = (Zh - mu) / sd
    C = np.cov(Z.T) + SHRINK * np.eye(Z.shape[1])
    Cinv = np.linalg.inv(C)
    return mu, sd, Cinv


def maha(X, mu, sd, Cinv):
    Z = (X - mu) / sd
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, Cinv, Z))


def transform(f, tp, cond, coef, tref):
    if cond == "OFF":
        return f
    g = ampnorm(f)
    return regress(g, tp, coef, tref) if coef is not None else g


def run(test_months):
    """Realistic SHM deployment: characterise the pristine structure from the FIRST deployment
    YEAR's healthy scans (spanning its full seasonal temperature cycle), then detect deviations
    in the LATER years. A full-temperature baseline avoids the degenerate 'everything is novel'
    failure of a single-month baseline; within-deployment seasonal temperature is handled by
    compensation. FPR on later-year healthy is fully out-of-sample (no leakage)."""
    cache = cache_features(test_months)
    months = [m for m in test_months if m in cache]
    base_year = min(m.split("_")[0] for m in months)
    baseline_months = [m for m in months if m.split("_")[0] == base_year]
    detect_months = [m for m in months if m.split("_")[0] != base_year]
    print(f"  baseline_year={base_year} ({len(baseline_months)}mo, healthy samples) "
          f"detect={len(detect_months)}mo (later years)", flush=True)
    Hf, Ht = [], []
    for m in baseline_months:
        f, y, tp = cache[m]; h = (y == 0)
        if h.any():
            Hf.append(f[h]); Ht.append(tp[h])
    Hraw = np.concatenate(Hf); Htp = np.concatenate(Ht)
    coef, tref = fit_drift(ampnorm(Hraw), Htp)  # drift on ampnormed baseline-year healthy

    results = {}
    for cond in ("OFF", "ON"):
        Zh = flat(transform(Hraw, Htp, cond, coef, tref))
        idx = rng.permutation(len(Zh)); half = len(idx) // 2
        mu, sd, Cinv = fit_oneclass(Zh[idx[:half]])
        thr = np.percentile(maha(Zh[idx[half:]], mu, sd, Cinv), 95)  # FPR 5% on baseline-healthy
        rows = []
        for m in detect_months:
            if m not in cache:
                continue
            f0, y, tp = cache[m]
            d = maha(flat(transform(f0, tp, cond, coef, tref)), mu, sd, Cinv); pred = (d >= thr)
            rows.append({"month": m, "tempC": round(float(tp.mean()), 1),
                         "n_dmg": int((y == 1).sum()), "n_heal": int((y == 0).sum()),
                         "recall": float(pred[y == 1].mean()) if (y == 1).any() else None,
                         "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None})
        results[cond] = {"thr": float(thr), "rows": rows}
    return results


if __name__ == "__main__":
    avail = sorted(os.path.basename(f).replace("measurements_", "").replace(".pickle", "")
                   for f in glob.glob(f"{LT}/measurements_*.pickle"))
    test_months = [m for m in avail if m.split("_")[0] in TEST_YEARS]
    print(f"[one-class] test_years={TEST_YEARS} ({len(test_months)}mo)")
    res = run(test_months)
    print(f"\n{'month':9s}{'temp':>5s}{'nDmg':>7s}{'nHeal':>7s} | "
          f"{'recOFF':>7s}{'recON':>7s} | {'fprOFF':>7s}{'fprON':>7s}")
    on = {r["month"]: r for r in res["ON"]["rows"]}
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    for r in res["OFF"]["rows"]:
        o = on.get(r["month"], {})
        print(f"{r['month']:9s}{r['tempC']:5.0f}{r['n_dmg']:7d}{r['n_heal']:7d} | "
              f"{fmt(r['recall']):>7s}{fmt(o.get('recall')):>7s} | {fmt(r['fpr']):>7s}{fmt(o.get('fpr')):>7s}",
              flush=True)
    out = os.path.join(HERE, "..", "results", "ogw", "novelty_oneclass.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nsaved {out}  (thrOFF={res['OFF']['thr']:.3f} thrON={res['ON']['thr']:.3f})")
