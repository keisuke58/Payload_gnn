#!/usr/bin/env python3
"""Experiment B: damage detection over time (years/seasons), compensation ON vs OFF.

The OGW long-term "damage tag" is a SEVERITY level (0=healthy, >0 damaged; reaches 6 in
2021). Damage accumulates and worsens over years while temperature cycles seasonally.

Experiment A (multi-year supervised): train on several YEARS (so the detector sees multiple
damage states + a full temperature span), then deploy on a held-out FUTURE year across all
its seasons. Question: does environmental compensation (amplitude-norm + temperature-
regression, see TEMPERATURE_ROBUSTNESS.md) keep false alarms down across the test year's
seasons WITHOUT sacrificing recall on its (progressively worse) damage? We report recall (on
damaged, tag>0) and FPR (on healthy, tag==0) per test month, compensation ON vs OFF.

Train : all months of B_TRAIN_YEARS (default 2018,2019) — damage + healthy.
Drift fit (ON): ALL healthy samples across the training years (full seasonal temp span).
Test  : all months of B_TEST_YEARS (default 2020,2021,2022).
(For the brittle single-year framing, set B_TRAIN_YEARS=2018 B_TEST_YEARS=2019,2021.)

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$HOME/miniconda3/lib python3 scripts/longterm_detection_over_time.py
"""
import os, sys, glob, pickle, json
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from longterm_gnn import path_features, build_graphs
from build_gw_graph import build_edge_index
from train_gw import build_graph_level_model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
torch.manual_seed(0); rng = np.random.default_rng(0)
LT = os.path.join(HERE, "..", "data", "external", "ogw_longterm")
ei = build_edge_index(8, list(range(8)), "full")
AMP = [0, 1, 3, 4, 5]
EPOCHS = int(os.environ.get("B_EPOCHS", "40"))
# Experiment A: train on multiple YEARS (sees several damage states + temperatures),
# test on held-out future year(s). Defaults: train 2018+2019 -> test 2020/2021/2022.
TRAIN_YEARS = os.environ.get("B_TRAIN_YEARS", "2018,2019").split(",")
TEST_YEARS = os.environ.get("B_TEST_YEARS", "2020,2021,2022").split(",")


def load(m):
    d = pickle.load(open(f"{LT}/measurements_{m}.pickle", "rb"))
    y = d["damage tag"].astype(int)
    return d["guided wave"], (y > 0).astype(int), np.asarray(d["temperature"], float)


def try_load(m):
    try:
        return load(m)
    except Exception:
        return None


def cache_features(months):
    """Load each month ONCE, compute raw path_features (tiny vs the 1.9GB waveform), free
    the waveform. Returns {month: (raw_feats, y_bin, temp)}; all months ~300MB total."""
    cache = {}
    for m in months:
        r = try_load(m)
        if r is None:
            continue
        gw, y, tp = r
        cache[m] = (path_features(gw), y, tp); del gw, r
    return cache


def ampnorm(f):
    f = f.copy(); g = f[:, :, 0].mean(1, keepdims=True); g[g < 1e-9] = 1e-9
    for k in AMP:
        f[:, :, k] = f[:, :, k] / g
    f[:, :, 2] = f[:, :, 2] / (g ** 2)
    return f


def fit_drift(fh, th, deg=2):
    n, F = fh.shape[1], fh.shape[2]; c = np.zeros((n, F, deg + 1))
    for j in range(n):
        for k in range(F):
            c[j, k] = np.polyfit(th, fh[:, j, k], deg)
    return c, float(th.mean())


def regress(f, t, c, tref):
    o = f.copy()
    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            o[:, j, k] -= (np.polyval(c[j, k], t) - np.polyval(c[j, k], tref)).astype(np.float32)
    return o


@torch.no_grad()
def scores(model, feats, mu, sd):
    g = [Data(x=torch.tensor((feats[i] - mu) / sd, dtype=torch.float), edge_index=ei) for i in range(len(feats))]
    return np.concatenate([torch.softmax(model(b.x, b.edge_index, None, b.batch), 1)[:, 1].numpy()
                           for b in DataLoader(g, batch_size=256)])


def train_detector(Xtr, ytr):
    flat = Xtr.reshape(-1, Xtr.shape[2]); mu, sd = flat.mean(0), flat.std(0); sd[sd < 1e-8] = 1
    model = build_graph_level_model("sage", Xtr.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((ytr == 0).sum()) / max(1, (ytr == 1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(build_graphs(Xtr, ytr, mu, sd, ei), batch_size=128, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
    model.eval()
    return model, mu, sd


def run(train_months, test_months):
    # ---- load every needed month ONCE (raw features cached in RAM) ----
    cache = cache_features(train_months + test_months)
    tm = [m for m in train_months if m in cache]
    Xraw = np.concatenate([cache[m][0] for m in tm])
    ytr = np.concatenate([cache[m][1] for m in tm])
    # ---- drift fit pool: ALL healthy samples across the training years (full temp span) ----
    pf, pt = [], []
    for m in tm:
        f, y, tp = cache[m]; fa = ampnorm(f); h = (y == 0)
        if h.any():
            pf.append(fa[h]); pt.append(tp[h])
    coef, tref = (fit_drift(np.concatenate(pf), np.concatenate(pt)) if pf else (None, None))

    results = {}
    for cond in ("OFF", "ON"):
        Xtr_t = Xraw if cond == "OFF" else ampnorm(Xraw)
        mdl, mu, sd = train_detector(Xtr_t, ytr)
        thr = np.percentile(scores(mdl, Xtr_t[ytr == 0], mu, sd), 95)  # FPR 5% on train-healthy
        rows = []
        for m in test_months:
            if m not in cache:
                continue
            f0, y, tp = cache[m]; f = f0
            if cond == "ON":
                f = ampnorm(f0)
                if coef is not None:
                    f = regress(f, tp, coef, tref)
            s = scores(mdl, f, mu, sd); pred = (s >= thr)
            rows.append({"month": m, "tempC": round(float(tp.mean()), 1),
                         "n_dmg": int((y == 1).sum()), "n_heal": int((y == 0).sum()),
                         "recall": float(pred[y == 1].mean()) if (y == 1).any() else None,
                         "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None})
        results[cond] = {"thr": float(thr), "rows": rows}
    return results


if __name__ == "__main__":
    avail = sorted(os.path.basename(f).replace("measurements_", "").replace(".pickle", "")
                   for f in glob.glob(f"{LT}/measurements_*.pickle"))
    train_months = [m for m in avail if m.split("_")[0] in TRAIN_YEARS]
    test_months = [m for m in avail if m.split("_")[0] in TEST_YEARS]
    print(f"train_years={TRAIN_YEARS} ({len(train_months)}mo)  "
          f"test_years={TEST_YEARS} ({len(test_months)}mo)  epochs={EPOCHS}")
    res = run(train_months, test_months)
    print(f"\n{'month':9s}{'temp':>5s}{'nDmg':>7s}{'nHeal':>7s} | "
          f"{'recOFF':>7s}{'recON':>7s} | {'fprOFF':>7s}{'fprON':>7s}")
    on = {r["month"]: r for r in res["ON"]["rows"]}
    for r in res["OFF"]["rows"]:
        o = on.get(r["month"], {})
        f = lambda v: "  -  " if v is None else f"{v:.3f}"
        print(f"{r['month']:9s}{r['tempC']:5.0f}{r['n_dmg']:7d}{r['n_heal']:7d} | "
              f"{f(r['recall']):>7s}{f(o.get('recall')):>7s} | {f(r['fpr']):>7s}{f(o.get('fpr')):>7s}", flush=True)
    out = os.path.join(HERE, "..", "results", "ogw", "detection_over_time.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nsaved {out}  (thrOFF={res['OFF']['thr']:.3f} thrON={res['ON']['thr']:.3f})")
