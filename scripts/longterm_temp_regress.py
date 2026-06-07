#!/usr/bin/env python3
"""Regression-based temperature compensation for OGW long-term false alarms.

Why the naive fix failed (longterm_temp_compensate): per-month z-score re-standardizes
EVERY month to mean0/var1. That helps a far month (cold) but DISTORTS a month already
near the training marginal (summer 7.7%->54%).

Standard SHM fix (regression/cointegration compensation): model the temperature-induced
drift of each healthy feature, Delta_f(T), and subtract only the drift RELATIVE to the
training reference temperature. At the training temp the correction is ~0 (no distortion);
cold gets a large shift, summer a small one -> consistent across the whole range.

Leakage-free: temperature varies WITHIN every month (2018_03 spans 0-22C), so Delta_f(T)
is fit on a HEALTHY calibration pool; FPR is tested on held-out healthy samples. Damage
labels are never used by the compensator (realistic: baseline scans exist per season).

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$HOME/miniconda3/lib python3 scripts/longterm_temp_regress.py
"""
import os, sys, pickle
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


def load(m):
    d = pickle.load(open(f"{LT}/measurements_{m}.pickle", "rb"))
    return d["guided wave"], d["damage tag"].astype(int), np.asarray(d["temperature"], dtype=float)


@torch.no_grad()
def scores(model, feats, mu, sd):
    g = [Data(x=torch.tensor((feats[i] - mu) / sd, dtype=torch.float), edge_index=ei)
         for i in range(feats.shape[0])]
    out = [torch.softmax(model(b.x, b.edge_index, None, b.batch), 1)[:, 1].numpy()
           for b in DataLoader(g, batch_size=256, shuffle=False)]
    return np.concatenate(out)


def fit_drift(feats_h, temps_h, deg=2):
    """Per (node,feature) polynomial T->feature on healthy samples. Returns coeff array
    (8,F,deg+1) and the reference temperature (pool mean)."""
    n, F = feats_h.shape[1], feats_h.shape[2]
    coef = np.zeros((n, F, deg + 1))
    for j in range(n):
        for k in range(F):
            coef[j, k] = np.polyfit(temps_h, feats_h[:, j, k], deg)
    return coef, float(temps_h.mean())


def compensate(feats, temps, coef, t_ref):
    """x_comp = x - (poly(T) - poly(t_ref)), per sample/node/feature."""
    out = feats.copy()
    n, F = feats.shape[1], feats.shape[2]
    for j in range(n):
        for k in range(F):
            drift = np.polyval(coef[j, k], temps) - np.polyval(coef[j, k], t_ref)
            out[:, j, k] -= drift.astype(np.float32)
    return out


if __name__ == "__main__":
    # --- train detector on 2018_03 (mixed temps, has damage) ---
    gw, y, tp = load("2018_03"); feats = path_features(gw)
    idx = rng.permutation(len(y)); ntr = int(0.7 * len(y)); tr, te = idx[:ntr], idx[ntr:]
    flat = feats[tr].reshape(-1, feats.shape[2]); mu0, sd0 = flat.mean(0), flat.std(0); sd0[sd0 < 1e-8] = 1
    model = build_graph_level_model("sage", feats.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(1, (y[tr] == 1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(build_graphs(feats[tr], y[tr], mu0, sd0, ei), batch_size=128, shuffle=True)
    for ep in range(40):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
    model.eval()
    thr = np.percentile(scores(model, feats[tr][y[tr] == 0], mu0, sd0), 95)
    print(f"trained 2018_03 (mean {tp.mean():.0f}C). thr(FPR5%)={thr:.3f}")

    # --- temperature-drift regression: fit on a healthy calibration pool spanning temps ---
    # realistic: pre-deployment baseline scans collected across seasons (healthy only).
    pool_f, pool_t = [feats[tr][y[tr] == 0]], [tp[tr][y[tr] == 0]]
    test_months = [("2018_03", te), ("2018_07", None), ("2018_12", None), ("2021_01", None)]
    held = {}
    for m, te_idx in test_months[1:]:
        try:
            gw2, y2, tp2 = load(m)
        except FileNotFoundError:
            continue
        f2 = path_features(gw2); ci = rng.permutation(len(f2))
        cal, tst = ci[:len(ci) // 2], ci[len(ci) // 2:]
        pool_f.append(f2[cal]); pool_t.append(tp2[cal])
        held[m] = (f2[tst], tp2[tst], y2[tst]); del gw2, f2
    Fpool = np.concatenate(pool_f, 0); Tpool = np.concatenate(pool_t, 0)
    coef, t_ref = fit_drift(Fpool, Tpool, deg=2)
    print(f"drift fit on {len(Tpool)} healthy samples, T_ref={t_ref:.1f}C, range[{Tpool.min():.0f},{Tpool.max():.0f}]")

    # in-dist reference (held-out 2018_03 healthy)
    sh = scores(model, feats[te][y[te] == 0], mu0, sd0)
    print(f"\n{'month':9s}{'tempC':>6s}{'N':>7s}{'uncomp':>8s}{'zscore':>8s}{'regress':>8s}")
    print(f"{'03(ref)':9s}{tp.mean():6.0f}{(y[te]==0).sum():7d}{(sh>=thr).mean():8.3f}{'-':>8s}{'-':>8s}")
    for m in ("2018_07", "2018_12", "2021_01"):
        if m not in held:
            continue
        f2, t2, y2 = held[m]; hmask = (y2 == 0)
        fh, th = f2[hmask], t2[hmask]
        # uncompensated
        s_un = scores(model, fh, mu0, sd0)
        # naive own-month z-score (the failed method)
        fl = fh.reshape(-1, fh.shape[2]); muT, sdT = fl.mean(0), fl.std(0); sdT[sdT < 1e-8] = 1
        s_z = scores(model, fh, muT, sdT)
        # regression compensation
        s_r = scores(model, compensate(fh, th, coef, t_ref), mu0, sd0)
        print(f"{m:9s}{th.mean():6.0f}{len(fh):7d}{(s_un>=thr).mean():8.3f}"
              f"{(s_z>=thr).mean():8.3f}{(s_r>=thr).mean():8.3f}", flush=True)
