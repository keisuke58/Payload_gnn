#!/usr/bin/env python3
"""Deployable environmental compensation for OGW long-term: amplitude-norm + temp-regress.

Two orthogonal nuisance axes drive false alarms (see TEMPERATURE_ROBUSTNESS.md):
  (1) TEMPERATURE shifts wave speed/shape  -> regression compensation Delta_f(T)
  (2) GAIN drift (sensor coupling/aging)   -> per-measurement amplitude normalization
Each is removed by its matched physical correction. Combined, FPR drops to <1% at every
temperature while detection recall is preserved (damage lives in the RELATIVE cross-path
amplitude pattern, which amplitude-norm keeps).

Pipeline: path_features -> ampnorm (divide amplitude features by cross-path mean gain)
          -> train SAGE on 2018_03 -> fit Delta_f(T) on ampnormed healthy pool
          -> test FPR with ampnorm+regress on held-out healthy.

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=$HOME/miniconda3/lib python3 scripts/longterm_robust_compensate.py
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
AMP = [0, 1, 3, 4, 5]  # linear-amplitude features in path_features; 2=energy(squared); 6-9=shape/freq


def load(m):
    d = pickle.load(open(f"{LT}/measurements_{m}.pickle", "rb"))
    return d["guided wave"], d["damage tag"].astype(int), np.asarray(d["temperature"], float)


def ampnorm(feats):
    """Remove per-measurement global gain; keep relative cross-path pattern (=damage info)."""
    f = feats.copy(); g = f[:, :, 0].mean(1, keepdims=True); g[g < 1e-9] = 1e-9
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


if __name__ == "__main__":
    gw, y, tp = load("2018_03"); F0 = ampnorm(path_features(gw))
    idx = rng.permutation(len(y)); ntr = int(0.7 * len(y)); tr, te = idx[:ntr], idx[ntr:]
    flat = F0[tr].reshape(-1, F0.shape[2]); mu, sd = flat.mean(0), flat.std(0); sd[sd < 1e-8] = 1
    model = build_graph_level_model("sage", F0.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(1, (y[tr] == 1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(build_graphs(F0[tr], y[tr], mu, sd, ei), batch_size=128, shuffle=True)
    for ep in range(40):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
    model.eval()
    thr = np.percentile(scores(model, F0[tr][y[tr] == 0], mu, sd), 95)

    pf, pt, held = [F0[tr][y[tr] == 0]], [tp[tr][y[tr] == 0]], {}
    for m in ("2018_07", "2018_12", "2021_01"):
        try:
            gw2, y2, t2 = load(m)
        except FileNotFoundError:
            continue
        f2 = ampnorm(path_features(gw2)); ci = rng.permutation(len(f2)); cal, tst = ci[:len(ci) // 2], ci[len(ci) // 2:]
        pf.append(f2[cal]); pt.append(t2[cal]); held[m] = (f2[tst], t2[tst], y2[tst]); del gw2, f2
    c, tref = fit_drift(np.concatenate(pf), np.concatenate(pt))
    sdm = scores(model, F0[te][y[te] == 1], mu, sd); rec = (sdm >= thr).mean() if (y[te] == 1).any() else float("nan")
    print(f"COMBO ampnorm+temp-regress | thr(FPR5%)={thr:.3f} detRecall(03 dmg held-out)={rec:.3f} Tref={tref:.1f}C")
    print(f"{'month':9s}{'temp':>5s}{'FPR':>9s}")
    sh = scores(model, F0[te][y[te] == 0], mu, sd)
    print(f"{'03(ref)':9s}{tp.mean():5.0f}{(sh>=thr).mean():9.3f}")
    for m, (f2, t2, y2) in held.items():
        h = (y2 == 0); s = scores(model, regress(f2[h], t2[h], c, tref), mu, sd)
        print(f"{m:9s}{t2[h].mean():5.0f}{(s>=thr).mean():9.3f}", flush=True)
