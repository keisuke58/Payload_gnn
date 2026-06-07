#!/usr/bin/env python3
"""Temperature-OOD / environmental robustness on OGW long-term.

Train on 2018_03 (~9C, with damage). Test FALSE-ALARM rate (FPR) on healthy data at
temperature extremes: 2018_07 (hot ~30C) and 2018_12 (cold ~0C). If FPR spikes at
extremes, temperature shift mimics damage = the core uncontrolled-condition problem.
Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/longterm_temp_ood.py
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
    d = pickle.load(open(f"{LT}/measurements_2018_{m}.pickle", "rb"))
    return d["guided wave"], d["damage tag"].astype(int), d["temperature"]


@torch.no_grad()
def scores(model, feats, mu, sd):
    out = []
    g = [Data(x=torch.tensor((feats[i]-mu)/sd, dtype=torch.float), edge_index=ei) for i in range(feats.shape[0])]
    for b in DataLoader(g, batch_size=256, shuffle=False):
        out.append(torch.softmax(model(b.x, b.edge_index, None, b.batch), 1)[:, 1].numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    gw, y, tp = load("03")
    feats = path_features(gw)
    idx = rng.permutation(len(y)); ntr = int(0.7*len(y)); tr = idx[:ntr]
    flat = feats[tr].reshape(-1, feats.shape[2]); mu, sd = flat.mean(0), flat.std(0); sd[sd<1e-8]=1
    g_tr = build_graphs(feats[tr], y[tr], mu, sd, ei)
    model = build_graph_level_model("sage", feats.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr]==0).sum())/max(1,(y[tr]==1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(g_tr, batch_size=128, shuffle=True)
    for ep in range(40):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
    model.eval()
    # threshold from 2018_03 healthy (train split) for FPR=5%
    s_h = scores(model, feats[tr][y[tr]==0], mu, sd); thr = np.percentile(s_h, 95)
    print(f"trained on 2018_03 (~9C). calibrated thr (FPR5% on train-healthy)={thr:.3f}")
    print(f"{'month':8s}{'tempC':>7s}{'N':>7s}{'FPR@0.5':>9s}{'FPR@cal':>9s}")
    # in-dist reference (2018_03 held-out healthy)
    te = idx[ntr:]; sh = scores(model, feats[te][y[te]==0], mu, sd)
    print(f"{'03(ref)':8s}{tp.mean():7.0f}{(y[te]==0).sum():7d}{(sh>=0.5).mean():9.3f}{(sh>=thr).mean():9.3f}")
    for m in ("07", "12"):
        gw2, y2, tp2 = load(m)
        f2 = path_features(gw2); s2 = scores(model, f2, mu, sd)  # all healthy
        print(f"{'20'+m:8s}{tp2.mean():7.0f}{len(y2):7d}{(s2>=0.5).mean():9.3f}{(s2>=thr).mean():9.3f}", flush=True)
        del gw2, f2
