#!/usr/bin/env python3
"""Temperature compensation for OGW long-term false alarms.

Problem (longterm_temp_ood): model trained at 9C false-alarms at cold (FPR 47% @ 0C).
Fix (temperature-matched baseline, OBS-like): standardize each test month's features by
THAT month's own healthy-baseline statistics, aligning the marginal to training. Report
FPR before vs after compensation. (Realistic SHM: a pristine baseline scan exists per
operating condition.)
Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/longterm_temp_compensate.py
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
    g = [Data(x=torch.tensor((feats[i]-mu)/sd, dtype=torch.float), edge_index=ei) for i in range(feats.shape[0])]
    out = [torch.softmax(model(b.x, b.edge_index, None, b.batch), 1)[:, 1].numpy()
           for b in DataLoader(g, batch_size=256, shuffle=False)]
    return np.concatenate(out)


if __name__ == "__main__":
    gw, y, tp = load("03"); feats = path_features(gw)
    idx = rng.permutation(len(y)); ntr = int(0.7*len(y)); tr = idx[:ntr]
    flat = feats[tr].reshape(-1, feats.shape[2]); mu0, sd0 = flat.mean(0), flat.std(0); sd0[sd0<1e-8]=1
    model = build_graph_level_model("sage", feats.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr]==0).sum())/max(1,(y[tr]==1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(build_graphs(feats[tr], y[tr], mu0, sd0, ei), batch_size=128, shuffle=True)
    for ep in range(40):
        model.train()
        for b in dl: opt.zero_grad(); crit(model(b.x,b.edge_index,None,b.batch), b.y).backward(); opt.step()
    model.eval()
    thr = np.percentile(scores(model, feats[tr][y[tr]==0], mu0, sd0), 95)
    print(f"trained 9C, thr(FPR5%)={thr:.3f}")
    print(f"{'month':7s}{'tempC':>6s}{'FPR_uncomp':>11s}{'FPR_comp':>10s}")
    for m in ("07", "12"):
        gw2, y2, tp2 = load(m); f2 = path_features(gw2)
        # uncompensated: train(9C) stats
        s_un = scores(model, f2, mu0, sd0)
        # compensated: this month's OWN healthy baseline stats (calib split)
        ci = rng.permutation(len(f2)); cal = ci[:len(ci)//2]; tst = ci[len(ci)//2:]
        fl = f2[cal].reshape(-1, f2.shape[2]); muT, sdT = fl.mean(0), fl.std(0); sdT[sdT<1e-8]=1
        s_co = scores(model, f2[tst], muT, sdT)
        print(f"{'20'+m:7s}{tp2.mean():6.0f}{(s_un>=thr).mean():11.3f}{(s_co>=thr).mean():10.3f}", flush=True)
        del gw2, f2
