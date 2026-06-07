#!/usr/bin/env python3
"""OGW long-term integration: 8-path guided waves -> graph -> GNN damage detection.

Each measurement = a graph of 8 sensor-path nodes (features from the 2000-sample
waveform), graph-level label = damage tag (0/1). Reports the multi-metric panel.
Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/longterm_gnn.py
"""
import os, sys, glob, pickle
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from train_gw import build_graph_level_model
from build_gw_graph import build_edge_index
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
torch.manual_seed(0); rng = np.random.default_rng(0)
LT = os.path.join(HERE, "..", "data", "external", "ogw_longterm")


def path_features(gw):  # gw: (N,8,2000) -> (N,8,F)
    x = gw.astype(np.float64)
    rms = np.sqrt((x**2).mean(2)); mx = np.abs(x).max(2); en = (x**2).sum(2)
    std = x.std(2); ptp = x.max(2) - x.min(2); mab = np.abs(x).mean(2)
    amax = x.argmax(2) / x.shape[2]
    F = np.fft.rfft(x, axis=2); P = np.abs(F)
    domf = P.argmax(2) / P.shape[2]
    cent = (np.arange(P.shape[2]) * P).sum(2) / (P.sum(2) + 1e-9) / P.shape[2]
    hi = P[:, :, P.shape[2]//2:].sum(2) / (P.sum(2) + 1e-9)
    feats = np.stack([rms, mx, en, std, ptp, mab, amax, domf, cent, hi], axis=2)
    return feats.astype(np.float32)


def load_month(f):
    d = pickle.load(open(f, "rb"))
    return d["guided wave"], d["damage tag"].astype(int)


def build_graphs(feats, y, mu, sd, ei):
    out = []
    for i in range(feats.shape[0]):
        x = torch.tensor((feats[i] - mu) / sd, dtype=torch.float)
        out.append(Data(x=x, edge_index=ei, y=torch.tensor([int(y[i])], dtype=torch.long)))
    return out


def evaluate(model, graphs):
    model.eval(); Y, P, PD = [], [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=128, shuffle=False):
            o = model(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch)
            sm = torch.softmax(o, 1)
            P += o.argmax(1).tolist(); Y += b.y.tolist(); PD += sm[:, 1].tolist()
    Y, P, PD = np.array(Y), np.array(P), np.array(PD)
    tp=int(((P==1)&(Y==1)).sum()); fp=int(((P==1)&(Y==0)).sum()); fn=int(((P==0)&(Y==1)).sum()); tn=int(((P==0)&(Y==0)).sum())
    rec=tp/(tp+fn) if tp+fn else 0; fpr=fp/(fp+tn) if fp+tn else 0; prec=tp/(tp+fp) if tp+fp else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    auc=roc_auc_score(Y,PD) if Y.any() and (Y==0).any() else float("nan")
    aup=average_precision_score(Y,PD) if Y.any() else float("nan")
    print(f"  acc={(Y==P).mean():.3f} recall(1-FNR)={rec:.3f} FPR={fpr:.3f} prec={prec:.3f} "
          f"F1={f1:.3f} AUPRC={aup:.3f} AUC={auc:.3f}")


if __name__ == "__main__":
    files = sorted(glob.glob(f"{LT}/measurements_2018_03.pickle"))
    gw, y = load_month(files[0])
    print(f"loaded {os.path.basename(files[0])}: gw={gw.shape} damage={int(y.sum())}/{len(y)}")
    feats = path_features(gw)               # (N,8,10)
    # stratified 70/30 split
    idx = rng.permutation(len(y)); n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    flat = feats[tr].reshape(-1, feats.shape[2])
    mu, sd = flat.mean(0), flat.std(0); sd[sd < 1e-8] = 1
    ei = build_edge_index(8, list(range(8)), "full")
    g_tr = build_graphs(feats[tr], y[tr], mu, sd, ei)
    g_te = build_graphs(feats[te], y[te], mu, sd, ei)
    print(f"train {len(g_tr)} (dmg {int(y[tr].sum())}) | test {len(g_te)} (dmg {int(y[te].sum())})")
    model = build_graph_level_model("sage", feats.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr]==0).sum())/max(1,(y[tr]==1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(g_tr, batch_size=128, shuffle=True)
    for ep in range(60):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
        if (ep+1) % 20 == 0:
            print(f"[ep{ep+1}] train:", end=""); evaluate(model, g_tr)
    print("=== TEST (held-out, multi-metric panel) ==="); evaluate(model, g_te)
