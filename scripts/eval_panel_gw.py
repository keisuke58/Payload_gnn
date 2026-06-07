#!/usr/bin/env python3
"""Multi-metric panel evaluation for the Payload GW classifier (WCCM insight applied).

Graph-level healthy/defect. Reports recall(=1-FNR), FPR, precision, F1, AUPRC, AUC, acc
on (a) FEM val (in-dist) and (b) OGW real data (sim-to-real, bootstrapped sensors).

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/eval_panel_gw.py
"""
from __future__ import annotations
import os, sys, glob
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from parse_ogw_wavefield import load_ogw
from build_gw_graph import extract_time_features, build_edge_index
from train_gw import build_graph_level_model
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score, roc_auc_score

CK = os.path.join(HERE, "..", "runs", "gw_sage_20260607_141917", "best_model.pt")
FEM = os.path.join(HERE, "..", "data", "processed_gw_92_comp")
OGWX = os.path.join(HERE, "..", "data", "external", "ogw_stringer", "extracted")
SCEN = [("Intact", 0), ("FirstImpact", 1), ("SecondImpact", 1)]
NSENS, NBOOT, FEAT = 9, 40, "comprehensive"
rng = np.random.default_rng(0)


def load_model():
    sd = torch.load(CK, map_location="cpu")
    m = build_graph_level_model("sage", 24, edge_attr_dim=0, hidden=128, num_classes=2, num_layers=3)
    m.load_state_dict(sd); m.eval(); return m


def fem_val():
    d = torch.load(os.path.join(FEM, "val.pt"), weights_only=False)
    return d if isinstance(d, list) else d.get("data", d)


def ogw_boot():
    gs = []
    for tag, y in SCEN:
        excs = sorted(glob.glob(os.path.join(OGWX, f"OGW_CFRP_Stringer_Wavefield_{tag}", "BURST_100kHz*")))
        if not excs:
            continue
        coords, wf, t = load_ogw(excs[0]); N = coords.shape[0]
        for _ in range(NBOOT):
            sidx = rng.choice(N, NSENS, replace=False)
            sd = {i: wf[:, s] for i, s in enumerate(sidx)}
            feats = np.asarray(extract_time_features(t, sd, FEAT), dtype=np.float32)
            pos = [float(coords[s, 0] * 1000) for s in sidx]
            ei = build_edge_index(NSENS, pos, "full")
            gs.append(Data(x=torch.tensor(feats), edge_index=ei, y=torch.tensor([y], dtype=torch.long)))
        del wf
    return gs


@torch.no_grad()
def run(m, graphs):
    Y, P, PD = [], [], []
    for g in graphs:
        bt = torch.zeros(g.x.size(0), dtype=torch.long)
        prob = torch.softmax(m(g.x, g.edge_index, getattr(g, "edge_attr", None), bt), -1).flatten()
        Y.append(int(g.y)); P.append(int(prob.argmax())); PD.append(float(prob[1]))
    return np.array(Y), np.array(P), np.array(PD)


def panel(name, yt, yp, pd):
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0; fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = float((yt == yp).mean())
    auprc = average_precision_score(yt, pd) if yt.any() and (yt == 0).any() else float("nan")
    auc = roc_auc_score(yt, pd) if yt.any() and (yt == 0).any() else float("nan")
    print(f"[{name:18s}] N={len(yt):3d} acc={acc:.3f} recall(1-FNR)={rec:.3f} FPR={fpr:.3f} "
          f"prec={prec:.3f} F1={f1:.3f} AUPRC={auprc:.3f} AUC={auc:.3f}")


if __name__ == "__main__":
    m = load_model()
    yt, yp, pd = run(m, fem_val()); panel("FEM val (in-dist)", yt, yp, pd)
    g = ogw_boot()
    if g:
        yt, yp, pd = run(m, g); panel("OGW (sim-to-real)", yt, yp, pd)
    else:
        print("OGW extracted 100kHz not found")
