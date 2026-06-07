#!/usr/bin/env python3
"""Robust sim-to-real DA for OGW: sensor bootstrap + multi-excitation + few-shot.

- Bootstrap many 9-sensor subsets per (scenario x excitation) -> many real graphs.
- Standardize features by FEM-train stats; train a SAGE classifier on FEM data.
- Zero-shot: evaluate on all real graphs (per-class).
- Few-shot: hold out one excitation as test, fine-tune on the rest (real labels),
  evaluate on the held-out excitation = cross-excitation generalization.

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/ogw_robust_da.py
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from parse_ogw_wavefield import load_ogw                              # noqa: E402
from build_gw_graph import extract_time_features, build_edge_index   # noqa: E402
from train_gw import build_graph_level_model                          # noqa: E402

BASE = os.path.join(HERE, "..", "data", "external", "ogw_stringer", "extracted")
FEM = os.path.join(HERE, "..", "data", "processed_gw_92_comp")
SCEN = [("Intact", 0), ("FirstImpact", 1), ("SecondImpact", 1)]
NSENS, NBOOT, FEAT = 9, 30, "comprehensive"
rng = np.random.default_rng(0)
torch.manual_seed(0)


def make_graph(coords, wf, t, sidx, label):
    sd = {i: wf[:, s] for i, s in enumerate(sidx)}
    feats = np.asarray(extract_time_features(t, sd, FEAT), dtype=np.float32)
    pos = [float(coords[s, 0] * 1000) for s in sidx]
    ei = build_edge_index(len(sidx), pos, "full")
    return Data(x=torch.tensor(feats), edge_index=ei, y=torch.tensor([label], dtype=torch.long))


def build_real():
    """Return list of (graph, excitation_tag). Bootstrap over available excitations."""
    graphs = []
    for tag, y in SCEN:
        excs = sorted(glob.glob(os.path.join(BASE, f"OGW_CFRP_Stringer_Wavefield_{tag}", "BURST_*")))
        for ex in excs:
            exname = os.path.basename(ex)
            try:
                coords, wf, t = load_ogw(ex)
            except Exception:
                continue
            N = coords.shape[0]
            for _ in range(NBOOT):
                sidx = rng.choice(N, NSENS, replace=False)
                graphs.append((make_graph(coords, wf, t, sidx, y), exname))
            del wf
    return graphs


def load_fem(split):
    d = torch.load(os.path.join(FEM, f"{split}.pt"), weights_only=False)
    return d if isinstance(d, list) else d.get("data", d)


def stats(graphs):
    X = torch.cat([g.x for g in graphs], 0).numpy().astype(np.float64)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-8] = 1.0
    return torch.tensor(mu, dtype=torch.float), torch.tensor(sd, dtype=torch.float)


def norm(graphs, mu, sd):
    out = []
    for g in graphs:
        g2 = g.clone(); g2.x = (g2.x - mu) / sd; out.append(g2)
    return out


def fit(model, graphs, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(graphs, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in dl:
            opt.zero_grad()
            crit(model(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch), b.y).backward()
            opt.step()
    return model


def evaluate(model, graphs):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for g in graphs:
            bt = torch.zeros(g.x.size(0), dtype=torch.long)
            p = int(model(g.x, g.edge_index, None, bt).argmax(-1))
            yt.append(int(g.y)); yp.append(p)
    yt, yp = np.array(yt), np.array(yp)
    acc = (yt == yp).mean()
    # per-class recall
    rec = {c: (yp[yt == c] == c).mean() if (yt == c).any() else float("nan") for c in (0, 1)}
    return acc, rec, yt, yp


if __name__ == "__main__":
    print("building bootstrap real graphs ...")
    real = build_real()
    excs = sorted({e for _, e in real})
    from collections import Counter
    print(f"real graphs: {len(real)} | excitations: {[e.split('_')[1] for e in excs]} | "
          f"labels {dict(Counter(int(g.y) for g, _ in real))}")

    fem_tr, fem_va = load_fem("train"), load_fem("val")
    mu, sd = stats(fem_tr)
    model = build_graph_level_model("sage", 24, edge_attr_dim=0, hidden=128, num_classes=2, num_layers=3)
    model = fit(model, norm(fem_tr, mu, sd), epochs=120, lr=1e-3)
    print(f"FEM val acc: {evaluate(model, norm(fem_va, mu, sd))[0]:.3f}")

    real_n = [(g, e) for g, e in zip(norm([g for g, _ in real], mu, sd), [e for _, e in real])]

    # zero-shot
    acc, rec, _, _ = evaluate(model, [g for g, _ in real_n])
    print(f"\n[zero-shot]  acc={acc:.3f}  recall(healthy)={rec[0]:.3f}  recall(defect)={rec[1]:.3f}")

    # few-shot: hold out 100kHz as test, fine-tune on the rest
    test_key = "100kHz"
    test = [g for g, e in real_n if test_key in e]
    ft = [g for g, e in real_n if test_key not in e]
    if ft and test:
        import copy
        fs = fit(copy.deepcopy(model), ft, epochs=40, lr=3e-4)
        a0, r0, _, _ = evaluate(model, test)
        a1, r1, _, _ = evaluate(fs, test)
        print(f"\n[held-out {test_key}] zero-shot acc={a0:.3f} (H{r0[0]:.2f}/D{r0[1]:.2f})  "
              f"-> few-shot acc={a1:.3f} (H{r1[0]:.2f}/D{r1[1]:.2f})")
    else:
        print("\n(only one excitation available; extract 50/200/300kHz for few-shot)")
