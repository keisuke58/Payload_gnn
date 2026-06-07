#!/usr/bin/env python3
"""Sim-to-real domain adaptation (DA) probe via feature standardization.

Baseline (no norm) gave 0/3 on OGW. Here we z-score node features and test:
  - src-norm : standardize OGW by FEM-train stats (consistent transform)
  - tgt-norm : standardize OGW by OGW's own stats (marginal alignment = simple DA)
The classifier is retrained on FEM-train standardized by FEM stats.

Run: OMP_NUM_THREADS=4 LD_LIBRARY_PATH=.../lib python3 scripts/ogw_da_experiment.py
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from build_gw_graph import build_gw_graph            # noqa: E402
from train_gw import build_graph_level_model         # noqa: E402
from torch_geometric.loader import DataLoader        # noqa: E402

DATA = os.path.join(HERE, "..", "data", "processed_gw_92_comp")
OGW = os.path.join(HERE, "..", "data", "external", "ogw_stringer", "csv9")
SCEN = [("Intact", 0), ("FirstImpact", 1), ("SecondImpact", 1)]
torch.manual_seed(0)


def load(split):
    d = torch.load(os.path.join(DATA, f"{split}.pt"), weights_only=False)
    return d if isinstance(d, list) else d.get("data", d)


def feat_stats(graphs):
    X = torch.cat([g.x for g in graphs], 0).numpy().astype(np.float64)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return torch.tensor(mu, dtype=torch.float), torch.tensor(sd, dtype=torch.float)


def apply_norm(graphs, mu, sd):
    out = []
    for g in graphs:
        g2 = g.clone(); g2.x = (g2.x - mu) / sd; out.append(g2)
    return out


def ogw_graphs():
    gs = []
    for tag, y in SCEN:
        c = glob.glob(os.path.join(OGW, f"OGW_{tag}_*.csv"))[0]
        gs.append((tag, y, build_gw_graph(c, label=y, feature_set="comprehensive")))
    return gs


def train(train_g, val_g):
    model = build_graph_level_model("sage", 24, edge_attr_dim=0, hidden=128,
                                    num_classes=2, num_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(train_g, batch_size=32, shuffle=True)
    best, best_state = -1, None
    for ep in range(120):
        model.train()
        for b in tl:
            opt.zero_grad()
            out = model(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch)
            crit(out, b.y).backward(); opt.step()
        # val
        model.eval(); correct = 0; tot = 0
        with torch.no_grad():
            for g in val_g:
                bt = torch.zeros(g.x.size(0), dtype=torch.long)
                p = model(g.x, g.edge_index, getattr(g, "edge_attr", None), bt).argmax(-1)
                correct += int(p.item() == int(g.y)); tot += 1
        acc = correct / tot
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  trained: best val acc {best:.3f}")
    return model


def eval_ogw(model, gs, mu, sd, tag):
    model.eval(); n = 0
    print(f"  [{tag}]")
    with torch.no_grad():
        for name, y, g in gs:
            g2 = g.clone(); g2.x = (g2.x - mu) / sd
            bt = torch.zeros(g2.x.size(0), dtype=torch.long)
            prob = torch.softmax(model(g2.x, g2.edge_index, None, bt), -1).flatten()
            pred = int(prob.argmax()); n += int(pred == y)
            print(f"    {name:13s} true={y} pred={pred} p(defect)={prob[1]:.3f} "
                  f"{'OK' if pred==y else 'X'}")
    print(f"  => OGW acc ({tag}): {n}/{len(gs)}")
    return n


if __name__ == "__main__":
    tr, va = load("train"), load("val")
    src_mu, src_sd = feat_stats(tr)
    model = train(apply_norm(tr, src_mu, src_sd), apply_norm(va, src_mu, src_sd))
    gs = ogw_graphs()
    tgt_mu, tgt_sd = feat_stats([g for _, _, g in gs])
    print("=== OGW sim-to-real with feature standardization ===")
    eval_ogw(model, gs, src_mu, src_sd, "src-norm (FEM stats)")
    eval_ogw(model, gs, tgt_mu, tgt_sd, "tgt-norm (OGW stats = simple DA)")
