#!/usr/bin/env python3
"""
Does diffusion-synthesized damage data improve scarce-damage GNN training?
---------------------------------------------------------------------------
Clean protocol (no leakage):
  TRAIN (real):  2018_03 — scarce damage regime (4.5%, 211/4681)
  AUGMENT:       + synthetic damaged waveforms from the conditional DDPM/FM
                 (generator was trained on 2018_03/12, 2019_07, 2022_01/07)
  TEST (held-out): 2020_12 (all healthy, winter) + 2022_02 (all damaged, winter)
                 — these months were NEVER seen by the generator or classifier

Feature consistency: real GW is downsampled 2000→256 and z-scored with the
generator's saved stats, so real and synthetic share one representation.

Run (vancouver):
  python3 scripts/augment_eval.py --synth results/payload_ddpm/synthetic_damaged.npz
"""
import os, sys, glob, pickle, argparse
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from train_gw import build_graph_level_model
from build_gw_graph import build_edge_index
from payload_diffusion import downsample, RESULTS_DIR
from longterm_gnn import path_features
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(0)
rng = np.random.default_rng(0)
LT  = os.path.join(HERE, "..", "data", "external", "ogw_longterm")

TRAIN_MONTH = "2018_03"
# v2 protocol: WITHIN-MONTH mixed test months (healthy+damaged in the same
# month → no year/aging confound), both held out from generator training.
# v1 (2020_12 healthy vs 2022_02 damaged) was confounded by 14 months of
# drift: baseline AUC 0.41 = classifier learned year effects, not damage.
TEST_MONTHS = ["2019_10", "2021_05"]   # dmg 53.0% / 47.8%


def load_month_z(month, mu, sig):
    """Load one month, downsample to 256, z-score with generator stats."""
    f = os.path.join(LT, f"measurements_{month}.pickle")
    d = pickle.load(open(f, "rb"))
    gw  = downsample(d["guided wave"].astype(np.float32))     # (N,8,256)
    y   = (np.asarray(d["damage tag"]) > 0).astype(int)
    gw  = (gw - mu) / sig
    return gw, y


def build_graphs(feats, y, f_mu, f_sd, ei):
    out = []
    for i in range(feats.shape[0]):
        x = torch.tensor((feats[i] - f_mu) / f_sd, dtype=torch.float)
        out.append(Data(x=x, edge_index=ei, y=torch.tensor([int(y[i])], dtype=torch.long)))
    return out


def panel(model, graphs, tag=""):
    model.eval(); Y, P, PD = [], [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=256, shuffle=False):
            o  = model(b.x, b.edge_index, getattr(b, "edge_attr", None), b.batch)
            sm = torch.softmax(o, 1)
            P += o.argmax(1).tolist(); Y += b.y.tolist(); PD += sm[:, 1].tolist()
    Y, P, PD = np.array(Y), np.array(P), np.array(PD)
    tp = int(((P==1)&(Y==1)).sum()); fp = int(((P==1)&(Y==0)).sum())
    fn = int(((P==0)&(Y==1)).sum()); tn = int(((P==0)&(Y==0)).sum())
    rec  = tp/(tp+fn) if tp+fn else 0.0
    fpr  = fp/(fp+tn) if fp+tn else 0.0
    prec = tp/(tp+fp) if tp+fp else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    auc  = roc_auc_score(Y, PD) if Y.any() and (Y==0).any() else float("nan")
    aup  = average_precision_score(Y, PD) if Y.any() else float("nan")
    bal  = 0.5*(rec + (1-fpr))
    res  = dict(acc=(Y==P).mean(), recall=rec, FPR=fpr, prec=prec,
                F1=f1, AUPRC=aup, AUC=auc, balacc=bal)
    print(f"  [{tag}] " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
    return res


def train_classifier(g_tr, in_dim, epochs=60, seed=0):
    torch.manual_seed(seed)
    model = build_graph_level_model("sage", in_dim, edge_attr_dim=0,
                                    hidden=64, num_classes=2, num_layers=3)
    y_tr  = np.array([int(g.y) for g in g_tr])
    w     = torch.tensor([1.0, float((y_tr==0).sum())/max(1,(y_tr==1).sum())],
                         dtype=torch.float)
    crit  = nn.CrossEntropyLoss(weight=w)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl    = DataLoader(g_tr, batch_size=128, shuffle=True)
    for ep in range(epochs):
        model.train()
        for b in dl:
            opt.zero_grad()
            crit(model(b.x, b.edge_index, None, b.batch), b.y).backward()
            opt.step()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth",   default=f"{RESULTS_DIR}/synthetic_damaged.npz")
    p.add_argument("--n_synth",  type=int, default=2000, help="synthetic samples per class to add")
    p.add_argument("--seeds",    type=int, default=3, help="repeat with N seeds")
    p.add_argument("--balanced", action="store_true",
                   help="also add synthetic HEALTHY (prevents 'everything is damaged' collapse)")
    args = p.parse_args()

    stats = np.load(f"{RESULTS_DIR}/norm_stats.npz")
    mu, sig = stats["mu"], stats["sig"]

    # ── real data ────────────────────────────────────────────────────────────
    gw_tr, y_tr = load_month_z(TRAIN_MONTH, mu, sig)
    print(f"train {TRAIN_MONTH}: {len(y_tr)} samples, dmg {y_tr.sum()} ({y_tr.mean():.1%})")

    test_sets = {}
    for m in TEST_MONTHS:
        g, y = load_month_z(m, mu, sig)
        test_sets[m] = (g, y)
        print(f"test  {m}: {len(y)} samples, dmg {y.sum()} ({y.mean():.1%})")

    # ── synthetic data ───────────────────────────────────────────────────────
    sd      = np.load(args.synth)
    gw_syn  = sd["gw"][:args.n_synth].astype(np.float32)   # already z-scored, (n,8,256)
    print(f"synthetic damaged: {len(gw_syn)} samples (guidance={float(sd['guidance'])})")

    gw_syn_h = None
    if args.balanced:
        hp = args.synth.replace("synthetic_damaged", "synthetic_healthy")
        if os.path.exists(hp):
            gw_syn_h = np.load(hp)["gw"][:args.n_synth].astype(np.float32)
            print(f"synthetic healthy: {len(gw_syn_h)} samples (balanced augmentation)")
        else:
            print(f"  [warn] {hp} not found — damaged-only augmentation")

    # ── features (shared representation) ────────────────────────────────────
    ft_tr  = path_features(gw_tr)
    ft_syn = path_features(gw_syn)
    in_dim = ft_tr.shape[2]
    ei     = build_edge_index(8, list(range(8)), "full")

    # feature normalisation from REAL TRAIN only
    flat = ft_tr.reshape(-1, in_dim)
    f_mu, f_sd = flat.mean(0), flat.std(0); f_sd[f_sd < 1e-8] = 1

    g_te = {m: build_graphs(path_features(g), y, f_mu, f_sd, ei)
            for m, (g, y) in test_sets.items()}

    ft_aug = [ft_tr, ft_syn]
    y_aug  = [y_tr, np.ones(len(ft_syn), dtype=int)]
    if gw_syn_h is not None:
        ft_aug.append(path_features(gw_syn_h))
        y_aug.append(np.zeros(len(gw_syn_h), dtype=int))
    ft_aug = np.concatenate(ft_aug); y_aug = np.concatenate(y_aug)

    results = {m: {"baseline": [], "augmented": []} for m in TEST_MONTHS}
    for seed in range(args.seeds):
        g_tr_base = build_graphs(ft_tr, y_tr, f_mu, f_sd, ei)
        m_base    = train_classifier(g_tr_base, in_dim, seed=seed)
        g_tr_aug  = build_graphs(ft_aug, y_aug, f_mu, f_sd, ei)
        m_aug     = train_classifier(g_tr_aug, in_dim, seed=seed)
        for m in TEST_MONTHS:
            results[m]["baseline"].append(panel(m_base, g_te[m], f"seed{seed} base {m}"))
            results[m]["augmented"].append(panel(m_aug, g_te[m], f"seed{seed} aug  {m}"))

    # ── summary ──────────────────────────────────────────────────────────────
    keys = ["AUC", "AUPRC", "recall", "FPR", "F1", "balacc"]
    for m in TEST_MONTHS:
        print(f"\n=== SUMMARY {m} (mean ± std over {args.seeds} seeds) ===")
        print(f"{'':12s}" + "".join(f"{k:>16s}" for k in keys))
        for name, rs in results[m].items():
            row = f"{name:12s}"
            for k in keys:
                v = np.array([r[k] for r in rs])
                row += f"  {v.mean():.3f}±{v.std():.3f}"
            print(row)


if __name__ == "__main__":
    main()
