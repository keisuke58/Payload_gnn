#!/usr/bin/env python3
"""Full cross-month temperature compensation evaluation for fairing Track A.

Uses ampnorm + 2nd-order temperature regression (from longterm_robust_compensate.py).
Trains GNN on 2018_03, fits drift model on ALL available months' healthy-calibration half,
then evaluates FPR(before) vs FPR(after) and recall across every available month.

Output: results/ogw/temp_compensate_full.json
"""
import os, sys, pickle, json
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
RES = os.path.join(HERE, "..", "results", "ogw")
ei = build_edge_index(8, list(range(8)), "full")
AMP = [0, 1, 3, 4, 5]


def load(m):
    path = os.path.join(LT, f"measurements_{m}.pickle")
    d = pickle.load(open(path, "rb"))
    return d["guided wave"], d["damage tag"].astype(int), np.asarray(d["temperature"], float)


def ampnorm(feats):
    f = feats.copy()
    g = f[:, :, 0].mean(1, keepdims=True); g[g < 1e-9] = 1e-9
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
    g = [Data(x=torch.tensor((feats[i] - mu) / sd, dtype=torch.float), edge_index=ei)
         for i in range(len(feats))]
    return np.concatenate([torch.softmax(model(b.x, b.edge_index, None, b.batch), 1)[:, 1].numpy()
                           for b in DataLoader(g, batch_size=256)])


if __name__ == "__main__":
    # 1. Train on 2018_03
    gw, y, tp = load("2018_03")
    F0 = ampnorm(path_features(gw))
    idx = rng.permutation(len(y)); ntr = int(0.7 * len(y)); tr = idx[:ntr]
    flat = F0[tr].reshape(-1, F0.shape[2]); mu, sd = flat.mean(0), flat.std(0); sd[sd < 1e-8] = 1
    model = build_graph_level_model("sage", F0.shape[2], edge_attr_dim=0, hidden=64, num_classes=2, num_layers=3)
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(1, (y[tr] == 1).sum())], dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(build_graphs(F0[tr], y[tr], mu, sd, ei), batch_size=128, shuffle=True)
    for ep in range(40):
        model.train()
        for b in dl:
            opt.zero_grad(); crit(model(b.x, b.edge_index, None, b.batch), b.y).backward(); opt.step()
    model.eval()
    thr = np.percentile(scores(model, F0[tr][y[tr] == 0], mu, sd), 95)
    print(f"GNN trained on 2018_03 | thr(FPR5%)={thr:.3f} | Tref={tp.mean():.1f}C")

    # 2. Collect calibration pool (first half of healthy records in each month)
    all_months = sorted([
        f.replace("measurements_", "").replace(".pickle", "")
        for f in os.listdir(LT) if f.startswith("measurements_") and f.endswith(".pickle")
    ])
    cal_feats, cal_temps = [F0[tr][y[tr] == 0]], [tp[tr][y[tr] == 0]]
    held = {}  # month -> (test_feats, test_temps, test_labels)
    bad_months = []
    try:
        bad = open(os.path.join(LT, "KNOWN_BAD_2018_05.txt")).read()
    except Exception:
        bad = ""

    for m in all_months:
        if m == "2018_03":
            continue
        try:
            gw2, y2, t2 = load(m)
        except Exception:
            continue
        if "2018_05" in m:  # known bad
            bad_months.append(m)
            continue
        f2 = ampnorm(path_features(gw2))
        ci = rng.permutation(len(f2)); cal_i, tst_i = ci[:len(ci) // 2], ci[len(ci) // 2:]
        cal_feats.append(f2[cal_i]); cal_temps.append(t2[cal_i])
        held[m] = (f2[tst_i], t2[tst_i], y2[tst_i])
        del gw2, f2

    # 3. Fit drift model on calibration pool
    c, tref = fit_drift(np.concatenate(cal_feats), np.concatenate(cal_temps))
    print(f"Drift model fit on {len(cal_feats)} months, Tref={tref:.1f}C")

    # 4. Evaluate per month
    rows = []
    for m, (f2, t2, y2) in sorted(held.items()):
        h = (y2 == 0); d_ = (y2 == 1)
        # raw (uncompensated)
        s_raw = scores(model, f2, mu, sd)
        fpr_raw = float((s_raw[h] >= thr).mean()) if h.any() else float("nan")
        rec_raw = float((s_raw[d_] >= thr).mean()) if d_.any() else float("nan")
        # compensated
        f2c = regress(f2, t2, c, tref)
        s_comp = scores(model, f2c, mu, sd)
        fpr_comp = float((s_comp[h] >= thr).mean()) if h.any() else float("nan")
        rec_comp = float((s_comp[d_] >= thr).mean()) if d_.any() else float("nan")
        rows.append({"month": m, "tempC": float(t2.mean()), "n_heal": int(h.sum()), "n_dmg": int(d_.sum()),
                     "fpr_raw": fpr_raw, "fpr_comp": fpr_comp,
                     "recall_raw": rec_raw, "recall_comp": rec_comp})
        print(f"  {m}  T={t2.mean():5.1f}C  FPR {fpr_raw:.3f}->{fpr_comp:.3f}  "
              f"rec {rec_raw if rec_raw==rec_raw else 'N/A'}", flush=True)

    # Summary
    fpr_raw_vals  = [r["fpr_raw"]  for r in rows if r["fpr_raw"]  == r["fpr_raw"]]
    fpr_comp_vals = [r["fpr_comp"] for r in rows if r["fpr_comp"] == r["fpr_comp"]]
    rec_raw_vals  = [r["recall_raw"]  for r in rows if r["recall_raw"]  == r["recall_raw"]]
    rec_comp_vals = [r["recall_comp"] for r in rows if r["recall_comp"] == r["recall_comp"]]
    summary = {
        "n_months": len(rows),
        "mean_fpr_raw":  float(np.mean(fpr_raw_vals)),
        "mean_fpr_comp": float(np.mean(fpr_comp_vals)),
        "mean_recall_raw":  float(np.mean(rec_raw_vals)) if rec_raw_vals else None,
        "mean_recall_comp": float(np.mean(rec_comp_vals)) if rec_comp_vals else None,
        "threshold": float(thr),
        "Tref": float(tref),
    }
    print(f"\n=== SUMMARY ===")
    print(f"Mean FPR   raw={summary['mean_fpr_raw']:.3f}  compensated={summary['mean_fpr_comp']:.3f}")
    print(f"Mean Recall raw={summary['mean_recall_raw']}  compensated={summary['mean_recall_comp']}")

    out = {"summary": summary, "rows": rows}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "temp_compensate_full.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/ogw/temp_compensate_full.json")
