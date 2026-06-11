#!/usr/bin/env python3
"""Joint posterior p(defect TYPE, LOCATION | guided-wave measurement) — Stage-2 head.

Heads
-----
1. TYPE  p(tag | x_ogw):   OGW long-term damage tags (multiclass over the 14
   damage-state IDs 0..13 found in the monthly pickles). Real measured data,
   richest type set available. CAVEAT: tags were introduced sequentially over
   4.5 years, so tag identity is confounded with season/temperature/ageing —
   a random split CANNOT separate "damage signature" from "time signature".
   We report (a) per-month test accuracy and (b) a temperature-only confound
   probe to quantify this honestly (see INSIGHTS_FROM_WCCM_OOD.md, item 1/2).
2. BINARY p(defect | x_fem): FEM fairing GW dataset processed_gw_92_comp
   (9 sensors x 24 comprehensive features, train 172 / val 42).
3. LOCATION p(loc | x_fem): processed_gw_5loc, 5 classes
   (0=healthy, quadrants of z x theta: 1=z<1500,th<15 2=z<1500,th>=15
    3=z>=1500,th<15 4=z>=1500,th>=15; verified against doe_gw_fairing.json).
4. CONTINUOUS (z, theta) FMPE: conditional flow-matching posterior
   q(z, theta | PCA(x_fem)) on the defect samples joined to
   doe_gw_fairing.json continuous positions (same recipe as the companion
   fmpe_defect.py: Gaussian-path conditional FM, Euler ODE sampling,
   90%-interval coverage as the calibration metric).

All classifier heads are calibrated by temperature scaling on a held-out
calibration split; we report ECE before/after plus the multi-metric panel
(accuracy / macro-F1 / per-class recall / detection recall / FPR / AUPRC) —
never accuracy alone.

Joint API (Stage-2 output for the Payload side):
    DefectCharacterizer.characterize(x_fem, x_ogw=None) ->
        {"p_defect", "type_probs", "location_probs", "top1", "joint", ...}
Type and location heads live in different measurement domains (real OGW plate
8 paths x 2000 samples vs FEM fairing 9 sensors x 24 features), so the joint
is assembled under conditional independence given the two measurements:
    p(type, loc | x) ~= p(type | x_ogw) * p(loc | x_fem).

Usage:
    python3 scripts/defect_type_posterior.py scan          # cache OGW features (heavy IO, ~56 x 2GB pickles)
    python3 scripts/defect_type_posterior.py train         # train + calibrate all heads, figures + metrics
    python3 scripts/defect_type_posterior.py demo          # characterize() on validation samples
Run via scripts/qsub_defect_type_posterior.sh on frontale (Torque).
"""
import argparse
import collections
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from longterm_gnn import path_features  # (N,8,2000) -> (N,8,10)

OUT = os.path.join(ROOT, "results", "defect_type_posterior")
LT = os.path.join(ROOT, "data", "external", "ogw_longterm")
CACHE = os.path.join(OUT, "ogw_type_cache.npz")
CKPT = os.path.join(OUT, "heads.pt")
METRICS = os.path.join(OUT, "metrics.json")
SEED = 42
DEVICE = "cpu"  # all heads are small; frontale nodes are CPU
LOC_NAMES = {0: "healthy", 1: "z<1500,th<15", 2: "z<1500,th>=15",
             3: "z>=1500,th<15", 4: "z>=1500,th>=15"}
KNOWN_BAD = {"2018_05"}  # figshare source corrupt, see KNOWN_BAD_2018_05.txt
PER_MONTH_TAG_CAP = 800  # subsample cap per (month, tag) — pickles are ~2GB each


# ───────────────────────── stage 1: OGW scan/cache ──────────────────────────
def scan_ogw():
    """Extract compact per-path features + damage tag + temperature from every
    monthly pickle, subsampled to PER_MONTH_TAG_CAP per (month, tag)."""
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(LT, "measurements_*.pickle")))
    feats, tags, temps, month_idx, months = [], [], [], [], []
    hist = {}
    for f in files:
        month = re.search(r"measurements_(\d{4}_\d{2})\.pickle", f).group(1)
        if month in KNOWN_BAD:
            print(f"[scan] {month}: skipped (known bad)", flush=True)
            continue
        try:
            d = pickle.load(open(f, "rb"))
            gw = d["guided wave"]
            tag = d["damage tag"].astype(int)
            tp = d["temperature"].astype(np.float32)
        except Exception as e:  # corrupt / partial file
            print(f"[scan] {month}: FAILED to load ({e})", flush=True)
            continue
        keep = []
        for t in np.unique(tag):
            idx = np.where(tag == t)[0]
            if len(idx) > PER_MONTH_TAG_CAP:
                idx = rng.choice(idx, PER_MONTH_TAG_CAP, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        fe = path_features(gw[keep])
        feats.append(fe)
        tags.append(tag[keep])
        temps.append(tp[keep])
        month_idx.append(np.full(len(keep), len(months), dtype=np.int32))
        hist[month] = {int(t): int(c) for t, c in
                       zip(*np.unique(tag, return_counts=True))}
        months.append(month)
        print(f"[scan] {month}: N={len(tag)} kept={len(keep)} "
              f"tags={hist[month]}", flush=True)
        del d, gw
    feats = np.concatenate(feats)
    np.savez_compressed(
        CACHE, feats=feats, tags=np.concatenate(tags),
        temps=np.concatenate(temps), month_idx=np.concatenate(month_idx),
        months=np.array(months))
    json.dump(hist, open(os.path.join(OUT, "ogw_tag_histogram.json"), "w"),
              indent=1, sort_keys=True)
    print(f"[scan] cached {feats.shape} -> {CACHE}")


# ─────────────────────────── shared model pieces ────────────────────────────
class MLPHead(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, n_classes, epochs=300, lr=1e-3, wd=1e-4, hidden=128,
              class_weight=True, seed=SEED):
    torch.manual_seed(seed)
    model = MLPHead(Xtr.shape[1], n_classes, hidden=hidden).to(DEVICE)
    w = None
    if class_weight:
        cnt = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / n_classes,
                         dtype=torch.float)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xtr, dtype=torch.float)
    y = torch.tensor(ytr, dtype=torch.long)
    n = len(y)
    bs = min(256, n)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(X[b]), y[b]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def logits_of(model, X):
    return model(torch.tensor(X, dtype=torch.float)).numpy()


def fit_temperature(logits_cal, y_cal):
    """Scalar temperature scaling (Guo et al. 2017) on the calibration split."""
    lg = torch.tensor(logits_cal, dtype=torch.float)
    y = torch.tensor(y_cal, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lg / torch.exp(logT), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def softmax_np(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins=15):
    """Expected calibration error (top-label, equal-width bins)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metric_panel(probs, y, healthy_class=0):
    """Multi-metric panel — never accuracy alone (feedback_eval_metrics)."""
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, recall_score, roc_auc_score)
    pred = probs.argmax(1)
    classes = np.arange(probs.shape[1])
    panel = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_recall": {int(c): float(recall_score(y == c, pred == c, zero_division=0))
                             for c in classes},
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
    }
    # detection view: healthy vs any-defect
    if (y == healthy_class).any() and (y != healthy_class).any():
        is_def = (y != healthy_class).astype(int)
        p_def = 1.0 - probs[:, healthy_class]
        pred_def = (pred != healthy_class).astype(int)
        tp = int(((pred_def == 1) & (is_def == 1)).sum())
        fp = int(((pred_def == 1) & (is_def == 0)).sum())
        fn = int(((pred_def == 0) & (is_def == 1)).sum())
        tn = int(((pred_def == 0) & (is_def == 0)).sum())
        panel["det_recall"] = tp / (tp + fn) if tp + fn else None
        panel["det_fpr"] = fp / (fp + tn) if fp + tn else None
        panel["det_auprc"] = float(average_precision_score(is_def, p_def))
        panel["det_auc"] = float(roc_auc_score(is_def, p_def))
        mask = y != healthy_class
        panel["defect_only_f1"] = float(f1_score(
            y[mask], pred[mask], average="macro", zero_division=0))
    return panel


def reliability_fig(ax, probs_raw, probs_cal, y, title, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    for probs, lab, col in ((probs_raw, "uncalibrated", "tab:red"),
                            (probs_cal, "temp-scaled", "tab:blue")):
        conf = probs.max(1)
        acc = (probs.argmax(1) == y).astype(float)
        xs, ys = [], []
        for lo, hi, mid in zip(edges[:-1], edges[1:], mids):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean())
                ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", color=col,
                label=f"{lab} (ECE={ece(probs, y):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)


def confusion_fig(ax, cm, labels, title):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=7, color="black" if norm[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)


# ───────────────────────── FEM dataset loading ──────────────────────────────
def pool_graph(x):
    """Permutation-invariant pooling over sensor nodes -> fixed 96-dim vector.

    The processed_gw_* graphs have VARIABLE node counts (9 / 99 / 104 sensors)
    that correlate strongly with the label (healthy jobs were extracted with
    9 sensors, defect jobs with 99) — flattening would both fail and leak.
    Pooled stats reduce but do not remove that confound; we quantify the
    residual leakage with an n_nodes-only probe (see metrics confound_probe).
    """
    return np.concatenate([x.mean(0), x.std(0), x.max(0), x.min(0)])


def load_fem(name):
    """processed_gw_* PyG lists -> pooled (N, 96) arrays + labels + names + n_nodes."""
    out = {}
    for split in ("train", "val"):
        ds = torch.load(os.path.join(ROOT, "data", name, f"{split}.pt"),
                        weights_only=False)
        X = np.stack([pool_graph(g.x.numpy()) for g in ds])
        y = np.array([int(g.y) for g in ds])
        names = [g.name for g in ds]
        n_nodes = np.array([g.x.shape[0] for g in ds])
        out[split] = (X, y, names, n_nodes)
    return out


def doe_positions():
    doe = json.load(open(os.path.join(ROOT, "doe_gw_fairing.json")))
    return {s["job_name"]: s["defect_params"] for s in doe["samples"]}


def join_positions(names, y):
    """Map sample names to continuous (z_center, theta_deg); None if no DOE entry."""
    by_job = doe_positions()
    pos = []
    for nm, yy in zip(names, y):
        if yy == 0:
            pos.append(None)
            continue
        job = nm.replace("_sensors.csv", "").replace("_9s", "")
        p = by_job.get(job)
        pos.append((p["z_center"], p["theta_deg"]) if p else None)
    return pos


# ───────────────────── FMPE continuous (z, theta) head ──────────────────────
class VelocityNet(nn.Module):
    def __init__(self, theta_dim=2, cond_dim=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim + 1 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, theta_dim))

    def forward(self, th, t, c):
        return self.net(torch.cat([th, t[:, None], c], dim=1))


def train_fmpe(C_tr, th_tr, epochs=4000, lr=1e-3, wd=1e-4, seed=SEED):
    """Gaussian-path conditional flow matching: x_t = (1-t) th + t eps."""
    torch.manual_seed(seed)
    model = VelocityNet(theta_dim=th_tr.shape[1], cond_dim=C_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    C = torch.tensor(C_tr, dtype=torch.float)
    TH = torch.tensor(th_tr, dtype=torch.float)
    n = len(TH)
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(128, n),))
        th0, c = TH[idx], C[idx]
        t = torch.rand(len(idx))
        eps = torch.randn_like(th0)
        xt = (1 - t[:, None]) * th0 + t[:, None] * eps
        v_target = eps - th0  # d x_t / d t
        opt.zero_grad()
        ((model(xt, t, c) - v_target) ** 2).mean().backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def fmpe_sample(model, c, n_samples=512, steps=64):
    cB = torch.tensor(c, dtype=torch.float)[None].repeat(n_samples, 1)
    th = torch.randn(n_samples, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), 1.0 - i * dt)
        th = th - model(th, t, cB) * dt
    return th.numpy()


# ─────────────────────────────── training ───────────────────────────────────
def split_indices(y, frac_cal, frac_test, rng):
    """Stratified train/cal/test indices."""
    tr, ca, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te = max(1, int(round(frac_test * len(idx))))
        n_ca = max(1, int(round(frac_cal * len(idx))))
        te += idx[:n_te].tolist()
        ca += idx[n_te:n_te + n_ca].tolist()
        tr += idx[n_te + n_ca:].tolist()
    return np.array(tr), np.array(ca), np.array(te)


def standardize(Xtr, *rest):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return [(x - mu) / sd for x in (Xtr,) + rest] + [mu, sd]


def train_type_head(state, metrics, rng):
    """OGW damage-tag multiclass head with calibration + confound probes."""
    print("\n=== TYPE head: OGW long-term damage tags ===", flush=True)
    d = np.load(CACHE, allow_pickle=True)
    X = d["feats"].reshape(len(d["feats"]), -1)  # (M, 80)
    tags = d["tags"]
    temps = d["temps"]
    month_idx = d["month_idx"]
    months = d["months"]
    classes = np.unique(tags)
    cls_map = {int(c): i for i, c in enumerate(classes)}
    y = np.array([cls_map[int(t)] for t in tags])
    print(f"  cached N={len(y)}, classes={classes.tolist()}, "
          f"counts={np.bincount(y).tolist()}")

    tr, ca, te = split_indices(y, frac_cal=0.15, frac_test=0.2, rng=rng)
    Xtr, Xca, Xte, mu, sd = standardize(X[tr], X[ca], X[te])
    model = train_mlp(Xtr, y[tr], len(classes), epochs=60, hidden=256)
    lg_ca, lg_te = logits_of(model, Xca), logits_of(model, Xte)
    T = fit_temperature(lg_ca, y[ca])
    p_raw, p_cal = softmax_np(lg_te), softmax_np(lg_te, T)
    panel = metric_panel(p_cal, y[te], healthy_class=cls_map.get(0, 0))
    panel["ece_uncalibrated"] = ece(p_raw, y[te])
    panel["ece_calibrated"] = ece(p_cal, y[te])
    panel["temperature_T"] = T
    panel["classes"] = classes.tolist()
    panel["n_train/cal/test"] = [len(tr), len(ca), len(te)]

    # honesty probe 1: per-month test accuracy (tags ~ time => look for months
    # where the same tag is classified differently)
    per_month = {}
    pm = month_idx[te]
    for mi in np.unique(pm):
        m = pm == mi
        per_month[str(months[mi])] = {
            "n": int(m.sum()),
            "acc": float((p_cal[m].argmax(1) == y[te][m]).mean())}
    panel["per_month_test_acc"] = per_month

    # honesty probe 2: can temperature ALONE predict the tag? If yes, part of
    # the head's skill is seasonal context, not damage physics.
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=1000)
    probe.fit(temps[tr][:, None], y[tr])
    panel["confound_probe"] = {
        "temp_only_acc": float(probe.score(temps[te][:, None], y[te])),
        "majority_acc": float(np.bincount(y[tr]).max() / len(tr)),
        "note": ("damage states were introduced sequentially over 4.5 years; "
                 "tag identity is confounded with season/temperature/ageing. "
                 "temp_only_acc >> majority_acc means the waveform head's "
                 "accuracy partially reflects environmental context.")}
    metrics["type_head"] = panel
    state["type"] = {"state_dict": model.state_dict(), "T": T, "mu": mu,
                     "sd": sd, "classes": classes, "hidden": 256,
                     "d_in": X.shape[1]}
    state["_type_eval"] = (p_raw, p_cal, y[te])
    print(f"  test acc={panel['accuracy']:.3f} macroF1={panel['macro_f1']:.3f} "
          f"ECE {panel['ece_uncalibrated']:.3f}->{panel['ece_calibrated']:.3f} "
          f"(T={T:.2f}) | temp-only probe "
          f"{panel['confound_probe']['temp_only_acc']:.3f} "
          f"vs majority {panel['confound_probe']['majority_acc']:.3f}")


def train_fem_head(state, metrics, rng, dataset, key, n_classes, hidden=64):
    print(f"\n=== {key} head: {dataset} ===", flush=True)
    d = load_fem(dataset)
    Xtr_all, ytr_all, _, ntr_nodes = d["train"]
    Xte, yte, names_te, nte_nodes = d["val"]
    # carve calibration split out of train (val stays untouched as test)
    tr, ca, _ = split_indices(ytr_all, frac_cal=0.2, frac_test=0.0, rng=rng)
    Xtr, Xca, Xte_s, mu, sd = standardize(Xtr_all[tr], Xtr_all[ca], Xte)
    model = train_mlp(Xtr, ytr_all[tr], n_classes, epochs=300, hidden=hidden)
    lg_ca, lg_te = logits_of(model, Xca), logits_of(model, Xte_s)
    T = fit_temperature(lg_ca, ytr_all[ca])
    p_raw, p_cal = softmax_np(lg_te), softmax_np(lg_te, T)
    panel = metric_panel(p_cal, yte, healthy_class=0)
    panel["ece_uncalibrated"] = ece(p_raw, yte)
    panel["ece_calibrated"] = ece(p_cal, yte)
    panel["temperature_T"] = T
    panel["n_train/cal/test"] = [len(tr), len(ca), len(yte)]
    panel["test_class_counts"] = np.bincount(yte, minlength=n_classes).tolist()
    # honesty probe: sensor count correlates with label in this dataset
    # (healthy jobs were extracted with 9 sensors, defect jobs with 99).
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=1000)
    probe.fit(ntr_nodes[tr][:, None].astype(float), ytr_all[tr])
    panel["confound_probe"] = {
        "n_nodes_only_acc": float(probe.score(
            nte_nodes[:, None].astype(float), yte)),
        "majority_acc": float(np.bincount(ytr_all[tr]).max() / len(tr)),
        "note": ("graphs have 9/99/104 sensors and the count correlates with "
                 "the label (extraction campaigns differed); a high "
                 "n_nodes_only_acc means part of the head's skill may be an "
                 "extraction artefact, not defect physics.")}
    metrics[f"{key}_head"] = panel
    state[key] = {"state_dict": model.state_dict(), "T": T, "mu": mu, "sd": sd,
                  "hidden": hidden, "d_in": Xtr.shape[1],
                  "n_classes": n_classes}
    state[f"_{key}_eval"] = (p_raw, p_cal, yte)
    print(f"  test acc={panel['accuracy']:.3f} macroF1={panel['macro_f1']:.3f} "
          f"detRec={panel.get('det_recall')} detFPR={panel.get('det_fpr')} "
          f"AUPRC={panel.get('det_auprc'):.3f} "
          f"ECE {panel['ece_uncalibrated']:.3f}->{panel['ece_calibrated']:.3f}")


def train_fmpe_head(state, metrics):
    """Continuous (z, theta) posterior on FEM defect samples with DOE join."""
    print("\n=== FMPE head: continuous (z, theta) posterior ===", flush=True)
    d = load_fem("processed_gw_5loc")
    # PCA fit on train features (defect samples only)
    Xtr, ytr, ntr, _ = d["train"]
    pos_tr = join_positions(ntr, ytr)
    keep_tr = [i for i, p in enumerate(pos_tr) if p is not None]
    Xte, yte, nte, _ = d["val"]
    pos_te = join_positions(nte, yte)
    keep_te = [i for i, p in enumerate(pos_te) if p is not None]
    print(f"  joined defect samples: train {len(keep_tr)}, val {len(keep_te)} "
          f"(skipped Multi/Valid/Test jobs without DOE entries)")

    Xtr_d = Xtr[keep_tr]
    Xte_d = Xte[keep_te]
    Xtr_s, Xte_s, mu, sd = standardize(Xtr_d, Xte_d)
    k = min(12, len(keep_tr) - 1)
    U, S, Vt = np.linalg.svd(Xtr_s - Xtr_s.mean(0), full_matrices=False)
    P = Vt[:k].T  # (D, k)
    C_all = (Xtr_s - Xtr_s.mean(0)) @ P
    C_te = (Xte_s - Xtr_s.mean(0)) @ P
    c_sd = C_all.std(0)
    c_sd[c_sd < 1e-8] = 1
    C_all, C_te = C_all / c_sd, C_te / c_sd

    th_all = np.array([pos_tr[i] for i in keep_tr], dtype=np.float32)
    th_te = np.array([pos_te[i] for i in keep_te], dtype=np.float32)
    th_mu, th_sd = th_all.mean(0), th_all.std(0)

    # carve a conformal-calibration split out of train (the raw FM posterior
    # is overconfident at this sample size, ~80 train pairs)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(C_all))
    n_cal = max(10, int(0.2 * len(perm)))
    cal, fit = perm[:n_cal], perm[n_cal:]
    model = train_fmpe(C_all[fit], (th_all[fit] - th_mu) / th_sd, epochs=3000)

    def posterior(c):
        return fmpe_sample(model, c) * th_sd + th_mu  # (S, 2) physical units

    # split-conformal half-widths on |truth - posterior mean| (per dim, 90%)
    res = np.array([np.abs(posterior(C_all[i]).mean(0) - th_all[i])
                    for i in cal])
    q_idx = min(n_cal - 1, int(np.ceil(0.9 * (n_cal + 1))) - 1)
    q_conf = np.sort(res, axis=0)[q_idx]  # (2,)

    # evaluation: posterior mean error + 90% coverage (raw FM and conformal)
    errs, cover_raw, cover_conf = [], [], []
    for i in range(len(C_te)):
        s = posterior(C_te[i])
        m = s.mean(0)
        lo, hi = np.percentile(s, 5, axis=0), np.percentile(s, 95, axis=0)
        cover_raw.append((th_te[i] >= lo) & (th_te[i] <= hi))
        cover_conf.append(np.abs(m - th_te[i]) <= q_conf)
        errs.append(np.abs(m - th_te[i]))
    errs = np.array(errs)
    cover_raw = np.array(cover_raw, dtype=float)
    cover_conf = np.array(cover_conf, dtype=float)
    rng_z = th_all[:, 0].max() - th_all[:, 0].min()
    rng_t = th_all[:, 1].max() - th_all[:, 1].min()
    panel = {
        "n_train_fit": len(fit), "n_calibration": len(cal),
        "n_test": len(keep_te), "pca_k": int(k),
        "mae_z_mm": float(errs[:, 0].mean()),
        "mae_theta_deg": float(errs[:, 1].mean()),
        "mae_z_relative_to_range": float(errs[:, 0].mean() / rng_z),
        "mae_theta_relative_to_range": float(errs[:, 1].mean() / rng_t),
        "coverage90_raw_z": float(cover_raw[:, 0].mean()),
        "coverage90_raw_theta": float(cover_raw[:, 1].mean()),
        "coverage90_conformal_z": float(cover_conf[:, 0].mean()),
        "coverage90_conformal_theta": float(cover_conf[:, 1].mean()),
        "conformal_halfwidth": {"z_mm": float(q_conf[0]),
                                "theta_deg": float(q_conf[1])},
        "prior_range_z_mm": [float(th_all[:, 0].min()), float(th_all[:, 0].max())],
        "prior_range_theta_deg": [float(th_all[:, 1].min()), float(th_all[:, 1].max())],
    }
    metrics["fmpe_head"] = panel
    state["fmpe"] = {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                     "pca_mean": Xtr_s.mean(0), "P": P, "c_sd": c_sd,
                     "th_mu": th_mu, "th_sd": th_sd, "cond_dim": int(k),
                     "q_conf": q_conf}
    state["_fmpe_eval"] = (model, C_te, th_te, th_mu, th_sd, q_conf)
    print(f"  MAE z={panel['mae_z_mm']:.0f} mm, theta={panel['mae_theta_deg']:.1f} deg | "
          f"90% coverage raw z={panel['coverage90_raw_z']:.2f}/"
          f"th={panel['coverage90_raw_theta']:.2f} -> conformal "
          f"z={panel['coverage90_conformal_z']:.2f}/"
          f"th={panel['coverage90_conformal_theta']:.2f} "
          f"(halfwidth z={q_conf[0]:.0f}mm th={q_conf[1]:.1f}deg)")


def make_figures(state, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
            axes, ("_type_eval", "_binary_eval", "_loc_eval"),
            ("TYPE: OGW damage tag", "BINARY: FEM healthy/defect",
             "LOCATION: FEM 5-class")):
        p_raw, p_cal, y = state[key]
        reliability_fig(ax, p_raw, p_cal, y, title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_reliability.png"), dpi=150)
    plt.close(fig)

    # 2. confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    confusion_fig(axes[0], metrics["type_head"]["confusion_matrix"],
                  [f"tag{c}" for c in metrics["type_head"]["classes"]],
                  "TYPE (OGW tags, calibrated)")
    confusion_fig(axes[1], metrics["binary_head"]["confusion_matrix"],
                  ["healthy", "defect"], "BINARY (FEM)")
    confusion_fig(axes[2], metrics["loc_head"]["confusion_matrix"],
                  [LOC_NAMES[i] for i in range(5)], "LOCATION (FEM 5-class)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=150)
    plt.close(fig)

    # 3. FMPE posterior examples + coverage
    from matplotlib.patches import Rectangle
    model, C_te, th_te, th_mu, th_sd, q_conf = state["_fmpe_eval"]
    n_show = min(6, len(C_te))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, ax in enumerate(axes.ravel()[:n_show]):
        s = fmpe_sample(model, C_te[j], n_samples=512) * th_sd + th_mu
        m = s.mean(0)
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.25, label="posterior draws")
        ax.add_patch(Rectangle((m[0] - q_conf[0], m[1] - q_conf[1]),
                               2 * q_conf[0], 2 * q_conf[1], fill=False,
                               edgecolor="tab:green", lw=1.2,
                               label="conformal 90% box"))
        ax.plot(th_te[j, 0], th_te[j, 1], "r*", ms=14, label="truth")
        ax.axvline(1500, color="gray", lw=0.5)
        ax.axhline(15, color="gray", lw=0.5)
        ax.set_xlim(500, 2500)
        ax.set_ylim(0, 30)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("theta [deg]")
        if j == 0:
            ax.legend(fontsize=8)
    fm = metrics["fmpe_head"]
    fig.suptitle(f"FMPE posterior q(z, theta | x_fem) — val examples | "
                 f"90% cov raw z={fm['coverage90_raw_z']:.2f}/"
                 f"th={fm['coverage90_raw_theta']:.2f}, conformal "
                 f"z={fm['coverage90_conformal_z']:.2f}/"
                 f"th={fm['coverage90_conformal_theta']:.2f}; "
                 f"MAE z={fm['mae_z_mm']:.0f}mm, th={fm['mae_theta_deg']:.1f}deg")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fmpe_posterior.png"), dpi=150)
    plt.close(fig)
    print(f"[fig] saved 3 figures to {OUT}")


def cmd_train():
    rng = np.random.default_rng(SEED)
    state, metrics = {}, {}
    train_type_head(state, metrics, rng)
    train_fem_head(state, metrics, rng, "processed_gw_92_comp", "binary", 2)
    train_fem_head(state, metrics, rng, "processed_gw_5loc", "loc", 5)
    train_fmpe_head(state, metrics)
    make_figures(state, metrics)
    metrics["caveats"] = [
        "TYPE head: OGW damage tags 0..13 were introduced sequentially over "
        "4.5 years -> tag identity is confounded with season/temperature/"
        "ageing; the random split measures in-distribution separability, NOT "
        "pure damage-signature skill. See confound_probe (temperature-only "
        "baseline) and per_month_test_acc.",
        "TYPE and LOCATION heads live in different measurement domains "
        "(real OGW stiffened plate vs FEM H3 fairing); the joint is a "
        "conditional-independence product, not a single learned posterior.",
        "LOCATION val split has only 2 samples of class 3 — per-class "
        "recall for that class is anecdotal.",
        "FEM graphs have 9/99/104 sensor nodes and the count correlates "
        "with the label (extraction campaigns differed between healthy and "
        "defect jobs); features are pooled to be permutation/count-"
        "invariant, but the residual artefact is quantified by the "
        "n_nodes-only confound probe in binary_head/loc_head metrics.",
        "FEM heads: train/val from the same DOE (no OOD axis); per "
        "INSIGHTS_FROM_WCCM_OOD.md expect precise localisation to degrade "
        "far more than detection under distribution shift.",
    ]
    ckpt = {k: v for k, v in state.items() if not k.startswith("_")}
    torch.save(ckpt, CKPT)
    json.dump(metrics, open(METRICS, "w"), indent=1)
    print(f"\n[train] checkpoint -> {CKPT}\n[train] metrics -> {METRICS}")


# ───────────────────────────── joint API ────────────────────────────────────
class DefectCharacterizer:
    """Stage-2 joint posterior p(defect TYPE, LOCATION | guided-wave x).

    x_fem : (n_sensors, 24) FEM fairing sensor features (processed_gw_* layout;
            pooled internally to a fixed 96-dim vector, any sensor count)
            -> p_defect (binary head), location_probs (5-class head),
               optional continuous (z, theta) posterior samples (FMPE head).
    x_ogw : (8, 2000) raw OGW waveforms OR (8, 10) path features
            -> type_probs over OGW damage tags.
    Joint: p(type, loc | x) ~= p(type | x_ogw) * p(loc | x_fem)  (cond. indep.)
    """

    def __init__(self, ckpt_path=CKPT):
        ck = torch.load(ckpt_path, weights_only=False)
        self.heads = {}
        for key in ("type", "binary", "loc"):
            c = ck[key]
            n_cls = len(c["classes"]) if key == "type" else c["n_classes"]
            m = MLPHead(c["d_in"], n_cls, hidden=c["hidden"])
            m.load_state_dict(c["state_dict"])
            m.eval()
            self.heads[key] = (m, c)
        c = ck["fmpe"]
        fm = VelocityNet(theta_dim=2, cond_dim=c["cond_dim"])
        fm.load_state_dict(c["state_dict"])
        fm.eval()
        self.fmpe = (fm, c)

    def _head_probs(self, key, x_flat):
        m, c = self.heads[key]
        z = (x_flat - c["mu"]) / c["sd"]
        lg = logits_of(m, z[None])
        return softmax_np(lg, c["T"])[0]

    def characterize(self, x_fem, x_ogw=None, n_posterior_samples=512):
        x_fem = np.asarray(x_fem, dtype=np.float32)
        if x_fem.ndim == 2:  # (n_sensors, 24) -> pooled 96-dim
            x_fem = pool_graph(x_fem).astype(np.float32)
        x_fem = x_fem.reshape(-1)
        p_bin = self._head_probs("binary", x_fem)
        p_loc = self._head_probs("loc", x_fem)
        out = {
            "p_defect": float(p_bin[1]),
            "location_probs": {LOC_NAMES[i]: float(p)
                               for i, p in enumerate(p_loc)},
            "type_probs": None,
            "top1": {"location": LOC_NAMES[int(p_loc.argmax())]},
        }
        # continuous (z, theta) posterior, conditioned on "defect present"
        fm, c = self.fmpe
        zx = (x_fem - c["mu"]) / c["sd"]
        cond = ((zx - c["pca_mean"]) @ c["P"]) / c["c_sd"]
        s = fmpe_sample(fm, cond, n_samples=n_posterior_samples)
        s = s * c["th_sd"] + c["th_mu"]
        q = c["q_conf"]
        m_z, m_t = float(s[:, 0].mean()), float(s[:, 1].mean())
        out["location_continuous"] = {
            "z_mm_mean": m_z,
            "z_mm_ci90_raw": [float(np.percentile(s[:, 0], 5)),
                              float(np.percentile(s[:, 0], 95))],
            "z_mm_ci90_conformal": [m_z - float(q[0]), m_z + float(q[0])],
            "theta_deg_mean": m_t,
            "theta_deg_ci90_raw": [float(np.percentile(s[:, 1], 5)),
                                   float(np.percentile(s[:, 1], 95))],
            "theta_deg_ci90_conformal": [m_t - float(q[1]), m_t + float(q[1])],
            "note": "valid conditionally on a defect being present; use the "
                    "conformal interval (the raw FM interval is overconfident "
                    "at this training-set size)",
        }
        if x_ogw is not None:
            x_ogw = np.asarray(x_ogw, dtype=np.float64)
            if x_ogw.ndim == 2 and x_ogw.shape[1] != 10:  # raw waveforms
                x_ogw = path_features(x_ogw[None])[0]
            _, ct = self.heads["type"]
            p_type = self._head_probs("type", x_ogw.reshape(-1).astype(np.float32))
            classes = ct["classes"]
            out["type_probs"] = {f"tag{int(c_)}": float(p)
                                 for c_, p in zip(classes, p_type)}
            out["top1"]["type"] = f"tag{int(classes[int(p_type.argmax())])}"
            # joint under conditional independence given the two measurements
            joint = p_bin[1] * np.outer(p_type, p_loc[1:] / max(p_loc[1:].sum(), 1e-9))
            out["joint"] = {
                "matrix_type_x_loc": joint.tolist(),
                "type_labels": [f"tag{int(c_)}" for c_ in classes],
                "loc_labels": [LOC_NAMES[i] for i in range(1, 5)],
                "assumption": "p(type,loc|x) ~= p_defect * p(type|x_ogw) * "
                              "p(loc|x_fem, defect)",
            }
        return out


def cmd_demo(n=3):
    """Run characterize() on validation samples from both domains."""
    ch = DefectCharacterizer()
    fem = torch.load(os.path.join(ROOT, "data", "processed_gw_5loc", "val.pt"),
                     weights_only=False)
    d = np.load(CACHE, allow_pickle=True)
    rng = np.random.default_rng(0)
    ogw_idx = rng.choice(len(d["tags"]), n, replace=False)
    picks = [g for g in fem if int(g.y) != 0][:n - 1] + \
            [g for g in fem if int(g.y) == 0][:1]
    demo_out = []
    for g, oi in zip(picks, ogw_idx):
        rep = ch.characterize(g.x.numpy(), x_ogw=d["feats"][oi])
        rep["_fem_sample"] = g.name
        rep["_fem_true_loc"] = LOC_NAMES[int(g.y)]
        rep["_ogw_sample_true_tag"] = int(d["tags"][oi])
        demo_out.append(rep)
        print(f"\n--- {g.name} (true loc: {LOC_NAMES[int(g.y)]}) "
              f"+ OGW sample (true tag {int(d['tags'][oi])}) ---")
        print(json.dumps({k: v for k, v in rep.items() if k != "joint"},
                         indent=1))
    json.dump(demo_out, open(os.path.join(OUT, "demo_characterize.json"), "w"),
              indent=1)
    print(f"\n[demo] saved -> {OUT}/demo_characterize.json")


def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["scan", "train", "demo", "all"])
    a = ap.parse_args()
    if a.stage in ("scan", "all"):
        if a.stage == "all" and os.path.exists(CACHE):
            print(f"[scan] cache exists, skipping ({CACHE})")
        else:
            scan_ogw()
    if a.stage in ("train", "all"):
        cmd_train()
    if a.stage in ("demo", "all"):
        cmd_demo()


if __name__ == "__main__":
    main()
