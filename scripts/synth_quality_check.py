#!/usr/bin/env python3
"""
Generative quality audit for synthetic GW data — BEFORE trusting augmentation.

Checks:
  1. C2ST (classifier two-sample test): can a GBM tell real-damaged from
     synthetic-damaged in feature space?  AUC≈0.5 good, AUC≈1.0 bad.
  2. Conditioning fidelity: direction/magnitude of (damaged − healthy) feature
     shift, real vs synthetic. The generator must reproduce the damage signature.
  3. Memorisation: nearest-neighbour distance synthetic→train vs train→train.
     Synth NN distance ≪ train NN distance ⇒ copying.
  4. Diversity: mean pairwise distance within synthetic vs within real.
  5. Visual: waveform overlays + PCA scatter.

Run: python3 scripts/synth_quality_check.py --synth results/payload_ddpm/synthetic_damaged.npz
"""
import os, sys, glob, pickle, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from payload_diffusion import downsample, RESULTS_DIR, LT_DIR
from longterm_gnn import path_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)


def load_real(month, mu, sig, want_dmg, n=2000):
    f = os.path.join(LT_DIR, f"measurements_{month}.pickle")
    d = pickle.load(open(f, "rb"))
    y = (np.asarray(d["damage tag"]) > 0).astype(int)
    idx = np.where(y == want_dmg)[0]
    if len(idx) == 0:
        return None
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    gw = downsample(d["guided wave"][idx].astype(np.float32))
    return (gw - mu) / sig


def flat_feats(gw):
    """(N,8,256) → (N, 8*10) feature matrix."""
    return path_features(gw).reshape(len(gw), -1)


def c2st(a, b, label=""):
    """Classifier two-sample test. Returns CV AUC."""
    X = np.concatenate([a, b]); y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean()
    print(f"  C2ST {label}: AUC={auc:.3f}  (0.5=indistinguishable, 1.0=trivially separable)")
    return auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", default=f"{RESULTS_DIR}/synthetic_damaged.npz")
    p.add_argument("--n",     type=int, default=2000)
    args = p.parse_args()

    sd  = np.load(args.synth)
    mu, sig = sd["mu"], sd["sig"]
    gw_syn  = sd["gw"][:args.n].astype(np.float32)
    print(f"synthetic: {gw_syn.shape}, guidance={float(sd['guidance'])}")

    # real reference sets (generator's training months → in-distribution check)
    real_dmg  = load_real("2022_07", mu, sig, want_dmg=1, n=args.n)   # damaged, summer
    real_dmg2 = load_real("2022_01", mu, sig, want_dmg=1, n=args.n)   # damaged, winter
    real_hlt  = load_real("2018_12", mu, sig, want_dmg=0, n=args.n)   # healthy, winter
    real_hlt2 = load_real("2019_07", mu, sig, want_dmg=0, n=args.n)   # healthy, summer

    f_syn  = flat_feats(gw_syn)
    f_dmg  = flat_feats(np.concatenate([real_dmg, real_dmg2]))
    f_hlt  = flat_feats(np.concatenate([real_hlt, real_hlt2]))

    print("\n── 1. C2ST: real vs synthetic ──")
    auc_rs = c2st(f_dmg[rng.choice(len(f_dmg), min(len(f_dmg), len(f_syn)), replace=False)],
                  f_syn, "real-dmg vs synth-dmg")
    # calibration reference: real-vs-real across seasons (irreducible env shift)
    auc_rr = c2st(flat_feats(real_dmg), flat_feats(real_dmg2),
                  "real-dmg(summer) vs real-dmg(winter) [reference]")

    print("\n── 2. Conditioning fidelity: damage signature ──")
    shift_real = f_dmg.mean(0) - f_hlt.mean(0)
    # synthetic healthy may still be generating — fall back to real healthy
    syn_hlt_path = f"{RESULTS_DIR}/synthetic_healthy.npz"
    if os.path.exists(syn_hlt_path):
        sh = np.load(syn_hlt_path)
        f_syn_hlt = flat_feats(sh["gw"][:args.n].astype(np.float32))
        shift_syn = f_syn.mean(0) - f_syn_hlt.mean(0)
        src = "synth-hlt"
    else:
        shift_syn = f_syn.mean(0) - f_hlt.mean(0)
        src = "real-hlt (synth-hlt not ready)"
    cos = np.dot(shift_real, shift_syn) / (np.linalg.norm(shift_real) * np.linalg.norm(shift_syn) + 1e-12)
    ratio = np.linalg.norm(shift_syn) / (np.linalg.norm(shift_real) + 1e-12)
    print(f"  damage-shift cosine(real, synth[{src}]) = {cos:.3f}  (1.0=same direction)")
    print(f"  damage-shift magnitude ratio synth/real = {ratio:.3f}")

    print("\n── 3. Memorisation check (waveform space) ──")
    sub_tr  = real_dmg[rng.choice(len(real_dmg), 500, replace=False)].reshape(500, -1)
    sub_syn = gw_syn[rng.choice(len(gw_syn), 500, replace=False)].reshape(500, -1)
    def nn_dist(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) ** 0.5
        return d
    d_st = nn_dist(sub_syn, sub_tr); np.fill_diagonal(d_st, np.inf) if d_st.shape[0]==d_st.shape[1] else None
    d_tt = nn_dist(sub_tr, sub_tr);  np.fill_diagonal(d_tt, np.inf)
    nn_syn = d_st.min(1).mean(); nn_real = d_tt.min(1).mean()
    print(f"  NN dist synth→train: {nn_syn:.2f}   train→train: {nn_real:.2f}   "
          f"ratio={nn_syn/nn_real:.2f}  (<0.5 ⇒ memorising)")

    print("\n── 4. Diversity ──")
    div_syn  = d_st.mean()  # cross dist as scale ref
    pd_syn   = nn_dist(sub_syn, sub_syn); np.fill_diagonal(pd_syn, np.nan)
    pd_real  = d_tt.copy()
    print(f"  mean pairwise dist: synth={np.nanmean(pd_syn):.2f}  real={np.nanmean(np.where(np.isinf(pd_real), np.nan, pd_real)):.2f}")

    # ── 5. visuals ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    t = np.arange(256)
    for i in range(4):
        axes[0, 0].plot(t, real_dmg[i, 0], alpha=0.7, lw=0.8)
    axes[0, 0].set_title("REAL damaged (path 0)")
    for i in range(4):
        axes[0, 1].plot(t, gw_syn[i, 0], alpha=0.7, lw=0.8)
    axes[0, 1].set_title("SYNTH damaged (path 0)")

    pca = PCA(n_components=2).fit(np.concatenate([f_dmg, f_syn]))
    pr, ps = pca.transform(f_dmg), pca.transform(f_syn)
    ph = pca.transform(f_hlt)
    axes[1, 0].scatter(*ph.T,  s=2, alpha=0.2, label="real healthy", c="gray")
    axes[1, 0].scatter(*pr.T,  s=2, alpha=0.3, label="real damaged", c="tab:red")
    axes[1, 0].scatter(*ps.T,  s=2, alpha=0.3, label="synth damaged", c="tab:blue")
    axes[1, 0].legend(markerscale=6); axes[1, 0].set_title("PCA of path features")

    axes[1, 1].hist(f_dmg[:, 0], bins=60, alpha=0.5, density=True, label="real dmg RMS")
    axes[1, 1].hist(f_syn[:, 0], bins=60, alpha=0.5, density=True, label="synth dmg RMS")
    axes[1, 1].legend(); axes[1, 1].set_title("RMS (path 0) distribution")
    plt.tight_layout()
    out = f"{RESULTS_DIR}/synth_quality.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nsaved → {out}")

    print("\n── verdict ──")
    ok = (auc_rs < max(0.75, auc_rr + 0.15)) and cos > 0.7 and nn_syn / nn_real > 0.5
    print("  PASS" if ok else "  FAIL — augmentation results should not be trusted yet",
          f"(C2ST={auc_rs:.2f} vs ref {auc_rr:.2f}, cos={cos:.2f}, NN-ratio={nn_syn/nn_real:.2f})")


if __name__ == "__main__":
    main()
