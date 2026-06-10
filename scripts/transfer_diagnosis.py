#!/usr/bin/env python3
"""
Why does cross-month damage detection fail? (baseline AUC 0.38 < chance)
Decompose the failure into: task difficulty / real-transfer gap / synthetic gap.

  A. CEILING   : train+test WITHIN 2019_10 (70/30)        — is the task learnable at all?
  B. REAL-XFER : train 2018_03 + REAL 2019_07 → 2019_10   — does even real cross-month
                                                            damage data transfer?
  C. SYNTH-XFER: train 2018_03 + synthetic    → 2019_10   — current augmentation
  D. CALIBRATED: like C but features re-normalised with a healthy slice of the
                 TEST month (deployment-realistic: one healthy calibration scan)

If B fails like C → transfer is broken at task level, synthetic quality is not
the bottleneck. If D recovers → per-month healthy calibration is the fix
(consistent with the earlier sim-to-real finding: 正規化が鍵).

Run: python3 scripts/transfer_diagnosis.py
"""
import os, sys, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from build_gw_graph import build_edge_index
from payload_diffusion import RESULTS_DIR
from longterm_gnn import path_features
from augment_eval import load_month_z, build_graphs, panel, train_classifier

rng = np.random.default_rng(0)


def feat_norm(ft):
    flat = ft.reshape(-1, ft.shape[2])
    mu, sd = flat.mean(0), flat.std(0); sd[sd < 1e-8] = 1
    return mu, sd


def run(tag, ft_tr, y_tr, ft_te, y_te, f_mu, f_sd, ei, seeds=3):
    in_dim = ft_tr.shape[2]
    rs = []
    for seed in range(seeds):
        g_tr = build_graphs(ft_tr, y_tr, f_mu, f_sd, ei)
        g_te = build_graphs(ft_te, y_te, f_mu, f_sd, ei)
        m = train_classifier(g_tr, in_dim, seed=seed)
        rs.append(panel(m, g_te, f"{tag} seed{seed}"))
    keys = ["AUC", "AUPRC", "recall", "FPR", "F1", "balacc"]
    print(f"  >> {tag}: " + "  ".join(
        f"{k}={np.mean([r[k] for r in rs]):.3f}" for k in keys))
    return rs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_synth", type=int, default=2000)
    args = p.parse_args()

    stats = np.load(f"{RESULTS_DIR}/norm_stats.npz")
    mu, sig = stats["mu"], stats["sig"]
    ei = build_edge_index(8, list(range(8)), "full")

    gw_03, y_03 = load_month_z("2018_03", mu, sig)
    gw_07, y_07 = load_month_z("2019_07", mu, sig)
    gw_10, y_10 = load_month_z("2019_10", mu, sig)

    ft_03, ft_07, ft_10 = path_features(gw_03), path_features(gw_07), path_features(gw_10)

    # ── A. ceiling: within 2019_10 ───────────────────────────────────────────
    idx = rng.permutation(len(y_10)); n_tr = int(0.7 * len(y_10))
    tr, te = idx[:n_tr], idx[n_tr:]
    f_mu, f_sd = feat_norm(ft_10[tr])
    print("\n── A. CEILING (within 2019_10, 70/30) ──")
    run("A", ft_10[tr], y_10[tr], ft_10[te], y_10[te], f_mu, f_sd, ei)

    # held-out test for B/C/D = same 30% slice of 2019_10 (comparable numbers)
    ft_te, y_te = ft_10[te], y_10[te]

    # ── B. real cross-month transfer ─────────────────────────────────────────
    ft_b = np.concatenate([ft_03, ft_07]); y_b = np.concatenate([y_03, y_07])
    f_mu, f_sd = feat_norm(ft_b)
    print("\n── B. REAL-XFER (2018_03 + real 2019_07 → 2019_10) ──")
    run("B", ft_b, y_b, ft_te, y_te, f_mu, f_sd, ei)

    # ── C. synthetic transfer (current augmentation) ─────────────────────────
    sd_d = np.load(f"{RESULTS_DIR}/synthetic_damaged.npz")
    sd_h = np.load(f"{RESULTS_DIR}/synthetic_healthy.npz")
    ft_sd = path_features(sd_d["gw"][:args.n_synth].astype(np.float32))
    ft_sh = path_features(sd_h["gw"][:args.n_synth].astype(np.float32))
    ft_c = np.concatenate([ft_03, ft_sd, ft_sh])
    y_c  = np.concatenate([y_03, np.ones(len(ft_sd), int), np.zeros(len(ft_sh), int)])
    f_mu, f_sd_n = feat_norm(np.concatenate([ft_03]))
    print("\n── C. SYNTH-XFER (2018_03 + synth → 2019_10) ──")
    run("C", ft_c, y_c, ft_te, y_te, f_mu, f_sd_n, ei)

    # ── D. healthy-calibrated normalisation ─────────────────────────────────
    # deployment-realistic: re-normalise features with a small HEALTHY slice
    # of the test month (one calibration scan), train set = same as C
    hl = np.where(y_10[tr] == 0)[0][:200]            # healthy calib from train slice
    f_mu_d, f_sd_d = feat_norm(ft_10[tr][hl])
    print("\n── D. CALIBRATED (C + test-month healthy normalisation) ──")
    run("D", ft_c, y_c, ft_te, y_te, f_mu_d, f_sd_d, ei)

    print("\nInterpretation: A=task ceiling; B≈C ⇒ transfer gap is task-level "
          "(damage states/env drift), synthetic quality not the bottleneck; "
          "D>>C ⇒ per-month healthy calibration is the practical fix.")


if __name__ == "__main__":
    main()
