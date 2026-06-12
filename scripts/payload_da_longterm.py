"""
payload_da_longterm.py — (B) domain adaptation on the REAL long-term OGW data,
across a TEMPERATURE domain shift, with the proper waveform features.

This is the real-data sequel to the (A) cross-frequency demo. (A) showed the
domain-adaptation toolkit closes the domain gap mechanically but crude FFT
band-energy features carry no defect signal. (B) uses the PROPER per-path
waveform features (rms/energy/std/ptp/argmax/dom-freq/spectral-centroid/hi-freq
ratio — the same 8x10 representation that reaches AUROC 0.94 in longterm_gnn.py),
on the documented severe shift: damage detection trained on a COOL month
(2018-03, ~9C, which carries damage labels) and evaluated on a HOT month
(2018-07, ~30C, healthy) — where an uncorrected detector false-alarms because the
temperature shift mimics damage (the core uncontrolled-condition problem).

We ask: does unsupervised alignment (per-domain standardize / CORAL / quantile
matching) reduce the cross-month feature gap and RECOVER the false-positive rate,
and how does it compare to plain z-scoring? This complements the synthetic-data
DANN prototype with a label-free, real-data baseline on the actual shift.

Honest scope: months 2018-03..08 are on disk (cool->hot; the ~0C winter extreme
is not), so the shift evaluated is cool->hot; the target month is healthy-only so
the metric is false-positive rate (FPR), exactly the temperature-OOD framing.

Usage
-----
    python scripts/payload_da_longterm.py            # 2018-03 -> 2018-07
    python scripts/payload_da_longterm.py --src-month 2018_03 --tgt-month 2018_06
    python scripts/payload_da_longterm.py --fig
    python scripts/payload_da_longterm.py --test
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import domain_adapt as da   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT = os.path.join(ROOT, "data", "external", "ogw_longterm")


# ═════════════════════════════════════════════════════════════════════════════
#  Loader + proper per-path waveform features (mirrors longterm_gnn.py)
# ═════════════════════════════════════════════════════════════════════════════

def path_features(gw: np.ndarray) -> np.ndarray:
    """(N,8,2000) waveforms -> (N,8,10) per-path features (time + spectral)."""
    x = gw.astype(np.float64)
    rms = np.sqrt((x ** 2).mean(2)); mx = np.abs(x).max(2); en = (x ** 2).sum(2)
    std = x.std(2); ptp = x.max(2) - x.min(2); mab = np.abs(x).mean(2)
    amax = x.argmax(2) / x.shape[2]
    P = np.abs(np.fft.rfft(x, axis=2))
    domf = P.argmax(2) / P.shape[2]
    cent = (np.arange(P.shape[2]) * P).sum(2) / (P.sum(2) + 1e-9) / P.shape[2]
    hi = P[:, :, P.shape[2] // 2:].sum(2) / (P.sum(2) + 1e-9)
    return np.stack([rms, mx, en, std, ptp, mab, amax, domf, cent, hi],
                    axis=2).astype(np.float32)


def load_month(month: str, max_n: int | None = None, seed: int = 0):
    """Load one long-term month -> (X (n,80) features, y (n,) damage, temp_mean)."""
    f = os.path.join(LT, f"measurements_{month}.pickle")
    d = pickle.load(open(f, "rb"))
    gw = d["guided wave"]
    y = d["damage tag"].astype(int)
    temp = np.asarray(d.get("temperature", np.full(len(y), np.nan)), float)
    if max_n and len(y) > max_n:
        idx = np.random.default_rng(seed).choice(len(y), max_n, replace=False)
        gw, y, temp = gw[idx], y[idx], temp[idx]
    X = path_features(gw).reshape(len(y), -1)          # (n, 80)
    return X, y, float(np.nanmean(temp))


# ═════════════════════════════════════════════════════════════════════════════
#  The temperature-shift DA experiment
# ═════════════════════════════════════════════════════════════════════════════

def fpr_at(scores_healthy_target: np.ndarray, thr: float) -> float:
    return float((scores_healthy_target > thr).mean())


def run(src_month: str = "2018_03", tgt_month: str = "2018_07",
        max_n: int | None = 3000, target_fpr: float = 0.05,
        verbose: bool = True) -> dict:
    """Train detection on the cool source month, evaluate FPR on the hot target
    month, and test whether each alignment recovers the temperature-induced
    false-alarm collapse."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    Xs, ys, ts = load_month(src_month, max_n)
    Xt, yt, tt = load_month(tgt_month, max_n)

    # in-domain detectability on the source (proper features should separate)
    auroc_in = (float(cross_val_score(LogisticRegression(max_iter=1000), Xs, ys,
                                      cv=5, scoring="roc_auc").mean())
                if len(np.unique(ys)) > 1 else float("nan"))

    src_healthy = Xs[ys == 0]
    tgt_healthy = Xt[yt == 0]
    res = {}
    for m in da.ALIGNERS:
        # fit the detector on aligned source, score aligned target healthy
        Xs_a = da.ALIGNERS[m](Xs, Xt)
        clf = LogisticRegression(max_iter=1000).fit(Xs_a, ys)
        # threshold calibrated to target_fpr on the SOURCE healthy (no target leak)
        Ssrc_h = clf.predict_proba(da._apply_target(m, Xs, Xs)[ys == 0])[:, 1] \
            if m == "standardize" else clf.predict_proba(Xs_a[ys == 0])[:, 1]
        thr = float(np.quantile(Ssrc_h, 1.0 - target_fpr))
        Xt_a = da._apply_target(m, Xs, Xt)
        St_h = clf.predict_proba(Xt_a[yt == 0])[:, 1]
        # recall-preservation: held-out recall on SOURCE damaged at the same
        # source-healthy-calibrated 5% threshold (does DA fix FPR WITHOUT going
        # blind to damage?). 5-fold CV avoids in-sample optimism.
        recall_src = np.nan
        if len(np.unique(ys)) > 1:
            from sklearn.model_selection import StratifiedKFold
            rc = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs_a, ys):
                c = LogisticRegression(max_iter=1000).fit(Xs_a[tr], ys[tr])
                t = float(np.quantile(c.predict_proba(Xs_a[tr][ys[tr] == 0])[:, 1],
                                      1.0 - target_fpr))
                if (ys[te] == 1).sum():
                    rc.append(float((c.predict_proba(Xs_a[te][ys[te] == 1])[:, 1] > t).mean()))
            recall_src = float(np.mean(rc)) if rc else np.nan
        res[m] = {
            "fpr_target": fpr_at(St_h, thr),
            "recall_src": recall_src,
            "mmd2": da.mmd2(da.ALIGNERS[m](src_healthy, tgt_healthy),
                            da._apply_target(m, src_healthy, tgt_healthy)),
            "pad": da.proxy_a_distance(da.ALIGNERS[m](src_healthy, tgt_healthy),
                                       da._apply_target(m, src_healthy,
                                                        tgt_healthy)),
        }
    out = {"src_month": src_month, "tgt_month": tgt_month,
           "temp_src": ts, "temp_tgt": tt, "auroc_in_source": auroc_in,
           "n_src": len(ys), "n_tgt": len(yt),
           "src_damage": int(ys.sum()), "tgt_damage": int(yt.sum()),
           "target_fpr_design": target_fpr, "by_method": res}
    if verbose:
        _report(out)
    return out


def _report(o: dict):
    bar = "=" * 72
    print(bar)
    print("  PAYLOAD (B): REAL long-term OGW domain adaptation across TEMPERATURE")
    print(bar)
    print(f"  source {o['src_month']} ({o['temp_src']:.1f}C, n={o['n_src']}, "
          f"{o['src_damage']} damaged)  ->  "
          f"target {o['tgt_month']} ({o['temp_tgt']:.1f}C, n={o['n_tgt']}, "
          f"{o['tgt_damage']} damaged)")
    print(f"  proper 8x10 waveform features (d=80); in-source detectability "
          f"AUROC = {o['auroc_in_source']:.3f}")
    print(f"  metric: false-positive rate on target healthy, threshold "
          f"calibrated to {o['target_fpr_design']*100:.0f}% on source healthy\n")
    print(f"  {'align':>12}{'target FPR':>12}{'recall(src)':>12}{'MMD2':>10}{'PAD':>8}")
    base = o["by_method"]["none"]
    for m, r in o["by_method"].items():
        rec = r.get("recall_src", float("nan"))
        print(f"  {m:>12}{r['fpr_target']*100:>11.1f}%{rec*100:>11.1f}%"
              f"{r['mmd2']:>10.3f}{r['pad']:>8.2f}")
    best = min(o["by_method"].items(), key=lambda kv: kv[1]["fpr_target"])
    print(f"\n  raw temperature shift: target FPR {base['fpr_target']*100:.1f}% "
          f"(design {o['target_fpr_design']*100:.0f}%), proxy-A {base['pad']:.2f}")
    print(f"  best alignment: {best[0]} -> target FPR "
          f"{best[1]['fpr_target']*100:.1f}% (gap PAD -> {best[1]['pad']:.2f})")
    print(bar)


def make_figure(o: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # thesis style: Times系セリフ + stix数式（本文フォントと整合）
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.linewidth": 0.7})
    LAB = {"none": "none", "standardize": "z-score", "coral": "CORAL", "quantile": "quantile"}
    methods = list(o["by_method"])
    design = o["target_fpr_design"] * 100
    fpr = [o["by_method"][m]["fpr_target"] * 100 for m in methods]
    pad = [o["by_method"][m]["pad"] for m in methods]
    labels = [LAB.get(m, m) for m in methods]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    # (a) FPR: 設計値以下=緑(良), 超過=赤(悪) で良否を一目で
    cols = ["#2e7d32" if v <= design * 1.2 else "#c62828" for v in fpr]
    b = ax[0].bar(labels, fpr, color=cols, edgecolor="k", linewidth=0.5)
    ax[0].axhline(design, color="0.35", ls="--", lw=1.0,
                  label=f"design target {design:.0f}%")
    rec = [o["by_method"][m].get("recall_src", float("nan")) * 100 for m in methods]
    for r, v, rc in zip(b, fpr, rec):
        lab = f"{v:.0f}" + (f"\n(rec {rc:.0f}%)" if rc == rc else "")
        ax[0].text(r.get_x()+r.get_width()/2, v+0.6, lab, ha="center", va="bottom", fontsize=7)
    ax[0].set_ylabel("target healthy FPR [%]", fontsize=10)
    ax[0].set_title(f"(a) temperature shift {o['temp_src']:.0f}$^{{\\circ}}$C $\\to$ "
                    f"{o['temp_tgt']:.0f}$^{{\\circ}}$C", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False); ax[0].tick_params(labelsize=8.5)
    ax[0].set_ylim(0, max(fpr)*1.18); ax[0].grid(alpha=0.25, axis="y")
    # (b) cross-month distribution gap
    ax[1].bar(labels, pad, color="#37474f", edgecolor="k", linewidth=0.5)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("proxy-$\\mathcal{A}$ distance", fontsize=10)
    ax[1].set_title("(b) cross-month distribution gap", fontsize=10)
    ax[1].tick_params(labelsize=8.5); ax[1].grid(alpha=0.25, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = os.path.splitext(out_path)[0]
    fig.savefig(base + ".pdf", bbox_inches="tight")               # vector (thesis primary)
    fig.savefig(base + ".png", bbox_inches="tight", dpi=200,
                facecolor="white")                                 # RGB raster (no alpha)
    plt.close(fig)
    return base + ".pdf"


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests (synthetic; no big pickle needed)
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    rng = np.random.default_rng(0)
    gw = rng.normal(size=(5, 8, 2000))
    f = path_features(gw)
    ok(f.shape == (5, 8, 10), "path_features shape (N,8,10)")
    ok(np.isfinite(f).all(), "path_features finite")
    spike = np.zeros((1, 8, 2000)); spike[0, :, 1000] = 5.0
    fs = path_features(spike)
    ok(fs[0, 0, 1] > path_features(np.full((1, 8, 2000), 1e-3))[0, 0, 1],
       "max-amplitude feature responds to a spike")

    ok(fpr_at(np.array([0.1, 0.9, 0.95]), 0.5) == 2 / 3, "fpr_at counts exceedances")

    # synthetic month-shift through the toolkit: alignment cuts the gap
    Xs, ys, Xt, yt = da.make_shifted_domains(n=200, d=80, shift=1.3, seed=2)
    g = da.domain_gap_report(Xs, Xt, verbose=False)
    ok(any(r.mmd2_after < r.mmd2_before for r in g[1:]), "an aligner cuts MMD2")
    ok(g[0].pad_before > 0.5, "synthetic domains are separable raw")

    print(f"payload_da_longterm: {n}/{n} unit tests passed")
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Real long-term OGW domain adaptation across temperature.")
    ap.add_argument("--src-month", default="2018_03")
    ap.add_argument("--tgt-month", default="2018_07")
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(); return
    o = run(args.src_month, args.tgt_month, args.max_n, args.target_fpr)
    if args.fig:
        p = make_figure(o, os.path.join(ROOT, "results", "payload_da_longterm.png"))
        print(f"\nfigure -> {p}")


if __name__ == "__main__":
    main()
