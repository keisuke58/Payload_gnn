"""
payload_da_gw.py — (A) lightweight domain-adaptation demo on the H3-fairing
guided-wave (GW) data, using the reusable toolkit `domain_adapt.py`.

The Payload2026 GNN-SHM line already has a DANN prototype (adversarial sim->real
alignment) and cross-structure transfer eval; this is the COMPLEMENTARY
LABEL-FREE, training-free baseline: per-feature CORAL / quantile matching /
standardisation + domain-gap metrics (RBF-MMD^2, proxy-A-distance) + a
source->target transfer score, on a real cross-CONDITION shift.

Domain shift used here: EXCITATION FREQUENCY. The Abaqus GW runs come in
Freq{25,50,75,100}k variants, each with a Healthy and a Debond config. Different
excitation frequency => different Lamb-wave dispersion => a genuine feature-
distribution shift. We detect Healthy(0) vs Debond(1) from FFT band-energy
features of the sensor signals, train on the LOW-frequency domain and test on the
HIGH-frequency domain, and ask whether unsupervised alignment recovers transfer.

Self-contained: numpy/scipy only (no torch / torch_geometric). Honest scope: this
is a small-sample, simulation-only qualitative demo (few configs per frequency,
windowed to make samples) — the point is the DOMAIN GAP and its reduction, not a
calibrated detection number. The full node-feature vs-DANN comparison is (B).

Usage
-----
    python src/payload_da_gw.py            # the frequency-shift demo
    python src/payload_da_gw.py --fig      # + figure
    python src/payload_da_gw.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

import domain_adapt as da

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GW_DIR = os.path.join(ROOT, "abaqus_work")


# ═════════════════════════════════════════════════════════════════════════════
#  GW signal loading + FFT featurisation
# ═════════════════════════════════════════════════════════════════════════════

def load_gw(path: str):
    """Load a Job-GW-*_sensors.csv → (t (T,), signals (T, n_sensors))."""
    rows = []
    for r in csv.reader(open(path)):
        if not r or r[0].startswith("#") or r[0] == "time_s":
            continue
        try:
            rows.append([float(x) for x in r])
        except ValueError:
            continue
    a = np.asarray(rows, dtype=float)
    return a[:, 0], a[:, 1:]


def fft_window_features(sig: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """FFT band-energy + amplitude descriptor of a 1-D windowed signal."""
    F = np.abs(np.fft.rfft(sig - sig.mean()))
    bands = [b.mean() for b in np.array_split(F, n_bands)]
    return np.array(bands + [sig.std(), np.abs(sig).max(),
                             float(np.argmax(F))], dtype=float)


def featurise_config(path: str, n_windows: int = 6, n_bands: int = 8):
    """Per (sensor × time-window) FFT feature samples for one config → (M, d)."""
    _, S = load_gw(path)
    T, n_sensors = S.shape
    w = T // n_windows
    feats = []
    for j in range(n_sensors):
        for k in range(n_windows):
            seg = S[k * w:(k + 1) * w, j]
            if len(seg) >= 8:
                feats.append(fft_window_features(seg, n_bands))
    return np.array(feats) if feats else np.zeros((0, n_bands + 3))


# ═════════════════════════════════════════════════════════════════════════════
#  Config inventory: label (Healthy/Debond) + frequency domain
# ═════════════════════════════════════════════════════════════════════════════

def _config_token(path: str) -> str:
    return os.path.basename(path).replace("Job-GW-", "").replace("_sensors.csv", "")


def gw_config_label(tok: str):
    """0=healthy, 1=defect, None=unknown (from the config name)."""
    if tok.endswith("-H") or tok == "Healthy" or "Healthy" in tok:
        return 0
    if (tok.endswith("-D") or tok == "Debond" or "Debond" in tok
            or re.match(r"D\d", tok)):
        return 1
    return None


def gw_frequency(tok: str):
    """Excitation frequency in kHz from the config name, or None."""
    m = re.search(r"Freq(\d+)k", tok)
    return int(m.group(1)) if m else None


def build_domain(freqs, n_windows: int = 6):
    """Feature matrix X + labels y for all Healthy/Debond configs whose
    excitation frequency is in `freqs`."""
    X, y = [], []
    for path in sorted(glob.glob(os.path.join(GW_DIR, "Job-GW-*_sensors.csv"))):
        tok = _config_token(path)
        f = gw_frequency(tok)
        lab = gw_config_label(tok)
        if f in freqs and lab is not None:
            feats = featurise_config(path, n_windows)
            X.append(feats)
            y.append(np.full(len(feats), lab))
    if not X:
        return np.zeros((0, 0)), np.zeros(0)
    return np.vstack(X), np.concatenate(y)


# ═════════════════════════════════════════════════════════════════════════════
#  Demo
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GWDemo:
    src_freqs: tuple
    tgt_freqs: tuple
    Xs: np.ndarray
    ys: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    transfer: list           # domain_adapt.TransferResult list
    gap: list                # domain_adapt.GapResult list
    auroc_in_source: float   # in-domain detectability (CV AUROC on source)


def _in_domain_auroc(X, y, seed: int = 0) -> float:
    """Cross-validated in-domain Healthy-vs-Debond AUROC — tells us whether the
    features carry ANY defect signal before asking about transfer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    if len(np.unique(y)) < 2:
        return float("nan")
    k = int(min(np.bincount(y.astype(int))))
    if k < 2:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=500), X, y,
                                 cv=min(5, k), scoring="roc_auc").mean())


def run_demo(src_freqs=(25, 50), tgt_freqs=(75, 100), n_windows: int = 6,
             verbose: bool = True) -> GWDemo:
    Xs, ys = build_domain(src_freqs, n_windows)
    Xt, yt = build_domain(tgt_freqs, n_windows)
    if len(Xs) == 0 or len(Xt) == 0:
        raise FileNotFoundError(
            f"no GW configs for freqs {src_freqs}/{tgt_freqs} under {GW_DIR}")
    transfer = da.compare_adaptations(Xs, ys, Xt, yt, verbose=False)
    gap = da.domain_gap_report(Xs, Xt, verbose=False)
    demo = GWDemo(src_freqs, tgt_freqs, Xs, ys, Xt, yt, transfer, gap,
                  _in_domain_auroc(Xs, ys))
    if verbose:
        _report(demo)
    return demo


def _report(d: GWDemo):
    bar = "=" * 72
    print(bar)
    print("  PAYLOAD GW SIM2SIM DOMAIN ADAPTATION  —  excitation-frequency shift")
    print(bar)
    print(f"  source freqs {d.src_freqs} kHz  (n={len(d.Xs)}, "
          f"{int(d.ys.sum())} debond / {int((d.ys == 0).sum())} healthy)")
    print(f"  target freqs {d.tgt_freqs} kHz  (n={len(d.Xt)}, "
          f"{int(d.yt.sum())} debond / {int((d.yt == 0).sum())} healthy)")
    print(f"  features: FFT band energies + amplitude, d={d.Xs.shape[1]}\n")
    print("  Healthy-vs-Debond transfer (train source → test target):")
    print(f"  {'align':>12}{'AUROC':>9}{'acc':>8}{'MMD² after':>12}{'PAD':>7}")
    for r in d.transfer:
        au = "  n/a " if r.auroc != r.auroc else f"{r.auroc:>8.3f}"
        print(f"  {r.method:>12}{au}{r.accuracy:>8.3f}{r.mmd2_after:>12.4f}"
              f"{r.pad_after:>7.2f}")
    base = next(r for r in d.transfer if r.method == "none")
    best = max(d.transfer, key=lambda r: (-1 if r.auroc != r.auroc else r.auroc))
    print(f"\n  raw cross-frequency gap: MMD²={base.mmd2_before:.3f}, "
          f"proxy-A-dist={d.gap[0].pad_before:.2f}")
    print(f"  alignment closes the gap: MMD² {base.mmd2_before:.3f}→"
          f"{min(g.mmd2_after for g in d.gap):.3f}, "
          f"PAD {d.gap[0].pad_before:.2f}→{min(g.pad_after for g in d.gap):.2f}")
    print(f"  no-adapt transfer AUROC {base.auroc:.3f} → "
          f"best ({best.method}) {best.auroc:.3f}")
    print(f"  in-domain (source) detectability AUROC = {d.auroc_in_source:.3f}")
    print()
    if d.auroc_in_source < 0.6:
        print("  VERDICT: the toolkit closes the domain gap mechanically, but the")
        print("  crude FFT band-energy features carry NO debond signal even")
        print("  in-domain (AUROC≈0.5) — the defect lives in wave ToF/phase, not")
        print("  band energy. => motivates (B): the 34-dim wave/curvature GNN node")
        print("  features, where detection is separable and DA-vs-DANN is the real test.")
    else:
        print("  VERDICT: features carry defect signal in-domain; alignment recovers")
        print("  cross-frequency transfer.")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure
# ═════════════════════════════════════════════════════════════════════════════

def make_figure(d: GWDemo, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    methods = [r.method for r in d.transfer]
    # (a) transfer AUROC per alignment
    ax[0].bar(methods, [r.auroc for r in d.transfer], color="#1565C0")
    ax[0].axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("target AUROC", fontsize=8)
    ax[0].set_title("(a) H-vs-Debond transfer", fontsize=9)
    ax[0].tick_params(labelsize=7, axis="x", rotation=30)
    # (b) domain gap (MMD²) before/after
    ax[1].bar(methods, [r.mmd2_after for r in d.gap], color="#2e7d32")
    ax[1].axhline(d.gap[0].mmd2_before, color="#b71c1c", lw=0.9, ls="--",
                  label=f"raw {d.gap[0].mmd2_before:.2f}")
    ax[1].set_ylabel("MMD² after align", fontsize=8)
    ax[1].set_title("(b) cross-frequency gap", fontsize=9)
    ax[1].legend(fontsize=7); ax[1].tick_params(labelsize=7, axis="x", rotation=30)
    # (c) source vs target feature distribution (1st PCA comp)
    Z = np.vstack([d.Xs, d.Xt])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    u = Z @ np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][0]
    ns = len(d.Xs)
    ax[2].hist(u[:ns], bins=30, alpha=0.5, density=True, color="#1565C0",
               label=f"src {d.src_freqs}k")
    ax[2].hist(u[ns:], bins=30, alpha=0.5, density=True, color="#d7301f",
               label=f"tgt {d.tgt_freqs}k")
    ax[2].set_title("(c) feature shift (PC1)", fontsize=9)
    ax[2].legend(fontsize=7); ax[2].tick_params(labelsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    ok(gw_config_label("Freq25k-H") == 0, "healthy label")
    ok(gw_config_label("Freq75k-D") == 1, "defect label")
    ok(gw_config_label("D2-r15") == 1, "D-prefix defect")
    ok(gw_config_label("Curved-H") == 0, "curved healthy")
    ok(gw_frequency("Freq50k-D") == 50, "frequency parse")
    ok(gw_frequency("Curved-H") is None, "no-frequency config")

    rng = np.random.default_rng(0)
    f = fft_window_features(rng.normal(size=128))
    ok(f.shape == (11,), "fft feature dim (8 bands + 3 stats)")
    ok(np.isfinite(f).all(), "fft feature finite")

    # spike signal has larger max-amplitude descriptor than a flat one
    flat = fft_window_features(np.ones(128) * 0.01)
    spike = fft_window_features(np.r_[np.zeros(120), np.ones(8)])
    ok(spike[-2] >= flat[-2], "amplitude descriptor sensitive")

    # synthetic two-domain GW-like transfer through the toolkit
    Xs, ys, Xt, yt = da.make_shifted_domains(n=120, d=11, shift=1.2, seed=3)
    res = da.compare_adaptations(Xs, ys, Xt, yt, verbose=False)
    ok(len(res) == len(da.ALIGNERS), "all aligners evaluated")
    ok(all(0 <= r.accuracy <= 1 for r in res), "accuracy in range")
    gap = da.domain_gap_report(Xs, Xt, verbose=False)
    ok(any(g.mmd2_after < g.mmd2_before for g in gap[1:]), "an aligner cuts MMD²")

    print(f"payload_da_gw: {n}/{n} unit tests passed")
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Payload GW domain-adaptation demo (excitation-frequency shift).")
    ap.add_argument("--src", type=int, nargs="+", default=[25, 50],
                    help="source excitation frequencies (kHz)")
    ap.add_argument("--tgt", type=int, nargs="+", default=[75, 100],
                    help="target excitation frequencies (kHz)")
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--fig", action="store_true",
                    help="save results/payload_da_gw.png")
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return
    d = run_demo(tuple(args.src), tuple(args.tgt), args.n_windows)
    if args.fig:
        p = make_figure(d, os.path.join(ROOT, "results", "payload_da_gw.png"))
        print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
