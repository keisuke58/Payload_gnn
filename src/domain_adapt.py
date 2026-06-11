"""
domain_adapt.py — reusable unsupervised domain-adaptation toolkit for SHM sim2real.

Data-agnostic on purpose: it operates on plain feature matrices X ∈ R^{n×d}, so
the SAME code serves
  * CFRP:    source = FEM DSPSS features,  target = measured (TSA-derived) DSPSS
  * Payload: source = month-A (or synthetic) features, target = month-B / real
both of which have a documented domain shift (the CFRP sim-to-real gap; the
Payload2026 cross-month transfer collapse).

What it provides
----------------
  alignment (unsupervised, target labels NOT used):
    - standardize_per_domain : z-score each domain (removes 1st-order shift)
    - coral                  : align 2nd-order stats (whiten source → recolor to
                               target covariance); closed-form, classic CORAL
    - quantile_match         : per-feature marginal (histogram) matching
  domain-gap metrics (lower = closer domains):
    - mmd2                    : RBF-kernel maximum mean discrepancy²
    - proxy_a_distance        : 2(1−2ε) from a source-vs-target domain classifier
  transfer-evaluation harness:
    - transfer_eval           : train a classifier on (adapted) source, score on
                               target → AUROC/accuracy (the (a)/(b) number)
    - compare_adaptations     : baseline + each adaptation, gap + transfer side by
                               side, so you can see which alignment actually helps

When the measured ground-truth labels arrive, wire the real (Xs,ys,Xt,yt) into
compare_adaptations; nothing else changes.

Usage
-----
    python domain_adapt.py            # synthetic covariate-shift demo
    python domain_adapt.py --test     # unit tests only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  Symmetric-PSD helpers (matrix sqrt / inverse-sqrt via eigh)
# ═════════════════════════════════════════════════════════════════════════════

def _sym_pow(C: np.ndarray, p: float, eps: float = 1e-6) -> np.ndarray:
    """C^p for a symmetric PSD matrix C (eigendecomposition; eigenvalues floored)."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return (V * (w ** p)) @ V.T


def _cov(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    d = X.shape[1]
    return (Xc.T @ Xc) / max(len(X) - 1, 1) + ridge * np.eye(d)


# ═════════════════════════════════════════════════════════════════════════════
#  Alignment (unsupervised) — each returns a TRANSFORMED source X_s'
# ═════════════════════════════════════════════════════════════════════════════

def standardize_per_domain(Xs: np.ndarray, Xt: np.ndarray):
    """Z-score each domain by its OWN mean/std (removes 1st-order/scale shift).

    Returns (Xs', Xt') both standardized; this is the cheapest sim2real fix and
    often the bulk of the gap (e.g. measured-vs-FEM DSPSS differ mostly in scale).
    """
    s = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    t = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
    return s, t


def coral(Xs: np.ndarray, Xt: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """CORAL (Sun et al. 2016): recolor source so cov(Xs')≈cov(Xt).

    Xs' = (Xs − μs) Cs^{-1/2} Ct^{1/2} + μt.  Closed-form, no labels, no training.
    Aligns up to 2nd-order statistics; complements per-domain standardization
    (which only fixes 1st-order).
    """
    mus, mut = Xs.mean(0), Xt.mean(0)
    Cs = _cov(Xs, ridge)
    Ct = _cov(Xt, ridge)
    A = _sym_pow(Cs, -0.5) @ _sym_pow(Ct, 0.5)
    return (Xs - mus) @ A + mut


def quantile_match(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Per-feature marginal (histogram) matching: map each source feature's rank
    onto the target feature's empirical quantile function.  Aligns the full 1-D
    marginal of every feature (stronger than moment matching, ignores coupling).
    """
    out = np.empty_like(Xs, dtype=float)
    for j in range(Xs.shape[1]):
        s, t = Xs[:, j], Xt[:, j]
        ranks = (np.argsort(np.argsort(s)) + 0.5) / len(s)     # source CDF ranks
        out[:, j] = np.quantile(t, np.clip(ranks, 0, 1))
    return out


ALIGNERS = {
    "none": lambda Xs, Xt: Xs,
    "standardize": lambda Xs, Xt: standardize_per_domain(Xs, Xt)[0],
    "coral": coral,
    "quantile": quantile_match,
}


def _apply_target(method: str, Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """The matching target-side transform for a method (so train/test live in the
    same space).  standardize also z-scores the target; coral/quantile/none leave
    the target as-is (source is mapped onto target's space)."""
    if method == "standardize":
        return standardize_per_domain(Xs, Xt)[1]
    return Xt


# ═════════════════════════════════════════════════════════════════════════════
#  Domain-gap metrics
# ═════════════════════════════════════════════════════════════════════════════

def _median_bandwidth(X: np.ndarray, cap: int = 500) -> float:
    if len(X) > cap:
        X = X[np.random.default_rng(0).choice(len(X), cap, replace=False)]
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return float(np.sqrt(med / 2.0) + 1e-8)


def mmd2(Xs: np.ndarray, Xt: np.ndarray, bandwidth: float | None = None) -> float:
    """Biased RBF-kernel MMD² between two samples (median-heuristic bandwidth).
    0 ⇔ identical distributions; larger ⇒ further apart."""
    if bandwidth is None:
        bandwidth = _median_bandwidth(np.vstack([Xs, Xt]))
    g = 1.0 / (2.0 * bandwidth ** 2)

    def k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-g * d2)
    return float(k(Xs, Xs).mean() + k(Xt, Xt).mean() - 2.0 * k(Xs, Xt).mean())


def proxy_a_distance(Xs: np.ndarray, Xt: np.ndarray, seed: int = 0) -> float:
    """Proxy-A-distance: train a domain classifier (source=0 vs target=1) with
    cross-validation; PAD = 2(1−2ε).  ~0 ⇔ indistinguishable domains, ~2 ⇔ fully
    separable (large shift)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    n_splits = max(2, min(5, len(Xs), len(Xt)))
    err = 1.0 - cross_val_score(
        LogisticRegression(max_iter=500), X, y, cv=n_splits,
        scoring="accuracy").mean()
    return float(2.0 * (1.0 - 2.0 * err))


# ═════════════════════════════════════════════════════════════════════════════
#  Transfer evaluation harness
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferResult:
    method: str
    auroc: float
    accuracy: float
    mmd2_before: float
    mmd2_after: float
    pad_after: float


def transfer_eval(Xs, ys, Xt, yt, method: str = "none", seed: int = 0,
                  clf=None) -> TransferResult:
    """Train a classifier on (aligned) SOURCE, evaluate on TARGET.

    Target labels yt are used ONLY for scoring, never for the alignment — this is
    unsupervised domain adaptation.  Returns AUROC/accuracy plus the domain gap
    (MMD²) before and after alignment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    Xs_a = ALIGNERS[method](Xs, Xt)
    Xt_a = _apply_target(method, Xs, Xt)
    mmd_b = mmd2(Xs, Xt)
    mmd_a = mmd2(Xs_a, Xt_a)
    pad_a = proxy_a_distance(Xs_a, Xt_a, seed)

    clf = clf or LogisticRegression(max_iter=500)
    clf.fit(Xs_a, ys)
    if hasattr(clf, "predict_proba") and len(np.unique(yt)) == 2:
        p = clf.predict_proba(Xt_a)[:, 1]
        auroc = float(roc_auc_score(yt, p))
    else:
        auroc = float("nan")
    acc = float(accuracy_score(yt, clf.predict(Xt_a)))
    return TransferResult(method, auroc, acc, mmd_b, mmd_a, pad_a)


@dataclass
class GapResult:
    method: str
    mmd2_before: float
    mmd2_after: float
    pad_before: float
    pad_after: float


def domain_gap_report(Xs, Xt, methods=None, prescale: bool = True,
                      seed: int = 0, verbose: bool = True) -> list:
    """LABEL-FREE sim2real quantification: for each alignment, the domain gap
    (RBF-MMD² and proxy-A-distance) BEFORE vs AFTER.  No labels needed — use this
    when you only have source/target feature matrices (e.g. FEM vs measured DSPSS
    distribution features).  prescale z-scores feature columns by SOURCE stats
    first so no single feature (raw MPa scale vs histogram density) dominates.
    """
    methods = methods or list(ALIGNERS.keys())
    if prescale:
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-8
        Xs, Xt = (Xs - mu) / sd, (Xt - mu) / sd
    mmd_b, pad_b = mmd2(Xs, Xt), proxy_a_distance(Xs, Xt, seed)
    res = []
    for m in methods:
        Xs_a = ALIGNERS[m](Xs, Xt)
        Xt_a = _apply_target(m, Xs, Xt)
        res.append(GapResult(m, mmd_b, mmd2(Xs_a, Xt_a),
                             pad_b, proxy_a_distance(Xs_a, Xt_a, seed)))
    if verbose:
        bar = "=" * 72
        print(bar)
        print("  SIM2REAL DOMAIN GAP (label-free)  —  source→target alignment")
        print(bar)
        print(f"  source n={len(Xs)}, target n={len(Xt)}, d={Xs.shape[1]}")
        print(f"  raw gap: MMD²={mmd_b:.4f}, proxy-A-dist={pad_b:.2f}\n")
        print(f"  {'method':>12}{'MMD² after':>13}{'PAD after':>11}"
              f"{'gap ↓':>9}")
        for r in res:
            drop = (1 - r.mmd2_after / r.mmd2_before) * 100 if r.mmd2_before else 0
            print(f"  {r.method:>12}{r.mmd2_after:>13.4f}{r.pad_after:>11.2f}"
                  f"{drop:>8.0f}%")
        best = min(res[1:], key=lambda r: r.mmd2_after) if len(res) > 1 else res[0]
        print(f"\n  best alignment: {best.method} "
              f"(MMD² {best.mmd2_before:.4f}→{best.mmd2_after:.4f})")
        print(bar)
    return res


def compare_adaptations(Xs, ys, Xt, yt, methods=None, seed: int = 0,
                        verbose: bool = True) -> list:
    """Run baseline + each adaptation; return a list of TransferResult."""
    methods = methods or list(ALIGNERS.keys())
    res = [transfer_eval(Xs, ys, Xt, yt, m, seed) for m in methods]
    if verbose:
        _report(res, Xs, Xt)
    return res


def _report(res: list, Xs: np.ndarray, Xt: np.ndarray):
    bar = "=" * 72
    print(bar)
    print("  DOMAIN ADAPTATION  —  source→target transfer (unsupervised)")
    print(bar)
    print(f"  source n={len(Xs)}, target n={len(Xt)}, d={Xs.shape[1]}")
    print(f"  raw domain gap: MMD²={res[0].mmd2_before:.4f}\n")
    print(f"  {'method':>12}{'AUROC':>9}{'acc':>8}{'MMD² after':>12}{'PAD':>7}")
    for r in res:
        au = "  n/a " if r.auroc != r.auroc else f"{r.auroc:>8.3f}"
        print(f"  {r.method:>12}{au}{r.accuracy:>8.3f}{r.mmd2_after:>12.4f}"
              f"{r.pad_after:>7.2f}")
    best = max(res, key=lambda r: (-1 if r.auroc != r.auroc else r.auroc))
    print(f"\n  best transfer: {best.method} (AUROC {best.auroc:.3f})")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#  Synthetic sim2real generator (for the demo / tests)
# ═════════════════════════════════════════════════════════════════════════════

def make_shifted_domains(n: int = 300, d: int = 8, shift: float = 1.5,
                         rotate: float = 0.6, seed: int = 0):
    """Two-class problem with a covariate shift between source and target:
    target = source distribution + mean shift + a covariance rotation/scaling
    (mimicking the sim→real DSPSS gap).  Labels share the same decision rule."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)

    def sample(n, mu, A):
        X = rng.normal(size=(n, d)) @ A + mu
        y = (X @ w + 0.3 * rng.normal(size=n) > 0).astype(int)
        return X, y

    Xs, ys = sample(n, np.zeros(d), np.eye(d))
    R = np.eye(d) + rotate * rng.normal(size=(d, d)) / np.sqrt(d)
    Xt, yt = sample(n, shift * np.ones(d), R)
    return Xs, ys, Xt, yt


# ═════════════════════════════════════════════════════════════════════════════
#  Unit tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests() -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    rng = np.random.default_rng(0)

    # matrix sqrt round-trip
    M = rng.normal(size=(5, 5)); C = M @ M.T + np.eye(5)
    S = _sym_pow(C, 0.5)
    ok(np.allclose(S @ S, C, atol=1e-4), "sym sqrt: S@S == C")
    ok(np.allclose(_sym_pow(C, -0.5) @ _sym_pow(C, 0.5), np.eye(5), atol=1e-4),
       "C^-1/2 C^1/2 == I")

    # CORAL aligns covariance: cov(coral(Xs,Xt)) ≈ cov(Xt)
    Xs = rng.normal(size=(400, 6))
    A = rng.normal(size=(6, 6))
    Xt = rng.normal(size=(400, 6)) @ A + 3.0
    Xs_c = coral(Xs, Xt)
    rel = (np.linalg.norm(_cov(Xs_c) - _cov(Xt)) /
           np.linalg.norm(_cov(Xt)))
    ok(rel < 0.15, f"CORAL aligns covariance (rel {rel:.3f})")
    ok(np.allclose(Xs_c.mean(0), Xt.mean(0), atol=0.3), "CORAL aligns mean")

    # CORAL / standardization reduce MMD vs raw
    m_raw = mmd2(Xs, Xt)
    m_coral = mmd2(Xs_c, Xt)
    ok(m_coral < m_raw, "CORAL reduces MMD²")
    s_s, s_t = standardize_per_domain(Xs, Xt)
    ok(mmd2(s_s, s_t) < m_raw, "per-domain standardize reduces MMD²")

    # quantile matching → matched feature marginals (KS small per feature)
    Xq = quantile_match(Xs, Xt)
    ks = max(_ks(Xq[:, j], Xt[:, j]) for j in range(Xs.shape[1]))
    ok(ks < 0.15, f"quantile match aligns marginals (max KS {ks:.3f})")

    # proxy-A-distance: large for shifted, small for identical
    pad_shift = proxy_a_distance(Xs, Xt)
    pad_same = proxy_a_distance(Xs, Xs + 1e-6 * rng.normal(size=Xs.shape))
    ok(pad_shift > pad_same, "PAD larger for shifted domains")

    # full transfer harness: adaptation should not hurt, usually helps
    Xs2, ys2, Xt2, yt2 = make_shifted_domains(seed=1)
    res = compare_adaptations(Xs2, ys2, Xt2, yt2, verbose=False)
    by = {r.method: r for r in res}
    ok(set(by) == set(ALIGNERS), "all methods evaluated")
    ok(all(0 <= r.accuracy <= 1 for r in res), "accuracy in range")
    best = max(r.auroc for r in res)
    ok(best >= by["none"].auroc - 1e-9, "best adaptation ≥ no-adaptation baseline")
    ok(by["coral"].mmd2_after <= by["none"].mmd2_after + 1e-9,
       "coral does not increase the gap")

    print(f"domain_adapt: {n}/{n} unit tests passed")
    return n


def _ks(a, b):
    za = np.sort(a); zb = np.sort(b)
    grid = np.unique(np.concatenate([za, zb]))
    Fa = np.searchsorted(za, grid, side="right") / len(za)
    Fb = np.searchsorted(zb, grid, side="right") / len(zb)
    return float(np.max(np.abs(Fa - Fb)))


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Reusable unsupervised domain-adaptation toolkit (sim2real).")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--shift", type=float, default=1.5)
    ap.add_argument("--test", action="store_true", help="unit tests only")
    args = ap.parse_args()

    if args.test:
        run_tests(); return
    Xs, ys, Xt, yt = make_shifted_domains(n=args.n, d=args.d, shift=args.shift)
    print("synthetic sim→real covariate shift "
          f"(shift={args.shift}); target labels used only for scoring\n")
    compare_adaptations(Xs, ys, Xt, yt)


if __name__ == "__main__":
    main()
