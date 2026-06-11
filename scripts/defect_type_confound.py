#!/usr/bin/env python3
"""NET defect-TYPE identifiability after removing temporal/temperature confounds.

Background
----------
scripts/defect_type_posterior.py found a type-head AUC 0.939 on the OGW
long-term set, BUT a confound probe showed temp_only_acc 0.792 ~= majority
0.791 on 2018_03 alone (binary tags 0/1). The richer signal — the OGW
'damage tag' is a damage-STATE ID 0..13 spanning 4.5 years — is dangerous:
different damage states were introduced in different YEARS, so a naive
classifier over the whole archive just learns "which year / which temperature"
= a pure confound, NOT damage physics.

This script does an honest, causal-style decomposition:

  1. ASSEMBLE a multi-type set where >=2 damage tags CO-OCCUR within
     overlapping temperature/time, so tag is not aliased with year. We inspect
     the tag x month occurrence table (the pickles' 'damage tag' uniques) and
     pick the largest subset of (month, tag) cells in which >=2 tags appear
     under comparable temperature ranges. If no such overlap exists, we SAY SO
     explicitly — that is itself the finding (types are temporally separated,
     so type is unidentifiable from this dataset without controlling time).

  2. TEMPERATURE as an explicit covariate. On the assembled set we fit four
     classifiers and report accuracy / macro-F1 / ECE for each:
        (a) temperature-only baseline
        (b) waveform-features-only
        (c) waveform + temperature
        (d) waveform with temperature REGRESSED OUT of every feature
            (residualise each feature on a smooth temp fit, then classify).
     The gap (c or d) minus (a) = NET type signal beyond temperature.

  3. TIME-BLOCK split (not random): train on earlier dates, test on later
     dates within the same tag set, to kill leakage from temporal
     autocorrelation.

  4. HONEST VERDICT: is there residual type-discrimination once temperature
     and time are controlled? Quantified as Delta accuracy over the temp-only
     baseline with a bootstrap CI.

Figures -> results/defect_type_posterior/:
  fig_confound_heatmap.png   tag x month occurrence (shows the confound structure)
  fig_confound_conditions.png  the four conditions, multi-metric (acc/macroF1/ECE)

Usage:
  python3 scripts/defect_type_confound.py scan     # cache features+temps+tags+month for ALL months (heavy IO)
  python3 scripts/defect_type_confound.py analyze  # build set, 4 conditions, time-block, CI, figures
  python3 scripts/defect_type_confound.py all
Run on frontale via scripts/qsub_defect_type_confound.sh (pickles ~2GB; subsampled).
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from longterm_gnn import path_features  # (N,8,2000) -> (N,8,10)

OUT = os.path.join(ROOT, "results", "defect_type_posterior")
LT = os.path.join(ROOT, "data", "external", "ogw_longterm")
CACHE = os.path.join(OUT, "confound_cache.npz")        # this script's own cache
METRICS = os.path.join(OUT, "confound_metrics.json")
HEATMAP = os.path.join(OUT, "fig_confound_heatmap.png")
CONDFIG = os.path.join(OUT, "fig_confound_conditions.png")
SEED = 42
KNOWN_BAD = {"2018_05"}            # figshare source corrupt
PER_MONTH_TAG_CAP = 400           # subsample cap per (month, tag); ~2GB pickles
N_BOOT = 2000                     # bootstrap resamples for CI


# ─────────────────────────── stage 1: scan all months ──────────────────────
def scan_all():
    """Per-(month, tag) subsampled path features + temperature + tag + month
    across EVERY monthly pickle (not just 2018_03)."""
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
        except Exception as e:
            print(f"[scan] {month}: FAILED to load ({e})", flush=True)
            continue
        # full-month tag histogram (BEFORE subsampling) — drives the heatmap
        u, c = np.unique(tag, return_counts=True)
        hist[month] = {int(t): int(cc) for t, cc in zip(u, c)}
        keep = []
        for t in u:
            idx = np.where(tag == t)[0]
            if len(idx) > PER_MONTH_TAG_CAP:
                idx = rng.choice(idx, PER_MONTH_TAG_CAP, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        feats.append(path_features(gw[keep]))
        tags.append(tag[keep])
        temps.append(tp[keep])
        month_idx.append(np.full(len(keep), len(months), dtype=np.int32))
        months.append(month)
        print(f"[scan] {month}: N={len(tag)} kept={len(keep)} tags={hist[month]} "
              f"temp[{tp.min():.1f},{tp.max():.1f}]", flush=True)
        del d, gw
    feats = np.concatenate(feats)
    np.savez_compressed(
        CACHE, feats=feats, tags=np.concatenate(tags),
        temps=np.concatenate(temps), month_idx=np.concatenate(month_idx),
        months=np.array(months))
    json.dump(hist, open(os.path.join(OUT, "confound_tag_histogram.json"), "w"),
              indent=1, sort_keys=True)
    print(f"[scan] cached {feats.shape} over {len(months)} months -> {CACHE}")


# ─────────────────────────── small ML utilities ────────────────────────────
def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    e, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fit_classifier(Xtr, ytr, Xte, n_classes):
    """Multinomial logistic regression (small, stable, well-calibrated enough
    for an honest-baseline comparison). Class-balanced. Returns test probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    # expand to full class set in case a class is absent from this train fold
    P = np.zeros((len(Xte), n_classes))
    for j, c in enumerate(clf.classes_):
        P[:, c] = proba[:, j]
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    return P


def panel(probs, y, n_classes):
    from sklearn.metrics import f1_score
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0,
                                   labels=np.arange(n_classes))),
        "ece": ece(probs, y),
    }


def partial_out_temperature(X, temp, deg=3, ref_fit=None):
    """Residualise each feature column on a smooth (polynomial) fit of temp.
    ref_fit: dict of per-column poly coeffs fitted on TRAIN; if None, fit here
    and return the coeffs so TEST uses the SAME temp->feature relationship."""
    t = np.asarray(temp, dtype=np.float64)
    fits = {} if ref_fit is None else ref_fit
    R = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        if ref_fit is None:
            coef = np.polyfit(t, X[:, j], deg)
            fits[j] = coef
        else:
            coef = fits[j]
        R[:, j] = X[:, j] - np.polyval(coef, t)
    return R, fits


# ─────────────────────── stage 2: confound-controlled set ──────────────────
def build_overlap_set(tags, temps, month_idx, months, hist):
    """Find the largest set of damage tags that CO-OCCUR within overlapping
    temperature so tag is not aliased with year.

    Strategy: tag is confounded with TIME when each tag lives in disjoint
    months. We instead require, for the candidate tag pair/group, that their
    per-tag temperature distributions OVERLAP substantially (shared support).
    We score every pair of tags by (a) temperature-range overlap fraction and
    (b) presence across multiple distinct months, then greedily grow the
    largest group whose pairwise temperature overlap stays > THRESH.

    Returns (idx, used_tags, diag) where idx selects rows of the cache.
    If no >=2-tag overlap exists, used_tags is [] and diag explains why.
    """
    OVL_THRESH = 0.40  # min fraction of shared temperature support
    all_tags = np.unique(tags)
    # per-tag temperature support (5th..95th pct) and month presence
    supp, tmonths = {}, {}
    for t in all_tags:
        m = tags == t
        lo, hi = np.percentile(temps[m], 5), np.percentile(temps[m], 95)
        supp[int(t)] = (float(lo), float(hi))
        tmonths[int(t)] = sorted(set(month_idx[m].tolist()))

    def ovl(a, b):
        la, ha = supp[a]; lb, hb = supp[b]
        inter = max(0.0, min(ha, hb) - max(la, lb))
        union = max(ha, hb) - min(la, lb)
        return inter / union if union > 0 else 0.0

    # temperature overlap matrix between tags
    pair_ovl = {}
    for i, a in enumerate(all_tags):
        for b in all_tags[i + 1:]:
            pair_ovl[(int(a), int(b))] = ovl(int(a), int(b))

    # ALSO require the candidate tags to actually share months (true co-occurrence
    # of damage classes within the same measurement campaign), OR at minimum
    # share temperature support so a within-temperature contrast exists.
    shared_months = {}
    for (a, b) in pair_ovl:
        sm = set(tmonths[a]) & set(tmonths[b])
        shared_months[(a, b)] = sorted(sm)

    # greedy: seed with the highest-overlap pair, grow while pairwise ovl ok
    if not pair_ovl:
        return np.array([], int), [], {"reason": "only one tag present"}
    seed = max(pair_ovl, key=lambda k: pair_ovl[k])
    group = set(seed)
    improved = True
    while improved:
        improved = False
        best_c, best_min = None, -1
        for c in all_tags:
            c = int(c)
            if c in group:
                continue
            mins = min(pair_ovl.get((min(c, g), max(c, g)), 0.0) for g in group)
            if mins >= OVL_THRESH and mins > best_min:
                best_min, best_c = mins, c
        if best_c is not None:
            group.add(best_c); improved = True
    group = sorted(group)

    diag = {
        "tag_temperature_support_p5_p95": {str(k): v for k, v in supp.items()},
        "tag_present_in_n_months": {str(k): len(v) for k, v in tmonths.items()},
        "best_pair": list(seed),
        "best_pair_temp_overlap": pair_ovl[seed],
        "overlap_threshold": OVL_THRESH,
        "selected_tags": group,
        "pairwise_temp_overlap_within_selected": {
            f"{a}_{b}": pair_ovl[(a, b)]
            for i, a in enumerate(group) for b in group[i + 1:]},
        "shared_months_within_selected": {
            f"{a}_{b}": [months[mi] for mi in shared_months[(a, b)]]
            for i, a in enumerate(group) for b in group[i + 1:]},
    }
    if len(group) < 2 or pair_ovl[seed] < OVL_THRESH:
        diag["reason"] = (
            "No >=2 damage tags share temperature support above threshold "
            f"{OVL_THRESH}: damage states are TEMPORALLY/THERMALLY SEPARATED. "
            "Type is NOT identifiable from this archive without controlling "
            "time — the apparent type signal is the year/temperature confound.")
        return np.array([], int), [], diag

    idx = np.where(np.isin(tags, group))[0]
    diag["n_samples_selected"] = int(len(idx))
    diag["per_tag_counts_selected"] = {
        str(int(t)): int((tags[idx] == t).sum()) for t in group}
    return idx, group, diag


# ─────────────────────── stage 2/3/4: run the decomposition ─────────────────
def bootstrap_delta_ci(p_full, p_temp, y, n_classes, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI of (acc_full - acc_temp) on the SAME test rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    af = (p_full.argmax(1) == y)
    at = (p_temp.argmax(1) == y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        deltas[b] = af[s].mean() - at[s].mean()
    return {
        "delta_acc": float(af.mean() - at.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)),
                 float(np.percentile(deltas, 97.5))],
        "p_delta_le_0": float((deltas <= 0).mean()),
    }


def run_conditions(X, temp, y_raw, group, split_idx, n_classes, label):
    """Fit conditions (a)-(d) for a given train/test split.
    split_idx = (tr, te) index arrays into the rows of X/temp/y_raw."""
    tr, te = split_idx
    cmap = {int(t): i for i, t in enumerate(group)}
    y = np.array([cmap[int(t)] for t in y_raw])
    Tcol = temp.reshape(-1, 1)

    # (a) temperature-only
    pa = fit_classifier(Tcol[tr], y[tr], Tcol[te], n_classes)
    # (b) waveform-only
    pb = fit_classifier(X[tr], y[tr], X[te], n_classes)
    # (c) waveform + temperature
    Xc = np.concatenate([X, Tcol], axis=1)
    pc = fit_classifier(Xc[tr], y[tr], Xc[te], n_classes)
    # (d) waveform with temperature regressed out (fit on train, apply to test)
    Rtr, fits = partial_out_temperature(X[tr], temp[tr])
    Rte, _ = partial_out_temperature(X[te], temp[te], ref_fit=fits)
    pd_ = fit_classifier(Rtr, y[tr], Rte, n_classes)

    res = {
        "label": label,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_class_counts": {str(int(t)): int((y[te] == cmap[int(t)]).sum())
                              for t in group},
        "majority_acc": float(np.bincount(y[tr], minlength=n_classes).max() / len(tr)),
        "a_temp_only": panel(pa, y[te], n_classes),
        "b_waveform_only": panel(pb, y[te], n_classes),
        "c_waveform_plus_temp": panel(pc, y[te], n_classes),
        "d_waveform_temp_regressed_out": panel(pd_, y[te], n_classes),
    }
    # NET signal: best waveform-based condition minus temp-only baseline, with CI
    res["net_signal_c_vs_a"] = bootstrap_delta_ci(pc, pa, y[te], n_classes)
    res["net_signal_d_vs_a"] = bootstrap_delta_ci(pd_, pa, y[te], n_classes)
    res["net_signal_b_vs_a"] = bootstrap_delta_ci(pb, pa, y[te], n_classes)
    return res


def analyze():
    d = np.load(CACHE, allow_pickle=True)
    X = d["feats"].reshape(len(d["feats"]), -1)  # (M, 80)
    tags = d["tags"]
    temps = d["temps"].astype(np.float64)
    month_idx = d["month_idx"]
    months = list(d["months"])
    hist = json.load(open(os.path.join(OUT, "confound_tag_histogram.json")))

    metrics = {
        "dataset": {
            "n_months": len(months), "months": months,
            "n_samples_cached": int(len(tags)),
            "tags_present": np.unique(tags).tolist(),
            "per_month_tag_cap": PER_MONTH_TAG_CAP,
        }
    }

    # ── confound structure: tag x month occurrence ──────────────────────────
    all_tags = sorted({int(t) for m in hist for t in hist[m]})
    occ = np.zeros((len(all_tags), len(months)), dtype=int)
    for ci, m in enumerate(months):
        for t, cnt in hist.get(m, {}).items():
            occ[all_tags.index(int(t)), ci] = cnt
    metrics["confound_structure"] = {
        "tags": all_tags,
        "n_months_each_tag_appears_in": {
            str(t): int((occ[i] > 0).sum()) for i, t in enumerate(all_tags)},
        "tags_per_month": {months[ci]: [all_tags[i] for i in np.where(occ[:, ci] > 0)[0]]
                           for ci in range(len(months))},
    }

    # ── assemble the confound-controlled multi-type set ─────────────────────
    idx, group, diag = build_overlap_set(tags, temps, month_idx, months, hist)
    metrics["overlap_set"] = diag

    make_heatmap(occ, all_tags, months, group)

    if len(group) < 2:
        metrics["verdict"] = {
            "net_type_identifiable": False,
            "reason": diag.get("reason", "no >=2-tag temperature overlap"),
            "note": ("Damage tags do not co-occur under comparable temperature: "
                     "type is aliased with year/temperature. The apparent "
                     "type-head skill reported earlier is the confound, not "
                     "damage discrimination."),
        }
        json.dump(metrics, open(METRICS, "w"), indent=1)
        # still emit a conditions figure placeholder? No — heatmap tells story.
        print("[analyze] NO >=2-tag temperature overlap. Verdict: type "
              "UNIDENTIFIABLE without controlling time. See heatmap + metrics.")
        print(json.dumps(metrics["overlap_set"], indent=1))
        return

    n_classes = len(group)
    Xs, ts, ys, mis = X[idx], temps[idx], tags[idx], month_idx[idx]

    # ── (2) RANDOM split: four conditions ───────────────────────────────────
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(idx))
    cut = int(0.7 * len(perm))
    rand_split = (perm[:cut], perm[cut:])
    metrics["random_split"] = run_conditions(
        Xs, ts, ys, group, rand_split, n_classes, "random 70/30")

    # ── (3) TIME-BLOCK split: earlier months train, later months test ───────
    order = np.argsort(mis, kind="stable")
    ord_months = mis[order]
    uniq = np.unique(ord_months)
    cut_m = uniq[int(0.7 * len(uniq))] if len(uniq) > 1 else uniq[0]
    tr_tb = order[ord_months < cut_m]
    te_tb = order[ord_months >= cut_m]
    if len(tr_tb) > 5 and len(te_tb) > 5 and \
            len(np.unique(ys[tr_tb])) >= 2 and len(np.unique(ys[te_tb])) >= 2:
        metrics["time_block_split"] = run_conditions(
            Xs, ts, ys, group, (tr_tb, te_tb), n_classes,
            f"time-block: train months<{months[cut_m]}, test>=")
    else:
        metrics["time_block_split"] = {
            "skipped": True,
            "reason": ("after time-block split a fold lacks >=2 tags — the "
                       "selected tags are still largely time-separated; the "
                       "within-temperature overlap is not enough to support a "
                       "later-time test. This itself weakens identifiability."),
            "n_train": int(len(tr_tb)), "n_test": int(len(te_tb)),
            "train_tags": np.unique(ys[tr_tb]).tolist(),
            "test_tags": np.unique(ys[te_tb]).tolist(),
        }

    # ── (4) honest verdict ──────────────────────────────────────────────────
    rs = metrics["random_split"]
    tb = metrics.get("time_block_split", {})
    verdict = {
        "selected_tags": group,
        "temp_only_acc_random": rs["a_temp_only"]["accuracy"],
        "waveform_plus_temp_acc_random": rs["c_waveform_plus_temp"]["accuracy"],
        "net_delta_acc_c_vs_a_random": rs["net_signal_c_vs_a"],
        "net_delta_acc_d_vs_a_random": rs["net_signal_d_vs_a"],
    }
    net_rand = rs["net_signal_d_vs_a"]
    rand_positive = net_rand["ci95"][0] > 0
    if not tb.get("skipped"):
        net_tb = tb["net_signal_d_vs_a"]
        verdict["net_delta_acc_d_vs_a_timeblock"] = net_tb
        tb_positive = net_tb["ci95"][0] > 0
        verdict["net_type_identifiable"] = bool(rand_positive and tb_positive)
        verdict["interpretation"] = (
            "POSITIVE: residual type discrimination survives temperature "
            "partial-out AND a forward-in-time split (CI lower bound > 0 in "
            "both)." if (rand_positive and tb_positive) else
            "WEAK/NULL: temperature-residualised, time-block net signal CI "
            "includes 0 -> type is not robustly identifiable beyond the "
            "temperature/time confound.")
    else:
        verdict["net_type_identifiable"] = bool(rand_positive) and False
        verdict["interpretation"] = (
            "INCONCLUSIVE for time: random-split net signal "
            + ("> 0" if rand_positive else "includes 0") +
            ", but a forward-in-time test could not be formed (tags too "
            "time-separated) -> identifiability not demonstrated under the "
            "stricter time-block control.")
    metrics["verdict"] = verdict

    make_conditions_fig(metrics)
    json.dump(metrics, open(METRICS, "w"), indent=1)
    print(f"[analyze] metrics -> {METRICS}")
    print(json.dumps(verdict, indent=1))


# ───────────────────────────── figures ─────────────────────────────────────
def make_heatmap(occ, all_tags, months, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(10, len(months) * 0.28),
                                    max(3, len(all_tags) * 0.5)))
    show = np.log10(occ + 1)
    im = ax.imshow(show, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=90, fontsize=6)
    ax.set_yticks(range(len(all_tags)))
    ax.set_yticklabels([f"tag {t}" for t in all_tags], fontsize=8)
    ax.set_xlabel("month")
    ax.set_ylabel("damage tag")
    sel = set(selected)
    for i, t in enumerate(all_tags):
        if t in sel:
            ax.get_yticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_fontweight("bold")
    title = "OGW damage-tag x month occurrence (log10 count+1)"
    if len(sel) >= 2:
        title += f"\nconfound-controlled set (red) tags={sorted(sel)}: co-occur under overlapping temperature"
    else:
        title += "\nNO >=2-tag temperature overlap -> tag is aliased with year (type unidentifiable)"
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("log10(count+1)", fontsize=8)
    fig.tight_layout()
    fig.savefig(HEATMAP, dpi=150)
    plt.close(fig)
    print(f"[fig] heatmap -> {HEATMAP}")


def make_conditions_fig(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = ["a_temp_only", "b_waveform_only",
             "c_waveform_plus_temp", "d_waveform_temp_regressed_out"]
    clabels = ["(a) temp\nonly", "(b) waveform\nonly",
               "(c) waveform\n+ temp", "(d) waveform\ntemp-regressed-out"]
    blocks = [("random_split", "Random 70/30 split"),
              ("time_block_split", "Time-block split (train past, test future)")]
    blocks = [(k, t) for k, t in blocks
              if k in metrics and not metrics[k].get("skipped")]
    metric_keys = [("accuracy", "accuracy"), ("macro_f1", "macro-F1"),
                   ("ece", "ECE (lower=better)")]
    fig, axes = plt.subplots(len(blocks), 3,
                             figsize=(15, 4.2 * len(blocks)), squeeze=False)
    colors = ["#bbbbbb", "#4c72b0", "#dd8452", "#55a868"]
    for r, (bkey, btitle) in enumerate(blocks):
        blk = metrics[bkey]
        maj = blk.get("majority_acc")
        for cj, (mk, mlab) in enumerate(metric_keys):
            ax = axes[r][cj]
            vals = [blk[c][mk] for c in conds]
            ax.bar(range(4), vals, color=colors)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            if mk == "accuracy" and maj is not None:
                ax.axhline(maj, color="k", ls="--", lw=1,
                           label=f"majority {maj:.3f}")
                ax.legend(fontsize=8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(clabels, fontsize=7)
            ax.set_title(f"{btitle}\n{mlab}", fontsize=9)
            if mk != "ece":
                ax.set_ylim(0, 1)
        # annotate net signal on the accuracy axis
        ns = blk.get("net_signal_d_vs_a", {})
        if ns:
            axes[r][0].text(
                0.5, -0.28,
                f"NET (d)-(a) Δacc={ns['delta_acc']:+.3f} "
                f"CI95[{ns['ci95'][0]:+.3f},{ns['ci95'][1]:+.3f}]",
                transform=axes[r][0].transAxes, ha="center", fontsize=8,
                color="darkred")
    fig.suptitle("Defect-type identifiability vs temperature/time confound — "
                 "four conditions, multi-metric (no accuracy-only)", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(CONDFIG, dpi=150)
    plt.close(fig)
    print(f"[fig] conditions -> {CONDFIG}")


def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["scan", "analyze", "all"])
    a = ap.parse_args()
    if a.stage in ("scan", "all"):
        if a.stage == "all" and os.path.exists(CACHE):
            print(f"[scan] cache exists, skipping ({CACHE})")
        else:
            scan_all()
    if a.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
