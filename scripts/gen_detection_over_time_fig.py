#!/usr/bin/env python3
"""Figure: damage detection over time (recall & FPR per month), compensation ON vs OFF.

Reads results/ogw/{detection_over_time.json (A, supervised multi-year),
novelty_oneclass.json (B', one-class)} and renders a 2x2 panel: rows = experiment (A / B'),
left = recall on damaged months, right = FPR on healthy months. Solid = compensation ON,
dashed = OFF. X axis = chronological month index.

Run: LD_LIBRARY_PATH=$HOME/miniconda3/lib python3 scripts/gen_detection_over_time_fig.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300, "figure.dpi": 120})
HERE = os.path.dirname(os.path.abspath(__file__))
OGW = os.path.join(HERE, "..", "results", "ogw")
SPECS = [("detection_over_time.json", "A: supervised, train 2018-2019"),
         ("novelty_oneclass.json", "B': one-class healthy-only")]


STYLE = {"OFF": ("o--", "#d62728", "no compensation"),
         "AMP": ("^-", "#2ca02c", "amplitude-norm only"),
         "FULL": ("s-", "#1f77b4", "amp + temp-regress"),
         "ON": ("s-", "#1f77b4", "compensation ON")}


def series(rows, key):
    x, v = [], []
    for i, r in enumerate(rows):
        if r.get(key) is not None:
            x.append(i); v.append(r[key])
    return np.array(x), np.array(v)


fig, axes = plt.subplots(len(SPECS), 2, figsize=(9.8, 5.4), squeeze=False)
for row, (fn, title) in enumerate(SPECS):
    p = os.path.join(OGW, fn)
    axL, axR = axes[row]
    if not os.path.exists(p):
        for ax in (axL, axR):
            ax.text(0.5, 0.5, f"{fn}\n(not yet run)", ha="center", va="center", color="gray")
        continue
    res = json.load(open(p))
    conds = [c for c in ("OFF", "AMP", "FULL", "ON") if c in res]
    months = [r["month"][2:] for r in res[conds[0]]["rows"]]
    for ax, key, ylab in [(axL, "recall", "Recall (damaged months)"),
                          (axR, "fpr", "FPR (healthy months)")]:
        for c in conds:
            mk, col, lab = STYLE[c]
            x, v = series(res[c]["rows"], key)
            ax.plot(x, v, mk, color=col, ms=3.5, lw=1.2, label=lab)
        ax.set_ylabel(ylab, fontsize=8); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(len(months))); ax.set_xticklabels(months, rotation=90, fontsize=5)
        ax.grid(alpha=0.25, lw=0.4)
        if key == "fpr":
            ax.axhline(0.10, ls=":", color="red", lw=0.8, alpha=0.6); ax.set_ylim(-0.01, 1.03)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axL.set_title(title, fontsize=9, loc="left")
    axL.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=1)
fig.suptitle("Damage detection over time: environmental compensation ON vs OFF", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OGW, "fig_detection_over_time.png")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
