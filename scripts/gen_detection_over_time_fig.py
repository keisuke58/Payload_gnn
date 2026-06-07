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


def series(rows, key):
    x, v, lab = [], [], []
    for i, r in enumerate(rows):
        if r.get(key) is not None:
            x.append(i); v.append(r[key]); lab.append(r["month"][2:])
    return np.array(x), np.array(v), lab


fig, axes = plt.subplots(len(SPECS), 2, figsize=(9.5, 5.2), squeeze=False)
for row, (fn, title) in enumerate(SPECS):
    p = os.path.join(OGW, fn)
    axL, axR = axes[row]
    if not os.path.exists(p):
        for ax in (axL, axR):
            ax.text(0.5, 0.5, f"{fn}\n(not yet run)", ha="center", va="center", color="gray")
        continue
    res = json.load(open(p))
    off, on = res["OFF"]["rows"], res["ON"]["rows"]
    months = [r["month"][2:] for r in off]
    for ax, key, ylab in [(axL, "recall", "Recall (damaged months)"),
                          (axR, "fpr", "FPR (healthy months)")]:
        xo, vo, _ = series(off, key); xn, vn, _ = series(on, key)
        ax.plot(xo, vo, "o--", color="#d62728", ms=4, lw=1.2, label="compensation OFF")
        ax.plot(xn, vn, "s-", color="#1f77b4", ms=4, lw=1.4, label="compensation ON")
        ax.set_ylabel(ylab, fontsize=8); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(len(months))); ax.set_xticklabels(months, rotation=90, fontsize=5)
        ax.grid(alpha=0.25, lw=0.4)
        if key == "fpr":
            ax.axhline(0.10, ls=":", color="red", lw=0.8, alpha=0.6); ax.set_ylim(-0.01, 0.55)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axL.set_title(title, fontsize=9, loc="left")
    if row == 0:
        axL.legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("Damage detection over time: environmental compensation ON vs OFF", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OGW, "fig_detection_over_time.png")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
