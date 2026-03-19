#!/usr/bin/env python3
"""
Plot separation results summary for portfolio.

Generates three figures from fairing separation FEM + GNN-SHM results:
  1. fig_separation_displacement.png — Z-displacement time history (Normal / Stuck3 / Stuck6)
  2. fig_separation_stress.png       — von Mises stress comparison across cases
  3. fig_gnn_summary.png             — GNN anomaly detection summary (AUC bar chart + metrics)

Usage:
    python3 scripts/plot_separation_results.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fall back to default

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "separation")

CSV_FILES = {
    "Normal": os.path.join(RESULTS_DIR, "Sep-v2-Normal_time_history.csv"),
    "Stuck3": os.path.join(RESULTS_DIR, "Sep-v2-Stuck3_time_history.csv"),
    "Stuck6": os.path.join(RESULTS_DIR, "Sep-v2-Stuck6_time_history.csv"),
}

# ---------------------------------------------------------------------------
# CSV reader (no pandas dependency)
# ---------------------------------------------------------------------------
def read_csv(filepath):
    """Read a CSV file and return a dict of column lists.

    Returns
    -------
    dict  :  {column_name: [values]}
        Numeric columns are converted to float where possible.
    """
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    header = lines[0].split(",")
    data = {col: [] for col in header}

    for line in lines[1:]:
        parts = line.split(",")
        for col, val in zip(header, parts):
            try:
                data[col].append(float(val))
            except ValueError:
                data[col].append(val)
    return data


def filter_rows(data, **kwargs):
    """Return a filtered copy of *data* where all key=value conditions match."""
    n = len(next(iter(data.values())))
    mask = [True] * n
    for col, target in kwargs.items():
        for i in range(n):
            if data[col][i] != target:
                mask[i] = False
    return {col: [v for v, m in zip(vals, mask) if m] for col, vals in data.items()}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading CSV data ...")
all_data = {}
for label, path in CSV_FILES.items():
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found — skipping {label}")
        continue
    all_data[label] = read_csv(path)
    print(f"  {label}: {len(all_data[label]['time_s'])} rows")

if not all_data:
    print("ERROR: No data files found. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "Normal": "#2196F3",
    "Stuck3": "#FF9800",
    "Stuck6": "#F44336",
}
INSTANCE_STYLES = {
    "Q1-INNERSKIN": {"ls": "-",  "marker": None, "label_suffix": " (Q1)"},
    "Q2-INNERSKIN": {"ls": "--", "marker": None, "label_suffix": " (Q2)"},
}

# ===================================================================
# Figure 1 — Displacement (u_z_max) time history
# ===================================================================
print("\nGenerating fig_separation_displacement.png ...")

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, case in zip(axes1, ["Normal", "Stuck3", "Stuck6"]):
    if case not in all_data:
        ax.set_title(f"{case} (no data)")
        continue

    for inst, style in INSTANCE_STYLES.items():
        d = filter_rows(all_data[case], instance=inst, step="Step-Separation")
        if not d["time_s"]:
            # Try without step filter in case naming differs
            d = filter_rows(all_data[case], instance=inst)
        if not d["time_s"]:
            continue
        ax.plot(
            np.array(d["time_s"]) * 1000,  # convert to ms
            d["u_z_max_mm"],
            ls=style["ls"],
            color=COLORS[case],
            linewidth=1.8,
            label=inst,
        )

    ax.set_title(case, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time [ms]", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=10)

axes1[0].set_ylabel("Max $u_z$ Displacement [mm]", fontsize=11)

fig1.suptitle(
    "Fairing Half-Shell Z-Displacement: Normal vs. Stuck Connectors",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig1.tight_layout()

out1 = os.path.join(RESULTS_DIR, "fig_separation_displacement.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {out1}")

# ===================================================================
# Figure 2 — Stress comparison (s_mises_max for Q1-INNERSKIN)
# ===================================================================
print("Generating fig_separation_stress.png ...")

fig2, ax2 = plt.subplots(figsize=(8, 5))

for case in ["Normal", "Stuck3", "Stuck6"]:
    if case not in all_data:
        continue
    d = filter_rows(all_data[case], instance="Q1-INNERSKIN", step="Step-Separation")
    if not d["time_s"]:
        d = filter_rows(all_data[case], instance="Q1-INNERSKIN")
    if not d["time_s"]:
        continue
    ax2.plot(
        np.array(d["time_s"]) * 1000,
        d["s_mises_max_MPa"],
        color=COLORS[case],
        linewidth=2.0,
        label=case,
    )

ax2.set_xlabel("Time [ms]", fontsize=12)
ax2.set_ylabel("Max von Mises Stress [MPa]", fontsize=12)
ax2.set_title(
    "Q1 Inner Skin — Peak von Mises Stress Comparison",
    fontsize=13,
    fontweight="bold",
)
ax2.legend(fontsize=11)
ax2.tick_params(labelsize=11)

fig2.tight_layout()
out2 = os.path.join(RESULTS_DIR, "fig_separation_stress.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {out2}")

# ===================================================================
# Figure 3 — GNN-SHM summary
# ===================================================================
print("Generating fig_gnn_summary.png ...")

fig3, (ax3a, ax3b) = plt.subplots(
    1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]}
)

# -- Left panel: bar chart --
approaches = ["From Scratch\n(random init)", "Fine-tuned\n(GW-SHM pretrained)"]
aucs = [0.45, 0.979]
bar_colors = ["#BDBDBD", "#4CAF50"]

bars = ax3a.bar(approaches, aucs, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)

for bar, val in zip(bars, aucs):
    ax3a.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"AUC = {val:.3f}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

ax3a.set_ylim(0, 1.15)
ax3a.set_ylabel("AUC (Area Under ROC Curve)", fontsize=12)
ax3a.set_title("GAT Anomaly Detection Performance", fontsize=13, fontweight="bold")
ax3a.axhline(0.5, color="gray", ls=":", lw=1, label="Random baseline")
ax3a.legend(fontsize=10, loc="upper left")
ax3a.tick_params(labelsize=11)
ax3a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

# -- Right panel: summary text --
ax3b.axis("off")

summary_lines = [
    ("Model", "GAT (Graph Attention Network)"),
    ("Pretraining", "GW-SHM: Lamb-wave debonding detection"),
    ("Fine-tuning", "Fairing separation node-level anomaly"),
    ("", ""),
    ("Training set", "2 graphs (Normal + Stuck3)"),
    ("Validation set", "1 graph (Stuck6)"),
    ("", ""),
    ("AUC (fine-tuned)", "0.979"),
    ("AUC (from scratch)", "0.45"),
    ("", ""),
    ("Anomaly nodes", ""),
    ("  Stuck3", "159 nodes (0.47% of mesh)"),
    ("  Stuck6", "153 nodes (0.45% of mesh)"),
]

y_pos = 0.95
for key, val in summary_lines:
    if key == "" and val == "":
        y_pos -= 0.04
        continue
    if val:
        ax3b.text(0.02, y_pos, f"{key}:", fontsize=11, fontweight="bold",
                  transform=ax3b.transAxes, va="top", family="monospace")
        ax3b.text(0.45, y_pos, val, fontsize=11,
                  transform=ax3b.transAxes, va="top", family="monospace")
    else:
        # section header
        ax3b.text(0.02, y_pos, key, fontsize=11, fontweight="bold",
                  transform=ax3b.transAxes, va="top", family="monospace")
    y_pos -= 0.065

ax3b.set_title("GNN-SHM Summary", fontsize=13, fontweight="bold")

# Add border around the text panel
for spine in ["top", "bottom", "left", "right"]:
    ax3b.spines[spine].set_visible(True)
    ax3b.spines[spine].set_linewidth(0.8)
    ax3b.spines[spine].set_color("#CCCCCC")

fig3.suptitle(
    "GNN-based Structural Health Monitoring for Fairing Separation",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig3.tight_layout()
out3 = os.path.join(RESULTS_DIR, "fig_gnn_summary.png")
fig3.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {out3}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("All figures generated:")
print(f"  1. {out1}")
print(f"  2. {out2}")
print(f"  3. {out3}")
print("=" * 60)
