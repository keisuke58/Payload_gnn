#!/usr/bin/env python3
"""
Mesh convergence check plot for H3 fairing model.
Mesh sizes: 10, 12, 25, 50, 100 mm
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data collection ---
mesh_configs = {
    100: {"path": None, "status": "not_generated"},
    50:  {"path": "dataset_output_ideal_50mm/sample_0000", "status": "ok"},
    25:  {"path": "dataset_output_ideal/sample_0000", "status": "ok"},
    12:  {"path": "dataset_output_ideal_12mm/sample_0000", "status": "ok"},
    10:  {"path": "dataset_output_ideal_10mm/sample_0000", "status": "failed"},
}

# Defect: theta=30 deg, z=2500 mm, r=50 mm (all same)
DEFECT_RADIUS = 50  # mm
# Fairing geometry (60-deg sector of cylinder R=2638mm, H=5000mm barrel)
FAIRING_R = 2638.0  # mm outer skin radius
FAIRING_ANGLE = np.radians(60)  # 60-deg sector
FAIRING_H_BARREL = 5000.0  # mm
FAIRING_H_NOSE = 5400.0  # mm (ogive)
# Approximate outer skin area (barrel + nose)
BARREL_AREA = FAIRING_R * FAIRING_ANGLE * FAIRING_H_BARREL  # arc_length * height
NOSE_AREA = 0.5 * FAIRING_R * FAIRING_ANGLE * FAIRING_H_NOSE  # rough estimate
TOTAL_AREA = BARREL_AREA + NOSE_AREA  # ~2.17e7 mm^2
# Defect area: circle of radius 50mm on surface
DEFECT_AREA = np.pi * DEFECT_RADIUS**2  # ~7854 mm^2

results = {}
for h, cfg in mesh_configs.items():
    if cfg["status"] != "ok":
        results[h] = {"n_nodes": None, "n_defect": None, "status": cfg["status"]}
        continue
    nodes_path = os.path.join(PROJECT, cfg["path"], "nodes.csv")
    if not os.path.exists(nodes_path):
        results[h] = {"n_nodes": None, "n_defect": None, "status": "missing"}
        continue
    df = pd.read_csv(nodes_path)
    n_nodes = len(df)
    n_defect = int((df["defect_label"] == 1).sum())
    results[h] = {"n_nodes": n_nodes, "n_defect": n_defect, "status": "ok"}

# --- Theoretical estimates ---
# Node count ~ Area / h^2 (quadrilateral elements)
h_available = sorted([h for h in results if results[h]["status"] == "ok"])
n_available = [results[h]["n_nodes"] for h in h_available]

# Fit power law: N = C * h^alpha
log_h = np.log(h_available)
log_n = np.log(n_available)
alpha, log_C = np.polyfit(log_h, log_n, 1)
C = np.exp(log_C)

h_theory = np.linspace(8, 120, 200)
n_theory = C * h_theory**alpha

# Defect nodes ~ defect_area / h^2
d_available = [results[h]["n_defect"] for h in h_available]
log_d = np.log([max(d, 0.5) for d in d_available])
alpha_d, log_Cd = np.polyfit(log_h, log_d, 1)
Cd = np.exp(log_Cd)
d_theory = Cd * h_theory**alpha_d

# Theoretical estimate for missing meshes
for h in [100, 10]:
    if results[h]["status"] != "ok":
        results[h]["n_nodes_est"] = C * h**alpha
        results[h]["n_defect_est"] = Cd * h**alpha_d

# --- Print summary ---
print("=" * 70)
print("Mesh Convergence Summary  (defect: θ=30°, z=2500mm, r=50mm)")
print("=" * 70)
print(f"{'h [mm]':>8}  {'Nodes':>10}  {'Defect':>8}  {'Density':>12}  {'Status'}")
print("-" * 70)
for h in sorted(results.keys()):
    r = results[h]
    if r["status"] == "ok":
        density = r["n_nodes"] / (TOTAL_AREA / 1e6)  # nodes/m^2
        print(f"{h:>8}  {r['n_nodes']:>10,}  {r['n_defect']:>8}  {density:>10,.0f}/m²  OK")
    elif "n_nodes_est" in r:
        est_n = int(r["n_nodes_est"])
        est_d = max(0, int(r["n_defect_est"]))
        print(f"{h:>8}  ~{est_n:>9,}  ~{est_d:>7}  {'---':>12}  {r['status']} (estimated)")
    else:
        print(f"{h:>8}  {'---':>10}  {'---':>8}  {'---':>12}  {r['status']}")

print(f"\nPower-law fit: N ∝ h^{alpha:.2f}  (expect ≈ -2 for 2D mesh)")
print(f"Power-law fit: N_defect ∝ h^{alpha_d:.2f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("H3 Fairing – Mesh Convergence Check\n"
             "(Defect: θ=30°, z=2500 mm, r=50 mm)",
             fontsize=13, fontweight="bold")

# Colors
c_data = "#2563EB"
c_theory = "#94A3B8"
c_missing = "#EF4444"
c_est = "#F59E0B"

# --- (a) Total node count ---
ax = axes[0]
ax.plot(h_theory, n_theory, "-", color=c_theory, lw=1.5, alpha=0.6,
        label=f"Fit: N ∝ h$^{{{alpha:.2f}}}$")
for h in h_available:
    ax.plot(h, results[h]["n_nodes"], "o", color=c_data, ms=9, zorder=5)
    ax.annotate(f"{results[h]['n_nodes']:,}",
                (h, results[h]["n_nodes"]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color=c_data)
# Estimated points
for h in [100, 10]:
    r = results[h]
    if "n_nodes_est" in r:
        marker = "x" if r["status"] == "failed" else "D"
        color = c_missing if r["status"] == "failed" else c_est
        ax.plot(h, r["n_nodes_est"], marker, color=color, ms=9, zorder=5)
        label_txt = f"~{int(r['n_nodes_est']):,}\n({r['status']})"
        ax.annotate(label_txt, (h, r["n_nodes_est"]),
                    textcoords="offset points", xytext=(8, -15 if h == 10 else 8),
                    fontsize=7, color=color)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Element size h [mm]")
ax.set_ylabel("Total node count N")
ax.set_title("(a) Node Count vs Element Size")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([10, 12, 25, 50, 100])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.invert_xaxis()
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- (b) Defect node count ---
ax = axes[1]
ax.plot(h_theory, np.maximum(d_theory, 0.5), "-", color=c_theory, lw=1.5, alpha=0.6,
        label=f"Fit: N$_d$ ∝ h$^{{{alpha_d:.2f}}}$")
for h in h_available:
    ax.plot(h, results[h]["n_defect"], "s", color=c_data, ms=9, zorder=5)
    ax.annotate(f"{results[h]['n_defect']}",
                (h, max(results[h]["n_defect"], 0.5)),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color=c_data, fontweight="bold")

# Minimum detection threshold
ax.axhline(y=3, color="#10B981", ls="--", lw=1.2, alpha=0.7)
ax.text(105, 3.3, "Min detection\n(3 nodes)", fontsize=7, color="#10B981", ha="right")

# Estimated points
for h in [100, 10]:
    r = results[h]
    if "n_defect_est" in r:
        marker = "x" if r["status"] == "failed" else "D"
        color = c_missing if r["status"] == "failed" else c_est
        est_d = max(r["n_defect_est"], 0.5)
        ax.plot(h, est_d, marker, color=color, ms=9, zorder=5)
        ax.annotate(f"~{max(0,int(r['n_defect_est']))}\n({r['status']})",
                    (h, est_d),
                    textcoords="offset points", xytext=(8, -15 if h == 10 else 8),
                    fontsize=7, color=color)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Element size h [mm]")
ax.set_ylabel("Defect node count N$_d$")
ax.set_title("(b) Defect Resolution vs Element Size")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([10, 12, 25, 50, 100])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.invert_xaxis()
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- (c) Convergence ratio (Richardson extrapolation style) ---
ax = axes[2]

# h/D ratio (mesh size / defect diameter) — key for resolution
h_all = np.array([100, 50, 25, 12, 10], dtype=float)
D = 2 * DEFECT_RADIUS  # defect diameter = 100mm
h_over_D = h_all / D

# Defect detection criteria
for h in h_available:
    nd = results[h]["n_defect"]
    ratio = h / D
    ax.plot(ratio, nd, "o", color=c_data, ms=10, zorder=5)
    ax.annotate(f"h={h}mm\nN$_d$={nd}",
                (ratio, max(nd, 0.5)),
                textcoords="offset points", xytext=(10, 5),
                fontsize=8, color=c_data)

# Threshold regions
ax.axvspan(0, 0.25, alpha=0.08, color="#10B981", label="Excellent (h/D < 0.25)")
ax.axvspan(0.25, 0.5, alpha=0.08, color="#F59E0B", label="Good (h/D < 0.5)")
ax.axvspan(0.5, 1.2, alpha=0.08, color="#EF4444", label="Poor (h/D ≥ 0.5)")

ax.axhline(y=3, color="#10B981", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("h / D  (element size / defect diameter)")
ax.set_ylabel("Defect node count N$_d$")
ax.set_title("(c) Resolution Quality: h/D Ratio")
ax.set_xlim(0, 1.1)
ax.set_ylim(0, max(60, max(d_available) * 1.2))
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(PROJECT, "figures", "mesh_convergence.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close()
