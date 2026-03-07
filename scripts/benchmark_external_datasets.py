#!/usr/bin/env python3
"""
External Datasets Benchmark — Cross-Dataset Comparison

Compare guided wave characteristics across:
  1. This project (Abaqus Explicit, CFRP/Al-HC fairing)
  2. DINS-SHM (SFEM, isotropic + composite waveguide)
  3. CONCEPT (Experimental, CFRP plate, PZT)

Output: figures/external_benchmark/ with comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import scipy.io as sio
from scipy.signal import welch

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "external_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

DINS_DIR = ROOT / "data" / "external" / "dins_shm"
CONCEPT_DIR = ROOT / "data" / "external" / "concept" / "DATASET_PLATEUN01" / "data"
GW_DIR = ROOT / "abaqus_work" / "gw_fairing_dataset"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
})


def load_project_gw(n_samples=5):
    """Load this project's GW sensor data (healthy + defect)."""
    healthy_files = sorted(GW_DIR.glob("Job-GW-Fair-Healthy-A*_sensors.csv"))[:n_samples]
    defect_files = sorted(GW_DIR.glob("Job-GW-Fair-[0-9]*_sensors.csv"))

    data = {"healthy": [], "defect": []}
    for f in healthy_files:
        df = pd.read_csv(f, comment="#")
        data["healthy"].append(df.iloc[:, 1].values)  # sensor_0_Ur
    for f in defect_files:
        df = pd.read_csv(f, comment="#")
        data["defect"].append(df.iloc[:, 1].values)

    # Time from first healthy file
    df0 = pd.read_csv(healthy_files[0], comment="#")
    data["time_s"] = df0["time_s"].values
    data["dt"] = data["time_s"][1] - data["time_s"][0]
    data["n_timesteps"] = len(data["time_s"])
    data["duration_s"] = data["time_s"][-1]
    return data


def load_dins_composite():
    """Load DINS-SHM composite waveguide data."""
    comp_dir = DINS_DIR / "composite" / "0_DataSet"
    labels = np.loadtxt(DINS_DIR / "composite" / "2_Labels" / "labels.csv", delimiter=",")

    data = {}
    # Sample evenly across all 2500 rows for diversity
    n_total = 2500
    n_sample = 100
    sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=int)

    for name, fname in [
        ("ax_damaged", "Ax2500Comp_D_07Aug20.txt"),
        ("ax_healthy", "Ax2500Comp_UD_07Aug20.txt"),
        ("flex_damaged", "Flex2500Comp_D_07Aug20.txt"),
        ("flex_healthy", "Flex2500Comp_UD_07Aug20.txt"),
    ]:
        # Load all rows then subsample
        all_rows = np.genfromtxt(comp_dir / fname, delimiter=",")
        data[name] = all_rows[sample_idx]

    data["labels"] = labels[sample_idx]
    data["n_timesteps"] = data["ax_damaged"].shape[1]
    return data


def load_concept():
    """Load CONCEPT experimental lamb wave data."""
    data = {}

    # Healthy baseline (T=30°C)
    h30 = sio.loadmat(CONCEPT_DIR / "H_T30.mat")
    data["healthy_30C"] = h30["data"]  # (1000, 4, 100)

    # Damaged state 1 (smallest damage)
    d1 = sio.loadmat(CONCEPT_DIR / "D1.mat")
    data["damage_1"] = d1["data"]

    # Damaged state 11 (largest damage)
    d11 = sio.loadmat(CONCEPT_DIR / "D11.mat")
    data["damage_11"] = d11["data"]

    # Multiple temperatures
    for t in [0, 10, 20, 30, 40, 50, 60]:
        ht = sio.loadmat(CONCEPT_DIR / f"H_T{t}.mat")
        data[f"healthy_T{t}"] = ht["data"]

    data["fs"] = 5e6  # 5 MHz sampling rate
    data["n_timesteps"] = 1000
    data["duration_s"] = data["n_timesteps"] / data["fs"]
    return data


# ── Figure 1: Waveform Comparison ──────────────────────────
def plot_waveform_comparison(proj, dins, concept):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Row 1: This project
    t_proj = proj["time_s"] * 1e6  # to µs
    ax = axes[0, 0]
    for i, w in enumerate(proj["healthy"][:3]):
        ax.plot(t_proj[:len(w)], w * 1e3, alpha=0.7, label=f"H-{i}")
    ax.set_title("This Project — Healthy (Abaqus Explicit)")
    ax.set_ylabel("Radial Disp. [mm]")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, w in enumerate(proj["defect"][:2]):
        ax.plot(t_proj[:len(w)], w * 1e3, alpha=0.7, label=f"D-{i}")
    if proj["healthy"]:
        ax.plot(t_proj[:len(proj["healthy"][0])], proj["healthy"][0] * 1e3,
                "k--", alpha=0.3, label="Healthy ref")
    ax.set_title("This Project — Defect vs Healthy")
    ax.set_ylabel("Radial Disp. [mm]")
    ax.legend(fontsize=8)

    # Row 2: DINS-SHM
    n_dins = dins["n_timesteps"]
    t_dins = np.arange(n_dins)
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(t_dins, dins["ax_healthy"][i], alpha=0.7, label=f"UD-{i}")
    ax.set_title("DINS-SHM Composite — Axial Healthy (SFEM)")
    ax.set_ylabel("Response")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_dins, dins["ax_damaged"][0], "r", alpha=0.7, label="Damaged")
    ax.plot(t_dins, dins["ax_healthy"][0], "b--", alpha=0.5, label="Healthy")
    ax.set_title("DINS-SHM Composite — Damaged vs Healthy")
    ax.set_ylabel("Response")
    ax.legend(fontsize=8)

    # Row 3: CONCEPT
    t_concept = np.arange(1000) / 5e6 * 1e6  # to µs
    ax = axes[2, 0]
    for i in range(3):
        ax.plot(t_concept, concept["healthy_30C"][:, 1, i], alpha=0.7,
                label=f"Meas-{i}")
    ax.set_title("CONCEPT Experimental — Healthy T=30°C, PZT-2")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [V]")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    h_mean = concept["healthy_30C"][:, 1, :10].mean(axis=1)
    d1_mean = concept["damage_1"][:, 1, :10].mean(axis=1)
    d11_mean = concept["damage_11"][:, 1, :10].mean(axis=1)
    ax.plot(t_concept, h_mean, "b", label="Healthy")
    ax.plot(t_concept, d1_mean, "orange", label="Damage-1 (small)")
    ax.plot(t_concept, d11_mean, "r", label="Damage-11 (large)")
    ax.set_title("CONCEPT Experimental — Damage Progression, PZT-2")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [V]")
    ax.legend(fontsize=8)

    fig.suptitle("Fig. B1: Guided Wave Waveform Comparison — Three Datasets",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "01_waveform_comparison.png", bbox_inches="tight")
    print(f"  Saved {OUT / '01_waveform_comparison.png'}")
    plt.close(fig)


# ── Figure 2: Frequency Content Comparison ─────────────────
def plot_frequency_comparison(proj, dins, concept):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # This project
    ax = axes[0]
    for w in proj["healthy"][:3]:
        f, psd = welch(w, fs=1.0/proj["dt"], nperseg=min(512, len(w)))
        ax.semilogy(f / 1e3, psd, alpha=0.5)
    ax.set_title("This Project (Abaqus)")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD")
    ax.set_xlim(0, 500)

    # DINS-SHM (no known fs, use normalized)
    ax = axes[1]
    for i in range(3):
        f, psd = welch(dins["ax_healthy"][i], nperseg=512)
        ax.semilogy(f, psd, alpha=0.5)
    ax.set_title("DINS-SHM Composite (SFEM)")
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("PSD")

    # CONCEPT
    ax = axes[2]
    h_mean = concept["healthy_30C"][:, 1, :10].mean(axis=1)
    d11_mean = concept["damage_11"][:, 1, :10].mean(axis=1)
    f, psd_h = welch(h_mean, fs=5e6, nperseg=256)
    f, psd_d = welch(d11_mean, fs=5e6, nperseg=256)
    ax.semilogy(f / 1e3, psd_h, "b", label="Healthy")
    ax.semilogy(f / 1e3, psd_d, "r", label="Damage-11")
    ax.set_title("CONCEPT Experimental")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD")
    ax.set_xlim(0, 500)
    ax.legend(fontsize=8)

    fig.suptitle("Fig. B2: Frequency Content Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "02_frequency_comparison.png", bbox_inches="tight")
    print(f"  Saved {OUT / '02_frequency_comparison.png'}")
    plt.close(fig)


# ── Figure 3: CONCEPT Temperature Effect ──────────────────
def plot_concept_temperature(concept):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    temps = [0, 10, 20, 30, 40, 50, 60]
    t_us = np.arange(1000) / 5e6 * 1e6
    cmap = plt.cm.coolwarm

    ax = axes[0]
    for i, t in enumerate(temps):
        mean_sig = concept[f"healthy_T{t}"][:, 1, :10].mean(axis=1)
        color = cmap(i / (len(temps) - 1))
        ax.plot(t_us, mean_sig, color=color, alpha=0.8, label=f"{t}°C")
    ax.set_title("Temperature Effect on Healthy Signal (PZT-2)")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [V]")
    ax.legend(fontsize=7, ncol=2)

    # RMS vs temperature
    ax = axes[1]
    rms_vals = []
    for t in temps:
        sig = concept[f"healthy_T{t}"][:, 1, :]  # (1000, 100)
        rms = np.sqrt(np.mean(sig ** 2, axis=0))
        rms_vals.append(rms)
    bp = ax.boxplot(rms_vals, labels=[f"{t}°C" for t in temps], patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / (len(temps) - 1)))
    ax.set_title("Signal RMS vs Temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("RMS Voltage")

    fig.suptitle("Fig. B3: CONCEPT — Temperature Sensitivity of Lamb Waves",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_concept_temperature.png", bbox_inches="tight")
    print(f"  Saved {OUT / '03_concept_temperature.png'}")
    plt.close(fig)


# ── Figure 4: DINS-SHM Damage Index Map ──────────────────
def plot_dins_damage_map(dins):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = dins["labels"]  # (N, 2): position, size
    n = len(labels)

    # Compute damage index (RMS ratio)
    di_ax = []
    for i in range(min(n, 50)):
        rms_d = np.sqrt(np.mean(dins["ax_damaged"][i] ** 2))
        rms_h = np.sqrt(np.mean(dins["ax_healthy"][i] ** 2))
        di_ax.append(rms_d / rms_h if rms_h > 0 else 1.0)

    di_flex = []
    for i in range(min(n, 50)):
        rms_d = np.sqrt(np.mean(dins["flex_damaged"][i] ** 2))
        rms_h = np.sqrt(np.mean(dins["flex_healthy"][i] ** 2))
        di_flex.append(rms_d / rms_h if rms_h > 0 else 1.0)

    n_plot = min(len(di_ax), len(labels))

    ax = axes[0]
    sc = ax.scatter(labels[:n_plot, 0], labels[:n_plot, 1],
                    c=di_ax[:n_plot], cmap="hot_r", s=30, edgecolors="k", linewidths=0.3)
    ax.set_title("Axial Wave — Damage Index (RMS ratio)")
    ax.set_xlabel("Damage Position (normalized)")
    ax.set_ylabel("Damage Size (normalized)")
    plt.colorbar(sc, ax=ax, label="DI")

    ax = axes[1]
    sc = ax.scatter(labels[:n_plot, 0], labels[:n_plot, 1],
                    c=di_flex[:n_plot], cmap="hot_r", s=30, edgecolors="k", linewidths=0.3)
    ax.set_title("Flexural Wave — Damage Index (RMS ratio)")
    ax.set_xlabel("Damage Position (normalized)")
    ax.set_ylabel("Damage Size (normalized)")
    plt.colorbar(sc, ax=ax, label="DI")

    fig.suptitle("Fig. B4: DINS-SHM Composite — Damage Sensitivity Map",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "04_dins_damage_map.png", bbox_inches="tight")
    print(f"  Saved {OUT / '04_dins_damage_map.png'}")
    plt.close(fig)


# ── Figure 5: Cross-Dataset Summary ──────────────────────
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    columns = ["Property", "This Project", "DINS-SHM (Composite)", "CONCEPT (Experimental)"]
    rows = [
        ["Structure", "CFRP/Al-HC Fairing\n(curved, 1/6 sector)", "Composite Waveguide\n(1D, plate)", "CFRP Plate\n(flat, 0° unidirectional)"],
        ["Method", "Abaqus/Explicit FEA\n(CZM + C3D10)", "Spectral FEM\n(reduced-order)", "Experimental\n(PZT pitch-catch)"],
        ["Defect", "Debonding\n(skin-core interface)", "Generic damage\n(position + size)", "Delamination\n(adhesive putty)"],
        ["Samples", "96 healthy + 6 defect\n(augmented)", "2500 D + 2500 UD\n(per wave mode)", "7 temp × 100 meas\n+ 11 damage × 100"],
        ["Time Steps", "3,922\n(~3.9 ms, Δt≈1µs)", "8,192\n(normalized)", "1,000\n(200µs, Fs=5MHz)"],
        ["Channels", "9-99 sensors\n(radial Ur)", "1 response\n(Ax + Flex)", "4 PZT channels\n(voltage)"],
        ["Frequency", "50 kHz excitation", "Not specified", "250 kHz, 5-cycle burst"],
        ["Temperature", "Thermal gradient\n(100-200°C)", "N/A", "0-60°C\n(7 levels)"],
    ]

    table = ax.table(cellText=rows, colLabels=columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#D6E4F0" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)

    fig.suptitle("Fig. B5: Cross-Dataset Comparison Summary",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(OUT / "05_cross_dataset_summary.png", bbox_inches="tight")
    print(f"  Saved {OUT / '05_cross_dataset_summary.png'}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("External Datasets Benchmark")
    print("=" * 60)

    print("\n[1/5] Loading datasets...")
    proj = load_project_gw()
    print(f"  Project: {len(proj['healthy'])} healthy, {len(proj['defect'])} defect, "
          f"{proj['n_timesteps']} timesteps")

    dins = load_dins_composite()
    print(f"  DINS-SHM: {dins['n_timesteps']} timesteps per sample")

    concept = load_concept()
    print(f"  CONCEPT: {concept['n_timesteps']} timesteps, Fs={concept['fs']/1e6:.0f} MHz")

    print("\n[2/5] Waveform comparison...")
    plot_waveform_comparison(proj, dins, concept)

    print("[3/5] Frequency content comparison...")
    plot_frequency_comparison(proj, dins, concept)

    print("[4/5] CONCEPT temperature effect...")
    plot_concept_temperature(concept)

    print("[5/5] DINS-SHM damage sensitivity...")
    plot_dins_damage_map(dins)

    print("\n[Bonus] Cross-dataset summary table...")
    plot_summary_table()

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {OUT}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
