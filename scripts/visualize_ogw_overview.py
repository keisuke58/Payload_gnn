#!/usr/bin/env python3
"""Clean overview figure of the OGW omega-stringer wavefield dataset.

Shows what the data is: sensor layout on the scanned plate, the guided-wave field
propagating over time, and the debond signature (RMS difference vs intact).
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from parse_ogw_wavefield import load_ogw            # noqa: E402

BASE = os.path.join(HERE, "..", "data", "external", "ogw_stringer", "extracted")
EXC = "BURST_100kHz_5HC_9Vpp_x10"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "font.family": "serif"})


def folder(tag):
    return os.path.join(BASE, f"OGW_CFRP_Stringer_Wavefield_{tag}", EXC)


def field_panel(ax, coords, z, title, vmax, cmap="coolwarm"):
    sc = ax.scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, c=z, s=1.2, marker="s",
                    cmap=cmap, vmin=-vmax, vmax=vmax, linewidths=0)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    return sc


def main():
    coords, wf, t = load_ogw(folder("Intact"))
    # debond signature (RMS diff vs intact) using FirstImpact
    _, wf_d, _ = load_ogw(folder("FirstImpact"))
    rms_i = np.sqrt((wf**2).mean(0)); rms_d = np.sqrt((wf_d**2).mean(0))
    del wf_d
    diff = rms_d - rms_i

    # three propagation snapshots (early / mid / late after the burst)
    idx = [80, 200, 450]
    vmax = np.percentile(np.abs(wf[idx[1]]), 99)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for j, k in enumerate(idx):
        sc = field_panel(ax[0, j], coords, wf[k], f"Guided wavefield  t = {t[k]*1e6:.0f} us", vmax)
        fig.colorbar(sc, ax=ax[0, j], fraction=0.046, pad=0.04, shrink=0.8, label="u_z")

    # sensor layout (9 grid sensors) over the scanned region
    from ogw_to_sensor_csv import pick_sensor_grid
    sidx = pick_sensor_grid(coords, 9)
    ax[1, 0].scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, c="0.85", s=1.0, marker="s", linewidths=0)
    ax[1, 0].scatter(coords[sidx, 0]*1e3, coords[sidx, 1]*1e3, c="#c62828", s=80,
                     marker="^", edgecolors="k", linewidths=0.6, label="9 sensors", zorder=3)
    ax[1, 0].scatter([coords[:, 0].mean()*1e3], [coords[:, 1].mean()*1e3], c="#1565c0",
                     s=120, marker="*", edgecolors="k", linewidths=0.6, label="actuator", zorder=4)
    ax[1, 0].set_aspect("equal"); ax[1, 0].set_title("Scanned plate & sensor layout")
    ax[1, 0].set_xlabel("x [mm]"); ax[1, 0].set_ylabel("y [mm]"); ax[1, 0].legend(loc="upper right", fontsize=9)

    # RMS energy (intact) and debond signature
    rvmax = np.percentile(rms_i, 99)
    s2 = ax[1, 1].scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, c=rms_i, s=1.2, marker="s",
                          cmap="viridis", vmin=0, vmax=rvmax, linewidths=0)
    ax[1, 1].set_aspect("equal"); ax[1, 1].set_title("RMS energy (intact)")
    ax[1, 1].set_xlabel("x [mm]"); ax[1, 1].set_ylabel("y [mm]")
    fig.colorbar(s2, ax=ax[1, 1], fraction=0.046, pad=0.04, shrink=0.8)

    dvmax = np.percentile(np.abs(diff), 99)
    s3 = ax[1, 2].scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, c=diff, s=1.2, marker="s",
                          cmap="RdBu_r", vmin=-dvmax, vmax=dvmax, linewidths=0)
    ax[1, 2].set_aspect("equal"); ax[1, 2].set_title("Debond signature (RMS diff vs intact)")
    ax[1, 2].set_xlabel("x [mm]"); ax[1, 2].set_ylabel("y [mm]")
    fig.colorbar(s3, ax=ax[1, 2], fraction=0.046, pad=0.04, shrink=0.8)

    fig.suptitle("OGW CFRP omega-stringer dataset (100 kHz burst): 233k-point SLDV wavefield, "
                 "1024 time steps", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "results", "ogw", "ogw_dataset_overview.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("overview ->", out)


if __name__ == "__main__":
    main()
