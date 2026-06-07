#!/usr/bin/env python3
"""Compare OGW omega-stringer scenarios (Intact / local debond / large debond).

Computes the per-point RMS energy of the out-of-plane wavefield (sqrt of the
time-mean of z^2). The stringer guides/reflects waves and a debond traps/scatters
them, so the RMS map and its difference vs. Intact reveal the defect signature.

Usage:
    python3 scripts/compare_ogw_scenarios.py [--exc BURST_100kHz_5HC_9Vpp_x10] [--out fig.png]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from parse_ogw_wavefield import load_ogw  # noqa: E402

BASE = os.path.join(HERE, "..", "data", "external", "ogw_stringer", "extracted")
SCEN = [("Intact", "Intact"), ("Local debond", "FirstImpact"), ("Large debond", "SecondImpact")]


def rms_map(exc):
    out = []
    coords_ref = None
    for label, tag in SCEN:
        folder = os.path.join(BASE, f"OGW_CFRP_Stringer_Wavefield_{tag}", exc)
        coords, wavefield, _ = load_ogw(folder)
        rms = np.sqrt((wavefield.astype(np.float64) ** 2).mean(axis=0))  # (N,)
        del wavefield
        out.append((label, coords, rms))
        coords_ref = coords
        print(f"{label:14s}: RMS [{rms.min():.2e}, {rms.max():.2e}] mean {rms.mean():.2e}")
    return out, coords_ref


def plot(maps, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = np.percentile(np.concatenate([m[2] for m in maps]), 99)
    intact_rms = maps[0][2]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for j, (label, coords, rms) in enumerate(maps):
        sc = ax[0, j].scatter(coords[:, 0], coords[:, 1], c=rms, s=1.0, marker="s",
                              cmap="viridis", vmin=0, vmax=vmax, linewidths=0)
        ax[0, j].set_title(f"RMS energy — {label}")
        fig.colorbar(sc, ax=ax[0, j], fraction=0.046, pad=0.04, shrink=0.85)
        # difference vs intact (defect signature)
        if j == 0:
            ax[1, j].axis("off")
            ax[1, j].text(0.5, 0.5, "reference\n(Intact)", ha="center", va="center")
        else:
            diff = rms - intact_rms
            dvmax = np.percentile(np.abs(diff), 99)
            sd = ax[1, j].scatter(coords[:, 0], coords[:, 1], c=diff, s=1.0, marker="s",
                                  cmap="RdBu_r", vmin=-dvmax, vmax=dvmax, linewidths=0)
            ax[1, j].set_title(f"RMS difference vs Intact — {label}")
            fig.colorbar(sd, ax=ax[1, j], fraction=0.046, pad=0.04, shrink=0.85)
        for a in (ax[0, j], ax[1, j]):
            a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    fig.suptitle("OGW CFRP omega-stringer — wavefield RMS energy & debond signature (100 kHz burst)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"figure -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exc", default="BURST_100kHz_5HC_9Vpp_x10")
    ap.add_argument("--out", default=os.path.join(BASE, "..", "ogw_scenario_comparison_100kHz.png"))
    a = ap.parse_args()
    maps, _ = rms_map(a.exc)
    plot(maps, a.out)
