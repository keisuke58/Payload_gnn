#!/usr/bin/env python3
"""flight_clearance_demo.py — reusable-vehicle go/no-go from NDT posteriors.

Payload2026 framing: a CFRP fairing / interstage panel of a reusable launch
vehicle is inspected between flights (guided-wave / DSPSS NDT).  The WCCM
Stage-2 FMPE model (wccm2026-cfrp-gnn/fmpe_defect.py) turns one measurement
into a posterior over the defect theta = (cx, cy, layer, log2size); the
Stage-3 phase-field prognosis layer (cfrp_phasefield_2d.py) propagates that
posterior through an anisotropic laminate fracture simulation and returns a
flight-clearance decision:

    OK      — fly again as-is
    REPAIR  — refurbish the panel before the next flight
    RETIRE  — pull the structure (vehicle loss >> refurbishment cost)

This demo runs flight_clearance() for three load levels (normalised
transverse peel-strain proxy — see the module docstring; mapping real flight
loads to this local driver needs an outer structural analysis) and prints
the decision table + saves a figure.

Run:  python3 scripts/flight_clearance_demo.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

# import the Stage-3 module from the WCCM repo
WCCM_REPO = "/home/nishioka/git/wccm2026-cfrp-gnn"
sys.path.insert(0, WCCM_REPO)

from cfrp_phasefield_2d import (  # noqa: E402
    LaminateConfig,
    flight_clearance,
    seed_defect,
    simulate_growth,
)

PAYLOAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(PAYLOAD_ROOT, "results", "flight_clearance_demo.png")

# calibrated normalised load scale of the Stage-3 module
LOAD_LEVELS = {"nominal ascent": 0.06, "max-Q gust": 0.10, "abort/overload": 0.14}
N_FLIGHTS = 10
N_DRAWS = 16
SEED = 42


def load_posterior() -> tuple[np.ndarray, str]:
    """Load a saved FMPE posterior if one exists, else build a mock one.

    A real posterior would be saved by fmpe_defect.py as an .npz with an
    array of shape (N, 4) = (cx, cy, layer, log2size).
    """
    for pattern in (os.path.join(WCCM_REPO, "results", "fmpe_posterior*.npz"),
                    os.path.join(WCCM_REPO, "fmpe_posterior*.npz")):
        hits = sorted(glob.glob(pattern))
        if hits:
            with np.load(hits[0]) as z:
                arr = z[z.files[0]]
            if arr.ndim == 2 and arr.shape[1] == 4:
                return arr.astype(float), f"saved FMPE posterior: {hits[0]}"

    # mock posterior mimicking FMPE output for a mid-laminate ~2x2 defect
    # (spreads comparable to the FMPE test-set RMSEs)
    rng = np.random.default_rng(SEED)
    n = 256
    post = np.column_stack([
        rng.normal(0.55, 0.04, n),          # cx
        rng.normal(0.45, 0.04, n),          # cy
        rng.normal(9.5, 1.2, n),            # layer (depth)
        rng.normal(1.0, 0.35, n),           # log2size (~2x2 elements)
    ])
    return post, "mock FMPE posterior (no saved npz found)"


def main() -> None:
    posterior, source = load_posterior()
    cfg = LaminateConfig()

    print("=" * 72)
    print("Reusable-vehicle flight clearance — Stage-3 phase-field prognosis")
    print("=" * 72)
    print(f"posterior source : {source}  (N={len(posterior)})")
    print(f"laminate         : {cfg.n_plies} plies {cfg.ply_angles_deg}, "
          f"grid {cfg.nx}x{cfg.ny}")
    print(f"posterior mean   : cx={posterior[:,0].mean():.2f} "
          f"cy={posterior[:,1].mean():.2f} layer={posterior[:,2].mean():.1f} "
          f"size~{2**posterior[:,3].mean():.1f} elements")
    print(f"MC draws / case  : {N_DRAWS}, horizon: {N_FLIGHTS} flights\n")

    rows = []
    header = (f"{'load case':>16} {'load':>6} {'P(growth)':>10} "
              f"{'P(survive %d)' % N_FLIGHTS:>14} {'E[flights]':>11} "
              f"{'decision':>9}")
    print(header)
    print("-" * len(header))
    for name, load in LOAD_LEVELS.items():
        t0 = time.perf_counter()
        out = flight_clearance(posterior,
                               load_profile=[0.5 * load, load],
                               n_flights=N_FLIGHTS,
                               cfg=cfg, n_draws=N_DRAWS,
                               rng=np.random.default_rng(SEED))
        dt = time.perf_counter() - t0
        rows.append((name, load, out))
        print(f"{name:>16} {load:>6.3f} {out['p_growth_next']:>10.2f} "
              f"{out['p_survive_n']:>14.3f} "
              f"{out['expected_remaining_flights']:>11.1f} "
              f"{out['decision']:>9}   [{dt:.0f}s]")

    print("\nCaveats: quasi-static AT2 (no fatigue — Carrara 2020 = future "
          "work);\nnormalised load proxy; thresholds alpha/beta are placeholders"
          " for an\nexpected-cost decision rule (vehicle loss >> refurbishment).")

    _make_figure(rows, posterior, cfg)
    print(f"\nfigure saved: {OUT_PNG}")


def _make_figure(rows, posterior, cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    dec_color = {"OK": "#2a9d3a", "REPAIR": "#e9a800", "RETIRE": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"width_ratios": [1.1, 1.0, 1.4]})

    # (a) P(growth) bar chart with decisions
    ax = axes[0]
    names = [r[0] for r in rows]
    loads = [r[1] for r in rows]
    ps = [r[2]["p_growth_next"] for r in rows]
    decs = [r[2]["decision"] for r in rows]
    bars = ax.bar(range(len(rows)), ps,
                  color=[dec_color[d] for d in decs], width=0.6)
    for b, d, p in zip(bars, decs, ps):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.03, d,
                ha="center", fontsize=10, fontweight="bold",
                color=dec_color[d])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{n}\n(load {l:.2f})" for n, l in zip(names, loads)],
                       fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P(growth next flight)")
    ax.axhline(0.05, ls=":", c="gray", lw=1)
    ax.axhline(0.30, ls="--", c="gray", lw=1)
    ax.text(2.45, 0.055, r"$\alpha$=0.05", fontsize=7, color="gray")
    ax.text(2.45, 0.305, r"$\beta$=0.30", fontsize=7, color="gray")
    ax.set_title("(a) Stage-3 clearance decision")

    # (b) FMPE posterior scatter (depth vs size)
    ax = axes[1]
    sc = ax.scatter(2.0 ** posterior[:, 3], posterior[:, 2],
                    c=posterior[:, 0], s=12, alpha=0.6, cmap="viridis")
    ax.set_xlabel("defect size [elements]")
    ax.set_ylabel("layer (depth)")
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="cx")
    ax.set_title("(b) Stage-2 FMPE posterior")

    # (c) example damage field at the intermediate load
    ax = axes[2]
    theta_mean = posterior.mean(axis=0)
    d0 = seed_defect(*theta_mean, cfg)
    res = simulate_growth(d0, rows[1][1], cfg)
    im = ax.imshow(res.d_final, origin="lower", aspect="auto",
                   extent=[0, cfg.Lx, 0, cfg.Ly], cmap="inferno",
                   vmin=0, vmax=1)
    ply_t = cfg.Ly / cfg.n_plies
    for k in range(1, cfg.n_plies):
        ax.axhline(k * ply_t, color="w", lw=0.4, alpha=0.5)
    fig.colorbar(im, ax=ax, label="damage d")
    ax.set_xlabel("x (in-plane)")
    ax.set_ylabel("y (thickness)")
    ax.set_title(f"(c) damage field, posterior-mean defect "
                 f"@ load {rows[1][1]:.2f} "
                 f"(rel. growth {res.rel_growth:.1f})")

    fig.suptitle("CFRP panel flight clearance: NDT posterior → phase-field "
                 "prognosis → go/no-go", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
