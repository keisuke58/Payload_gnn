#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cryogenic Hydrogen-Tank DOE Generation — LHS over defect + operating conditions.

Design note: docs/HYDROGEN_TANK_SHM.md (geometry, materials, loading, defects).
This is the *first* code step of the hydrogen-tank SHM line (memo §7, milestone M1).

Pure-Python (no Abaqus) — runnable/verifiable outside the cluster. Produces a JSON
DOE consumed by src/generate_cryotank_dataset.py (Abaqus), mirroring the fairing
DOE schema (samples[].defect_params + metadata) so downstream tooling is reusable.

Defect types (memo §4):
  - healthy            — no defect (baseline)
  - weld_flaw          — porosity / lack-of-fusion on a weld line (located ON a weld)
  - microcrack         — thermal-cycle micro-cracking (local stiffness loss)
  - insulation_debond  — foam/MLI separation (same interface type as skin-core debond)

Operating point (memo §2-§3):
  - temperature_K : 20 K (LH2 cryo) or 293 K (RT), stratified
  - pressure_MPa  : operational internal pressure

Usage:
  python src/generate_cryotank_doe.py --n_samples 1 --output doe_cryotank_1sample.json
  python src/generate_cryotank_doe.py --n_samples 100 --output doe_cryotank_100.json
  python src/generate_cryotank_doe.py --healthy_only --output doe_cryotank_healthy.json
"""

import argparse
import json

import numpy as np

try:
    from scipy.stats.qmc import LatinHypercube
    _HAVE_QMC = True
except Exception:  # scipy missing / too old — fall back to uniform RNG
    _HAVE_QMC = False

# ---- Geometry bounds (design assumption; see memo §1) ------------------------
# Al-Li cylindrical LH2 tank, barrel section (representative, dimensions non-public).
THETA_RANGE = (5.0, 85.0)        # deg, margin from symmetry edges (1/12--1/6 sector)
Z_RANGE = (400.0, 3600.0)        # mm, along barrel, avoid dome junctions
WELD_THETA_DEG = [15.0, 45.0, 75.0]  # longitudinal weld-line angular positions

# ---- Defect size tiers (mm): (name, r_min, r_max, fraction) ------------------
SIZE_TIERS = [
    ("Small", 20.0, 60.0, 0.35),
    ("Medium", 60.0, 120.0, 0.40),
    ("Large", 120.0, 220.0, 0.25),
]

# ---- Operating point (memo §2-§3; design assumptions) ------------------------
TEMPERATURE_K = [20.0, 293.0]    # LH2 cryo / room-temperature
PRESSURE_MPA_RANGE = (0.2, 0.5)  # operational internal pressure

DEFECT_TYPES = ["weld_flaw", "microcrack", "insulation_debond"]


def _lhs(n, d, seed):
    """Latin Hypercube samples in [0,1)^ (n x d); uniform fallback if no scipy.qmc."""
    if _HAVE_QMC:
        return LatinHypercube(d=d, seed=seed).random(n)
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _pick_tier(u):
    """Map u in [0,1) to a size tier by cumulative fraction, return (name, rmin, rmax)."""
    acc = 0.0
    for name, rmin, rmax, frac in SIZE_TIERS:
        acc += frac
        if u < acc:
            return name, rmin, rmax
    name, rmin, rmax, _ = SIZE_TIERS[-1]
    return name, rmin, rmax


def generate(n_samples, seed=42, healthy_only=False, defect_types=None):
    """Return a DOE dict: {'samples': [...], 'metadata': {...}}."""
    defect_types = list(defect_types) if defect_types else list(DEFECT_TYPES)
    samples = []

    if healthy_only:
        for i in range(n_samples):
            t_u = _lhs(n_samples, 2, seed)[i]
            samples.append({
                "id": i,
                "job_name": "CryoTank_Healthy_%04d" % i,
                "defect_params": {
                    "defect_type": "healthy",
                    "temperature_K": float(TEMPERATURE_K[int(t_u[0] * len(TEMPERATURE_K)) % len(TEMPERATURE_K)]),
                    "pressure_MPa": round(float(PRESSURE_MPA_RANGE[0] + t_u[1] *
                                                (PRESSURE_MPA_RANGE[1] - PRESSURE_MPA_RANGE[0])), 4),
                },
            })
        return {"samples": samples, "metadata": {"n_samples": n_samples,
                                                 "healthy_only": True, "seed": seed}}

    # 5 design dims: [defect_type, theta, z, size, operating(temp,press packed via 2 dims)]
    u = _lhs(n_samples, 6, seed)
    for i in range(n_samples):
        dtype = defect_types[int(u[i, 0] * len(defect_types)) % len(defect_types)]

        # weld_flaw sits ON a weld line; others use continuous theta
        if dtype == "weld_flaw":
            theta = float(WELD_THETA_DEG[int(u[i, 1] * len(WELD_THETA_DEG)) % len(WELD_THETA_DEG)])
        else:
            theta = round(float(THETA_RANGE[0] + u[i, 1] * (THETA_RANGE[1] - THETA_RANGE[0])), 2)

        z = round(float(Z_RANGE[0] + u[i, 2] * (Z_RANGE[1] - Z_RANGE[0])), 1)
        tier, rmin, rmax = _pick_tier(u[i, 3])
        radius = round(float(rmin + u[i, 4] * (rmax - rmin)), 1)
        temp = float(TEMPERATURE_K[int(u[i, 5] * len(TEMPERATURE_K)) % len(TEMPERATURE_K)])
        press = round(float(PRESSURE_MPA_RANGE[0] + (u[i, 5] * 7.0 % 1.0) *
                            (PRESSURE_MPA_RANGE[1] - PRESSURE_MPA_RANGE[0])), 4)

        samples.append({
            "id": i,
            "job_name": "CryoTank_%s_%04d" % (dtype, i),
            "defect_params": {
                "defect_type": dtype,
                "theta_deg": theta,
                "z_center": z,
                "radius": radius,
                "size_tier": tier,
                "temperature_K": temp,
                "pressure_MPa": press,
            },
        })

    return {"samples": samples,
            "metadata": {"n_samples": n_samples, "seed": seed,
                         "defect_types": defect_types,
                         "size_tiers": [t[0] for t in SIZE_TIERS]}}


def main():
    p = argparse.ArgumentParser(description="Cryogenic hydrogen-tank DOE generator")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--output", type=str, default="doe_cryotank_1sample.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--healthy_only", action="store_true", default=False)
    p.add_argument("--defect_types", nargs="+", default=None, choices=DEFECT_TYPES)
    args = p.parse_args()

    doe = generate(args.n_samples, seed=args.seed,
                   healthy_only=args.healthy_only, defect_types=args.defect_types)
    with open(args.output, "w") as f:
        json.dump(doe, f, indent=2)
    print("Wrote %d samples -> %s" % (len(doe["samples"]), args.output))


if __name__ == "__main__":
    main()
