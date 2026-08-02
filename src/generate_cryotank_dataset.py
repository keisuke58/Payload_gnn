# -*- coding: utf-8 -*-
# generate_cryotank_dataset.py
# Abaqus CAE scaffold — H3 cryogenic LH2 tank (Al-Li) FEM with SHM defects.
#
# Design note: docs/HYDROGEN_TANK_SHM.md
# This is the SCAFFOLD for milestone M1 (memo §7): generate ONE healthy sample,
# eyeball the graph, then batch. Abaqus model-building bodies are marked TODO and
# should reuse the fairing generator logic (generate_fairing_dataset.py:
# sector part, symmetry BCs, mesh control; generate_cohesive_fairing.py /
# generate_czm_sector12.py: cohesive/CZM interfaces for debond & weld flaw).
#
# The Abaqus import is guarded so this module can be imported (and --dry-run
# executed) WITHOUT Abaqus — the cluster runs the real build:
#   abaqus cae noGUI=src/generate_cryotank_dataset.py -- --job <name> --defect <json>
# Local plan check (no Abaqus):
#   python src/generate_cryotank_dataset.py --dry-run --defect doe_cryotank_1sample.json

import argparse
import json
import sys

try:
    from abaqus import *              # noqa: F401,F403
    from abaqusConstants import *     # noqa: F401,F403
    from caeModules import *          # noqa: F401,F403
    from driverUtils import executeOnCaeStartup
    _HAVE_ABAQUS = True
except Exception:
    _HAVE_ABAQUS = False

# ==============================================================================
# PARAMETERS — Al-Li LH2 tank (design assumption; memo §1-§2, dimensions non-public)
# ==============================================================================
RADIUS = 2600.0        # mm, φ ≈ 5.2 m barrel
H_BARREL = 4000.0      # mm, barrel section modeled
SECTOR_DEG = 30.0      # deg, symmetric sector (1/12)
T_WALL = 4.0           # mm, wall thickness (design assumption)

# Al-Li material — RT baseline + cryogenic shift (memo §2; DESIGN ASSUMPTION, verify).
MAT_ALLI = {
    "name": "AlLi_2219",
    "rt": {"E_MPa": 75000.0, "nu": 0.33, "yield_MPa": 400.0, "alpha_per_C": 23.0e-6,
           "density_t_mm3": 2.70e-9},
    # cryogenic (20 K) multipliers relative to RT
    "cryo_factor": {"E": 1.10, "yield": 1.20, "alpha": 0.55},
}

REFERENCE_TEMP_K = 293.0


def material_at(temperature_K):
    """Return effective Al-Li properties at a temperature (RT interpolation endpoints)."""
    rt = MAT_ALLI["rt"]
    if temperature_K <= 20.0:
        cf = MAT_ALLI["cryo_factor"]
        return {"E_MPa": rt["E_MPa"] * cf["E"], "nu": rt["nu"],
                "yield_MPa": rt["yield_MPa"] * cf["yield"],
                "alpha_per_C": rt["alpha_per_C"] * cf["alpha"],
                "density_t_mm3": rt["density_t_mm3"], "temperature_K": temperature_K}
    return {"E_MPa": rt["E_MPa"], "nu": rt["nu"], "yield_MPa": rt["yield_MPa"],
            "alpha_per_C": rt["alpha_per_C"], "density_t_mm3": rt["density_t_mm3"],
            "temperature_K": temperature_K}


def resolve_plan(defect_params):
    """Build a human-readable analysis plan from defect_params (no Abaqus needed)."""
    dtype = defect_params.get("defect_type", "healthy")
    temp = float(defect_params.get("temperature_K", REFERENCE_TEMP_K))
    press = float(defect_params.get("pressure_MPa", 0.3))
    mat = material_at(temp)
    plan = {
        "geometry": {"radius_mm": RADIUS, "h_barrel_mm": H_BARREL,
                     "sector_deg": SECTOR_DEG, "wall_mm": T_WALL},
        "material": mat,
        "loading": {"internal_pressure_MPa": press,
                    "thermal": {"reference_K": REFERENCE_TEMP_K, "operating_K": temp,
                                "delta_K": temp - REFERENCE_TEMP_K}},
        "defect": {"type": dtype},
    }
    if dtype != "healthy":
        plan["defect"].update({
            "theta_deg": defect_params.get("theta_deg"),
            "z_center_mm": defect_params.get("z_center"),
            "radius_mm": defect_params.get("radius"),
            "model": {
                "weld_flaw": "cohesive/element weakening on weld line",
                "microcrack": "local stiffness loss + partial contact discontinuity",
                "insulation_debond": "interface cohesive degradation (reuse CZM)",
            }.get(dtype, "TODO"),
        })
    return plan


# ==============================================================================
# Abaqus build (cluster only) — bodies TODO, reuse fairing generator logic
# ==============================================================================
def build_model(job_name, defect_params):
    if not _HAVE_ABAQUS:
        raise RuntimeError("Abaqus not available. Use --dry-run for a local plan check, "
                           "or run via: abaqus cae noGUI=generate_cryotank_dataset.py -- ...")
    executeOnCaeStartup()
    plan = resolve_plan(defect_params)
    # TODO(M1): create sector shell part (reuse generate_fairing_dataset.py sector logic)
    # TODO(M1): assign Al-Li material from plan["material"]; set expansion (alpha) + ref temp
    # TODO(M1): sections + assembly + symmetry BCs on circumferential edges
    # TODO(M1): Step-1 static: internal pressure (plan["loading"]) + Predefined Field
    #           temperature (operating_K) for cryogenic thermal stress
    # TODO(M2): insert defect per plan["defect"]["model"]
    #           - insulation_debond / weld_flaw: cohesive interface (generate_cohesive_fairing.py,
    #             generate_czm_sector12.py)
    #           - microcrack: local stiffness reduction + seam
    # TODO(M1): mesh (continuum/solid shell) + element type
    # TODO(M1): create + write Job(job_name)
    raise NotImplementedError("Abaqus model build is a scaffold — implement M1 bodies.")


def main():
    p = argparse.ArgumentParser(description="Cryogenic tank FEM generator (Abaqus)")
    p.add_argument("--job", type=str, default="CryoTank_Healthy_0000")
    p.add_argument("--defect", type=str, default=None,
                   help="JSON: single defect_params, or a DOE {'samples':[...]} (uses sample 0)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print the resolved analysis plan without Abaqus")
    # Abaqus passes script args after '--'; argparse needs them isolated.
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = p.parse_args(argv)

    defect_params = {"defect_type": "healthy", "temperature_K": 20.0, "pressure_MPa": 0.3}
    if args.defect:
        with open(args.defect) as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            defect_params = data["samples"][0]["defect_params"]
        elif isinstance(data, dict) and "defect_params" in data:
            defect_params = data["defect_params"]
        else:
            defect_params = data

    if args.dry_run or not _HAVE_ABAQUS:
        plan = resolve_plan(defect_params)
        print(json.dumps({"job": args.job, "abaqus_available": _HAVE_ABAQUS, "plan": plan},
                         indent=2))
        if not args.dry_run and not _HAVE_ABAQUS:
            print("\n[note] Abaqus not found — printed plan only. "
                  "Run on the cluster to build the model.", file=sys.stderr)
        return

    build_model(args.job, defect_params)


if __name__ == "__main__":
    main()
