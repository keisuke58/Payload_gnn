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

BOTTOM_BC_CHOICES = ("skirt", "axial")


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
    closed_end = bool(defect_params.get("closed_end", True))
    bottom_bc = defect_params.get("bottom_bc", "skirt")
    if bottom_bc not in BOTTOM_BC_CHOICES:
        raise ValueError("bottom_bc must be one of %s, got %r" % (BOTTOM_BC_CHOICES, bottom_bc))
    plan = {
        "geometry": {"radius_mm": RADIUS, "h_barrel_mm": H_BARREL,
                     "sector_deg": SECTOR_DEG, "wall_mm": T_WALL},
        "material": mat,
        "loading": {"internal_pressure_MPa": press,
                    # Closed-end (dome) thrust carried by the barrel: N_z = pR/2 [N/mm],
                    # applied as an axial edge traction on the top edge.
                    "closed_end": closed_end,
                    "end_cap_thrust_N_per_mm": press * RADIUS / 2.0 if closed_end else 0.0,
                    "thermal": {"reference_K": REFERENCE_TEMP_K, "operating_K": temp,
                                "delta_K": temp - REFERENCE_TEMP_K}},
        # 'skirt': bottom edge fixed radially + axially (skirt/Y-ring attachment) ->
        #          local cryogenic thermal stress and edge bending at the junction.
        # 'axial': bottom edge fixed axially only -> free contraction, ~zero thermal stress.
        "boundary": {"bottom": bottom_bc},
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
# Abaqus build (cluster only)
# ==============================================================================
# M2 baseline defect model: Young's modulus knockdown factor applied to a z-band
# of the shell around the defect (stiffness-loss proxy; CZM/cohesive comes later).
DEFECT_E_KNOCKDOWN = {"weld_flaw": 0.5, "microcrack": 0.3, "insulation_debond": 0.2}

PI = 3.141592653589793


def _add_material(model, name, props, ref_C):
    """Create an isotropic Abaqus material (density, elastic, thermal expansion).

    Units: tonne/mm^3, MPa, 1/degC. `zero=ref_C` sets the stress-free reference
    temperature for the CTE so a Predefined Field at op_C drives contraction.
    Runs only inside Abaqus (model is an Abaqus Model object).
    """
    m = model.Material(name=name)
    m.Density(table=((props["density_t_mm3"],),))
    m.Elastic(table=((props["E_MPa"], props["nu"]),))
    m.Expansion(table=((props["alpha_per_C"],),), zero=ref_C)
    return m


def build_model(job_name, defect_params):
    """Build the Al-Li cryogenic tank barrel SECTOR shell model and write the INP.

    *** Requires cluster validation (milestone M1) — cannot be executed here. ***
    Key assumptions:
      - Single-part shell (S4R by default), revolved about global Y (Y = axial).
      - Units: mm, N, MPa, tonne; temperatures converted to degC for CTE.
      - Internal pressure applied on `side1Faces` of the revolved shell. The shell
        normal of BaseShellRevolve is NOT guaranteed to point inward; if the ODB
        shows hoop compression, flip to `side2Faces` (or negate magnitude).
      - Defect (if any) = z-band stiffness knockdown (Sec-Defect over the full
        sector arc between z_lo..z_hi). TODO(M2): localized (theta, z) patch / CZM.
      - Symmetry BCs: theta=0 cut edges selected by bounding box around the global
        z~0 plane; theta=sector cut edges selected by the angle of edge.pointOn
        (tolerant to revolve direction sign).
      - Closed-end thrust pR/2 [N/mm] applied as a +Y ShellEdgeLoad on the top edge
        (disable with --open-end). Expect membrane sigma_z = pR/2t mid-barrel.
      - Bottom edge: 'skirt' (default, u_r = u_axial = 0, cylindrical CSYS) or
        'axial' (u_axial = 0 only; free contraction -> ~zero thermal stress).
        Corner nodes shared with the symmetry edges get consistent zero constraints
        from two CSYSs; Abaqus may warn but the system is not over-determined in value.
    """
    if not _HAVE_ABAQUS:
        raise RuntimeError("Abaqus not available; use --dry-run or run via abaqus cae noGUI=...")
    # Local imports: keep module import Abaqus-free and dependency-free.
    import os
    from math import atan2, cos, radians, sin

    executeOnCaeStartup()
    plan = resolve_plan(defect_params)
    model = mdb.models['Model-1']

    geo = plan["geometry"]
    mat = plan["material"]
    load = plan["loading"]
    dfx = plan["defect"]
    R = float(geo["radius_mm"])
    Hb = float(geo["h_barrel_mm"])
    sector_deg = float(geo["sector_deg"])
    tw = float(geo["wall_mm"])
    press = float(load["internal_pressure_MPa"])
    ref_C = REFERENCE_TEMP_K - 273.15
    op_C = float(load["thermal"]["operating_K"]) - 273.15
    dtype = dfx["type"]
    print("build_model: job=%s defect=%s R=%.1f Hb=%.1f sector=%.1f tw=%.2f P=%.3f MPa "
          "T: %.1fC -> %.1fC" % (job_name, dtype, R, Hb, sector_deg, tw, press, ref_C, op_C))

    # ---------------- Materials & sections ----------------
    _add_material(model, 'AlLi', mat, ref_C)
    model.HomogeneousShellSection(name='Sec-Wall', material='AlLi', thickness=tw,
                                  idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT,
                                  integrationRule=SIMPSON, numIntPts=5)
    if dtype != 'healthy':
        mat_d = dict(mat)
        mat_d["E_MPa"] = mat["E_MPa"] * DEFECT_E_KNOCKDOWN.get(dtype, 0.5)
        _add_material(model, 'AlLi_Defect', mat_d, ref_C)
        model.HomogeneousShellSection(name='Sec-Defect', material='AlLi_Defect', thickness=tw,
                                      idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT,
                                      integrationRule=SIMPSON, numIntPts=5)

    # ---------------- Part: barrel sector (revolve a vertical line about Y) ----------------
    s = model.ConstrainedSketch(name='prof', sheetSize=max(2 * R, Hb) * 2.0)
    s.Line(point1=(R, 0.0), point2=(R, Hb))
    p = model.Part(name='Tank', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShellRevolve(sketch=s, angle=sector_deg, flipRevolveDirection=OFF)

    region = p.Set(faces=p.faces, name='Set-All')
    p.SectionAssignment(region=region, sectionName='Sec-Wall')

    # ---------------- Defect partition (M2 baseline: z-band knockdown) ----------------
    # The defect is approximated as a circumferential z-band [z_lo, z_hi] of reduced
    # stiffness spanning the whole sector arc. TODO(M2): localized (theta, z) patch.
    if dtype != 'healthy':
        try:
            z_center = float(dfx["z_center_mm"])
            radius = float(dfx["radius_mm"])
            z_lo = max(1.0, z_center - radius)
            z_hi = min(Hb - 1.0, z_center + radius)
            for off in (z_lo, z_hi):
                dp = p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=off)
                p.PartitionFaceByDatumPlane(datumPlane=p.datums[dp.id], faces=p.faces)
            band = p.faces.getByBoundingBox(xMin=-1e9, yMin=z_lo - 0.1, zMin=-1e9,
                                            xMax=1e9, yMax=z_hi + 0.1, zMax=1e9)
            if band:
                reg_d = p.Set(faces=band, name='Set-Defect')
                p.SectionAssignment(region=reg_d, sectionName='Sec-Defect')
                print("Defect band z=[%.1f, %.1f] -> %d face(s) assigned Sec-Defect"
                      % (z_lo, z_hi, len(band)))
            else:
                print("WARNING: defect band selection empty (z=[%.1f, %.1f])" % (z_lo, z_hi))
        except Exception as e:
            print("WARNING: defect partition failed (%s); continuing with healthy section" % e)

    # ---------------- Assembly & mesh ----------------
    a = model.rootAssembly
    inst = a.Instance(name='Tank-1', part=p, dependent=OFF)
    arc = R * sector_deg * PI / 180.0
    seed = max(arc, Hb) / 60.0
    a.seedPartInstance(regions=(inst,), size=seed, deviationFactor=0.1)
    a.generateMesh(regions=(inst,))
    print("Mesh: seed=%.2f mm, %d elements" % (seed, len(inst.elements)))

    # ---------------- Step & outputs ----------------
    model.StaticStep(name='Step-1', previous='Initial')
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF', 'TEMP', 'E'))

    # ---------------- Internal pressure ----------------
    # NOTE: side1Faces assumed to be the inner (pressurized) side. Verify sign on the
    # cluster (hoop stress must be tensile); flip to side2Faces if not.
    surf = a.Surface(side1Faces=inst.faces, name='Surf-Inner')
    model.Pressure(name='Internal_P', createStepName='Step-1', region=surf,
                   distributionType=UNIFORM, magnitude=press)

    # ---------------- Closed-end (dome) thrust on the top edge ----------------
    # The barrel alone is an open cylinder; a real tank's domes pull it axially with
    # N_z = pR/2. Apply that as a +Y edge traction on the top edge (y = Hb) so the
    # membrane axial stress is pR/2t. Bottom edge reacts it (axial BC below).
    thrust = float(load["end_cap_thrust_N_per_mm"])
    if thrust > 0.0:
        try:
            et = inst.edges.getByBoundingBox(xMin=-1e9, yMin=Hb - 0.1, zMin=-1e9,
                                             xMax=1e9, yMax=Hb + 0.1, zMax=1e9)
            surf_top = a.Surface(side1Edges=et, name='Surf-TopEdge')
            model.ShellEdgeLoad(name='EndCap_Thrust', createStepName='Step-1',
                                region=surf_top, magnitude=thrust,
                                distributionType=UNIFORM, traction=GENERAL,
                                directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                                follower=OFF, resultant=OFF)
            print("Load EndCap_Thrust: %.1f N/mm on %d top edge(s)" % (thrust, len(et)))
        except Exception as e:
            print("WARNING: end-cap thrust failed (%s)" % e)

    # ---------------- Thermal: ref (Initial) -> operating (Step-1) ----------------
    model.Temperature(name='T_ref', createStepName='Initial', region=inst.sets['Set-All'],
                      distributionType=UNIFORM, magnitudes=(ref_C,))
    model.Temperature(name='T_op', createStepName='Step-1', region=inst.sets['Set-All'],
                      distributionType=UNIFORM, magnitudes=(op_C,))

    # ---------------- Symmetry BCs (sector cut planes) ----------------
    th = radians(sector_deg)
    # theta=0 cut: edges lie in global plane z~0 (x>0).
    try:
        e0 = inst.edges.getByBoundingBox(xMin=1.0, yMin=-0.1, zMin=-0.1,
                                         xMax=1e9, yMax=Hb + 0.1, zMax=0.1)
        set0 = a.Set(edges=e0, name='BC_Sym_T0')
        model.DisplacementBC(name='Sym_T0', createStepName='Initial', region=set0,
                             u3=0.0, ur1=0.0, ur2=0.0)
        print("BC Sym_T0: %d edge(s)" % len(e0))
    except Exception as e:
        print("WARNING: Sym_T0 BC failed (%s)" % e)

    # theta=sector cut: local Cartesian CSYS with local Z normal to the cut plane
    # (same construction as generate_fairing_dataset.py). Edges picked by angle of
    # edge.pointOn, accepting +-th since the revolve direction sign is uncertain.
    try:
        cs = a.DatumCsysByThreePoints(name='CSYS-SymTheta', coordSysType=CARTESIAN,
                                      origin=(0.0, 0.0, 0.0),
                                      point1=(cos(th), 0.0, sin(th)),   # local X = tangent
                                      point2=(0.0, 1.0, 0.0))           # local Z = normal
        tol = radians(0.5)
        matched = []
        for edge in inst.edges:
            x, y, z = edge.pointOn[0]
            if (x * x + z * z) < 1.0:   # skip anything on the axis
                continue
            ang = atan2(z, x)
            if abs(abs(ang) - th) < tol:
                matched.append(edge)
        if matched:
            set_a = a.Set(edges=_edge_array(inst, matched), name='BC_Sym_Ta')
            model.DisplacementBC(name='Sym_Ta', createStepName='Initial', region=set_a,
                                 u3=0.0, ur1=0.0, ur2=0.0, localCsys=a.datums[cs.id])
            print("BC Sym_Ta: %d edge(s) at theta=%.1f deg" % (len(matched), sector_deg))
        else:
            print("WARNING: no edges found at theta=%.1f deg for Sym_Ta" % sector_deg)
    except Exception as e:
        print("WARNING: Sym_Ta BC failed (%s)" % e)

    # Bottom edge y~0: 'axial' = axial fix only (free contraction);
    # 'skirt' = radial + axial fix in a cylindrical CSYS (local 1 = R, local 3 = global Y),
    # i.e. a stiff skirt/Y-ring attachment that resists cryogenic contraction.
    bottom_bc = plan["boundary"]["bottom"]
    try:
        eb = inst.edges.getByBoundingBox(xMin=-1e9, yMin=-0.1, zMin=-1e9,
                                         xMax=1e9, yMax=0.1, zMax=1e9)
        setb = a.Set(edges=eb, name='BC_Bottom')
        if bottom_bc == "skirt":
            # local X = (1,0,0); point2 in local XY plane -> local Z = X x (0,0,-1) = +Y
            cyl = a.DatumCsysByThreePoints(name='CSYS-Cyl', coordSysType=CYLINDRICAL,
                                           origin=(0.0, 0.0, 0.0),
                                           point1=(1.0, 0.0, 0.0),
                                           point2=(0.0, 0.0, -1.0))
            model.DisplacementBC(name='Fix_Bottom_Skirt', createStepName='Initial',
                                 region=setb, u1=0.0, u3=0.0,
                                 localCsys=a.datums[cyl.id])
            print("BC Fix_Bottom_Skirt (u_r = u_axial = 0): %d edge(s)" % len(eb))
        else:
            model.DisplacementBC(name='Fix_Bottom_Axial', createStepName='Initial',
                                 region=setb, u2=0.0)
            print("BC Fix_Bottom_Axial: %d edge(s)" % len(eb))
    except Exception as e:
        print("WARNING: bottom BC failed (%s)" % e)

    # ---------------- Job ----------------
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT)
    mdb.saveAs(pathName=job_name + '.cae')
    print("Writing INP for '%s'..." % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    print("INP written: %s" % os.path.abspath(job_name + '.inp'))


def _edge_array(inst, edges):
    """Combine individual Edge objects into one EdgeArray (slicing by index).

    Abaqus Sets require an EdgeArray; `inst.edges[i:i+1]` yields a 1-element
    EdgeArray and EdgeArrays support `+`. Cluster-only helper.
    """
    arr = None
    for e in edges:
        piece = inst.edges[e.index:e.index + 1]
        arr = piece if arr is None else arr + piece
    return arr


def main():
    p = argparse.ArgumentParser(description="Cryogenic tank FEM generator (Abaqus)")
    p.add_argument("--job", type=str, default="CryoTank_Healthy_0000")
    p.add_argument("--defect", type=str, default=None,
                   help="JSON: single defect_params, or a DOE {'samples':[...]} (uses sample 0)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print the resolved analysis plan without Abaqus")
    p.add_argument("--open-end", action="store_true", default=False,
                   help="Omit the closed-end (dome) axial thrust on the top edge")
    p.add_argument("--bottom-bc", choices=BOTTOM_BC_CHOICES, default=None,
                   help="Bottom edge BC: skirt (radial+axial fixed, default) or axial only")
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
    defect_params = dict(defect_params)
    if args.open_end:
        defect_params["closed_end"] = False
    if args.bottom_bc:
        defect_params["bottom_bc"] = args.bottom_bc

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
