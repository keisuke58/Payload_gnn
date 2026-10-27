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
import math
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

# Spray-on foam insulation (SOFI) on the outer wall + its bond line (CZM layer).
# DESIGN ASSUMPTIONS — polyurethane-foam order of magnitude (NCFI/BX class); verify
# against material data. The bond's strength is foam-limited (~0.3 MPa), not epoxy.
FOAM = {"t_mm": 25.0, "E_MPa": 7.0, "nu": 0.30, "density_t_mm3": 3.8e-11,
        "alpha_per_C": 5.4e-5}
BOND = {"t_mm": 0.2, "Kn": 1.0e4, "Ks": 5.0e3, "tn": 0.35, "ts": 0.30,
        "GIc": 0.10, "GIIc": 0.30, "BK_eta": 2.284}
BOND_DEBONDED_FACTOR = 1.0e-3   # pre-debonded patch stiffness = K/1000 (as cohesive_fairing)


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


def defect_patch(theta_deg, z_center, radius):
    """Localized (theta, z) patch bounding a defect of `radius` [mm] on the barrel.

    The circular defect is approximated by its bounding box in (arc, z): half-width
    radius/RADIUS [rad] in theta and `radius` in z, clamped inside the sector and barrel.
    Raises ValueError if the centre lies outside the modelled sector or barrel.
    """
    theta = float(theta_deg)
    zc = float(z_center)
    r = float(radius)
    if not (0.0 < theta < SECTOR_DEG):
        raise ValueError("defect theta_deg=%.2f outside the %.0f deg model sector"
                         % (theta, SECTOR_DEG))
    if not (0.0 < zc < H_BARREL):
        raise ValueError("defect z_center=%.1f outside the barrel [0, %.0f] mm" % (zc, H_BARREL))
    half_th = math.degrees(r / RADIUS)
    return {"theta_deg": [round(max(0.5, theta - half_th), 4),
                          round(min(SECTOR_DEG - 0.5, theta + half_th), 4)],
            "z_mm": [round(max(1.0, zc - r), 3), round(min(H_BARREL - 1.0, zc + r), 3)]}


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
    # Insulation layer: always on for insulation_debond (the defect lives in its bond
    # line); optional for other samples (e.g. matching healthy baselines).
    insulation = bool(defect_params.get("insulation", dtype == "insulation_debond"))
    if dtype == "insulation_debond" and not insulation:
        raise ValueError("insulation_debond requires the insulation layer (insulation=True)")
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
        "insulation": ({"enabled": True, "foam": dict(FOAM), "bond": dict(BOND)}
                       if insulation else {"enabled": False}),
        "defect": {"type": dtype},
    }
    if dtype != "healthy":
        plan["defect"].update({
            "theta_deg": defect_params.get("theta_deg"),
            "z_center_mm": defect_params.get("z_center"),
            "radius_mm": defect_params.get("radius"),
            "patch": defect_patch(defect_params.get("theta_deg"),
                                  defect_params.get("z_center"),
                                  defect_params.get("radius")),
            # How the defect enters the FE model:
            #   stiffness_knockdown: wall E x DEFECT_E_KNOCKDOWN on the patch
            #   czm_debond: bond-line cohesive elements pre-debonded (K x BOND_DEBONDED_FACTOR)
            "mechanism": "czm_debond" if dtype == "insulation_debond" else "stiffness_knockdown",
            "model": {
                "weld_flaw": "wall stiffness knockdown on the weld-line patch",
                "microcrack": "wall stiffness knockdown on the patch",
                "insulation_debond": "pre-debonded CZM bond line between wall and foam",
            }.get(dtype, "TODO"),
        })
    return plan


# ==============================================================================
# Abaqus build (cluster only)
# ==============================================================================
# M2 baseline defect model: Young's modulus knockdown factor applied to a (theta, z) patch
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
      - Defect (if any) on a localized (theta, z) patch, the bounding box of the defect
        circle (see defect_patch): weld_flaw / microcrack = wall stiffness knockdown;
        insulation_debond = pre-debonded cohesive (COH3D8) bond line between the wall
        and a foam shell (K x BOND_DEBONDED_FACTOR), the rest of the bond carries
        MAXS initiation + BK energy evolution so debond growth is representable.
      - Insulation (bond + foam) is added for insulation_debond or with --insulation;
        foam CTE >> Al-Li CTE, so cooling loads the bond line (the debond driver).
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
    mechanism = dfx.get("mechanism")
    if mechanism == "stiffness_knockdown":
        mat_d = dict(mat)
        mat_d["E_MPa"] = mat["E_MPa"] * DEFECT_E_KNOCKDOWN.get(dtype, 0.5)
        _add_material(model, 'AlLi_Defect', mat_d, ref_C)
        model.HomogeneousShellSection(name='Sec-Defect', material='AlLi_Defect', thickness=tw,
                                      idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT,
                                      integrationRule=SIMPSON, numIntPts=5)

    # ---------------- Part: barrel sector (revolve a vertical line about Y) ----------------
    p = _revolve_part(model, 'Tank', [((R, 0.0), (R, Hb))], sector_deg, Hb, solid=False)
    region = p.Set(faces=p.faces, name='Set-All')
    p.SectionAssignment(region=region, sectionName='Sec-Wall')
    # Revolve direction sign: which side of the z=0 plane the sector sweeps into.
    # Used for every radial datum plane / CSYS below so they match the geometry.
    sgn = 1.0 if p.faces[0].pointOn[0][2] >= 0.0 else -1.0

    # ---------------- Defect (localized (theta, z) patch) ----------------
    if mechanism == "stiffness_knockdown":
        try:
            inside = _partition_patch(p, dfx["patch"], sgn, cells=False)
            if inside:
                reg_d = p.Set(faces=_seq_array(p.faces, inside), name='Set-Defect')
                p.SectionAssignment(region=reg_d, sectionName='Sec-Defect')
                print("Defect patch (wall knockdown): %d face(s) %s" % (len(inside), dfx["patch"]))
            else:
                print("WARNING: defect patch selection empty %s" % dfx["patch"])
        except Exception as e:
            print("WARNING: defect partition failed (%s); continuing with healthy section" % e)

    # ---------------- Insulation: bond line (CZM) + foam ----------------
    ins = plan["insulation"]
    p_bond = p_foam = None
    if ins["enabled"]:
        p_bond, p_foam = _build_insulation_parts(model, ins, R, tw, Hb, sector_deg, ref_C)
        if mechanism == "czm_debond":
            try:
                inside = _partition_patch(p_bond, dfx["patch"], sgn, cells=True)
                if inside:
                    reg_d = p_bond.Set(cells=_seq_array(p_bond.cells, inside),
                                       name='Set-Debond')
                    p_bond.SectionAssignment(region=reg_d, sectionName='Sec-Bond-Debonded')
                    print("Defect patch (pre-debonded bond): %d cell(s) %s"
                          % (len(inside), dfx["patch"]))
                else:
                    print("WARNING: debond patch selection empty %s" % dfx["patch"])
            except Exception as e:
                print("WARNING: debond partition failed (%s); bond left intact" % e)

    # ---------------- Assembly & mesh ----------------
    a = model.rootAssembly
    inst = a.Instance(name='Tank-1', part=p, dependent=OFF)
    arc = R * sector_deg * PI / 180.0
    seed = max(arc, Hb) / 60.0
    a.seedPartInstance(regions=(inst,), size=seed, deviationFactor=0.1)
    a.generateMesh(regions=(inst,))
    print("Mesh wall: seed=%.2f mm, %d elements" % (seed, len(inst.elements)))
    shells = [inst]
    solids = []
    if ins["enabled"]:
        inst_foam = a.Instance(name='Foam-1', part=p_foam, dependent=OFF)
        a.seedPartInstance(regions=(inst_foam,), size=seed, deviationFactor=0.1)
        a.generateMesh(regions=(inst_foam,))
        inst_bond = a.Instance(name='Bond-1', part=p_bond, dependent=OFF)
        _mesh_bond(a, inst_bond, min(seed, 25.0))
        _tie_insulation(model, a, inst, inst_bond, inst_foam, ins, R, tw)
        shells.append(inst_foam)
        solids.append(inst_bond)

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
    for it in shells + solids:
        model.Temperature(name='T_ref_%s' % it.name, createStepName='Initial',
                          region=it.sets['Set-All'], distributionType=UNIFORM,
                          magnitudes=(ref_C,))
        model.Temperature(name='T_op_%s' % it.name, createStepName='Step-1',
                          region=it.sets['Set-All'], distributionType=UNIFORM,
                          magnitudes=(op_C,))

    # ---------------- Symmetry BCs (sector cut planes) ----------------
    # Shells (wall, foam): cut-plane EDGES, u_n = 0 and the two in-plane rotations.
    # Solids (bond line): cut-plane FACES, u_n = 0 only (no rotational DOFs).
    # theta=0 cut is the global plane z=0. theta=sector cut uses a local Cartesian CSYS
    # whose local Z is the cut-plane normal (as generate_fairing_dataset.py), built with
    # the same revolve sign `sgn` as the geometry.
    th = radians(sector_deg)
    # Cut-plane entities sit on the plane to numerical precision; a tight tolerance keeps
    # short partition pieces near the cut (e.g. bottom-arc segments) from being caught.
    tol = radians(0.05)
    cs = a.DatumCsysByThreePoints(name='CSYS-SymTheta', coordSysType=CARTESIAN,
                                  origin=(0.0, 0.0, 0.0),
                                  point1=(cos(th), 0.0, sgn * sin(th)),  # local X in plane
                                  point2=(0.0, 1.0, 0.0))               # local Z = normal
    for it in shells + solids:
        is_solid = it in solids
        seq = it.faces if is_solid else it.edges
        for tag, target, csys in (("T0", 0.0, None), ("Ta", th, a.datums[cs.id])):
            try:
                hits = []
                for ent in seq:
                    x, _, z = ent.pointOn[0]
                    if (x * x + z * z) < 1.0:   # skip anything on the axis
                        continue
                    if abs(abs(atan2(z, x)) - target) < tol:
                        hits.append(ent)
                if not hits:
                    print("WARNING: no cut-plane entities for Sym_%s on %s" % (tag, it.name))
                    continue
                arr = _seq_array(seq, hits)
                bc_set = (a.Set(faces=arr, name='BC_Sym_%s_%s' % (tag, it.name)) if is_solid
                          else a.Set(edges=arr, name='BC_Sym_%s_%s' % (tag, it.name)))
                kw = {"u3": 0.0} if is_solid else {"u3": 0.0, "ur1": 0.0, "ur2": 0.0}
                if csys is not None:
                    kw["localCsys"] = csys
                model.DisplacementBC(name='Sym_%s_%s' % (tag, it.name),
                                     createStepName='Initial', region=bc_set, **kw)
                print("BC Sym_%s on %s: %d entit(ies)" % (tag, it.name, len(hits)))
            except Exception as e:
                print("WARNING: Sym_%s BC on %s failed (%s)" % (tag, it.name, e))

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


def _revolve_part(model, name, lines, angle_deg, h_max, solid):
    """Revolve a closed (solid) or open (shell) profile about the global Y axis.

    The sketch carries the revolve axis as a ConstructionLine (x = 0), as in
    generate_fairing_dataset.py — BaseShellRevolve/BaseSolidRevolve need it.
    `lines` is a list of ((x1, y1), (x2, y2)) segments in the (r, axial) plane.
    """
    s = model.ConstrainedSketch(name='prof_%s' % name, sheetSize=20000.0)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, h_max + 1000.0))
    for p1, p2 in lines:
        s.Line(point1=p1, point2=p2)
    part = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    if solid:
        part.BaseSolidRevolve(sketch=s, angle=angle_deg, flipRevolveDirection=OFF)
    else:
        part.BaseShellRevolve(sketch=s, angle=angle_deg, flipRevolveDirection=OFF)
    return part


def _partition_patch(part, patch, sgn, cells):
    """Cut `part` along the defect patch and return the faces/cells inside it.

    Two axial-height planes (z_lo, z_hi) and two radial planes through the axis
    (th_lo, th_hi, swept to the `sgn` side of z = 0). Selection uses pointOn.
    """
    from math import atan2, cos, degrees, radians, sin
    th_lo, th_hi = patch["theta_deg"]
    z_lo, z_hi = patch["z_mm"]
    planes = [part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=off)
              for off in (z_lo, z_hi)]
    for ang in (th_lo, th_hi):
        phi = radians(ang)
        planes.append(part.DatumPlaneByThreePoints(point1=(0.0, 0.0, 0.0),
                                                   point2=(0.0, 1.0, 0.0),
                                                   point3=(cos(phi), 0.0, sgn * sin(phi))))
    for dp in planes:
        if cells:
            part.PartitionCellByDatumPlane(datumPlane=part.datums[dp.id], cells=part.cells)
        else:
            part.PartitionFaceByDatumPlane(datumPlane=part.datums[dp.id], faces=part.faces)
    inside = []
    for ent in (part.cells if cells else part.faces):
        x, y, z = ent.pointOn[0]
        if z_lo < y < z_hi and th_lo < degrees(abs(atan2(z, x))) < th_hi:
            inside.append(ent)
    return inside


def _build_insulation_parts(model, ins, R, tw, Hb, sector_deg, ref_C):
    """Bond line (solid, cohesive) + foam (shell) outside the wall; returns (bond, foam).

    Radial stack: wall mid-surface R | wall outer R + tw/2 = r_w | bond [r_w, r_b] |
    foam mid-surface r_b + t_foam/2. Materials/sections mirror generate_cohesive_fairing.py
    (TRACTION elastic + MAXS initiation + BK energy evolution; pre-debonded = K x factor).
    """
    foam, bond = ins["foam"], ins["bond"]
    r_w = R + tw / 2.0
    r_b = r_w + bond["t_mm"]
    r_f = r_b + foam["t_mm"] / 2.0

    _add_material(model, 'Foam', foam, ref_C)
    model.HomogeneousShellSection(name='Sec-Foam', material='Foam', thickness=foam["t_mm"],
                                  idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT,
                                  integrationRule=SIMPSON, numIntPts=5)

    Kn, Ks = bond["Kn"], bond["Ks"]
    m = model.Material(name='Mat-Bond')
    m.Elastic(type=TRACTION, table=((Kn, Ks, Ks),))
    m.Density(table=((1.2e-9,),))
    m.Expansion(table=((foam["alpha_per_C"],),), zero=ref_C)
    m.MaxsDamageInitiation(table=((bond["tn"], bond["ts"], bond["ts"]),))
    m.maxsDamageInitiation.DamageEvolution(
        type=ENERGY, mixedModeBehavior=BK, power=bond["BK_eta"],
        table=((bond["GIc"], bond["GIIc"], bond["GIIc"]),))
    f = BOND_DEBONDED_FACTOR
    md = model.Material(name='Mat-Bond-Debonded')
    md.Elastic(type=TRACTION, table=((Kn * f, Ks * f, Ks * f),))
    md.Density(table=((1.2e-9,),))
    md.Expansion(table=((foam["alpha_per_C"],),), zero=ref_C)
    for sec, mat_name in (('Sec-Bond', 'Mat-Bond'), ('Sec-Bond-Debonded', 'Mat-Bond-Debonded')):
        model.CohesiveSection(name=sec, material=mat_name, response=TRACTION_SEPARATION,
                              initialThicknessType=SPECIFY, initialThickness=bond["t_mm"])

    p_foam = _revolve_part(model, 'Foam', [((r_f, 0.0), (r_f, Hb))], sector_deg, Hb,
                           solid=False)
    p_foam.SectionAssignment(region=p_foam.Set(faces=p_foam.faces, name='Set-All'),
                             sectionName='Sec-Foam')
    p_bond = _revolve_part(model, 'Bond',
                           [((r_w, 0.0), (r_w, Hb)), ((r_w, Hb), (r_b, Hb)),
                            ((r_b, Hb), (r_b, 0.0)), ((r_b, 0.0), (r_w, 0.0))],
                           sector_deg, Hb, solid=True)
    p_bond.SectionAssignment(region=p_bond.Set(cells=p_bond.cells, name='Set-All'),
                             sectionName='Sec-Bond')
    print("Insulation: bond %.2f mm (CZM) at r=[%.2f, %.2f], foam %.1f mm shell at r=%.2f"
          % (bond["t_mm"], r_w, r_b, foam["t_mm"], r_f))
    return p_bond, p_foam


def _mesh_bond(a, inst, seed):
    """SWEEP + COH3D8 for the bond line; C3D10 fallback warns that CZM is inactive."""
    from mesh import ElemType
    a.seedPartInstance(regions=(inst,), size=seed, deviationFactor=0.1)
    try:
        a.setMeshControls(regions=inst.cells, elemShape=HEX, technique=SWEEP)
        a.setElementType(regions=(inst.cells,),
                         elemTypes=(ElemType(elemCode=COH3D8, elemLibrary=STANDARD),
                                    ElemType(elemCode=COH3D6, elemLibrary=STANDARD)))
        a.generateMesh(regions=(inst,))
        print("Mesh bond: SWEEP COH3D8, %d elements" % len(inst.elements))
        return
    except Exception as e:
        print("WARNING: bond SWEEP/COH3D8 failed (%s); falling back to C3D10" % e)
    try:
        a.deleteMesh(regions=(inst,))
    except Exception:
        pass
    a.setMeshControls(regions=inst.cells, elemShape=TET, technique=FREE)
    a.setElementType(regions=(inst.cells,),
                     elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),))
    a.generateMesh(regions=(inst,))
    print("WARNING: bond meshed with C3D10 — CZM NOT active (debond not represented)")


def _tie_insulation(model, a, inst_wall, inst_bond, inst_foam, ins, R, tw):
    """Tie wall-outer <-> bond-inner and bond-outer <-> foam-inner (thickness=ON).

    Main = bond solid faces, secondary = shell surfaces, as generate_cohesive_fairing.py.
    Shell sides follow the pressure convention (side1 = inner): wall outer = side2,
    foam inner = side1 (foam is revolved the same way as the wall).
    """
    from math import sqrt
    t_b = ins["bond"]["t_mm"]
    t_f = ins["foam"]["t_mm"]
    r_w = R + tw / 2.0
    r_b = r_w + t_b

    def faces_at(inst, r):
        hits = [fc for fc in inst.faces
                if abs(sqrt(fc.pointOn[0][0] ** 2 + fc.pointOn[0][2] ** 2) - r) < 0.25 * t_b]
        return _seq_array(inst.faces, hits), len(hits)

    try:
        bond_in, n_in = faces_at(inst_bond, r_w)
        bond_out, n_out = faces_at(inst_bond, r_b)
        s_bin = a.Surface(side1Faces=bond_in, name='Surf-Bond-Inner')
        s_bout = a.Surface(side1Faces=bond_out, name='Surf-Bond-Outer')
        s_wall = a.Surface(side2Faces=inst_wall.faces, name='Surf-Wall-Outer')
        s_foam = a.Surface(side1Faces=inst_foam.faces, name='Surf-Foam-Inner')
        model.Tie(name='Tie-Wall-Bond', main=s_bin, secondary=s_wall,
                  positionToleranceMethod=SPECIFIED, positionTolerance=tw + t_b,
                  adjust=ON, tieRotations=ON, thickness=ON)
        model.Tie(name='Tie-Bond-Foam', main=s_bout, secondary=s_foam,
                  positionToleranceMethod=SPECIFIED, positionTolerance=0.5 * t_f + t_b,
                  adjust=ON, tieRotations=ON, thickness=ON)
        print("Ties: wall<->bond (%d faces), bond<->foam (%d faces)" % (n_in, n_out))
    except Exception as e:
        print("WARNING: insulation ties failed (%s)" % e)


def _seq_array(seq, items):
    """Combine individual Edge/Face objects into one Edge/FaceArray (slicing by index).

    Abaqus Sets require an EdgeArray/FaceArray; `seq[i:i+1]` yields a 1-element
    array and these arrays support `+`. Abaqus-only helper.
    """
    arr = None
    for it in items:
        piece = seq[it.index:it.index + 1]
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
    p.add_argument("--insulation", action="store_true", default=False,
                   help="Add the foam + CZM bond layer (automatic for insulation_debond)")
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
    if args.insulation:
        defect_params["insulation"] = True

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
