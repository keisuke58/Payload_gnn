#!/usr/bin/env python3
"""
Generate Abaqus/Explicit INP for CompDam_DGD flat plate impact validation.

Model: CFRP/Al-HC sandwich panel with ply-by-ply C3D8R solid elements
Purpose: Validate progressive damage and compare with equivalent stiffness reduction

Usage:
    python scripts/generate_compdam_flatplate.py
    cd abaqus_work/compdam_flatplate
    bash run_compdam.sh
"""

import numpy as np
import os
import shutil
from pathlib import Path

# ============================================================
# Model Parameters
# ============================================================

# Plate geometry (mm)
PLATE_LX = 100.0
PLATE_LY = 100.0

# Layup [45/0/-45/90]s — bottom (inner skin) to top (outer skin)
PLY_ANGLES = [45, 0, -45, 90, 90, -45, 0, 45]
PLY_T = 0.125  # mm per ply (total skin = 1.0 mm)

# Al honeycomb core (simplified)
CORE_T = 10.0   # mm (reduced from 38mm for manageable model)
CORE_NZ = 5     # elements through core thickness

# In-plane element size
ELEM_XY = 1.0   # mm (refined for DGD crack kinematics resolution)

# Impact parameters
IMP_R = 6.35            # impact zone radius (mm)
PEAK_PRESSURE = 12.0    # MPa (below ~15 MPa DGD limit, 1mm mesh gives more damage)
PULSE_DURATION = 0.001  # s (1.0 ms half-sine — slower ramp for DGD stability)

# Analysis
TIME_PERIOD = 0.002     # s (2 ms total — longer pulse + response)
MASS_SCALING = 2        # factor (sqrt(2)=1.4x speedup, minimal)

# CompDam feature flags:
#   1xxxxx = matrix damage (CDM), 2=cohesive
#   x1xxxx = shear nonlinearity (Ramberg-Osgood)
#   xx1xxx = fiber tension damage
#   xxx1xx = fiber compression damage
#   xxxx0x = (reserved)
#   xxxxx1 = friction
FEATURE_FLAGS = 111101

# CFRP density (IM7-8552, 190 gsm)
CFRP_DENSITY = 1.57e-09  # tonne/mm^3

# Honeycomb core properties (Al 5052, consistent with project)
CORE_PROPS = {
    'density': 5.0e-11,  # tonne/mm^3 (50 kg/m^3)
    'E1': 1000.0, 'E2': 1.0, 'E3': 1.0,
    'nu12': 0.01, 'nu13': 0.01, 'nu23': 0.01,
    'G12': 400.0, 'G13': 240.0, 'G23': 1.0,
}


def layer_name(layer_idx, n_plies, n_core):
    """Generate descriptive element set name for each layer."""
    if layer_idx < n_plies:
        angle = PLY_ANGLES[layer_idx]
        return f"InnerPly{layer_idx + 1}_{angle:+d}"
    elif layer_idx < n_plies + n_core:
        ci = layer_idx - n_plies
        return f"Core{ci + 1}"
    else:
        pi = layer_idx - n_plies - n_core
        angle = PLY_ANGLES[pi]
        return f"OuterPly{pi + 1}_{angle:+d}"


def write_set(f, items, per_line=16):
    """Write node/element set items, 16 per line."""
    for i, item in enumerate(items):
        f.write(f"{item}")
        if i < len(items) - 1:
            f.write(", ")
            if (i + 1) % per_line == 0:
                f.write("\n")
        else:
            f.write("\n")


def main():
    outdir = Path("abaqus_work/compdam_flatplate")
    outdir.mkdir(parents=True, exist_ok=True)

    n_plies = len(PLY_ANGLES)  # 8
    n_core = CORE_NZ           # 5
    n_layers = 2 * n_plies + n_core  # 21

    nx = int(PLATE_LX / ELEM_XY)  # 50
    ny = int(PLATE_LY / ELEM_XY)  # 50
    n_nodes_xy = (nx + 1) * (ny + 1)  # 2601

    # ----------------------------------------------------------
    # Build z-plane coordinates (22 planes for 21 layers)
    # ----------------------------------------------------------
    z_planes = []
    z = 0.0

    # Inner skin: 8 plies (9 planes)
    for i in range(n_plies + 1):
        z_planes.append(z)
        if i < n_plies:
            z += PLY_T

    # Core: 5 layers (5 additional planes, bottom shared with inner skin top)
    core_dz = CORE_T / n_core
    for i in range(n_core):
        z += core_dz
        z_planes.append(z)

    # Outer skin: 8 plies (8 additional planes, bottom shared with core top)
    for i in range(n_plies):
        z += PLY_T
        z_planes.append(z)

    total_thickness = z_planes[-1]
    n_z_planes = len(z_planes)
    assert n_z_planes == n_layers + 1, f"Expected {n_layers + 1} z-planes, got {n_z_planes}"

    print(f"Model summary:")
    print(f"  Plate:      {PLATE_LX} x {PLATE_LY} mm")
    print(f"  Stack:      {n_plies} inner plies + {n_core} core + {n_plies} outer plies")
    print(f"  Thickness:  {total_thickness:.3f} mm")
    print(f"  Elem size:  {ELEM_XY} mm (in-plane)")
    print(f"  Z-planes:   {n_z_planes}")
    print(f"  Elements:   {n_layers * nx * ny}")
    print(f"  Nodes:      {n_z_planes * n_nodes_xy}")

    # ----------------------------------------------------------
    # Generate nodes
    # ----------------------------------------------------------
    nodes = []
    for iz in range(n_z_planes):
        zv = z_planes[iz]
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nid = iz * n_nodes_xy + iy * (nx + 1) + ix + 1
                nodes.append((nid, ix * ELEM_XY, iy * ELEM_XY, zv))

    # ----------------------------------------------------------
    # Generate elements by layer (C3D8R)
    # ----------------------------------------------------------
    elements = {}  # layer_idx -> [(eid, n1..n8), ...]
    eid = 1
    for li in range(n_layers):
        elems = []
        for iy in range(ny):
            for ix in range(nx):
                n1 = li * n_nodes_xy + iy * (nx + 1) + ix + 1
                n2 = n1 + 1
                n3 = n1 + (nx + 1) + 1
                n4 = n1 + (nx + 1)
                n5 = n1 + n_nodes_xy
                n6 = n2 + n_nodes_xy
                n7 = n3 + n_nodes_xy
                n8 = n4 + n_nodes_xy
                elems.append((eid, n1, n2, n3, n4, n5, n6, n7, n8))
                eid += 1
        elements[li] = elems
    total_elems = eid - 1

    # Layer classification
    inner_layers = list(range(n_plies))
    core_layers = list(range(n_plies, n_plies + n_core))
    outer_layers = list(range(n_plies + n_core, n_layers))

    # ----------------------------------------------------------
    # Identify impact zone elements (top ply, within IMP_R of center)
    # ----------------------------------------------------------
    cx, cy = PLATE_LX / 2.0, PLATE_LY / 2.0
    top_ply_layer = n_layers - 1  # outermost ply

    impact_elem_ids = []
    for eid_e, *_ in elements[top_ply_layer]:
        # Element centroid from (ix, iy)
        local_eid = eid_e - top_ply_layer * nx * ny
        iy_e = (local_eid - 1) // nx
        ix_e = (local_eid - 1) % nx
        ex = (ix_e + 0.5) * ELEM_XY
        ey = (iy_e + 0.5) * ELEM_XY
        r = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        if r <= IMP_R:
            impact_elem_ids.append(eid_e)

    # Impact zone nodes on top surface
    top_z_idx = n_z_planes - 1
    impact_top_nodes = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = ix * ELEM_XY
            y = iy * ELEM_XY
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if r <= IMP_R:
                nid = top_z_idx * n_nodes_xy + iy * (nx + 1) + ix + 1
                impact_top_nodes.append(nid)

    print(f"  Impact zone: {len(impact_elem_ids)} elements, {len(impact_top_nodes)} nodes")

    # ----------------------------------------------------------
    # Write INP
    # ----------------------------------------------------------
    inp_path = outdir / "compdam_flatplate_impact.inp"
    with open(inp_path, 'w') as f:
        # Header
        f.write("*Heading\n")
        f.write(f"** CompDam_DGD Flat Plate Impact Validation\n")
        f.write(f"** Plate: {PLATE_LX}x{PLATE_LY}mm, [45/0/-45/90]s x2 skins\n")
        f.write(f"** Core: Al-HC {CORE_T}mm, Total: {total_thickness:.3f}mm\n")
        f.write(f"** Elements: {total_elems} C3D8R, h_xy={ELEM_XY}mm\n")
        f.write(f"** Feature flags: {FEATURE_FLAGS}\n")
        f.write("**\n")

        # ==== Part ====
        f.write("*Part, name=Plate\n")

        # Nodes
        f.write("*Node\n")
        for nid, x, y, zv in nodes:
            f.write(f"{nid}, {x:.6f}, {y:.6f}, {zv:.6f}\n")

        # Elements per layer
        for li in range(n_layers):
            lname = layer_name(li, n_plies, n_core)
            f.write(f"*Element, type=C3D8R, elset={lname}\n")
            for row in elements[li]:
                f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, "
                        f"{row[5]}, {row[6]}, {row[7]}, {row[8]}\n")

        # All elements
        f.write("*Elset, elset=AllElements, generate\n")
        f.write(f"1, {total_elems}, 1\n")

        # Skin / core element sets
        for label, layers in [("InnerSkin", inner_layers),
                              ("CoreAll", core_layers),
                              ("OuterSkin", outer_layers)]:
            eids = []
            for li in layers:
                eids.extend([e[0] for e in elements[li]])
            f.write(f"*Elset, elset={label}\n")
            write_set(f, eids)

        # CFRP element set (all skin plies)
        cfrp_eids = []
        for li in inner_layers + outer_layers:
            cfrp_eids.extend([e[0] for e in elements[li]])
        f.write("*Elset, elset=AllCFRP\n")
        write_set(f, cfrp_eids)

        # Impact zone element set (top ply)
        f.write("*Elset, elset=ImpactZoneElems\n")
        write_set(f, impact_elem_ids)

        # ---- Node sets ----

        # Bottom surface
        bot_nodes = list(range(1, n_nodes_xy + 1))
        f.write("*Nset, nset=BotSurf\n")
        write_set(f, bot_nodes)

        # Top surface
        top_start = top_z_idx * n_nodes_xy + 1
        top_nodes = list(range(top_start, top_start + n_nodes_xy))
        f.write("*Nset, nset=TopSurf\n")
        write_set(f, top_nodes)

        # Impact zone nodes
        f.write("*Nset, nset=ImpactZone\n")
        write_set(f, impact_top_nodes)

        # All edge nodes (all z-planes)
        all_edge = []
        for iz in range(n_z_planes):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    if ix == 0 or ix == nx or iy == 0 or iy == ny:
                        nid = iz * n_nodes_xy + iy * (nx + 1) + ix + 1
                        all_edge.append(nid)
        f.write("*Nset, nset=AllEdges\n")
        write_set(f, all_edge)

        # Corner node (for rigid body motion constraint)
        corner1 = 1                         # (0,0,0)
        corner2 = nx + 1                    # (Lx,0,0)
        corner3 = ny * (nx + 1) + 1         # (0,Ly,0)

        # ---- Surface for impact pressure ----
        lname_top = layer_name(top_ply_layer, n_plies, n_core)
        f.write("*Surface, name=ImpactSurf, type=ELEMENT\n")
        for eid_e in impact_elem_ids:
            f.write(f"{eid_e}, S2\n")  # S2 = top face of C3D8R

        # ---- Orientations per ply angle ----
        # Default orientation for core (global XYZ)
        f.write("*Orientation, name=Ori_Core\n")
        f.write("1., 0., 0., 0., 1., 0.\n")
        f.write("3, 0.\n")

        unique_angles = sorted(set(PLY_ANGLES))
        for angle in unique_angles:
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Orientation, name={oname}\n")
            f.write(f"{c:.8f}, {s:.8f}, 0., {-s:.8f}, {c:.8f}, 0.\n")
            f.write("3, 0.\n")

        # ---- Sections (inside Part) ----
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Sections\n")
        f.write("** ============================================================\n")

        # CFRP ply sections (inner skin)
        for li in inner_layers:
            angle = PLY_ANGLES[li]
            lname = layer_name(li, n_plies, n_core)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Solid Section, elset={lname}, orientation={oname}, "
                    f"material=IM7-8552, controls=SectionCtrl-1\n")
            f.write(",\n")

        # Core sections (with default orientation for anisotropic material)
        for li in core_layers:
            lname = layer_name(li, n_plies, n_core)
            f.write(f"*Solid Section, elset={lname}, orientation=Ori_Core, material=HC-Core\n")
            f.write(",\n")

        # CFRP ply sections (outer skin)
        for li in outer_layers:
            pi = li - n_plies - n_core
            angle = PLY_ANGLES[pi]
            lname = layer_name(li, n_plies, n_core)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Solid Section, elset={lname}, orientation={oname}, "
                    f"material=IM7-8552, controls=SectionCtrl-1\n")
            f.write(",\n")

        f.write("*End Part\n")

        # ==== Assembly ====
        f.write("**\n")
        f.write("*Assembly, name=Assembly\n")
        f.write("*Instance, name=Plate-1, part=Plate\n")
        f.write("*End Instance\n")
        f.write("*End Assembly\n")

        # ---- Model-level keywords (outside Part/Assembly) ----

        # Section controls (required for CompDam VUMAT)
        f.write("**\n")
        f.write("*Section controls, name=SectionCtrl-1, distortion control=YES\n")

        # CompDam VUMAT for CFRP (reads IM7-8552.props from working dir)
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Material: IM7-8552 (CompDam VUMAT)\n")
        f.write("** ============================================================\n")
        f.write("*Material, name=IM7-8552\n")
        f.write("*Density\n")
        f.write(f"{CFRP_DENSITY:.2e},\n")
        f.write(f"*User material, constants=1\n")
        f.write(f"          {FEATURE_FLAGS},\n")
        f.write("*Depvar, delete=11\n")
        f.write("  19,\n")
        sdv_names = [
            "CDM_d2", "CDM_Fb1", "CDM_Fb2", "CDM_Fb3", "CDM_B",
            "CDM_Lc1", "CDM_Lc2", "CDM_Lc3", "CDM_FIm", "CDM_alpha",
            "CDM_STATUS", "CDM_Plas12", "CDM_Inel12", "CDM_FIfT",
            "CDM_slide1", "CDM_slide2", "CDM_FIfC", "CDM_d1T", "CDM_d1C",
        ]
        for i, name in enumerate(sdv_names, 1):
            f.write(f"  {i}, {name}\n")
        f.write("*Characteristic Length, definition=USER, components=3\n")

        # Elastic orthotropic for honeycomb core
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Material: Al Honeycomb Core\n")
        f.write("** ============================================================\n")
        f.write("*Material, name=HC-Core\n")
        f.write("*Density\n")
        f.write(f"{CORE_PROPS['density']:.2e},\n")
        f.write("*Elastic, type=ENGINEERING CONSTANTS\n")
        f.write(f"{CORE_PROPS['E1']}, {CORE_PROPS['E2']}, {CORE_PROPS['E3']}, "
                f"{CORE_PROPS['nu12']}, {CORE_PROPS['nu13']}, {CORE_PROPS['nu23']}, "
                f"{CORE_PROPS['G12']}, {CORE_PROPS['G13']},\n")
        f.write(f"{CORE_PROPS['G23']},\n")

        # ---- Initial Conditions: CompDam SDVs ----
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Initial Conditions\n")
        f.write("** ============================================================\n")
        f.write("*Initial Conditions, Type=Solution\n")
        for li in inner_layers + outer_layers:
            lname = layer_name(li, n_plies, n_core)
            # 19 SDVs: all 0 except CDM_STATUS (pos 11) = 1
            f.write(f"Plate-1.{lname}, "
                    f" 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0,\n")
            f.write(f"                  "
                    f" 0.d0, 0.d0, 0.d0,    1, 0.d0, 0.d0, 0.d0,\n")
            f.write(f"                  "
                    f" 0.d0, 0.d0, 0.d0, 0.d0, 0.d0\n")

        # ---- Boundary Conditions ----
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Boundary Conditions (Simply Supported)\n")
        f.write("** ============================================================\n")
        f.write("*Boundary\n")
        # Pin all edges (UZ = 0)
        f.write("Plate-1.AllEdges, 3, 3, 0.0\n")
        # Fix corners to prevent rigid body motion (UX, UY)
        f.write(f"Plate-1.{corner1}, 1, 2, 0.0\n")
        f.write(f"Plate-1.{corner2}, 2, 2, 0.0\n")

        # ---- Step ----
        f.write("**\n")
        f.write("** ============================================================\n")
        f.write("** Step: Impact\n")
        f.write("** ============================================================\n")
        f.write("*Step, name=Impact, nlgeom=YES\n")
        f.write("*Dynamic, Explicit\n")
        f.write(f", {TIME_PERIOD}\n")
        f.write("*Bulk Viscosity\n")
        f.write("0.10, 1.5\n")
        f.write(f"*Fixed Mass Scaling, factor={MASS_SCALING}\n")

        # Amplitude: smooth half-sine pulse
        t1 = PULSE_DURATION / 2.0
        t2 = PULSE_DURATION
        f.write("**\n")
        f.write("*Amplitude, name=ImpactPulse, definition=SMOOTH STEP\n")
        f.write(f"0.0, 0.0, {t1:.6e}, 1.0, {t2:.6e}, 0.0, {TIME_PERIOD}, 0.0\n")

        # Pressure load on impact zone surface
        f.write("**\n")
        f.write(f"** Impact pressure: {PEAK_PRESSURE} MPa over R={IMP_R}mm zone\n")
        f.write("*Dsload, amplitude=ImpactPulse\n")
        f.write(f"Plate-1.ImpactSurf, P, {PEAK_PRESSURE:.1f}\n")

        # ---- Output ----
        f.write("**\n")
        f.write("** Output\n")
        f.write("*Output, field, number interval=50\n")
        f.write("*Node Output\n")
        f.write("U, V, A, RF\n")
        f.write("*Element Output, elset=Plate-1.AllCFRP\n")
        f.write("S, LE, SDV\n")
        f.write("*Element Output, elset=Plate-1.CoreAll\n")
        f.write("S, LE\n")
        f.write("**\n")
        f.write("*Output, history, frequency=10\n")
        f.write("*Node Output, nset=Plate-1.ImpactZone\n")
        f.write("U3, V3, RF3\n")
        f.write("*Energy Output\n")
        f.write("ALLKE, ALLSE, ALLPD, ALLAE, ETOTAL\n")
        f.write("**\n")
        f.write("*End Step\n")

    print(f"\nINP written: {inp_path}")

    # ----------------------------------------------------------
    # Copy IM7-8552.props to working directory
    # ----------------------------------------------------------
    props_src = Path("external/CompDam_DGD/tests/IM7-8552.props")
    props_dst = outdir / "IM7-8552.props"
    if props_src.exists():
        shutil.copy2(props_src, props_dst)
        print(f"Props copied: {props_dst}")
    else:
        print(f"WARNING: {props_src} not found!")

    # ----------------------------------------------------------
    # Write run script
    # ----------------------------------------------------------
    vumat_abs = os.path.abspath("external/CompDam_DGD/for/CompDam_DGD.for")
    run_script = outdir / "run_compdam.sh"
    with open(run_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# CompDam flat plate impact simulation\n")
        f.write("# Run from: abaqus_work/compdam_flatplate/\n")
        f.write("#\n")
        f.write(f"# Model: {PLATE_LX}x{PLATE_LY}mm, [45/0/-45/90]s, Core={CORE_T}mm\n")
        f.write(f"# Elements: {total_elems} C3D8R\n")
        f.write(f"# VUMAT: CompDam_DGD (feature flags={FEATURE_FLAGS})\n")
        f.write("#\n")
        f.write('cd "$(dirname "$0")"\n')
        f.write("\n")
        f.write("ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus\n")
        f.write(f'VUMAT="{vumat_abs}"\n')
        f.write("\n")
        f.write('LD_PRELOAD=/home/nishioka/libfake_x11.so \\\n')
        f.write('$ABAQUS job=compdam_flatplate_impact \\\n')
        f.write('  user="$VUMAT" \\\n')
        f.write('  double=both \\\n')
        f.write('  cpus=4 \\\n')
        f.write('  interactive\n')
    os.chmod(run_script, 0o755)
    print(f"Run script: {run_script}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Ready to run!")
    print(f"  cd abaqus_work/compdam_flatplate")
    print(f"  bash run_compdam.sh")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
