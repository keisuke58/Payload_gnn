#!/usr/bin/env python3
"""
Generate parametric CompDam flat plate impact DOE (50 cases).

Varies: impact pressure, location, radius, layup
Purpose: Cross-structure generalization training data for GNN-SHM

Usage:
    python scripts/generate_compdam_parametric.py [--n_cases 50] [--seed 42]
    # Then run on frontale:
    cd abaqus_work/compdam_parametric
    bash run_all.sh
"""

import argparse
import json
import os
import shutil
import numpy as np
from pathlib import Path


# ============================================================
# DOE Parameter Ranges
# ============================================================

LAYUPS = {
    0: {'name': 'QI_45',   'angles': [45, 0, -45, 90, 90, -45, 0, 45]},
    1: {'name': 'CP_0_90', 'angles': [0, 90, 0, 90, 90, 0, 90, 0]},
    2: {'name': 'QI_std',  'angles': [0, 45, -45, 90, 90, -45, 45, 0]},
    3: {'name': 'QI_60',   'angles': [60, 0, -60, 90, 90, -60, 0, 60]},
}

# Fixed parameters
PLATE_LX = 100.0
PLATE_LY = 100.0
PLY_T = 0.125       # mm per ply
CORE_T = 10.0       # mm
CORE_NZ = 5
ELEM_XY = 2.0       # mm
PULSE_DURATION = 0.0005  # s
TIME_PERIOD = 0.001      # s
MASS_SCALING = 4
FEATURE_FLAGS = 111101
CFRP_DENSITY = 1.57e-09  # tonne/mm^3

CORE_PROPS = {
    'density': 5.0e-11,
    'E1': 1000.0, 'E2': 1.0, 'E3': 1.0,
    'nu12': 0.01, 'nu13': 0.01, 'nu23': 0.01,
    'G12': 400.0, 'G13': 240.0, 'G23': 1.0,
}


# ============================================================
# DOE Generation (Latin Hypercube Sampling)
# ============================================================

def generate_doe(n_cases, seed=42):
    """Generate DOE with LHS across 5D parameter space."""
    rng = np.random.default_rng(seed)

    # LHS in [0,1]^5
    n_dim = 5
    samples = np.zeros((n_cases, n_dim))
    for d in range(n_dim):
        perm = rng.permutation(n_cases)
        for i in range(n_cases):
            samples[perm[i], d] = (i + rng.uniform()) / n_cases

    cases = []
    for i in range(n_cases):
        s = samples[i]

        # Dim 0: Impact pressure (5-30 MPa)
        pressure = 5.0 + s[0] * 25.0

        # Dim 1: Impact radius (5-15 mm)
        impact_r = 5.0 + s[1] * 10.0

        # Dim 2: Impact X offset (-20 to +20 mm from center)
        x_offset = -20.0 + s[2] * 40.0

        # Dim 3: Impact Y offset (-20 to +20 mm from center)
        y_offset = -20.0 + s[3] * 40.0

        # Dim 4: Layup index (0-3, discrete)
        layup_idx = int(s[4] * len(LAYUPS))
        layup_idx = min(layup_idx, len(LAYUPS) - 1)

        layup = LAYUPS[layup_idx]
        case = {
            'case_id': i,
            'pressure': round(pressure, 2),
            'impact_r': round(impact_r, 2),
            'x_offset': round(x_offset, 2),
            'y_offset': round(y_offset, 2),
            'layup_idx': layup_idx,
            'layup_name': layup['name'],
            'ply_angles': layup['angles'],
        }
        cases.append(case)

    return cases


# ============================================================
# INP Generation (adapted from generate_compdam_flatplate.py)
# ============================================================

def layer_name(layer_idx, n_plies, n_core, ply_angles):
    """Generate descriptive element set name for each layer."""
    if layer_idx < n_plies:
        angle = ply_angles[layer_idx]
        return f"InnerPly{layer_idx + 1}_{angle:+d}"
    elif layer_idx < n_plies + n_core:
        ci = layer_idx - n_plies
        return f"Core{ci + 1}"
    else:
        pi = layer_idx - n_plies - n_core
        angle = ply_angles[pi]
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


def generate_inp(case, outdir):
    """Generate INP file for a single DOE case."""
    pressure = case['pressure']
    impact_r = case['impact_r']
    x_offset = case['x_offset']
    y_offset = case['y_offset']
    ply_angles = case['ply_angles']
    case_id = case['case_id']

    n_plies = len(ply_angles)
    n_core = CORE_NZ
    n_layers = 2 * n_plies + n_core

    nx = int(PLATE_LX / ELEM_XY)
    ny = int(PLATE_LY / ELEM_XY)
    n_nodes_xy = (nx + 1) * (ny + 1)

    # Z-planes
    z_planes = []
    z = 0.0
    for i in range(n_plies + 1):
        z_planes.append(z)
        if i < n_plies:
            z += PLY_T
    core_dz = CORE_T / n_core
    for i in range(n_core):
        z += core_dz
        z_planes.append(z)
    for i in range(n_plies):
        z += PLY_T
        z_planes.append(z)

    total_thickness = z_planes[-1]
    n_z_planes = len(z_planes)

    # Nodes
    nodes = []
    for iz in range(n_z_planes):
        zv = z_planes[iz]
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nid = iz * n_nodes_xy + iy * (nx + 1) + ix + 1
                nodes.append((nid, ix * ELEM_XY, iy * ELEM_XY, zv))

    # Elements
    elements = {}
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

    inner_layers = list(range(n_plies))
    core_layers = list(range(n_plies, n_plies + n_core))
    outer_layers = list(range(n_plies + n_core, n_layers))

    # Impact zone (offset from center)
    cx = PLATE_LX / 2.0 + x_offset
    cy = PLATE_LY / 2.0 + y_offset
    top_ply_layer = n_layers - 1

    impact_elem_ids = []
    for eid_e, *_ in elements[top_ply_layer]:
        local_eid = eid_e - top_ply_layer * nx * ny
        iy_e = (local_eid - 1) // nx
        ix_e = (local_eid - 1) % nx
        ex = (ix_e + 0.5) * ELEM_XY
        ey = (iy_e + 0.5) * ELEM_XY
        r = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        if r <= impact_r:
            impact_elem_ids.append(eid_e)

    # Fallback: if offset pushes impact zone off plate, use at least 1 element
    if not impact_elem_ids:
        # Find closest element to target center
        best_eid = None
        best_dist = 1e9
        for eid_e, *_ in elements[top_ply_layer]:
            local_eid = eid_e - top_ply_layer * nx * ny
            iy_e = (local_eid - 1) // nx
            ix_e = (local_eid - 1) % nx
            ex = (ix_e + 0.5) * ELEM_XY
            ey = (iy_e + 0.5) * ELEM_XY
            d = np.sqrt((ex - cx)**2 + (ey - cy)**2)
            if d < best_dist:
                best_dist = d
                best_eid = eid_e
        impact_elem_ids = [best_eid]

    # Impact zone nodes on top surface
    top_z_idx = n_z_planes - 1
    impact_top_nodes = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = ix * ELEM_XY
            y = iy * ELEM_XY
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if r <= impact_r:
                nid = top_z_idx * n_nodes_xy + iy * (nx + 1) + ix + 1
                impact_top_nodes.append(nid)
    if not impact_top_nodes:
        # Use nearest node
        best_nid = None
        best_dist = 1e9
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                x = ix * ELEM_XY
                y = iy * ELEM_XY
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                if d < best_dist:
                    best_dist = d
                    best_nid = top_z_idx * n_nodes_xy + iy * (nx + 1) + ix + 1
        impact_top_nodes = [best_nid]

    job_name = f"compdam_case_{case_id:03d}"
    inp_path = outdir / f"{job_name}.inp"

    with open(inp_path, 'w') as f:
        f.write("*Heading\n")
        f.write(f"** CompDam Parametric Case {case_id:03d}\n")
        f.write(f"** Pressure={pressure}MPa, R={impact_r}mm, "
                f"Offset=({x_offset},{y_offset}), Layup={case['layup_name']}\n")
        f.write(f"** Plate: {PLATE_LX}x{PLATE_LY}mm, Total: {total_thickness:.3f}mm\n")
        f.write(f"** Elements: {total_elems} C3D8R\n")
        f.write("**\n")

        # Part
        f.write("*Part, name=Plate\n")
        f.write("*Node\n")
        for nid, x, y, zv in nodes:
            f.write(f"{nid}, {x:.6f}, {y:.6f}, {zv:.6f}\n")

        for li in range(n_layers):
            lname = layer_name(li, n_plies, n_core, ply_angles)
            f.write(f"*Element, type=C3D8R, elset={lname}\n")
            for row in elements[li]:
                f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, "
                        f"{row[5]}, {row[6]}, {row[7]}, {row[8]}\n")

        f.write("*Elset, elset=AllElements, generate\n")
        f.write(f"1, {total_elems}, 1\n")

        for label, layers in [("InnerSkin", inner_layers),
                              ("CoreAll", core_layers),
                              ("OuterSkin", outer_layers)]:
            eids = []
            for li in layers:
                eids.extend([e[0] for e in elements[li]])
            f.write(f"*Elset, elset={label}\n")
            write_set(f, eids)

        cfrp_eids = []
        for li in inner_layers + outer_layers:
            cfrp_eids.extend([e[0] for e in elements[li]])
        f.write("*Elset, elset=AllCFRP\n")
        write_set(f, cfrp_eids)

        f.write("*Elset, elset=ImpactZoneElems\n")
        write_set(f, impact_elem_ids)

        # Node sets
        bot_nodes = list(range(1, n_nodes_xy + 1))
        f.write("*Nset, nset=BotSurf\n")
        write_set(f, bot_nodes)

        top_start = top_z_idx * n_nodes_xy + 1
        top_nodes = list(range(top_start, top_start + n_nodes_xy))
        f.write("*Nset, nset=TopSurf\n")
        write_set(f, top_nodes)

        f.write("*Nset, nset=ImpactZone\n")
        write_set(f, impact_top_nodes)

        all_edge = []
        for iz in range(n_z_planes):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    if ix == 0 or ix == nx or iy == 0 or iy == ny:
                        nid = iz * n_nodes_xy + iy * (nx + 1) + ix + 1
                        all_edge.append(nid)
        f.write("*Nset, nset=AllEdges\n")
        write_set(f, all_edge)

        corner1 = 1
        corner2 = nx + 1

        # Impact surface
        lname_top = layer_name(top_ply_layer, n_plies, n_core, ply_angles)
        f.write("*Surface, name=ImpactSurf, type=ELEMENT\n")
        for eid_e in impact_elem_ids:
            f.write(f"{eid_e}, S2\n")

        # Orientations
        f.write("*Orientation, name=Ori_Core\n")
        f.write("1., 0., 0., 0., 1., 0.\n")
        f.write("3, 0.\n")

        unique_angles = sorted(set(ply_angles))
        for angle in unique_angles:
            rad = np.radians(angle)
            c, s = np.cos(rad), np.sin(rad)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Orientation, name={oname}\n")
            f.write(f"{c:.8f}, {s:.8f}, 0., {-s:.8f}, {c:.8f}, 0.\n")
            f.write("3, 0.\n")

        # Sections
        f.write("**\n** Sections\n")
        for li in inner_layers:
            angle = ply_angles[li]
            lname = layer_name(li, n_plies, n_core, ply_angles)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Solid Section, elset={lname}, orientation={oname}, "
                    f"material=IM7-8552, controls=SectionCtrl-1\n,\n")

        for li in core_layers:
            lname = layer_name(li, n_plies, n_core, ply_angles)
            f.write(f"*Solid Section, elset={lname}, orientation=Ori_Core, material=HC-Core\n,\n")

        for li in outer_layers:
            pi = li - n_plies - n_core
            angle = ply_angles[pi]
            lname = layer_name(li, n_plies, n_core, ply_angles)
            oname = f"Ori_{angle:+04d}"
            f.write(f"*Solid Section, elset={lname}, orientation={oname}, "
                    f"material=IM7-8552, controls=SectionCtrl-1\n,\n")

        f.write("*End Part\n")

        # Assembly
        f.write("**\n*Assembly, name=Assembly\n")
        f.write("*Instance, name=Plate-1, part=Plate\n")
        f.write("*End Instance\n")
        f.write("*End Assembly\n")

        # Section controls
        f.write("**\n*Section controls, name=SectionCtrl-1, distortion control=YES\n")

        # CFRP Material (CompDam VUMAT)
        f.write("**\n*Material, name=IM7-8552\n")
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

        # Core Material
        f.write("**\n*Material, name=HC-Core\n")
        f.write("*Density\n")
        f.write(f"{CORE_PROPS['density']:.2e},\n")
        f.write("*Elastic, type=ENGINEERING CONSTANTS\n")
        f.write(f"{CORE_PROPS['E1']}, {CORE_PROPS['E2']}, {CORE_PROPS['E3']}, "
                f"{CORE_PROPS['nu12']}, {CORE_PROPS['nu13']}, {CORE_PROPS['nu23']}, "
                f"{CORE_PROPS['G12']}, {CORE_PROPS['G13']},\n")
        f.write(f"{CORE_PROPS['G23']},\n")

        # Initial Conditions
        f.write("**\n*Initial Conditions, Type=Solution\n")
        for li in inner_layers + outer_layers:
            lname = layer_name(li, n_plies, n_core, ply_angles)
            f.write(f"Plate-1.{lname}, "
                    f" 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0,\n")
            f.write(f"                  "
                    f" 0.d0, 0.d0, 0.d0,    1, 0.d0, 0.d0, 0.d0,\n")
            f.write(f"                  "
                    f" 0.d0, 0.d0, 0.d0, 0.d0, 0.d0\n")

        # Boundary Conditions
        f.write("**\n*Boundary\n")
        f.write("Plate-1.AllEdges, 3, 3, 0.0\n")
        f.write(f"Plate-1.{corner1}, 1, 2, 0.0\n")
        f.write(f"Plate-1.{corner2}, 2, 2, 0.0\n")

        # Step
        f.write("**\n*Step, name=Impact, nlgeom=YES\n")
        f.write("*Dynamic, Explicit\n")
        f.write(f", {TIME_PERIOD}\n")
        f.write("*Bulk Viscosity\n")
        f.write("0.06, 1.2\n")
        f.write(f"*Fixed Mass Scaling, factor={MASS_SCALING}\n")

        t1 = PULSE_DURATION / 2.0
        t2 = PULSE_DURATION
        f.write("**\n*Amplitude, name=ImpactPulse, definition=SMOOTH STEP\n")
        f.write(f"0.0, 0.0, {t1:.6e}, 1.0, {t2:.6e}, 0.0, {TIME_PERIOD}, 0.0\n")

        f.write("**\n*Dsload, amplitude=ImpactPulse\n")
        f.write(f"Plate-1.ImpactSurf, P, {pressure:.1f}\n")

        # Output
        f.write("**\n*Output, field, number interval=20\n")
        f.write("*Node Output\n")
        f.write("U, V, A, RF\n")
        f.write("*Element Output, elset=Plate-1.AllCFRP\n")
        f.write("S, LE, SDV\n")
        f.write("*Element Output, elset=Plate-1.CoreAll\n")
        f.write("S, LE\n")
        f.write("**\n*Output, history, frequency=20\n")
        f.write("*Node Output, nset=Plate-1.ImpactZone\n")
        f.write("U3, V3, RF3\n")
        f.write("*Energy Output\n")
        f.write("ALLKE, ALLSE, ALLPD, ALLAE, ETOTAL\n")
        f.write("**\n*End Step\n")

    return job_name, len(impact_elem_ids)


# ============================================================
# Batch Extraction Script (Abaqus Python)
# ============================================================

def write_extract_script(outdir, cases):
    """Write Abaqus Python extraction script for all cases."""
    script_path = outdir / "extract_all_nodes.py"
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python
"""Extract node-level data from all CompDam parametric ODBs.
Run with: abaqus python extract_all_nodes.py
"""
import os
import sys
import csv
import math
from collections import defaultdict
from odbAccess import openOdb

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_case(odb_path, csv_path, elem_csv_path):
    """Extract node data from a single ODB."""
    if not os.path.exists(odb_path):
        print("  SKIP (not found): {}".format(odb_path))
        return False

    try:
        odb = openOdb(odb_path, readOnly=True)
    except Exception as e:
        print("  ERROR opening {}: {}".format(odb_path, e))
        return False

    step = odb.steps['Impact']
    last_frame = step.frames[-1]

    instance = odb.rootAssembly.instances['PLATE-1']
    nodes = instance.nodes

    node_coords = {}
    for n in nodes:
        node_coords[n.label] = (n.coordinates[0], n.coordinates[1], n.coordinates[2])

    # Displacement
    node_disp = {}
    if 'U' in last_frame.fieldOutputs:
        u_field = last_frame.fieldOutputs['U']
        for val in u_field.values:
            nid = val.nodeLabel
            u1, u2, u3 = float(val.data[0]), float(val.data[1]), float(val.data[2])
            umag = math.sqrt(u1**2 + u2**2 + u3**2)
            node_disp[nid] = (u1, u2, u3, umag)

    # Element connectivity
    elements = instance.elements
    elem_to_nodes = {}
    elem_data = []
    for e in elements:
        conn = [int(c) for c in e.connectivity]
        elem_data.append({'elem_id': e.label, 'type': str(e.type), 'nodes': conn})
        elem_to_nodes[e.label] = conn

    # Stress → node average
    node_stress = defaultdict(lambda: {'S11': [], 'S22': [], 'S33': [], 'S12': []})
    if 'S' in last_frame.fieldOutputs:
        s_field = last_frame.fieldOutputs['S']
        for val in s_field.values:
            eid = val.elementLabel
            if eid in elem_to_nodes:
                s11 = float(val.data[0])
                s22 = float(val.data[1])
                s33 = float(val.data[2])
                s12 = float(val.data[3])
                for nid in elem_to_nodes[eid]:
                    node_stress[nid]['S11'].append(s11)
                    node_stress[nid]['S22'].append(s22)
                    node_stress[nid]['S33'].append(s33)
                    node_stress[nid]['S12'].append(s12)

    # Damage SDVs → node average
    node_damage = defaultdict(lambda: {'d2': [], 'd1T': [], 'd1C': []})
    sdv_map = {
        'SDV_CDM_D2': 'd2',
        'SDV_CDM_D1T': 'd1T',
        'SDV_CDM_D1C': 'd1C',
    }
    for sdv_label, damage_key in sdv_map.items():
        if sdv_label in last_frame.fieldOutputs:
            field = last_frame.fieldOutputs[sdv_label]
            for val in field.values:
                eid = val.elementLabel
                if eid in elem_to_nodes:
                    dval = float(val.data)
                    for nid in elem_to_nodes[eid]:
                        node_damage[nid][damage_key].append(dval)

    # Write node CSV
    header = ['node_id', 'x', 'y', 'z',
              'U1', 'U2', 'U3', 'Umag',
              'S11', 'S22', 'S33', 'S12', 'Mises',
              'CDM_d2', 'CDM_d1T', 'CDM_d1C', 'damage_label']

    with open(csv_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        for nid in sorted(node_coords.keys()):
            x, y, z = node_coords[nid]
            u1, u2, u3, umag = node_disp.get(nid, (0, 0, 0, 0))

            if nid in node_stress:
                s11 = sum(node_stress[nid]['S11']) / len(node_stress[nid]['S11'])
                s22 = sum(node_stress[nid]['S22']) / len(node_stress[nid]['S22'])
                s33 = sum(node_stress[nid]['S33']) / len(node_stress[nid]['S33'])
                s12 = sum(node_stress[nid]['S12']) / len(node_stress[nid]['S12'])
            else:
                s11 = s22 = s33 = s12 = 0.0
            mises = math.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*s12**2))

            if nid in node_damage:
                d2 = sum(node_damage[nid]['d2']) / len(node_damage[nid]['d2']) if node_damage[nid]['d2'] else 0
                d1T = sum(node_damage[nid]['d1T']) / len(node_damage[nid]['d1T']) if node_damage[nid]['d1T'] else 0
                d1C = sum(node_damage[nid]['d1C']) / len(node_damage[nid]['d1C']) if node_damage[nid]['d1C'] else 0
            else:
                d2 = d1T = d1C = 0.0

            damage_label = 1 if (d2 > 0.01 or d1T > 0.01 or d1C > 0.01) else 0
            writer.writerow([nid, x, y, z, u1, u2, u3, umag,
                            s11, s22, s33, s12, mises,
                            d2, d1T, d1C, damage_label])

    # Write element CSV
    with open(elem_csv_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['elem_id', 'type', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8'])
        for ed in elem_data:
            row = [ed['elem_id'], ed['type']] + ed['nodes']
            writer.writerow(row)

    odb.close()
    return True


def main():
    print("=" * 60)
    print("CompDam Parametric Batch Extraction")
    print("=" * 60)

    success = 0
    fail = 0

    # Scan for ODB files
    for name in sorted(os.listdir(CASE_DIR)):
        if name.startswith('compdam_case_') and name.endswith('.odb'):
            base = name.replace('.odb', '')
            odb_path = os.path.join(CASE_DIR, name)
            csv_path = os.path.join(CASE_DIR, base + '_nodes.csv')
            elem_csv = os.path.join(CASE_DIR, base + '_elements.csv')

            print("\\nExtracting: {}".format(name))
            ok = extract_case(odb_path, csv_path, elem_csv)
            if ok:
                success += 1
                print("  OK: {} + {}".format(csv_path, elem_csv))
            else:
                fail += 1

    print("\\n" + "=" * 60)
    print("Done: {} success, {} failed".format(success, fail))


if __name__ == '__main__':
    main()
''')
    print(f"  Extract script: {script_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CompDam parametric DOE generator")
    parser.add_argument('--n_cases', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='abaqus_work/compdam_parametric')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CompDam Parametric DOE: {args.n_cases} cases")
    print("=" * 60)

    # Generate DOE
    cases = generate_doe(args.n_cases, args.seed)

    # Save DOE summary
    doe_path = outdir / "doe_summary.json"
    with open(doe_path, 'w') as f:
        json.dump(cases, f, indent=2)
    print(f"DOE saved: {doe_path}")

    # Print DOE summary table
    print(f"\n{'ID':>4} {'Pressure':>10} {'Radius':>8} {'X_off':>7} {'Y_off':>7} {'Layup':>10}")
    print("-" * 50)
    for c in cases:
        print(f"{c['case_id']:>4d} {c['pressure']:>9.1f}  {c['impact_r']:>7.1f} "
              f"{c['x_offset']:>7.1f} {c['y_offset']:>7.1f}  {c['layup_name']:>9s}")

    # Generate INPs
    print(f"\nGenerating INP files...")
    props_src = Path("external/CompDam_DGD/tests/IM7-8552.props")
    if props_src.exists():
        shutil.copy2(props_src, outdir / "IM7-8552.props")
        print(f"  Props copied: IM7-8552.props")

    job_names = []
    for c in cases:
        job_name, n_impact = generate_inp(c, outdir)
        job_names.append(job_name)
        print(f"  Case {c['case_id']:03d}: {job_name}.inp "
              f"(P={c['pressure']:.0f}MPa, R={c['impact_r']:.0f}mm, "
              f"impact_elems={n_impact})")

    # Write batch run script
    vumat_abs = os.path.abspath("external/CompDam_DGD/for/CompDam_DGD.for")
    run_script = outdir / "run_all.sh"
    with open(run_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# CompDam parametric batch runner\n")
        f.write(f"# {args.n_cases} cases, 4 cpus each\n")
        f.write("# Run 6 jobs in parallel (24 cores total on frontale)\n")
        f.write("#\n")
        f.write("# Usage:\n")
        f.write("#   cd abaqus_work/compdam_parametric\n")
        f.write("#   bash run_all.sh 2>&1 | tee run_all.log\n")
        f.write("#\n\n")
        f.write('cd "$(dirname "$0")"\n\n')
        f.write("ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus\n")
        f.write(f'VUMAT="{vumat_abs}"\n')
        f.write("MAX_PARALLEL=6\n")
        f.write("CPUS_PER_JOB=4\n\n")

        f.write("# Job queue\n")
        f.write("JOBS=(\n")
        for jn in job_names:
            f.write(f'  "{jn}"\n')
        f.write(")\n\n")

        f.write("N_TOTAL=${#JOBS[@]}\n")
        f.write("N_DONE=0\n")
        f.write("N_FAIL=0\n")
        f.write('echo "Starting $N_TOTAL CompDam jobs (max $MAX_PARALLEL parallel)"\n')
        f.write('echo "$(date)"\n')
        f.write('echo ""\n\n')

        # Parallel execution with job slot management
        f.write("run_job() {\n")
        f.write("  local JOB=$1\n")
        f.write('  echo "[$(date +%H:%M:%S)] START: $JOB"\n')
        f.write("  LD_PRELOAD=/home/nishioka/libfake_x11.so \\\n")
        f.write('  $ABAQUS job=$JOB user="$VUMAT" double=both cpus=$CPUS_PER_JOB interactive > ${JOB}.log 2>&1\n')
        f.write("  local RC=$?\n")
        f.write("  if [ $RC -eq 0 ]; then\n")
        f.write('    echo "[$(date +%H:%M:%S)] DONE:  $JOB (OK)"\n')
        f.write("  else\n")
        f.write('    echo "[$(date +%H:%M:%S)] FAIL:  $JOB (rc=$RC)"\n')
        f.write("  fi\n")
        f.write("  return $RC\n")
        f.write("}\n\n")

        f.write("# Run with GNU parallel-style job control\n")
        f.write("PIDS=()\n")
        f.write("RUNNING=0\n\n")

        f.write("for JOB in \"${JOBS[@]}\"; do\n")
        f.write("  # Wait if at max parallel\n")
        f.write("  while [ $RUNNING -ge $MAX_PARALLEL ]; do\n")
        f.write("    for i in \"${!PIDS[@]}\"; do\n")
        f.write("      if ! kill -0 ${PIDS[$i]} 2>/dev/null; then\n")
        f.write("        wait ${PIDS[$i]}\n")
        f.write("        RC=$?\n")
        f.write("        if [ $RC -eq 0 ]; then\n")
        f.write("          N_DONE=$((N_DONE + 1))\n")
        f.write("        else\n")
        f.write("          N_FAIL=$((N_FAIL + 1))\n")
        f.write("        fi\n")
        f.write("        unset PIDS[$i]\n")
        f.write("        RUNNING=$((RUNNING - 1))\n")
        f.write("      fi\n")
        f.write("    done\n")
        f.write("    sleep 2\n")
        f.write("  done\n\n")
        f.write("  run_job $JOB &\n")
        f.write("  PIDS+=($!)\n")
        f.write("  RUNNING=$((RUNNING + 1))\n")
        f.write("done\n\n")

        f.write("# Wait for remaining\n")
        f.write("for pid in \"${PIDS[@]}\"; do\n")
        f.write("  wait $pid\n")
        f.write("  RC=$?\n")
        f.write("  if [ $RC -eq 0 ]; then\n")
        f.write("    N_DONE=$((N_DONE + 1))\n")
        f.write("  else\n")
        f.write("    N_FAIL=$((N_FAIL + 1))\n")
        f.write("  fi\n")
        f.write("done\n\n")

        f.write('echo ""\n')
        f.write('echo "========================================"\n')
        f.write('echo "Batch complete: $N_DONE OK, $N_FAIL FAIL / $N_TOTAL total"\n')
        f.write('echo "$(date)"\n')
        f.write('echo "========================================"\n')

    os.chmod(run_script, 0o755)
    print(f"\nBatch run script: {run_script}")

    # Write extraction script
    write_extract_script(outdir, cases)

    # Write single-job run script (for testing 1 case)
    test_script = outdir / "run_single.sh"
    with open(test_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Run a single CompDam case for testing\n")
        f.write("# Usage: bash run_single.sh compdam_case_000\n\n")
        f.write('cd "$(dirname "$0")"\n\n')
        f.write("ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus\n")
        f.write(f'VUMAT="{vumat_abs}"\n')
        f.write("JOB=${1:-compdam_case_000}\n\n")
        f.write('echo "Running: $JOB"\n')
        f.write("LD_PRELOAD=/home/nishioka/libfake_x11.so \\\n")
        f.write('$ABAQUS job=$JOB user="$VUMAT" double=both cpus=4 interactive\n')
    os.chmod(test_script, 0o755)
    print(f"Test script: {test_script}")

    # Summary
    pressures = [c['pressure'] for c in cases]
    radii = [c['impact_r'] for c in cases]
    print(f"\n{'=' * 60}")
    print(f"DOE Summary:")
    print(f"  Cases:     {args.n_cases}")
    print(f"  Pressure:  {min(pressures):.1f} - {max(pressures):.1f} MPa")
    print(f"  Radius:    {min(radii):.1f} - {max(radii):.1f} mm")
    print(f"  Layups:    {len(LAYUPS)} variants")
    print(f"  Est. time: ~{args.n_cases * 8 / 6:.0f} min ({args.n_cases} jobs / 6 parallel)")
    print(f"\nNext steps:")
    print(f"  1. Transfer to frontale: scp -r {outdir} frontale02:~/Payload2026/")
    print(f"  2. cd abaqus_work/compdam_parametric && bash run_all.sh")
    print(f"  3. abaqus python extract_all_nodes.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
