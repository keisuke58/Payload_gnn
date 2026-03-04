# -*- coding: utf-8 -*-
# extract_odb_results.py
# Abaqus Python script to extract nodal and element data from ODB
# Multi-class defect labeling. Academic refs: docs/DEFECT_MODELS_ACADEMIC.md
# 0=healthy, 1=debonding, 2=fod, 3=impact, 4=delamination, 5=inner_debond,
# 6=thermal_progression, 7=acoustic_fatigue
#
# Usage: abaqus python extract_odb_results.py --odb <odb_path> --output <output_dir>
#        abaqus python extract_odb_results.py --odb <odb> --output <dir> --defect_json <params.json>

import sys
import os
import csv
import json
import math
import argparse
from odbAccess import *
from abaqusConstants import *

# Multi-class defect type mapping
DEFECT_TYPE_MAP = {
    'healthy': 0,
    'debonding': 1,
    'fod': 2,
    'impact': 3,
    'delamination': 4,
    'inner_debond': 5,
    'thermal_progression': 6,
    'acoustic_fatigue': 7,
}


def _enforce_symmetry(node_rows, step_name=''):
    """Enforce symmetry on sector boundary nodes (displacement + stress).

    Only for STATIC analyses — dynamic/explicit analyses may have non-symmetric
    wave propagation at boundaries, so symmetry enforcement is skipped.

    TIE slave skin nodes may have non-zero normal displacement and asymmetric
    stress at symmetry boundaries. This:
    1. Zeros out normal displacement at theta=0 and theta=theta_max.
    2. Averages scalar stress (smises) between matched theta=0 / theta_max
       node pairs so that 360-degree sector mirroring is smooth.

    node_rows: list of [nid, x, y, z, ux, uy, uz, u_mag, temp,
                        s11, s22, s12, smises, thermal_smises,
                        le11, le22, le12, defect_label]
    Indices: ux=4, uy=5, uz=6, u_mag=7, s11=9, s22=10, s12=11, smises=12,
             thermal_smises=13
    """
    if not node_rows:
        return

    # Skip for analyses with asymmetric excitation (guided wave actuator etc.)
    # Symmetric loading (uniform pressure, thermal) is fine even in Explicit.
    su = step_name.upper()
    if any(kw in su for kw in ('GUIDEDWAVE', 'GW-', 'ACTUAT')):
        return

    # Auto-detect sector angle from node geometry
    theta_max = 0.0
    for row in node_rows:
        x, z = row[1], row[3]
        r = math.sqrt(x * x + z * z)
        if r < 1.0:
            continue
        theta = math.atan2(z, x)
        if theta > theta_max:
            theta_max = theta

    # Skip if not a sector model
    if theta_max < 0.01 or theta_max > math.pi / 2:
        return

    tol_rad = math.radians(0.5)
    sin_t = math.sin(theta_max)
    cos_t = math.cos(theta_max)

    # Collect boundary nodes: {(y_bin, r_bin): row_index}
    t0_idx = []   # list of (y, r, row_idx)
    ta_idx = []

    for i, row in enumerate(node_rows):
        x, z = row[1], row[3]
        r = math.sqrt(x * x + z * z)
        if r < 1.0:
            continue
        theta = math.atan2(z, x)

        if abs(theta) < tol_rad:
            # Enforce uz=0 at theta=0
            row[6] = 0.0
            row[7] = math.sqrt(row[4]**2 + row[5]**2)
            t0_idx.append((row[2], r, i))

        elif abs(theta - theta_max) < tol_rad:
            # Project out normal component at theta=max
            ux, uz = row[4], row[6]
            u_n = -sin_t * ux + cos_t * uz
            row[4] = ux + sin_t * u_n
            row[6] = uz - cos_t * u_n
            row[7] = math.sqrt(row[4]**2 + row[5]**2 + row[6]**2)
            ta_idx.append((row[2], r, i))

    # Average SCALAR stress between matched theta=0 / theta=max pairs.
    # Only smises (12) and thermal_smises (13) — NOT tensor components
    # (s11, s22, s12) because they reference different global directions
    # at theta=0 vs theta=max and cannot be naively averaged.
    stress_cols = [12, 13]
    n_pairs = 0
    for y0, r0, i0 in t0_idx:
        best_d = 999.0
        best_ia = -1
        for ya, ra, ia in ta_idx:
            d = math.sqrt((y0 - ya)**2 + (r0 - ra)**2)
            if d < best_d:
                best_d = d
                best_ia = ia
        if best_ia >= 0 and best_d < 5.0:
            for c in stress_cols:
                avg = (node_rows[i0][c] + node_rows[best_ia][c]) / 2.0
                node_rows[i0][c] = avg
                node_rows[best_ia][c] = avg
            n_pairs += 1

    if t0_idx or ta_idx:
        print("  Symmetry enforced: %d nodes (t0) + %d (ta), %d stress pairs averaged"
              % (len(t0_idx), len(ta_idx), n_pairs))


def _is_node_in_defect(x, y, z, defect_params):
    """
    Check if node (x,y,z) lies inside the circular defect zone on the cylindrical surface.
    Abaqus Revolve: Y=axial, XZ=radial. r=sqrt(x^2+z^2), theta=atan2(z,x), axial=y.
    """
    theta_deg = defect_params['theta_deg']
    z_center = defect_params['z_center']
    radius = defect_params['radius']

    r_node = math.sqrt(x * x + z * z)
    if r_node < 1.0:
        return False
    theta_node_rad = math.atan2(z, x)
    theta_center_rad = math.radians(theta_deg)
    arc_mm = r_node * abs(theta_node_rad - theta_center_rad)
    dy = y - z_center
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= radius


def extract_results(odb_path, output_dir, defect_params=None, strict=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Opening ODB: " + odb_path)
    try:
        odb = openOdb(path=odb_path)
    except Exception as e:
        print("Error opening ODB: " + str(e))
        sys.exit(1)
        
    # Get analysis steps — supports:
    #   GT 3-step: Step-Cure / Step-Thermal / Step-Mechanical
    #   Legacy 2-step: Step-1 (thermal) / Step-2 (thermal+mechanical)
    #   Legacy 1-step: single step
    step_keys = list(odb.steps.keys())
    if not step_keys:
        print("Error: ODB has no steps (job may have failed)")
        sys.exit(1)

    print("ODB steps: %s" % str(step_keys))

    # --- Detect step naming pattern ---
    step_cure = step_thermal = step_mechanical = None
    step1_name = step2_name = None

    for k in step_keys:
        ku = k.upper()
        if 'CURE' in ku:
            step_cure = k
        elif 'THERMAL' in ku:
            step_thermal = k
        elif 'MECHANICAL' in ku:
            step_mechanical = k
        elif ku == 'STEP-1':
            step1_name = k
        elif ku == 'STEP-2':
            step2_name = k

    # --- Determine main extraction frame and thermal-only frame ---
    if step_mechanical and len(odb.steps[step_mechanical].frames) > 0:
        # GT 3-step model: main=Step-Mechanical, thermal=Step-Thermal (or Step-Cure)
        step_name = step_mechanical
        frame = odb.steps[step_mechanical].frames[-1]
        thermal_step = step_thermal or step_cure
        frame_thermal = odb.steps[thermal_step].frames[-1] if thermal_step else None
        print("3-step GT analysis: cure=%s, thermal=%s, mechanical=%s" % (
            step_cure, step_thermal, step_mechanical))

    elif step2_name and len(odb.steps[step2_name].frames) > 0:
        # Legacy 2-step model: main=Step-2, thermal=Step-1
        step_name = step2_name
        frame = odb.steps[step2_name].frames[-1]
        frame_thermal = odb.steps[step1_name].frames[-1] if step1_name else None
        print("2-step analysis detected: thermal=%s, total=%s" % (step1_name, step2_name))

    else:
        # Legacy 1-step (or fallback): thermal = total
        fallback = step1_name
        if fallback is None:
            for k in step_keys:
                if k.upper() != 'INITIAL':
                    fallback = k
                    break
        if fallback is None:
            print("Error: No analysis step found in ODB")
            sys.exit(1)
        step_name = fallback
        step_obj = odb.steps[fallback]
        if len(step_obj.frames) == 0:
            for k in step_keys:
                if len(odb.steps[k].frames) > 0:
                    step_name = k
                    step_obj = odb.steps[k]
                    break
        if len(step_obj.frames) == 0:
            print("Error: No frames in any step (job may have failed)")
            sys.exit(1)
        frame = step_obj.frames[-1]
        frame_thermal = None  # will fallback to total = thermal
        print("1-step analysis (legacy): thermal_smises = smises")

    print("Extracting data from Step: " + step_name + ", Frame: " + str(frame.frameId))
    
    # ---------------------------------------------------------
    # 1. Nodal Data (Coordinates, Displacement, Temperature)
    # ---------------------------------------------------------
    # Get the instance (assuming one assembly instance or specific parts)
    # For this model, we have 'Part-OuterSkin-1', 'Part-Core-1', 'Part-InnerSkin-1'
    # We will extract data for the Outer Skin as it's the primary surface for inspection
    
    instance_name = 'PART-OUTERSKIN-1'
    all_keys = list(odb.rootAssembly.instances.keys())
    if instance_name not in all_keys:
        # Fallback: find OuterSkin variant or use first instance
        for k in all_keys:
            if 'OUTER' in k.upper():
                instance_name = k
                break
        else:
            instance_name = all_keys[0]

    instance = odb.rootAssembly.instances[instance_name]
    print("Extracting instance: " + instance_name)

    # Field Outputs
    disp_field = frame.fieldOutputs['U']
    # Nodal temperature: prefer NT11 (scalar nodal temp) over TEMP (element temp)
    temp_field = None
    for key in ('NT11', 'NT'):
        if key in frame.fieldOutputs:
            temp_field = frame.fieldOutputs[key]
            break
    # Fallback: if total frame lacks nodal NT, try thermal frame (Step-1)
    if temp_field is None and frame_thermal is not None:
        for key in ('NT11', 'NT'):
            if key in frame_thermal.fieldOutputs:
                temp_field = frame_thermal.fieldOutputs[key]
                print("  Temperature from Step-1 (not in Step-2)")
                break

    # Stress field (per-node averaging)
    stress_field = None
    if 'S' in frame.fieldOutputs:
        stress_field = frame.fieldOutputs['S']

    # Optional: Strain — LE (logarithmic) or E (engineering) from ODB
    strain_field = None
    for key in ('LE', 'E', 'LE11', 'E11'):
        if key in frame.fieldOutputs:
            strain_field = frame.fieldOutputs[key]
            print("  Using strain field: %s" % key)
            break
    # Subset to instance
    disp_sub = disp_field.getSubset(region=instance)
    temp_sub = temp_field.getSubset(region=instance) if temp_field else None

    # Create a map for displacements
    disp_values = {}
    for val in disp_sub.values:
        disp_values[val.nodeLabel] = val.data

    temp_values = {}
    if temp_sub:
        for val in temp_sub.values:
            nid = getattr(val, 'nodeLabel', None)
            if nid is not None:
                d = val.data
                t = d if isinstance(d, (int, float)) else (d[0] if d else 0.0)
                temp_values[nid] = t

    # Per-node stress: average ELEMENT_NODAL values across contributing elements
    # {node_label: [sum_s11, sum_s22, sum_s12, sum_mises, count]}
    node_stress = {}
    if stress_field:
        try:
            stress_sub = stress_field.getSubset(region=instance, position=ELEMENT_NODAL)
            for val in stress_sub.values:
                nid = val.nodeLabel
                # val.data = (S11, S22, S33, S12, S13, S23) for 3D
                # For shells: (S11, S22, S12) — indices depend on element type
                s = val.data
                mises = val.mises
                s11 = s[0] if len(s) > 0 else 0.0
                s22 = s[1] if len(s) > 1 else 0.0
                s12 = s[3] if len(s) > 3 else (s[2] if len(s) > 2 else 0.0)
                if nid not in node_stress:
                    node_stress[nid] = [0.0, 0.0, 0.0, 0.0, 0]
                node_stress[nid][0] += s11
                node_stress[nid][1] += s22
                node_stress[nid][2] += s12
                node_stress[nid][3] += mises
                node_stress[nid][4] += 1
        except Exception as e:
            print("Warning: Stress extraction: " + str(e)[:80])

    # Thermal-only stress from Step-1 (for 2-step analysis)
    # {node_label: [sum_mises, count]}
    node_thermal_stress = {}
    if frame_thermal is not None:
        thermal_s_field = frame_thermal.fieldOutputs['S'] if 'S' in frame_thermal.fieldOutputs else None
        if thermal_s_field:
            try:
                ts_sub = thermal_s_field.getSubset(region=instance, position=ELEMENT_NODAL)
                for val in ts_sub.values:
                    nid = val.nodeLabel
                    if nid not in node_thermal_stress:
                        node_thermal_stress[nid] = [0.0, 0]
                    node_thermal_stress[nid][0] += val.mises
                    node_thermal_stress[nid][1] += 1
                print("  Extracted thermal stress (Step-1) for %d nodes" % len(node_thermal_stress))
            except Exception as e:
                print("Warning: Thermal stress extraction: " + str(e)[:80])

    # Optional: Per-node strain (LE) - average from ELEMENT_NODAL if available
    node_strain = {}
    if strain_field:
        try:
            strain_sub = strain_field.getSubset(region=instance, position=ELEMENT_NODAL)
            for val in strain_sub.values:
                nid = val.nodeLabel
                d = val.data  # (LE11, LE22, LE33, LE12, LE13, LE23) or similar
                le11 = d[0] if len(d) > 0 else 0.0
                le22 = d[1] if len(d) > 1 else 0.0
                le12 = d[3] if len(d) > 3 else (d[2] if len(d) > 2 else 0.0)
                if nid not in node_strain:
                    node_strain[nid] = [0.0, 0.0, 0.0, 0]
                node_strain[nid][0] += le11
                node_strain[nid][1] += le22
                node_strain[nid][2] += le12
                node_strain[nid][3] += 1
            for nid in node_strain:
                c = node_strain[nid][3]
                if c > 0:
                    node_strain[nid][0] /= c
                    node_strain[nid][1] /= c
                    node_strain[nid][2] /= c
            print("  Extracted strain (LE) for %d nodes" % len(node_strain))
        except Exception as e:
            print("Warning: Strain extraction: " + str(e)[:60])
            node_strain = {}

    # Iterate nodes to get coordinates and defect labels
    node_rows = []
    n_defect = 0
    for node in instance.nodes:
        nid = node.label
        x, y, z = node.coordinates

        u = disp_values.get(nid, (0.0, 0.0, 0.0))
        t = temp_values.get(nid, 0.0)

        # Average stress at node
        if nid in node_stress and node_stress[nid][4] > 0:
            cnt = node_stress[nid][4]
            s11 = node_stress[nid][0] / cnt
            s22 = node_stress[nid][1] / cnt
            s12 = node_stress[nid][2] / cnt
            smises = node_stress[nid][3] / cnt
        else:
            s11 = s22 = s12 = smises = 0.0

        defect_label = 0
        if defect_params:
            if _is_node_in_defect(x, y, z, defect_params):
                # Multi-class label: 1=debonding, 2=fod, 3=impact
                defect_type = defect_params.get('defect_type', 'debonding')
                defect_label = DEFECT_TYPE_MAP.get(defect_type, 1)
                n_defect += 1

        # Derived: u_mag = |u|
        u_mag = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)

        # thermal_smises: Step-1 von Mises (thermal only) if 2-step, else total
        if nid in node_thermal_stress and node_thermal_stress[nid][1] > 0:
            thermal_smises = node_thermal_stress[nid][0] / node_thermal_stress[nid][1]
        else:
            thermal_smises = smises  # fallback: single-step (100% thermal)

        # Strain (LE11, LE22, LE12) if extracted
        le11 = le22 = le12 = 0.0
        if nid in node_strain:
            le11, le22, le12 = node_strain[nid][0], node_strain[nid][1], node_strain[nid][2]

        node_rows.append([nid, x, y, z, u[0], u[1], u[2], u_mag, t,
                          s11, s22, s12, smises, thermal_smises,
                          le11, le22, le12, defect_label])

    # Enforce symmetry on boundary nodes (TIE slave correction, static only)
    _enforce_symmetry(node_rows, step_name=step_name)

    csv_nodes = os.path.join(output_dir, 'nodes.csv')
    with open(csv_nodes, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y', 'z', 'ux', 'uy', 'uz', 'u_mag', 'temp',
                          's11', 's22', 's12', 'smises', 'thermal_smises',
                          'le11', 'le22', 'le12', 'defect_label'])
        writer.writerows(node_rows)

    print("Saved nodes to " + csv_nodes + " (n_defect_nodes=%d)" % n_defect)

    # Strict mode: fail if all physics are zero (ODB likely missing thermal/load)
    if strict and node_rows:
        ux_max = max(abs(r[4]) for r in node_rows)
        uy_max = max(abs(r[5]) for r in node_rows)
        uz_max = max(abs(r[6]) for r in node_rows)
        temp_nonzero = any(abs(r[8] - 20.0) > 0.1 for r in node_rows)  # expect ~120 after thermal (index 8 = temp)
        if ux_max < 1e-12 and uy_max < 1e-12 and uz_max < 1e-12 and not temp_nonzero:
            print("ERROR [--strict]: All ux,uy,uz,temp are zero. ODB may lack thermal load.")
            print("  Check: patch_inp_thermal applied? *Temperature in Step-1? NT in Node Output?")
            sys.exit(1)

    # metadata.csv
    meta_path = os.path.join(output_dir, 'metadata.csv')
    defect_type = defect_params.get('defect_type', 'debonding') if defect_params else 'healthy'
    defect_type_id = DEFECT_TYPE_MAP.get(defect_type, 0)
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'value'])
        writer.writerow(['defect_type', defect_type])
        writer.writerow(['defect_type_id', str(defect_type_id)])
        writer.writerow(['n_defect_nodes', str(n_defect)])
        writer.writerow(['n_total_nodes', str(len(node_rows))])
        writer.writerow(['instance', instance_name])
        if defect_params:
            writer.writerow(['theta_deg', str(defect_params['theta_deg'])])
            writer.writerow(['z_center', str(defect_params['z_center'])])
            writer.writerow(['radius', str(defect_params['radius'])])
            # Type-specific parameters
            if 'stiffness_factor' in defect_params:
                writer.writerow(['stiffness_factor', str(defect_params['stiffness_factor'])])
            if 'damage_ratio' in defect_params:
                writer.writerow(['damage_ratio', str(defect_params['damage_ratio'])])
            if 'delam_depth' in defect_params:
                writer.writerow(['delam_depth', str(defect_params['delam_depth'])])
            if 'fatigue_severity' in defect_params:
                writer.writerow(['fatigue_severity', str(defect_params['fatigue_severity'])])
    print("Saved metadata to " + meta_path + " (defect_type=%s, id=%d)" % (defect_type, defect_type_id))

    # ---------------------------------------------------------
    # 2. Element Data (Connectivity, Stress)
    # ---------------------------------------------------------
    csv_elems = os.path.join(output_dir, 'elements.csv')
    with open(csv_elems, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['elem_id', 'elem_type', 'n1', 'n2', 'n3', 'n4', 'mises_avg'])

        # Build stress map (average Mises per element for simplicity)
        elem_stress = {}
        if stress_field:
            try:
                stress_vals = stress_field.getSubset(region=instance, position=CENTROID).values
                for val in stress_vals:
                    elem_stress[val.elementLabel] = val.mises
            except Exception as e:
                print("Warning: Element stress: " + str(e)[:60])

        for elem in instance.elements:
            eid = elem.label
            nodes = elem.connectivity
            n_list = list(nodes)
            while len(n_list) < 4:
                n_list.append(-1)
            etype = 'S4R' if len(nodes) >= 4 else 'S3'
            mises = elem_stress.get(eid, 0.0)
            row = [eid, etype] + n_list[:4] + [mises]
            writer.writerow(row)

    print("Saved elements to " + csv_elems)

    odb.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odb', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--defect_json', type=str, default=None,
                        help='Path to JSON with defect_params (theta_deg, z_center, radius)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail if all ux,uy,uz,temp are zero (ODB missing thermal)')

    args, unknown = parser.parse_known_args()

    defect_params = None
    if args.defect_json and os.path.exists(args.defect_json):
        with open(args.defect_json, 'r') as f:
            defect_params = json.load(f)
        print("Defect params: theta=%.1f z=%.0f r=%.0f" %
              (defect_params['theta_deg'], defect_params['z_center'], defect_params['radius']))

    extract_results(args.odb, args.output, defect_params=defect_params, strict=args.strict)
