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
        
    # Get the last frame of the first analysis step (skip Initial if present)
    step_keys = list(odb.steps.keys())
    if not step_keys:
        print("Error: ODB has no steps (job may have failed)")
        sys.exit(1)
    # Prefer first non-Initial step
    step_name = step_keys[0]
    for k in step_keys:
        if k.upper() != 'INITIAL':
            step_name = k
            break
    step_obj = odb.steps[step_name]
    if len(step_obj.frames) == 0:
        # Try first step with frames
        for k in step_keys:
            if len(odb.steps[k].frames) > 0:
                step_name = k
                step_obj = odb.steps[k]
                break
    if len(step_obj.frames) == 0:
        print("Error: No frames in any step (job may have failed)")
        sys.exit(1)
    frame = step_obj.frames[-1]
    
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
    temp_field = None
    for key in ('NT11', 'NT12', 'NT', 'TEMP'):
        if key in frame.fieldOutputs:
            temp_field = frame.fieldOutputs[key]
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

        # thermal_smises: In this model, ALL loading is thermal (CTE mismatch).
        # Therefore thermal von Mises ≈ total von Mises.
        # Future: for mixed loading, extract STHERM field separately.
        thermal_smises = smises

        # Strain (LE11, LE22, LE12) if extracted
        le11 = le22 = le12 = 0.0
        if nid in node_strain:
            le11, le22, le12 = node_strain[nid][0], node_strain[nid][1], node_strain[nid][2]

        node_rows.append([nid, x, y, z, u[0], u[1], u[2], u_mag, t,
                          s11, s22, s12, smises, thermal_smises,
                          le11, le22, le12, defect_label])

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
