# -*- coding: utf-8 -*-
# extract_odb_results.py
# Abaqus Python script to extract nodal and element data from ODB
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


def extract_results(odb_path, output_dir, defect_params=None):
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
    if instance_name not in odb.rootAssembly.instances.keys():
         # Fallback to keys if case doesn't match or using different naming
         instance_name = odb.rootAssembly.instances.keys()[0]
         
    instance = odb.rootAssembly.instances[instance_name]
    
    # Field Outputs (NT11/NT12/NT=nodal temp, TEMP=element temp)
    disp_field = frame.fieldOutputs['U']
    temp_field = None
    for key in ('NT11', 'NT12', 'NT', 'TEMP'):
        if key in frame.fieldOutputs:
            temp_field = frame.fieldOutputs[key]
            break
    
    # Subset to instance
    disp_sub = disp_field.getSubset(region=instance)
    temp_sub = temp_field.getSubset(region=instance) if temp_field else None
    
    node_data = []
    
    # Map Node Labels to Values
    # Note: iterating bulk data is faster but this is clearer for script
    
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
            
    # Iterate nodes to get coordinates and defect labels
    # Coordinates in ODB are usually original. Deformed = Original + U
    # defect_label: 0=healthy, 1=inside defect zone (from defect_params geometry)

    node_rows = []
    n_defect = 0
    for node in instance.nodes:
        nid = node.label
        x, y, z = node.coordinates

        u = disp_values.get(nid, (0.0, 0.0, 0.0))
        t = temp_values.get(nid, 0.0)

        defect_label = 0
        if defect_params:
            if _is_node_in_defect(x, y, z, defect_params):
                defect_label = 1
                n_defect += 1

        node_rows.append([nid, x, y, z, u[0], u[1], u[2], t, defect_label])

    csv_nodes = os.path.join(output_dir, 'nodes.csv')
    with open(csv_nodes, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y', 'z', 'ux', 'uy', 'uz', 'temp', 'defect_label'])
        writer.writerows(node_rows)

    print("Saved nodes to " + csv_nodes + " (n_defect_nodes=%d)" % n_defect)

    # metadata.csv
    meta_path = os.path.join(output_dir, 'metadata.csv')
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'value'])
        writer.writerow(['defect_type', 'debonding' if defect_params else 'healthy'])
        writer.writerow(['n_defect_nodes', str(n_defect)])
        if defect_params:
            writer.writerow(['theta_deg', str(defect_params['theta_deg'])])
            writer.writerow(['z_center', str(defect_params['z_center'])])
            writer.writerow(['radius', str(defect_params['radius'])])
    print("Saved metadata to " + meta_path)

    # ---------------------------------------------------------
    # 2. Element Data (Connectivity, Stress)
    # ---------------------------------------------------------
    stress_field = frame.fieldOutputs['S']
    stress_sub = stress_field.getSubset(region=instance, position=ELEMENT_NODAL) 
    # ELEMENT_NODAL gives stress at nodes per element. 
    # For simple GNN, CENTROID or INTEGRATION_POINT might be easier, 
    # but let's stick to simple connectivity for now.
    
    # Actually, for GNN structure, we just need the topology (connectivity).
    # Stress is a feature we might want.
    
    csv_elems = os.path.join(output_dir, 'elements.csv')
    with open(csv_elems, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['elem_id', 'elem_type', 'n1', 'n2', 'n3', 'n4', 'mises_avg'])

        # Build stress map (average Mises per element for simplicity)
        stress_vals = stress_field.getSubset(region=instance, position=CENTROID).values
        elem_stress = {}
        for val in stress_vals:
            elem_stress[val.elementLabel] = val.mises

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

    args, unknown = parser.parse_known_args()

    defect_params = None
    if args.defect_json and os.path.exists(args.defect_json):
        with open(args.defect_json, 'r') as f:
            defect_params = json.load(f)
        print("Defect params: theta=%.1f z=%.0f r=%.0f" %
              (defect_params['theta_deg'], defect_params['z_center'], defect_params['radius']))

    extract_results(args.odb, args.output, defect_params=defect_params)
