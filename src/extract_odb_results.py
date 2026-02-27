# -*- coding: utf-8 -*-
# ODB Result Extraction - H3 Fairing FEM to CSV
#
# Extracts nodal field outputs (stress, displacement, coordinates) from an
# Abaqus ODB file and writes CSV files compatible with the GNN preprocessing
# pipeline (preprocess_fairing_data.py).
#
# Stress is extracted at ELEMENT_NODAL position (extrapolated from integration
# points to nodes), then averaged across elements sharing each node.
#
# Uses streaming writes per instance to avoid Abaqus Python memory/GC segfault.
#
# Runs in Abaqus Python environment:
#   cd abaqus_work
#   abaqus python ../src/extract_odb_results.py H3_Healthy.odb ../dataset_output/healthy_baseline
#   abaqus python ../src/extract_odb_results.py H3_Debond.odb ../dataset_output/sample_0001 --defect_json params.json

from odbAccess import *
from abaqusConstants import *
import math
import os
import sys
import csv
import json

# Geometry constants (must match generate_fairing_dataset.py)
RADIUS = 2600.0
FACE_T = 0.125 * 8  # 1.0 mm
CORE_T = 38.0
R_CORE_O = RADIUS - FACE_T / 2.0
R_CORE_I = R_CORE_O - CORE_T
R_OUTER = RADIUS
R_INNER = RADIUS - FACE_T - CORE_T

ELEM_TYPE_MAP = {
    'S4R': 'S4R', 'S4RT': 'S4RT', 'S3': 'S3', 'S3R': 'S3',
    'C3D8R': 'C3D8R', 'C3D8': 'C3D8',
    'C3D6': 'C3D6', 'C3D4': 'C3D4',
}

LABEL_OFFSETS = {
    'Skin_Outer': 0,
    'Skin_Inner': 100000,
    'Core': 200000,
}


def _get_part_name(inst_name):
    upper = inst_name.upper()
    if 'SKIN_OUTER' in upper:
        return 'Skin_Outer'
    elif 'SKIN_INNER' in upper:
        return 'Skin_Inner'
    elif 'CORE' in upper:
        return 'Core'
    return inst_name


def _is_in_debonding_zone(x, y, z, defect_params, surface_tol=5.0):
    r = math.sqrt(x**2 + y**2)
    near_outer_skin = abs(r - R_OUTER) < surface_tol
    near_core_outer = abs(r - R_CORE_O) < surface_tol
    if not (near_outer_skin or near_core_outer):
        return False
    theta = math.atan2(y, x)
    theta_c = math.radians(defect_params['theta_deg'])
    z_c = defect_params['z_center']
    r_def = defect_params['radius']
    d_theta = r_def / R_CORE_O
    d_z = r_def
    in_theta = (theta_c - d_theta) <= theta <= (theta_c + d_theta)
    in_z = (z_c - d_z) <= z <= (z_c + d_z)
    return in_theta and in_z


def extract_odb(odb_path, output_dir, defect_params=None):
    os.makedirs(output_dir, exist_ok=True)

    print("Opening ODB: %s" % odb_path)
    sys.stdout.flush()
    odb = openOdb(path=odb_path, readOnly=True)

    step_names = list(odb.steps.keys())
    step_name = step_names[-1]
    step = odb.steps[step_name]
    frame = step.frames[-1]
    print("Step: %s, Frame: %d" % (step_name, frame.incrementNumber))
    sys.stdout.flush()

    stress_field = frame.fieldOutputs['S']

    # ------------------------------------------------------------------
    # nodes.csv - stream per instance to control memory
    # ------------------------------------------------------------------
    nodes_path = os.path.join(output_dir, 'nodes.csv')
    n_nodes = 0
    n_defect = 0

    with open(nodes_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'x', 'y', 'z',
                     's11', 's22', 's12', 'dspss', 'defect_label'])

        for inst_name in sorted(odb.rootAssembly.instances.keys()):
            instance = odb.rootAssembly.instances[inst_name]
            if inst_name.upper() == 'ASSEMBLY':
                continue

            part_name = _get_part_name(inst_name)
            label_offset = LABEL_OFFSETS.get(part_name, 0)

            print("Instance: %s (%d nodes, %d elems)" %
                  (inst_name, len(instance.nodes), len(instance.elements)))
            sys.stdout.flush()

            # Stress: running sums per node
            sub = stress_field.getSubset(
                region=instance, position=ELEMENT_NODAL)
            node_sums = {}
            for v in sub.values:
                label = v.nodeLabel
                data = v.data
                s11 = float(data[0]) if len(data) > 0 else 0.0
                s22 = float(data[1]) if len(data) > 1 else 0.0
                if len(data) >= 6:
                    s12 = float(data[3])
                elif len(data) >= 3:
                    s12 = float(data[2])
                else:
                    s12 = 0.0
                mises = float(v.mises)
                if label in node_sums:
                    s = node_sums[label]
                    s[0] += s11
                    s[1] += s22
                    s[2] += s12
                    s[3] += mises
                    s[4] += 1
                else:
                    node_sums[label] = [s11, s22, s12, mises, 1]

            print("  Stress: %d nodes" % len(node_sums))
            sys.stdout.flush()

            # Write node rows directly to CSV
            count = 0
            for node in instance.nodes:
                label = node.label
                x = float(node.coordinates[0])
                y = float(node.coordinates[1])
                z = float(node.coordinates[2])

                sv = node_sums.get(label)
                if sv:
                    n = sv[4]
                    s11 = sv[0] / n
                    s22 = sv[1] / n
                    s12 = sv[2] / n
                    dspss = sv[3] / n
                else:
                    s11 = s22 = s12 = dspss = 0.0

                dl = 0
                if defect_params:
                    if _is_in_debonding_zone(x, y, z, defect_params):
                        dl = 1
                        n_defect += 1

                w.writerow([label + label_offset, x, y, z,
                            s11, s22, s12, dspss, dl])
                count += 1

            n_nodes += count
            # Free memory before next instance
            del node_sums
            del sub
            print("  Written: %d (total: %d)" % (count, n_nodes))
            sys.stdout.flush()

    print("nodes.csv: %d nodes" % n_nodes)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # elements.csv - stream directly
    # ------------------------------------------------------------------
    elems_path = os.path.join(output_dir, 'elements.csv')
    n_elems = 0

    with open(elems_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['elem_id', 'elem_type',
                     'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
                     'part_name'])
        for inst_name in sorted(odb.rootAssembly.instances.keys()):
            instance = odb.rootAssembly.instances[inst_name]
            if inst_name.upper() == 'ASSEMBLY':
                continue
            part_name = _get_part_name(inst_name)
            label_offset = LABEL_OFFSETS.get(part_name, 0)
            for elem in instance.elements:
                etype = ELEM_TYPE_MAP.get(elem.type, elem.type)
                conn = elem.connectivity
                nc = len(conn)
                nodes = []
                for k in range(8):
                    if k < nc:
                        nodes.append(conn[k] + label_offset)
                    else:
                        nodes.append(-1)
                if etype in ('S3', 'S3R'):
                    nodes[3:] = [-1] * 5
                elif etype in ('S4R', 'S4RT'):
                    nodes[4:] = [-1] * 4
                w.writerow([elem.label, etype] + nodes + [part_name])
                n_elems += 1

    print("elements.csv: %d elements" % n_elems)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # metadata.csv
    # ------------------------------------------------------------------
    meta_path = os.path.join(output_dir, 'metadata.csv')
    with open(meta_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value'])
        if defect_params:
            w.writerow(['defect_type', 'debonding'])
            w.writerow(['defect_theta_deg', str(defect_params['theta_deg'])])
            w.writerow(['defect_z_center', str(defect_params['z_center'])])
            w.writerow(['defect_radius', str(defect_params['radius'])])
            w.writerow(['interface', 'outer'])
        else:
            w.writerow(['defect_type', 'healthy'])
            w.writerow(['defect_radius', '0'])
        w.writerow(['odb_file', os.path.basename(odb_path)])
        w.writerow(['n_nodes', str(n_nodes)])
        w.writerow(['n_elements', str(n_elems)])
        w.writerow(['n_defect_nodes', str(n_defect)])

    print("metadata.csv written")
    print("  Defect nodes: %d / %d (%.1f%%)" % (
        n_defect, n_nodes, 100.0 * n_defect / max(n_nodes, 1)))
    sys.stdout.flush()

    odb.close()
    print("Done.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: abaqus python extract_odb_results.py "
              "<odb_file> <output_dir> [--defect_json <path>]")
        sys.exit(1)

    odb_path = sys.argv[1]
    output_dir = sys.argv[2]

    defect_params = None
    if '--defect_json' in sys.argv:
        idx = sys.argv.index('--defect_json')
        if idx + 1 < len(sys.argv):
            with open(sys.argv[idx + 1], 'r') as f:
                defect_params = json.load(f)

    extract_odb(odb_path, output_dir, defect_params)
