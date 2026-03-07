#!/usr/bin/env python3
"""Extract node-level data from CompDam ODB for PyG graph conversion.

Extracts:
  - Node coordinates (x, y, z)
  - Displacement (U1, U2, U3, Umag)
  - Element connectivity (C3D8R → node adjacency)
  - Element-to-node stress mapping (averaged)

Run with:  abaqus python scripts/extract_compdam_nodes.py

Output: abaqus_work/compdam_flatplate/compdam_node_data.csv
        abaqus_work/compdam_flatplate/compdam_elements.csv
"""

import os
import sys
import csv
import math
from collections import defaultdict

from odbAccess import openOdb

ODB_DIR = 'abaqus_work/compdam_flatplate'


def main():
    odb_path = os.path.join(ODB_DIR, 'compdam_flatplate_impact.odb')
    partial = os.path.join(ODB_DIR, 'compdam_flatplate_impact_partial.odb')
    if not os.path.exists(odb_path) and os.path.exists(partial):
        odb_path = partial

    print("Opening ODB: {}".format(odb_path))
    odb = openOdb(odb_path, readOnly=True)

    step = odb.steps['Impact']
    last_frame = step.frames[-1]
    print("Step: {}, Frame: {} (time={})".format(
        step.name, last_frame.frameId, last_frame.frameValue))

    # List available fields
    fields = list(last_frame.fieldOutputs.keys())
    print("Available fields: {}".format(', '.join(fields[:30])))

    # ── Extract node coordinates ──
    instance = odb.rootAssembly.instances['PLATE-1']
    nodes = instance.nodes
    print("Nodes: {}".format(len(nodes)))

    node_coords = {}
    for n in nodes:
        node_coords[n.label] = (n.coordinates[0], n.coordinates[1], n.coordinates[2])

    # ── Extract displacement (node-level) ──
    node_disp = {}
    if 'U' in last_frame.fieldOutputs:
        u_field = last_frame.fieldOutputs['U']
        for val in u_field.values:
            nid = val.nodeLabel
            u1, u2, u3 = float(val.data[0]), float(val.data[1]), float(val.data[2])
            umag = math.sqrt(u1**2 + u2**2 + u3**2)
            node_disp[nid] = (u1, u2, u3, umag)
        print("Displacement extracted: {} nodes".format(len(node_disp)))
    else:
        print("WARNING: U field not found")

    # ── Extract element connectivity ──
    elements = instance.elements
    print("Elements: {}".format(len(elements)))

    elem_data = []
    elem_to_nodes = {}  # elem_id → list of node IDs (O(1) lookup)
    for e in elements:
        conn = [int(c) for c in e.connectivity]
        elem_data.append({
            'elem_id': e.label,
            'type': str(e.type),
            'nodes': conn,
        })
        elem_to_nodes[e.label] = conn

    # ── Extract element-level stress + damage → map to nodes ──
    # Aggregate element values to nodes (average of surrounding elements)
    node_stress = defaultdict(lambda: {'S11': [], 'S22': [], 'S33': [], 'S12': []})
    node_damage = defaultdict(lambda: {'d2': [], 'd1T': [], 'd1C': []})

    # Stress
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
        print("Stress mapped to nodes: {} nodes".format(len(node_stress)))

    # SDV damage variables
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
            print("  {} mapped to nodes".format(sdv_label))

    # ── Write node CSV ──
    csv_path = os.path.join(ODB_DIR, 'compdam_node_data.csv')
    header = ['node_id', 'x', 'y', 'z',
              'U1', 'U2', 'U3', 'Umag',
              'S11', 'S22', 'S33', 'S12', 'Mises',
              'CDM_d2', 'CDM_d1T', 'CDM_d1C', 'damage_label']

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in sorted(node_coords.keys()):
            x, y, z = node_coords[nid]
            u1, u2, u3, umag = node_disp.get(nid, (0, 0, 0, 0))

            # Average stress from surrounding elements
            if nid in node_stress:
                s11 = sum(node_stress[nid]['S11']) / len(node_stress[nid]['S11'])
                s22 = sum(node_stress[nid]['S22']) / len(node_stress[nid]['S22'])
                s33 = sum(node_stress[nid]['S33']) / len(node_stress[nid]['S33'])
                s12 = sum(node_stress[nid]['S12']) / len(node_stress[nid]['S12'])
            else:
                s11 = s22 = s33 = s12 = 0.0
            # Von Mises
            mises = math.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*s12**2))

            # Average damage
            if nid in node_damage:
                d2 = sum(node_damage[nid]['d2']) / len(node_damage[nid]['d2']) if node_damage[nid]['d2'] else 0
                d1T = sum(node_damage[nid]['d1T']) / len(node_damage[nid]['d1T']) if node_damage[nid]['d1T'] else 0
                d1C = sum(node_damage[nid]['d1C']) / len(node_damage[nid]['d1C']) if node_damage[nid]['d1C'] else 0
            else:
                d2 = d1T = d1C = 0.0

            # Binary damage label: 1 if any damage > threshold
            damage_label = 1 if (d2 > 0.01 or d1T > 0.01 or d1C > 0.01) else 0

            writer.writerow([nid, x, y, z,
                            u1, u2, u3, umag,
                            s11, s22, s33, s12, mises,
                            d2, d1T, d1C, damage_label])

    print("\nNode CSV written: {} ({} nodes)".format(csv_path, len(node_coords)))

    # ── Write element connectivity CSV ──
    elem_csv = os.path.join(ODB_DIR, 'compdam_elements.csv')
    with open(elem_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['elem_id', 'type', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8'])
        for ed in elem_data:
            row = [ed['elem_id'], ed['type']] + ed['nodes']
            writer.writerow(row)
    print("Element CSV written: {} ({} elements)".format(elem_csv, len(elem_data)))

    # Summary
    n_damaged = sum(1 for nid in node_coords if nid in node_damage
                    and any(node_damage[nid][k] and max(node_damage[nid][k]) > 0.01
                            for k in ['d2', 'd1T', 'd1C']))
    print("\nDamaged nodes: {}/{} ({:.2f}%)".format(
        n_damaged, len(node_coords), 100*n_damaged/len(node_coords)))

    odb.close()
    print("Done!")


if __name__ == '__main__':
    main()
