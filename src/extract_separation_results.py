# -*- coding: utf-8 -*-
"""
Extract fairing separation ODB results for comparison visualization.

Extracts:
  1. Time-history CSV: max displacement, max stress, max velocity per instance
  2. Snapshot CSV: nodal displacement/stress at key time frames
  3. Energy history CSV: kinetic, strain, total energy

Usage:
  abaqus python src/extract_separation_results.py --odb abaqus_work/Sep-Test.odb
  abaqus python src/extract_separation_results.py --odb abaqus_work/Sep-Stuck3.odb
"""

import sys
import os
import csv
import math
import argparse
from odbAccess import *
from abaqusConstants import *


def extract_time_history(odb, output_dir, job_name):
    """Extract per-frame max displacement/stress/velocity per instance."""
    rows = []

    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        for i_frame, frame in enumerate(step.frames):
            t = frame.frameValue
            if step_name == 'Step-Separation':
                t += odb.steps['Step-Preload'].frames[-1].frameValue

            # Get field outputs
            u_field = frame.fieldOutputs['U'] if 'U' in frame.fieldOutputs else None
            v_field = frame.fieldOutputs['V'] if 'V' in frame.fieldOutputs else None
            s_field = frame.fieldOutputs['S'] if 'S' in frame.fieldOutputs else None

            if u_field is None:
                continue

            # Per-instance statistics
            instance_stats = {}
            for inst_name in odb.rootAssembly.instances.keys():
                inst = odb.rootAssembly.instances[inst_name]
                if len(inst.nodes) == 0:
                    continue

                # Displacement
                u_sub = u_field.getSubset(region=inst)
                u_mag_max = 0.0
                u_x_max = 0.0
                u_y_max = 0.0
                u_z_max = 0.0
                for val in u_sub.values:
                    mag = val.magnitude
                    if mag > u_mag_max:
                        u_mag_max = mag
                        u_x_max = val.data[0]
                        u_y_max = val.data[1]
                        u_z_max = val.data[2]

                # Velocity
                v_mag_max = 0.0
                if v_field is not None:
                    v_sub = v_field.getSubset(region=inst)
                    for val in v_sub.values:
                        if val.magnitude > v_mag_max:
                            v_mag_max = val.magnitude

                # Stress (Mises)
                s_mises_max = 0.0
                if s_field is not None:
                    s_sub = s_field.getSubset(region=inst)
                    for val in s_sub.values:
                        if val.mises > s_mises_max:
                            s_mises_max = val.mises

                instance_stats[inst_name] = {
                    'u_mag_max': u_mag_max,
                    'u_x_max': u_x_max,
                    'u_y_max': u_y_max,
                    'u_z_max': u_z_max,
                    'v_mag_max': v_mag_max,
                    's_mises_max': s_mises_max,
                }

            for inst_name, stats in instance_stats.items():
                rows.append([
                    step_name, t, inst_name,
                    stats['u_mag_max'], stats['u_x_max'],
                    stats['u_y_max'], stats['u_z_max'],
                    stats['v_mag_max'], stats['s_mises_max'],
                ])

    # Write CSV
    csv_path = os.path.join(output_dir, '%s_time_history.csv' % job_name)
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'time_s', 'instance',
            'u_mag_max_mm', 'u_x_max_mm', 'u_y_max_mm', 'u_z_max_mm',
            'v_mag_max_mm_s', 's_mises_max_MPa',
        ])
        writer.writerows(rows)

    print("  Time history: %s (%d rows)" % (csv_path, len(rows)))
    return csv_path


def extract_snapshots(odb, output_dir, job_name, snapshot_times=None):
    """Extract nodal displacement at specific time snapshots."""
    if snapshot_times is None:
        # Key moments: end of preload, early/mid/late separation
        snapshot_times = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]

    step_sep = odb.steps['Step-Separation'] if 'Step-Separation' in odb.steps else None
    if step_sep is None:
        print("  WARNING: Step-Separation not found")
        return

    t_preload = odb.steps['Step-Preload'].frames[-1].frameValue

    for t_target in snapshot_times:
        # Find closest frame
        best_frame = None
        best_dt = 1e30
        for step_name in odb.steps.keys():
            step = odb.steps[step_name]
            for frame in step.frames:
                t = frame.frameValue
                if step_name == 'Step-Separation':
                    t += t_preload
                dt = abs(t - t_target)
                if dt < best_dt:
                    best_dt = dt
                    best_frame = frame
                    best_step = step_name

        if best_frame is None:
            continue

        u_field = best_frame.fieldOutputs['U'] if 'U' in frame.fieldOutputs else None
        if u_field is None:
            continue

        rows = []
        for val in u_field.values:
            node = val.nodeLabel
            inst_name = val.instance.name if val.instance else 'ASSEMBLY'
            rows.append([
                inst_name, node,
                val.data[0], val.data[1], val.data[2], val.magnitude,
            ])

        csv_path = os.path.join(
            output_dir,
            '%s_snapshot_t%.3f.csv' % (job_name, t_target))
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['instance', 'node_id',
                              'u_x_mm', 'u_y_mm', 'u_z_mm', 'u_mag_mm'])
            writer.writerows(rows)

        print("  Snapshot t=%.3fs: %s (%d nodes)" % (
            t_target, csv_path, len(rows)))


def extract_energy(odb, output_dir, job_name):
    """Extract energy history (ALLKE, ALLSE, ETOTAL)."""
    rows = []

    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        history_region = None

        # Find assembly-level history region
        for hr_name, hr in step.historyRegions.items():
            if 'Assembly' in hr_name or hr_name == 'Assembly ASSEMBLY':
                history_region = hr
                break

        if history_region is None:
            # Try first available
            for hr_name, hr in step.historyRegions.items():
                history_region = hr
                break

        if history_region is None:
            continue

        # Get available energy outputs
        ke_data = history_region.historyOutputs['ALLKE'] if 'ALLKE' in history_region.historyOutputs else None
        se_data = history_region.historyOutputs['ALLSE'] if 'ALLSE' in history_region.historyOutputs else None
        et_data = history_region.historyOutputs['ETOTAL'] if 'ETOTAL' in history_region.historyOutputs else None

        if ke_data is None:
            continue

        for i, (t, ke_val) in enumerate(ke_data.data):
            se_val = se_data.data[i][1] if se_data and i < len(se_data.data) else 0.0
            et_val = et_data.data[i][1] if et_data and i < len(et_data.data) else 0.0
            rows.append([step_name, t, ke_val, se_val, et_val])

    csv_path = os.path.join(output_dir, '%s_energy.csv' % job_name)
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'time_s', 'ALLKE_mJ', 'ALLSE_mJ', 'ETOTAL_mJ'])
        writer.writerows(rows)

    print("  Energy: %s (%d rows)" % (csv_path, len(rows)))
    return csv_path


def extract_graph_data(odb, output_dir, job_name, t_target=0.2):
    """Extract nodes.csv + elements.csv for GNN graph construction.

    Outputs compatible with build_graph.py / build_curvature_graph():
      nodes.csv: node_id, x, y, z, ux, uy, uz, u_mag, s11, s22, s12, smises, defect_label
      elements.csv: elem_id, n1, n2, n3, n4, elem_type
    """
    # Find target frame
    best_frame = None
    best_dt = 1e30
    t_preload = odb.steps['Step-Preload'].frames[-1].frameValue
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        for frame in step.frames:
            t = frame.frameValue
            if step_name == 'Step-Separation':
                t += t_preload
            dt = abs(t - t_target)
            if dt < best_dt:
                best_dt = dt
                best_frame = frame

    if best_frame is None:
        print("  ERROR: No frame found near t=%.3f" % t_target)
        return

    # Collect per-instance: merge shell instances (InnerSkin, OuterSkin)
    # Skip Core (solid C3D8R) — use skin for GNN (shell mesh, 2D surface graph)
    target_instances = []
    for inst_name in odb.rootAssembly.instances.keys():
        if 'SKIN' in inst_name.upper():
            target_instances.append(inst_name)

    if not target_instances:
        # Fallback: all instances with elements
        for inst_name, inst in odb.rootAssembly.instances.items():
            if len(inst.elements) > 0:
                target_instances.append(inst_name)

    print("    Target instances: %s" % target_instances)

    # Get field outputs (Abaqus Repository uses [] not .get())
    u_field = best_frame.fieldOutputs['U'] if 'U' in best_frame.fieldOutputs else None
    s_field = best_frame.fieldOutputs['S'] if 'S' in best_frame.fieldOutputs else None
    v_field = best_frame.fieldOutputs['V'] if 'V' in best_frame.fieldOutputs else None

    # Build node data per instance, with global node ID remapping
    node_rows = []
    elem_rows = []
    global_nid = 1
    global_eid = 1
    inst_node_map = {}  # (inst_name, local_nid) -> global_nid

    for inst_name in sorted(target_instances):
        inst = odb.rootAssembly.instances[inst_name]

        # Node coordinates
        for node in inst.nodes:
            coords = node.coordinates
            inst_node_map[(inst_name, node.label)] = global_nid
            node_rows.append({
                'global_nid': global_nid,
                'inst': inst_name,
                'local_nid': node.label,
                'x': coords[0], 'y': coords[1], 'z': coords[2],
                'ux': 0.0, 'uy': 0.0, 'uz': 0.0, 'u_mag': 0.0,
                's11': 0.0, 's22': 0.0, 's12': 0.0, 'smises': 0.0,
                'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'v_mag': 0.0,
            })
            global_nid += 1

    # Fill displacement
    if u_field:
        for val in u_field.values:
            if val.instance is None:
                continue
            key = (val.instance.name, val.nodeLabel)
            if key in inst_node_map:
                gid = inst_node_map[key]
                row = node_rows[gid - 1]
                row['ux'] = val.data[0]
                row['uy'] = val.data[1]
                row['uz'] = val.data[2]
                row['u_mag'] = val.magnitude

    # Fill stress (extrapolated to nodes)
    if s_field:
        try:
            s_nodal = s_field.getSubset(position=ELEMENT_NODAL)
            # Average over integration points per node
            stress_accum = {}  # global_nid -> [sum_s11, sum_s22, sum_s12, sum_smises, count]
            for val in s_nodal.values:
                if val.instance is None:
                    continue
                key = (val.instance.name, val.nodeLabel)
                if key in inst_node_map:
                    gid = inst_node_map[key]
                    if gid not in stress_accum:
                        stress_accum[gid] = [0.0, 0.0, 0.0, 0.0, 0]
                    stress_accum[gid][0] += val.data[0]
                    stress_accum[gid][1] += val.data[1]
                    stress_accum[gid][2] += val.data[2] if len(val.data) > 2 else 0.0
                    stress_accum[gid][3] += val.mises
                    stress_accum[gid][4] += 1

            for gid, (s11, s22, s12, sm, cnt) in stress_accum.items():
                row = node_rows[gid - 1]
                row['s11'] = s11 / cnt
                row['s22'] = s22 / cnt
                row['s12'] = s12 / cnt
                row['smises'] = sm / cnt
        except Exception as e:
            print("    WARNING: Stress extraction failed: %s" % str(e))

    # Fill velocity
    if v_field:
        for val in v_field.values:
            if val.instance is None:
                continue
            key = (val.instance.name, val.nodeLabel)
            if key in inst_node_map:
                gid = inst_node_map[key]
                row = node_rows[gid - 1]
                row['vx'] = val.data[0]
                row['vy'] = val.data[1]
                row['vz'] = val.data[2]
                row['v_mag'] = val.magnitude

    # Element connectivity (remap to global node IDs)
    for inst_name in sorted(target_instances):
        inst = odb.rootAssembly.instances[inst_name]
        for elem in inst.elements:
            conn = list(elem.connectivity)
            mapped = []
            valid = True
            for local_nid in conn:
                key = (inst_name, local_nid)
                if key in inst_node_map:
                    mapped.append(inst_node_map[key])
                else:
                    valid = False
                    break
            if not valid:
                continue

            etype = str(elem.type)
            # Pad to 4 nodes (for triangles, 4th = 0)
            while len(mapped) < 4:
                mapped.append(0)

            elem_rows.append({
                'elem_id': global_eid,
                'n1': mapped[0], 'n2': mapped[1],
                'n3': mapped[2], 'n4': mapped[3],
                'elem_type': etype,
            })
            global_eid += 1

    # Write nodes.csv
    nodes_csv = os.path.join(output_dir, '%s_nodes.csv' % job_name)
    with open(nodes_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'node_id', 'x', 'y', 'z',
            'ux', 'uy', 'uz', 'u_mag',
            's11', 's22', 's12', 'smises',
            'vx', 'vy', 'vz', 'v_mag',
            'defect_label',
        ])
        for row in node_rows:
            writer.writerow([
                row['global_nid'], row['x'], row['y'], row['z'],
                row['ux'], row['uy'], row['uz'], row['u_mag'],
                row['s11'], row['s22'], row['s12'], row['smises'],
                row['vx'], row['vy'], row['vz'], row['v_mag'],
                0,  # defect_label: 0=normal, assigned later
            ])

    # Write elements.csv
    elems_csv = os.path.join(output_dir, '%s_elements.csv' % job_name)
    with open(elems_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['elem_id', 'n1', 'n2', 'n3', 'n4', 'elem_type'])
        for row in elem_rows:
            writer.writerow([
                row['elem_id'], row['n1'], row['n2'], row['n3'],
                row['n4'], row['elem_type'],
            ])

    print("  Graph data: %s (%d nodes)" % (nodes_csv, len(node_rows)))
    print("  Graph data: %s (%d elements)" % (elems_csv, len(elem_rows)))
    return nodes_csv, elems_csv


def main():
    parser = argparse.ArgumentParser(
        description='Extract fairing separation ODB results')
    parser.add_argument('--odb', required=True, help='ODB file path')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: same as ODB)')
    parser.add_argument('--graph', action='store_true',
                        help='Also extract graph data (nodes.csv + elements.csv)')
    parser.add_argument('--graph_time', type=float, default=0.2,
                        help='Time for graph snapshot (default: 0.2s)')
    args, _ = parser.parse_known_args()

    odb_path = args.odb
    if not os.path.exists(odb_path):
        print("ERROR: ODB not found: %s" % odb_path)
        sys.exit(1)

    job_name = os.path.splitext(os.path.basename(odb_path))[0]
    output_dir = args.output or os.path.dirname(odb_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 60)
    print("Extracting: %s" % odb_path)
    print("  Output: %s" % output_dir)
    print("=" * 60)

    odb = openOdb(odb_path, readOnly=True)

    print("\nSteps: %s" % list(odb.steps.keys()))
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        print("  %s: %d frames, t=[0, %.4f]" % (
            step_name, len(step.frames), step.frames[-1].frameValue))

    print("\n[1/3] Extracting time history...")
    extract_time_history(odb, output_dir, job_name)

    print("\n[2/3] Extracting snapshots...")
    extract_snapshots(odb, output_dir, job_name)

    print("\n[3/3] Extracting energy history...")
    extract_energy(odb, output_dir, job_name)

    if args.graph:
        print("\n[4/4] Extracting graph data (nodes + elements)...")
        extract_graph_data(odb, output_dir, job_name, t_target=args.graph_time)

    odb.close()
    print("\nDone: %s" % job_name)


if __name__ == '__main__':
    main()
