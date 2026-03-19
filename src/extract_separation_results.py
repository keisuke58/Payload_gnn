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


def main():
    parser = argparse.ArgumentParser(
        description='Extract fairing separation ODB results')
    parser.add_argument('--odb', required=True, help='ODB file path')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: same as ODB)')
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

    odb.close()
    print("\nDone: %s" % job_name)


if __name__ == '__main__':
    main()
