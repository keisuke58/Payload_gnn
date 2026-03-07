# -*- coding: utf-8 -*-
"""extract_gw_sensors.py — Extract sensor history from GW ODB files.

Reads History Output from Abaqus/Explicit GW ODB and writes _sensors.csv.
Each sensor has U1, U2, U3 → compute Ur (radial displacement).

Usage (Abaqus Python):
  abaqus python src/extract_gw_sensors.py --odb abaqus_work/Job-GW-Fair-0034.odb
  abaqus python src/extract_gw_sensors.py --odb_dir abaqus_work --jobs 0034 0068
"""

import os
import sys
import csv
import math
import argparse

from odbAccess import openOdb
from abaqusConstants import *


def extract_sensors(odb_path, output_csv=None):
    """Extract sensor history output from ODB to CSV.

    Reads all H-Output-S* history output requests, computes radial
    displacement Ur = sqrt(U1^2 + U2^2 + U3^2) for each sensor.
    Also saves sensor positions from node coordinates.
    """
    odb = openOdb(path=odb_path, readOnly=True)

    job_name = os.path.splitext(os.path.basename(odb_path))[0]
    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(odb_path),
                                  job_name + '_sensors.csv')

    # Find the explicit step
    step = None
    for s in odb.steps.values():
        step = s
        break
    if step is None:
        print("ERROR: No steps found in %s" % odb_path)
        odb.close()
        return None

    print("Step: %s, frames: %d" % (step.name, len(step.frames)))

    # Collect sensor history regions
    # Format: "Node PART-OUTERSKIN-1.12345" with U1/U2/U3 outputs
    # Skip "Assembly ASSEMBLY" (energy outputs only)
    sensor_regions = {}  # sensor_id -> history region key
    sid = 0
    for region_key in step.historyRegions.keys():
        if not region_key.startswith('Node '):
            continue
        region = step.historyRegions[region_key]
        out_keys = region.historyOutputs.keys()
        if 'U1' in out_keys:
            sensor_regions[sid] = region_key
            sid += 1

    n_sensors = len(sensor_regions)
    if n_sensors == 0:
        print("WARNING: No sensor history regions found in %s" % odb_path)
        odb.close()
        return None

    print("Found %d sensors" % n_sensors)

    # Extract time history for each sensor
    sensor_data = {}  # sid -> list of (time, Ur)
    sensor_positions = {}  # sid -> x_mm (arc distance from excitation)

    for sid in sorted(sensor_regions.keys()):
        region_key = sensor_regions[sid]
        region = step.historyRegions[region_key]

        # Get U1, U2, U3 (Abaqus Repository has no .get())
        u1_data = region.historyOutputs['U1'] if 'U1' in region.historyOutputs.keys() else None
        u2_data = region.historyOutputs['U2'] if 'U2' in region.historyOutputs.keys() else None
        u3_data = region.historyOutputs['U3'] if 'U3' in region.historyOutputs.keys() else None

        if u1_data is None:
            print("  Sensor %d: no U1 data, skipping" % sid)
            continue

        # Extract time-value pairs
        times_u1 = [t for t, v in u1_data.data]
        vals_u1 = [v for t, v in u1_data.data]
        vals_u2 = [v for t, v in u2_data.data] if u2_data else [0.0] * len(vals_u1)
        vals_u3 = [v for t, v in u3_data.data] if u3_data else [0.0] * len(vals_u1)

        # Compute radial displacement magnitude
        ur = [math.sqrt(u1**2 + u2**2 + u3**2) for u1, u2, u3
              in zip(vals_u1, vals_u2, vals_u3)]

        sensor_data[sid] = list(zip(times_u1, ur))

        # Get node position from assembly
        if region_key.startswith('Node '):
            node_ref = region_key[5:]
            parts = node_ref.split('.')
            if len(parts) == 2:
                inst_name, node_label = parts[0], int(parts[1])
                try:
                    inst = odb.rootAssembly.instances[inst_name]
                    node = inst.getNodeFromLabel(node_label)
                    coords = node.coordinates
                    # x_mm: arc distance from first sensor (simplified)
                    sensor_positions[sid] = coords
                except Exception:
                    sensor_positions[sid] = (0.0, 0.0, 0.0)

        print("  Sensor %d: %d time steps" % (sid, len(ur)))

    odb.close()

    if not sensor_data:
        print("ERROR: No sensor data extracted")
        return None

    # Compute arc distances from sensor 0 for position row
    pos_values = []
    ref_pos = sensor_positions.get(0, (0.0, 0.0, 0.0))
    for sid in sorted(sensor_data.keys()):
        if sid in sensor_positions:
            p = sensor_positions[sid]
            dist = math.sqrt((p[0]-ref_pos[0])**2 + (p[1]-ref_pos[1])**2 +
                             (p[2]-ref_pos[2])**2)
            pos_values.append(dist)
        else:
            pos_values.append(0.0)

    # Write CSV
    sorted_sids = sorted(sensor_data.keys())
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)

        # Header
        header = ['time_s'] + ['sensor_%d_Ur' % s for s in sorted_sids]
        writer.writerow(header)

        # Position row (comment)
        pos_row = ['# x_mm'] + ['%.1f' % p for p in pos_values]
        writer.writerow(pos_row)

        # Data rows
        n_steps = len(sensor_data[sorted_sids[0]])
        for i in range(n_steps):
            t = sensor_data[sorted_sids[0]][i][0]
            row = ['%.9e' % t]
            for sid in sorted_sids:
                row.append('%.9e' % sensor_data[sid][i][1])
            writer.writerow(row)

    print("Saved: %s (%d steps, %d sensors)" % (output_csv, n_steps, n_sensors))
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Extract GW sensor history from ODB')
    parser.add_argument('--odb', type=str, default=None,
                        help='Single ODB file path')
    parser.add_argument('--odb_dir', type=str, default=None,
                        help='Directory containing ODB files')
    parser.add_argument('--jobs', nargs='+', default=None,
                        help='Job indices to extract (e.g. 0034 0068)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSVs (default: same as ODB)')

    args, _ = parser.parse_known_args()

    if args.odb:
        extract_sensors(args.odb)
    elif args.odb_dir and args.jobs:
        for job_idx in args.jobs:
            odb_name = 'Job-GW-Fair-%s.odb' % job_idx
            odb_path = os.path.join(args.odb_dir, odb_name)
            if os.path.exists(odb_path):
                out_csv = None
                if args.output_dir:
                    out_csv = os.path.join(args.output_dir,
                                           'Job-GW-Fair-%s_sensors.csv' % job_idx)
                extract_sensors(odb_path, out_csv)
            else:
                print("SKIP: %s not found" % odb_path)
    else:
        print("Usage: abaqus python extract_gw_sensors.py --odb <path>")
        print("   or: abaqus python extract_gw_sensors.py --odb_dir <dir> --jobs 0034 0068")


if __name__ == '__main__':
    main()
