# -*- coding: utf-8 -*-
# extract_gw_field.py
# Abaqus Python script to extract U3 field output from guided wave ODB.
#
# Extracts outer skin node coordinates and U3 at selected frames
# for wave propagation animation.
#
# Usage: abaqus python extract_gw_field.py Job-GW-Healthy.odb [Job-GW-Debond.odb]
#
# Output: <odb_name>_field.npz (numpy archive with coords + U3 per frame)

import sys
import os
import csv
from odbAccess import openOdb


def extract_field(odb_path, output_dir=None, max_frames=100, instance_key='OUTERSKIN'):
    """Extract U3 field from outer skin at selected frames.

    Writes CSV files: <odb_name>_coords.csv and <odb_name>_frames.csv
    """
    if not os.path.exists(odb_path):
        print("ERROR: ODB not found: %s" % odb_path)
        return

    odb_name = os.path.splitext(os.path.basename(odb_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(odb_path) or '.'

    print("Opening ODB: %s" % odb_path)
    odb = openOdb(odb_path, readOnly=True)

    # Find outer skin instance
    target_inst = None
    for inst_name in odb.rootAssembly.instances.keys():
        if instance_key.upper() in inst_name.upper():
            target_inst = odb.rootAssembly.instances[inst_name]
            print("  Using instance: %s" % inst_name)
            break

    if target_inst is None:
        print("ERROR: No instance matching '%s' found" % instance_key)
        odb.close()
        return

    # Extract node coordinates (X, Y)
    node_labels = []
    node_x = []
    node_y = []
    for node in target_inst.nodes:
        node_labels.append(node.label)
        node_x.append(node.coordinates[0])
        node_y.append(node.coordinates[1])

    n_nodes = len(node_labels)
    print("  Outer skin nodes: %d" % n_nodes)

    # Build label -> index map
    label_to_idx = {}
    for idx, label in enumerate(node_labels):
        label_to_idx[label] = idx

    # Get step and frames
    step = odb.steps['Step-Wave']
    total_frames = len(step.frames)
    print("  Total frames: %d" % total_frames)

    # Select frames to extract (evenly spaced)
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        step_size = float(total_frames - 1) / (max_frames - 1)
        frame_indices = [int(round(i * step_size)) for i in range(max_frames)]
        # Deduplicate
        frame_indices = sorted(set(frame_indices))

    print("  Extracting %d frames..." % len(frame_indices))

    # Write coordinates CSV
    coords_path = os.path.join(output_dir, odb_name + '_coords.csv')
    with open(coords_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['node_label', 'x_mm', 'y_mm'])
        for i in range(n_nodes):
            writer.writerow([node_labels[i], '%.4f' % node_x[i], '%.4f' % node_y[i]])
    print("  Coords written: %s" % coords_path)

    # Write frames CSV: each row = one frame, columns = node U3 values
    frames_path = os.path.join(output_dir, odb_name + '_frames.csv')
    with open(frames_path, 'w') as f:
        writer = csv.writer(f)
        # Header: time, node_0, node_1, ...
        header = ['time_s'] + ['n%d' % label for label in node_labels]
        writer.writerow(header)

        for fi, frame_idx in enumerate(frame_indices):
            frame = step.frames[frame_idx]
            t = frame.frameValue

            # Get U3 field
            u3_field = frame.fieldOutputs['U']
            # Get subset for our instance
            u3_sub = u3_field.getSubset(region=target_inst)

            # Initialize all to 0
            u3_vals = [0.0] * n_nodes

            for val in u3_sub.values:
                label = val.nodeLabel
                if label in label_to_idx:
                    u3_vals[label_to_idx[label]] = val.data[2]  # U3 component

            row = ['%.9e' % t] + ['%.6e' % v for v in u3_vals]
            writer.writerow(row)

            if (fi + 1) % 20 == 0 or fi == 0:
                print("    Frame %d/%d: t=%.3e s" % (
                    fi + 1, len(frame_indices), t))
                sys.stdout.flush()

    print("  Frames written: %s (%d frames x %d nodes)" % (
        frames_path, len(frame_indices), n_nodes))
    odb.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: abaqus python extract_gw_field.py <odb1> [odb2] ...")
        sys.exit(1)

    for odb_path in sys.argv[1:]:
        print("\n" + "=" * 60)
        extract_field(odb_path, max_frames=80)

    print("\n" + "=" * 60)
    print("Field extraction complete.")
