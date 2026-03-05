#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_gw_sensor_subset.py — 100センサ CSV から 10/20/30/50 のサブセットを抽出

1回の解析（100センサ）で得た CSV から、先頭 k センサを抽出して別 CSV を出力。
再解析不要で 10, 20, 30, 50 の4パターンを生成。

Usage:
  python scripts/extract_gw_sensor_subset.py --input abaqus_work/gw_fairing_dataset_100
  python scripts/extract_gw_sensor_subset.py --input dir --csv Job-GW-Fair-Healthy_sensors.csv --k 20 --output out.csv
"""

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_subset(input_path, output_path, k):
    """Extract first k sensors (by ID) from CSV, write to output_path."""
    with open(input_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        print("ERROR: CSV too short: %s" % input_path)
        return False

    header = rows[0]
    pos_row = rows[1]
    data_rows = rows[2:]

    # Parse header: time_s, sensor_0_Ur, sensor_1_Ur, ...
    sensor_cols = []  # (sid, col_index)
    for i, col in enumerate(header):
        if i == 0:
            continue
        if col.startswith('sensor_'):
            try:
                sid = int(col.replace('sensor_', '').split('_')[0])
                sensor_cols.append((sid, i))
            except ValueError:
                pass

    sensor_cols.sort(key=lambda x: x[0])
    col_indices = [0] + [idx for sid, idx in sensor_cols[:k]]

    if len(col_indices) < 2:
        print("ERROR: No sensor columns for k=%d (found %d)" % (k, len(sensor_cols)))
        return False

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([header[i] for i in col_indices])
        writer.writerow([pos_row[i] if i < len(pos_row) else '' for i in col_indices])
        for row in data_rows:
            if len(row) >= max(col_indices) + 1:
                writer.writerow([row[i] for i in col_indices])

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract sensor subset (10/20/30/50) from 100-sensor CSV')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dir with *_sensors.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output base dir (default: input_dir)')
    parser.add_argument('--k', type=int, nargs='+', default=[10, 20, 30, 50],
                        help='Subset sizes (default: 10 20 30 50)')
    args = parser.parse_args()

    input_path = os.path.join(PROJECT_ROOT, args.input) if not os.path.isabs(args.input) else args.input
    output_base = args.output or input_path
    if args.output and not os.path.isabs(args.output):
        output_base = os.path.join(PROJECT_ROOT, args.output)

    files = [os.path.join(input_path, f) for f in os.listdir(input_path)
             if f.endswith('_sensors.csv')]

    if not files:
        print("ERROR: No *_sensors.csv found in %s" % input_path)
        sys.exit(1)

    for k in args.k:
        out_dir = os.path.join(output_base.rstrip('/'), 's%d' % k)
        os.makedirs(out_dir, exist_ok=True)
        for fp in sorted(files):
            base = os.path.basename(fp)
            out_path = os.path.join(out_dir, base)
            if extract_subset(fp, out_path, k):
                print("  %s -> s%d/%s" % (base, k, base))
        print("  k=%d done -> %s" % (k, out_dir))

    print("Done. Subsets: %s" % args.k)


if __name__ == '__main__':
    main()
