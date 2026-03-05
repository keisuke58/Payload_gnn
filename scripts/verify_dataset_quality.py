#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静解析データセット品質検証: 変位・温度の有無をカウント。

動解析 (GW) は scripts/verify_gw_dataset_quality.py を使用。

Usage:
  python scripts/verify_dataset_quality.py [--data_dir dataset_output]
"""
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_output')
    args = parser.parse_args()
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)

    n_total = 0
    n_disp = 0
    n_temp = 0
    n_good = 0

    for name in sorted(os.listdir(data_dir)):
        if not name.startswith('sample_'):
            continue
        nodes_csv = os.path.join(data_dir, name, 'nodes.csv')
        if not os.path.exists(nodes_csv):
            continue
        n_total += 1
        with open(nodes_csv) as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue
        ux_sum = temp_sum = 0
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    ux_sum += abs(float(parts[4]))
                    temp_sum += float(parts[7])
                except (ValueError, IndexError):
                    pass
        n_nodes = len(lines) - 1
        if n_nodes > 0:
            ux_mean = ux_sum / n_nodes
            temp_mean = temp_sum / n_nodes
            if ux_mean > 0.001:
                n_disp += 1
            if temp_mean > 10:
                n_temp += 1
            if ux_mean > 0.001 and temp_mean > 10:
                n_good += 1

    print("Dataset quality: %s" % data_dir)
    print("  Total samples: %d" % n_total)
    print("  With displacement (ux>0): %d" % n_disp)
    print("  With temperature (>10C): %d" % n_temp)
    print("  Good (both): %d" % n_good)
    if n_total > 0:
        pct = 100 * n_good / n_total
        score = int(20 + 80 * (n_good / n_total))  # 20 base + 80 for quality
        score = min(100, score)
        print("  Good ratio: %.1f%%" % pct)
        print("  Estimated score: %d/100" % score)


if __name__ == '__main__':
    main()
