#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_binary.py — 8クラスラベルを2クラス (healthy=0, defect=1) に変換

既存の processed データ (train.pt, val.pt) を読み込み、
y > 0 のラベルを全て 1 に置換して新ディレクトリに保存する。

Usage:
    python scripts/convert_to_binary.py \
        --input data/processed_s12_100 \
        --output data/processed_s12_100_binary
"""

import argparse
import os
import shutil
import sys

import torch


def convert(input_dir, output_dir):
    """Convert multi-class labels to binary (healthy=0, defect=1)."""
    os.makedirs(output_dir, exist_ok=True)

    for split in ('train', 'val'):
        src = os.path.join(input_dir, '%s.pt' % split)
        if not os.path.exists(src):
            print("WARNING: %s not found, skipping" % src)
            continue

        data_list = torch.load(src, weights_only=False)
        n_graphs = len(data_list)
        total_nodes = 0
        total_defect_before = 0
        total_defect_after = 0

        for d in data_list:
            total_nodes += d.y.size(0)
            total_defect_before += (d.y > 0).sum().item()
            # Remap: anything > 0 → 1
            d.y = (d.y > 0).long()
            total_defect_after += (d.y == 1).sum().item()

        dst = os.path.join(output_dir, '%s.pt' % split)
        torch.save(data_list, dst)

        pct = 100.0 * total_defect_after / max(total_nodes, 1)
        print("  %s: %d graphs, %d nodes, %d defect (%.2f%%)" % (
            split, n_graphs, total_nodes, total_defect_after, pct))

    # Copy norm_stats.pt (feature normalization is class-independent)
    stats_src = os.path.join(input_dir, 'norm_stats.pt')
    if os.path.exists(stats_src):
        shutil.copy2(stats_src, os.path.join(output_dir, 'norm_stats.pt'))
        print("  norm_stats.pt copied")

    print("Done: %s" % output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Convert 8-class labels to binary (healthy vs defect)')
    parser.add_argument('--input', required=True,
                        help='Input processed data dir (with train.pt, val.pt)')
    parser.add_argument('--output', required=True,
                        help='Output dir for binary-label data')
    args = parser.parse_args()

    print("Converting: %s -> %s" % (args.input, args.output))
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
