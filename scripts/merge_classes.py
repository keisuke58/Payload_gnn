# -*- coding: utf-8 -*-
"""merge_classes.py — Merge 8-class dataset into 5-class dataset.

Mapping:
  0 healthy              → 0 healthy
  1 debonding            → 1 debonding
  2 fod                  → 2 fod
  3 impact               → 3 impact
  4 delamination         → 4 delamination
  5 inner_debond         → 1 debonding      (same physical mechanism)
  6 thermal_progression  → 4 delamination   (thermal-induced layer separation)
  7 acoustic_fatigue     → 3 impact         (mechanical loading damage)

Usage:
  python scripts/merge_classes.py \
    --input data/processed_s12_thermal_600 \
    --output data/processed_s12_thermal_600_5class
"""

import argparse
import os
import torch

LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 4, 7: 3}
CLASS_NAMES = ['healthy', 'debonding', 'fod', 'impact', 'delamination']


def remap_labels(graphs):
    for g in graphs:
        new_y = g.y.clone()
        for old, new in LABEL_MAP.items():
            new_y[g.y == old] = new
        g.y = new_y
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    print("Loading from %s..." % args.input)
    train = torch.load(os.path.join(args.input, 'train.pt'), weights_only=False)
    val = torch.load(os.path.join(args.input, 'val.pt'), weights_only=False)
    norm_stats = torch.load(os.path.join(args.input, 'norm_stats.pt'), weights_only=False)

    print("Remapping labels: 8 → 5 classes")
    train = remap_labels(train)
    val = remap_labels(val)

    os.makedirs(args.output, exist_ok=True)
    torch.save(train, os.path.join(args.output, 'train.pt'))
    torch.save(val, os.path.join(args.output, 'val.pt'))
    torch.save(norm_stats, os.path.join(args.output, 'norm_stats.pt'))

    # Summary
    train_labels = torch.cat([g.y for g in train])
    val_labels = torch.cat([g.y for g in val])
    print("\nTrain: %d graphs, Val: %d graphs" % (len(train), len(val)))
    for c in range(5):
        nt = (train_labels == c).sum().item()
        nv = (val_labels == c).sum().item()
        print("  %d %-15s: train=%6d  val=%6d" % (c, CLASS_NAMES[c], nt, nv))

    n_healthy_graphs_train = sum(1 for g in train if (g.y > 0).any().item() == False)
    n_healthy_graphs_val = sum(1 for g in val if (g.y > 0).any().item() == False)
    print("\nHealthy-only graphs: train=%d, val=%d" % (n_healthy_graphs_train, n_healthy_graphs_val))
    print("Saved to %s" % args.output)


if __name__ == '__main__':
    main()
