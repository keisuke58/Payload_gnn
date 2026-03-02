#!/usr/bin/env python3
"""Compare GT healthy sample with baseline dataset statistics."""
import csv
import os
import glob
import sys
import statistics

def load_nodes_stats(csv_path):
    """Load nodes.csv and compute per-column statistics."""
    cols = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ['ux', 'uy', 'uz', 'u_mag', 'temp', 's11', 's22', 's12',
                       'smises', 'thermal_smises', 'le11', 'le22', 'le12']:
                if k not in cols:
                    cols[k] = []
                try:
                    cols[k].append(float(row[k]))
                except (ValueError, KeyError):
                    pass
    stats = {}
    for k, vals in cols.items():
        if vals:
            stats[k] = {
                'min': min(vals),
                'max': max(vals),
                'mean': statistics.mean(vals),
                'median': statistics.median(vals),
                'stdev': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'n': len(vals),
            }
    return stats


def main():
    base_dir = 'dataset_realistic_25mm_100'
    gt_dir = 'dataset_gt/sample_healthy'

    # --- GT stats ---
    gt_csv = os.path.join(gt_dir, 'nodes.csv')
    if not os.path.exists(gt_csv):
        print("ERROR: GT nodes.csv not found: %s" % gt_csv)
        sys.exit(1)
    gt_stats = load_nodes_stats(gt_csv)
    print("=" * 80)
    print("GT Healthy Sample (N_nodes=%d)" % gt_stats['ux']['n'])
    print("=" * 80)
    for k in ['ux', 'uy', 'uz', 'u_mag', 'temp', 'smises', 'thermal_smises',
              's11', 's22', 's12', 'le11', 'le22', 'le12']:
        s = gt_stats.get(k)
        if s:
            print("  %-18s min=%11.4f  max=%11.4f  mean=%11.4f  stdev=%11.4f" % (
                k, s['min'], s['max'], s['mean'], s['stdev']))

    # --- Baseline stats (aggregate across all healthy + defect samples) ---
    sample_dirs = sorted(glob.glob(os.path.join(base_dir, 'sample_*')))
    if not sample_dirs:
        print("\nNo baseline samples found in %s" % base_dir)
        return

    # Collect per-sample summaries
    baseline_summaries = []  # list of (defect_type, n_nodes, per-col stats)
    healthy_samples = []
    n_loaded = 0
    for sd in sample_dirs:
        nodes_csv = os.path.join(sd, 'nodes.csv')
        meta_csv = os.path.join(sd, 'metadata.csv')
        if not os.path.exists(nodes_csv):
            continue
        # Read defect type
        defect_type = 'unknown'
        if os.path.exists(meta_csv):
            with open(meta_csv, 'r') as f:
                for row in csv.DictReader(f):
                    if row['key'] == 'defect_type':
                        defect_type = row['value']
        stats = load_nodes_stats(nodes_csv)
        baseline_summaries.append((os.path.basename(sd), defect_type, stats))
        if defect_type == 'healthy':
            healthy_samples.append((os.path.basename(sd), stats))
        n_loaded += 1

    print("\n" + "=" * 80)
    print("Baseline Dataset: %d samples loaded (%d healthy)" % (n_loaded, len(healthy_samples)))
    print("=" * 80)

    # Aggregate stats across ALL baseline samples
    agg = {}
    for _, _, stats in baseline_summaries:
        for k in ['ux', 'uy', 'uz', 'u_mag', 'temp', 'smises', 'thermal_smises',
                   's11', 's22', 's12', 'le11', 'le22', 'le12']:
            s = stats.get(k)
            if s:
                if k not in agg:
                    agg[k] = {'mins': [], 'maxs': [], 'means': [], 'nodes': []}
                agg[k]['mins'].append(s['min'])
                agg[k]['maxs'].append(s['max'])
                agg[k]['means'].append(s['mean'])
                agg[k]['nodes'].append(s['n'])

    print("\nPer-sample ranges (min of mins → max of maxs), mean of means:")
    for k in ['ux', 'uy', 'uz', 'u_mag', 'temp', 'smises', 'thermal_smises',
              's11', 's22', 's12', 'le11', 'le22', 'le12']:
        a = agg.get(k)
        if a:
            print("  %-18s min=[%9.3f, %9.3f]  max=[%9.3f, %9.3f]  mean_of_means=%9.3f  n_nodes=[%d, %d]" % (
                k,
                min(a['mins']), max(a['mins']),
                min(a['maxs']), max(a['maxs']),
                statistics.mean(a['means']),
                min(a['nodes']), max(a['nodes']),
            ))

    # --- Comparison table ---
    print("\n" + "=" * 80)
    print("COMPARISON: GT vs Baseline (range)")
    print("=" * 80)
    print("%-18s | %-30s | %-30s | %s" % ("Field", "GT [min, max]", "Baseline [min, max]*", "Status"))
    print("-" * 110)
    for k in ['ux', 'uy', 'uz', 'u_mag', 'temp', 'smises', 'thermal_smises',
              's11', 's22', 's12', 'le11', 'le22', 'le12']:
        gs = gt_stats.get(k)
        a = agg.get(k)
        if gs and a:
            gt_range = "[%9.3f, %9.3f]" % (gs['min'], gs['max'])
            bl_range = "[%9.3f, %9.3f]" % (min(a['mins']), max(a['maxs']))
            # Check if GT range is within reasonable bounds of baseline
            bl_min = min(a['mins'])
            bl_max = max(a['maxs'])
            gt_min = gs['min']
            gt_max = gs['max']
            span = bl_max - bl_min if bl_max != bl_min else 1.0
            if gt_min >= bl_min - 0.2 * abs(span) and gt_max <= bl_max + 0.2 * abs(span):
                status = "OK (within range)"
            elif gt_max > bl_max * 2 or gt_min < bl_min * 2:
                status = "!! LARGE DIFF"
            else:
                status = "~  moderate diff"
            print("%-18s | %-30s | %-30s | %s" % (k, gt_range, bl_range, status))

    # --- Node count comparison ---
    print("\n" + "=" * 80)
    print("NODE COUNTS")
    print("=" * 80)
    print("  GT:       %d nodes (OuterSkin)" % gt_stats['ux']['n'])
    bl_nodes = [s['ux']['n'] for _, _, s in baseline_summaries if 'ux' in s]
    if bl_nodes:
        print("  Baseline: %d - %d nodes (median=%d)" % (
            min(bl_nodes), max(bl_nodes), statistics.median(bl_nodes)))

    print("\n* Baseline ranges are min/max across ALL %d samples (healthy + defect)" % n_loaded)


if __name__ == '__main__':
    main()
