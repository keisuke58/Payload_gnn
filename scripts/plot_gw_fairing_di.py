# -*- coding: utf-8 -*-
# plot_gw_fairing_di.py
# Damage Index bar chart: Fairing Healthy (H3) vs Defect (D3).
#
# Usage: python scripts/plot_gw_fairing_di.py

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    t, s, x = [], {}, []
    with open(path) as f:
        r = csv.reader(f)
        h = next(r)
        n = len(h) - 1
        for i in range(n):
            s[i] = []
        for row in r:
            if row[0].startswith('#'):
                x = [float(row[j]) if j + 1 < len(row) else j * 30
                     for j in range(1, n + 1)]
                continue
            try:
                t.append(float(row[0]))
                for i in range(n):
                    s[i].append(float(row[i + 1]) if row[i + 1] else 0)
            except (ValueError, IndexError):
                pass
    return np.array(t), {k: np.array(v) for k, v in s.items()}, x


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--healthy', default='abaqus_work/Job-GW-Fair-Test-H3_sensors.csv')
    p.add_argument('--defect', default='abaqus_work/Job-GW-Fair-Test-D3_sensors.csv')
    p.add_argument('--out', default='abaqus_work/gw_fairing_di.png')
    args = p.parse_args()

    t_h, s_h, x_h = load_csv(args.healthy)
    t_d, s_d, x_d = load_csv(args.defect)
    n = min(len(s_h), len(s_d))
    npts = min(len(t_h), len(t_d))
    x_pos = x_h if x_h else [0, 30, 90, 120, 60, 60, 60, 60, 60]

    di_list = []
    for i in range(n):
        inc = np.sqrt(np.mean(s_h[i][:npts] ** 2))
        scat = np.sqrt(np.mean((s_d[i][:npts] - s_h[i][:npts]) ** 2))
        di = scat / inc if inc > 1e-20 else 0
        di_list.append(di)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['S%d (%.0f)' % (i, x_pos[i]) if i < len(x_pos) else 'S%d' % i
              for i in range(n)]
    x = np.arange(n)
    bars = ax.bar(x, di_list, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axhline(np.mean(di_list), color='red', linestyle='--', linewidth=1.5,
               label='Mean DI = %.3f' % np.mean(di_list))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Damage Index', fontsize=11)
    ax.set_title('Fairing 30° Sector: Damage Index (H3 Healthy vs D3 Defect)\n'
                 'DI = RMS(scattered) / RMS(incident)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print("Saved: %s" % args.out)
    plt.close()


if __name__ == '__main__':
    main()
