#!/usr/bin/env python3
"""Convert an OGW full-field wavefield into a FEM-compatible sensor CSV.

Samples N sensors on a regular grid over the plate, takes the nearest OGW scan
point for each, and writes the `*_sensors.csv` schema the GW pipeline expects:

    row0 (header):  time_s, sensor_0_U3, sensor_1_U3, ...
    row1 (pos):     # x_mm, <x0_mm>, <x1_mm>, ...
    row2+:          <t_s>, <U3_0>, <U3_1>, ...

This lets the trained GW GNN ingest OGW real data for sim-to-real validation.

Usage:
    python3 scripts/ogw_to_sensor_csv.py <excitation_folder> --out out_sensors.csv [--n 99]
"""
from __future__ import annotations
import argparse, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from parse_ogw_wavefield import load_ogw  # noqa: E402


def pick_sensor_grid(coords, n=99):
    """Pick ~n sensors on a regular grid in the plate interior; return point indices."""
    x, y = coords[:, 0], coords[:, 1]
    x0, x1 = np.percentile(x, [5, 95]); y0, y1 = np.percentile(y, [5, 95])
    ncol = int(round(np.sqrt(n)));  nrow = int(np.ceil(n / ncol))
    gx = np.linspace(x0, x1, ncol); gy = np.linspace(y0, y1, nrow)
    targets = np.array([(px, py) for py in gy for px in gx])[:n]
    # nearest OGW point per target (vectorized argmin)
    idx = []
    for tx, ty in targets:
        idx.append(int(((x - tx) ** 2 + (y - ty) ** 2).argmin()))
    return np.array(idx)


def write_sensor_csv(folder, out_csv, n=99, tdecim=1):
    coords, wavefield, time = load_ogw(folder)          # coords(N,2), wf(T,N), time(T,)
    sidx = pick_sensor_grid(coords, n)
    xs_mm = coords[sidx, 0] * 1000.0                      # x position in mm
    wf = wavefield[::tdecim, sidx]                        # (T', n)
    t = time[::tdecim]
    header = ["time_s"] + [f"sensor_{i}_U3" for i in range(len(sidx))]
    pos_row = ["# x_mm"] + [f"{v:.1f}" for v in xs_mm]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header); w.writerow(pos_row)
        for k in range(wf.shape[0]):
            w.writerow([f"{t[k]:.9e}"] + [f"{v:.6e}" for v in wf[k]])
    print(f"sensors={len(sidx)} steps={wf.shape[0]} -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=99)
    ap.add_argument("--tdecim", type=int, default=1, help="time decimation factor")
    a = ap.parse_args()
    write_sensor_csv(a.folder, a.out, a.n, a.tdecim)
