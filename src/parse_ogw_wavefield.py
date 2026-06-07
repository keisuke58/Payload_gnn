#!/usr/bin/env python3
"""Parser for the Open Guided Waves (OGW) CFRP omega-stringer wavefield dataset.

Dataset: Zenodo 5105861 (Kudela et al.). Each excitation folder holds full-field
SLDV out-of-plane wavefields as HDF5:
    coordinates.h5  -> (2, N)      scan-point x,y  [m]
    data_z.h5       -> (T, N)      out-of-plane response per timestep
    time.h5         -> (T, 1)      time vector [s]
Three scenarios: Intact / FirstImpact (local debond) / SecondImpact (large debond).

Usage:
    python3 src/parse_ogw_wavefield.py <excitation_folder> [--snapshot out.png]
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import h5py


def _read(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f[list(f.keys())[0]])


def load_ogw(folder: str):
    """Return (coords[N,2], wavefield[T,N], time[T]) for one excitation folder."""
    coords = _read(os.path.join(folder, "coordinates.h5")).T          # (N,2)
    wavefield = _read(os.path.join(folder, "data_z.h5"))              # (T,N)
    time = _read(os.path.join(folder, "time.h5")).ravel()            # (T,)
    return coords, wavefield, time


def summarize(coords, wavefield, time):
    T, N = wavefield.shape
    dt = float(np.median(np.diff(time))) if time.size > 1 else float("nan")
    print(f"points N      : {N}")
    print(f"timesteps T   : {T}")
    print(f"dt            : {dt:.3e} s   (fs ~ {1/dt/1e3:.1f} kHz)")
    print(f"duration      : {time[-1]-time[0]:.3e} s")
    print(f"x range       : [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}] m")
    print(f"y range       : [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}] m")
    print(f"amplitude     : [{wavefield.min():.3e}, {wavefield.max():.3e}]")
    energy = (wavefield ** 2).sum(axis=1)
    t_peak = int(energy.argmax())
    print(f"peak-energy t : index {t_peak}  (t={time[t_peak]:.3e} s)")
    return t_peak


def snapshot(coords, wavefield, time, t_idx, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = wavefield[t_idx]
    vmax = np.percentile(np.abs(z), 99)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=z, s=1.0, cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"OGW wavefield snapshot (t = {time[t_idx]*1e6:.1f} us)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label("out-of-plane response")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"snapshot -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="excitation folder with coordinates/data_z/time .h5")
    ap.add_argument("--snapshot", default=None, help="output PNG for a peak-energy frame")
    a = ap.parse_args()
    coords, wavefield, time = load_ogw(a.folder)
    t_peak = summarize(coords, wavefield, time)
    if a.snapshot:
        snapshot(coords, wavefield, time, t_peak, a.snapshot)
