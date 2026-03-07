#!/usr/bin/env python3
"""Extract Chronos-Bolt embeddings from guided-wave sensor CSVs.

Pre-computes and caches time-series embeddings for later fusion with
static FEA graph features (34-dim → 34+proj_dim hybrid).

Output per sample:
    {sample_id}.pt = {
        'embeddings':       (n_sensors, 768)  float32  — raw Chronos-Bolt embeddings
        'sensor_positions': (n_sensors,)      float32  — arc distance (mm) from CSV
        'n_sensors':        int
        'source_csv':       str
    }

Usage:
    python src/extract_chronos_embeddings.py
    python src/extract_chronos_embeddings.py --data-dir abaqus_work/gw_fairing_dataset
    python src/extract_chronos_embeddings.py --device cuda --out-dir data/chronos_emb
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_sensor_positions(csv_path: Path) -> np.ndarray:
    """Read the comment line (line 2) to get sensor arc-distance positions."""
    with open(csv_path) as f:
        _header = f.readline()
        comment = f.readline().strip()

    # Format: "# x_mm,0.0,30.0,90.1,..."
    if comment.startswith("#"):
        parts = comment.lstrip("# ").split(",")
        # Skip the label ("x_mm")
        positions = []
        for p in parts[1:]:
            p = p.strip()
            try:
                positions.append(float(p))
            except ValueError:
                continue
        return np.array(positions, dtype=np.float32)
    return np.array([], dtype=np.float32)


def load_sensor_signals(
    csv_path: Path, max_sensors: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load sensor time-series from CSV.

    Returns:
        signals: (n_sensors, T) float32
        sensor_names: list of column names
    """
    df = pd.read_csv(csv_path, comment="#")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    sensor_cols = sorted(sensor_cols, key=lambda c: int(re.search(r"\d+", c).group()))
    if max_sensors is not None:
        sensor_cols = sensor_cols[:max_sensors]
    return df[sensor_cols].values.T.astype(np.float32), sensor_cols


def extract_sample_id(filename: str) -> str:
    """Extract a normalized sample ID from filename.

    Examples:
        Job-GW-Fair-Healthy-A042_sensors -> healthy_042
        Job-GW-Fair-0034_sensors         -> defect_0034
    """
    name = filename.replace("_sensors", "")
    m_healthy = re.search(r"Healthy-A(\d+)", name)
    if m_healthy:
        return f"healthy_{int(m_healthy.group(1)):04d}"
    m_defect = re.search(r"Fair-(\d+)", name)
    if m_defect:
        return f"defect_{m_defect.group(1)}"
    return name


def extract_embeddings(
    csv_files: list[Path],
    out_dir: Path,
    device: str = "cuda",
    max_sensors: int | None = None,
) -> None:
    """Extract and save Chronos-Bolt embeddings for each CSV."""
    from chronos import ChronosBoltPipeline

    model_name = "amazon/chronos-bolt-base"
    print(f"Loading {model_name} on {device}...")
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name, device_map=device, dtype=torch.float32
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, csv_path in enumerate(csv_files):
        sample_id = extract_sample_id(csv_path.stem)
        signals, sensor_names = load_sensor_signals(csv_path, max_sensors)
        sensor_positions = parse_sensor_positions(csv_path)
        n_sensors = signals.shape[0]

        # Truncate positions to match sensors if needed
        if len(sensor_positions) > n_sensors:
            sensor_positions = sensor_positions[:n_sensors]
        elif len(sensor_positions) < n_sensors:
            # Pad with indices if positions not available
            sensor_positions = np.arange(n_sensors, dtype=np.float32)

        # Embed each sensor channel
        sensor_tensors = [
            torch.tensor(signals[s], dtype=torch.float32) for s in range(n_sensors)
        ]
        emb, _ = pipeline.embed(sensor_tensors)
        # emb: (n_sensors, context_len+1, d_model=768)
        # Mean-pool over time → (n_sensors, 768)
        emb_pooled = emb.mean(dim=1).detach().cpu()

        result = {
            "embeddings": emb_pooled,                         # (n_sensors, 768)
            "sensor_positions": torch.tensor(sensor_positions),  # (n_sensors,)
            "n_sensors": n_sensors,
            "source_csv": csv_path.name,
            "sample_id": sample_id,
            "sensor_names": sensor_names,
        }
        torch.save(result, out_dir / f"{sample_id}.pt")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(csv_files)}] {sample_id}: "
                  f"{n_sensors} sensors, emb {emb_pooled.shape}")

    print(f"\nSaved {len(csv_files)} embeddings to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Chronos-Bolt embeddings from GW sensor CSVs"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("abaqus_work/gw_fairing_dataset"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/chronos_embeddings"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-sensors", type=int, default=None,
        help="Limit to first N sensors (None = use all)",
    )
    args = parser.parse_args()

    csv_files = sorted(args.data_dir.glob("*_sensors.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sensor CSVs in {args.data_dir}")
    print(f"Found {len(csv_files)} sensor CSVs")

    extract_embeddings(csv_files, args.out_dir, args.device, args.max_sensors)


if __name__ == "__main__":
    main()
