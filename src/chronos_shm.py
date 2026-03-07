#!/usr/bin/env python3
"""Chronos-2 Foundation Model for SHM Anomaly Detection.

Guided-wave sensor signals (9 sensors x 3922 time steps) from CFRP/Al-Honeycomb
fairing simulations are analyzed using Amazon Chronos-2 time-series foundation
model for zero-shot anomaly detection.

Two approaches:
  1. Embedding-based: Chronos-2 encoder embeddings + Isolation Forest
  2. Forecast-based:  Predict future steps, measure residual vs actual

Usage:
    python src/chronos_shm.py                          # default: embedding mode
    python src/chronos_shm.py --mode forecast           # forecast residual mode
    python src/chronos_shm.py --device cpu              # CPU inference
    python src/chronos_shm.py --downsample 4            # faster with downsampling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


N_SENSORS_COMMON = 9  # first 9 sensors are common across all files


def load_sensor_csv(
    path: Path, downsample: int = 1, n_sensors: int = N_SENSORS_COMMON
) -> np.ndarray:
    """Load a sensor CSV and return (n_sensors, T) array of Ur values."""
    df = pd.read_csv(path, comment="#")
    # Use only the first n_sensors sensor columns (common across all files)
    sensor_cols = [f"sensor_{i}_Ur" for i in range(n_sensors)]
    sensor_cols = [c for c in sensor_cols if c in df.columns]
    data = df[sensor_cols].values.T  # (n_sensors, T)
    if downsample > 1:
        data = data[:, ::downsample]
    return data.astype(np.float32)


def load_dataset(
    data_dir: Path, downsample: int = 1
) -> tuple[list[np.ndarray], list[str], list[int]]:
    """Load all sensor CSVs. Returns (signals, filenames, labels).

    Labels: 0 = healthy, 1 = defect.
    """
    signals, names, labels = [], [], []
    csv_files = sorted(data_dir.glob("Job-GW-Fair-*_sensors.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sensor CSVs found in {data_dir}")

    for f in csv_files:
        sig = load_sensor_csv(f, downsample)
        signals.append(sig)
        names.append(f.stem)
        label = 0 if "Healthy" in f.name else 1
        labels.append(label)

    print(f"Loaded {len(signals)} samples "
          f"(healthy={labels.count(0)}, defect={labels.count(1)}), "
          f"shape per sample: {signals[0].shape}")
    return signals, names, labels


# ── Embedding-based anomaly detection ────────────────────────────────


def extract_embeddings(
    signals: list[np.ndarray],
    model_name: str = "amazon/chronos-bolt-base",
    device: str = "cuda",
) -> np.ndarray:
    """Extract Chronos encoder embeddings for each sample.

    For each sample, embed every sensor channel independently,
    then mean-pool over time dimension and concatenate across sensors
    to produce a single fixed-size feature vector per sample.
    """
    from chronos import ChronosBoltPipeline

    print(f"Loading {model_name} on {device}...")
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float32,
    )

    n_samples = len(signals)
    n_sensors = signals[0].shape[0]
    all_embeddings = []

    for i, sig in enumerate(signals):
        # sig shape: (n_sensors, T)
        sensor_tensors = [torch.tensor(sig[s], dtype=torch.float32) for s in range(n_sensors)]
        # Batch all sensors at once for efficiency
        emb, _ = pipeline.embed(sensor_tensors)
        # emb shape: (n_sensors, context_len+1, d_model)
        # Mean-pool over time dimension → (n_sensors, d_model)
        emb_mean = emb.mean(dim=1)
        # Flatten to single vector: (n_sensors * d_model,)
        sample_emb = emb_mean.reshape(-1).detach().cpu().numpy()
        all_embeddings.append(sample_emb)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Embedded {i + 1}/{n_samples} "
                  f"(emb dim per sensor: {emb.shape[-1]})")

    embeddings = np.stack(all_embeddings)  # (n_samples, n_sensors * d_model)
    print(f"Embedding matrix: {embeddings.shape}")
    return embeddings


def run_embedding_mode(
    signals: list[np.ndarray],
    names: list[str],
    labels: list[int],
    device: str,
    out_dir: Path,
) -> None:
    """Embedding + Isolation Forest anomaly detection."""
    embeddings = extract_embeddings(signals, device=device)

    # Standardize
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    # Isolation Forest (unsupervised)
    iso = IsolationForest(
        contamination=0.05,  # expect ~5% anomalies
        random_state=42,
        n_estimators=200,
    )
    scores = iso.fit(emb_scaled).decision_function(emb_scaled)
    preds = iso.predict(emb_scaled)  # 1=normal, -1=anomaly

    # Report
    print("\n── Anomaly Detection Results (Embedding + Isolation Forest) ──")
    print(f"{'Sample':<45} {'Label':>8} {'Score':>8} {'Pred':>6}")
    print("-" * 72)
    for name, label, score, pred in zip(names, labels, scores, preds):
        lbl_str = "DEFECT" if label == 1 else "healthy"
        pred_str = "ANOMALY" if pred == -1 else "normal"
        print(f"{name:<45} {lbl_str:>8} {score:>8.4f} {pred_str:>6}")

    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(signals) - 1))
    coords = tsne.fit_transform(emb_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: by ground truth label
    ax = axes[0]
    colors_gt = ["#2196F3" if l == 0 else "#F44336" for l in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors_gt, s=60, alpha=0.7, edgecolors="k", linewidths=0.5)
    for i, (x, y) in enumerate(coords):
        if labels[i] == 1:
            ax.annotate(names[i].replace("Job-GW-Fair-", "").replace("_sensors", ""),
                       (x, y), fontsize=7, ha="center", va="bottom")
    ax.set_title("t-SNE of Chronos Embeddings (Ground Truth)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    from matplotlib.lines import Line2D
    legend_gt = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="Healthy"),
                 Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336", markersize=8, label="Defect")]
    ax.legend(handles=legend_gt, loc="best")

    # Plot 2: by anomaly score
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="RdYlGn", s=60, edgecolors="k", linewidths=0.5)
    for i, (x, y) in enumerate(coords):
        if preds[i] == -1:
            ax.annotate(names[i].replace("Job-GW-Fair-", "").replace("_sensors", ""),
                       (x, y), fontsize=7, ha="center", va="bottom")
    ax.set_title("t-SNE of Chronos Embeddings (Anomaly Score)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(sc, ax=ax, label="Anomaly Score (lower = more anomalous)")

    plt.tight_layout()
    fig_path = out_dir / "chronos_embedding_tsne.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()


# ── Forecast-based anomaly detection ─────────────────────────────────


def run_forecast_mode(
    signals: list[np.ndarray],
    names: list[str],
    labels: list[int],
    device: str,
    out_dir: Path,
    context_ratio: float = 0.7,
) -> None:
    """Forecast-based anomaly detection.

    Split each signal into context (first 70%) and horizon (last 30%).
    Use Chronos-2 to forecast the horizon from context.
    Anomaly score = normalized RMSE between forecast median and actual.
    """
    from chronos import ChronosBoltPipeline

    model_name = "amazon/chronos-bolt-base"
    print(f"Loading {model_name} on {device}...")
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float32,
    )

    n_sensors = signals[0].shape[0]
    T = signals[0].shape[1]
    ctx_len = int(T * context_ratio)
    horizon = T - ctx_len

    all_rmse = []

    for i, sig in enumerate(signals):
        sensor_rmses = []
        for s in range(n_sensors):
            context = torch.tensor(sig[s, :ctx_len], dtype=torch.float32)
            actual = sig[s, ctx_len:]

            forecast = pipeline.predict(
                inputs=[context],
                prediction_length=horizon,
            )
            # forecast shape: (1, n_samples, horizon) — take median
            median_forecast = forecast[0].median(dim=0).values.cpu().numpy()

            rmse = np.sqrt(np.mean((median_forecast - actual) ** 2))
            # Normalize by signal amplitude
            amp = np.std(sig[s])
            nrmse = rmse / (amp + 1e-12)
            sensor_rmses.append(nrmse)

        mean_nrmse = np.mean(sensor_rmses)
        all_rmse.append(mean_nrmse)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Forecast {i + 1}/{len(signals)}, NRMSE={mean_nrmse:.4f}")

    all_rmse = np.array(all_rmse)

    # Threshold: mean + 2*std of healthy samples
    healthy_rmse = all_rmse[np.array(labels) == 0]
    threshold = healthy_rmse.mean() + 2 * healthy_rmse.std()

    print(f"\n── Anomaly Detection Results (Forecast Residual) ──")
    print(f"Threshold (mean+2std of healthy): {threshold:.4f}")
    print(f"{'Sample':<45} {'Label':>8} {'NRMSE':>8} {'Pred':>6}")
    print("-" * 72)
    for name, label, rmse in zip(names, labels, all_rmse):
        lbl_str = "DEFECT" if label == 1 else "healthy"
        pred_str = "ANOMALY" if rmse > threshold else "normal"
        print(f"{name:<45} {lbl_str:>8} {rmse:>8.4f} {pred_str:>6}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#F44336" if l == 1 else "#2196F3" for l in labels]
    x = np.arange(len(all_rmse))
    ax.bar(x, all_rmse, color=colors, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Chronos-2 Forecast Residual Anomaly Scores")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Healthy"),
        Patch(facecolor="#F44336", label="Defect"),
    ]
    ax.legend(handles=legend_elements + [ax.get_lines()[0]], loc="best")
    plt.tight_layout()
    fig_path = out_dir / "chronos_forecast_anomaly.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    plt.close()

    # Detail plot: overlay forecast vs actual for defect samples
    defect_indices = [i for i, l in enumerate(labels) if l == 1]
    if defect_indices:
        n_defect = len(defect_indices)
        fig, axes = plt.subplots(n_defect, 3, figsize=(15, 4 * n_defect), squeeze=False)
        for row, idx in enumerate(defect_indices):
            sig = signals[idx]
            for col, s_id in enumerate([0, 4, 8]):  # near, mid, far sensors
                ax = axes[row, col]
                context = torch.tensor(sig[s_id, :ctx_len], dtype=torch.float32)
                forecast = pipeline.predict(
                    inputs=[context],
                    prediction_length=horizon,
                )
                median_fc = forecast[0].median(dim=0).values.cpu().numpy()

                t_ctx = np.arange(ctx_len)
                t_hor = np.arange(ctx_len, T)
                ax.plot(t_ctx, sig[s_id, :ctx_len], "b-", alpha=0.6, label="Context")
                ax.plot(t_hor, sig[s_id, ctx_len:], "g-", linewidth=1.5, label="Actual")
                ax.plot(t_hor, median_fc, "r--", linewidth=1.5, label="Forecast")
                ax.set_title(f"{names[idx]} — Sensor {s_id}")
                ax.legend(fontsize=7)
                ax.set_xlabel("Time step")
                ax.set_ylabel("Ur (m)")

        plt.tight_layout()
        fig_path = out_dir / "chronos_forecast_detail.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved: {fig_path}")
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Foundation Model for SHM Anomaly Detection"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("abaqus_work/gw_fairing_dataset"),
        help="Directory containing sensor CSVs",
    )
    parser.add_argument(
        "--mode",
        choices=["embedding", "forecast", "both"],
        default="embedding",
        help="Anomaly detection approach",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for time steps (1=full resolution)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/chronos_shm"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    signals, names, labels = load_dataset(args.data_dir, args.downsample)

    if args.mode in ("embedding", "both"):
        run_embedding_mode(signals, names, labels, args.device, args.out_dir)
    if args.mode in ("forecast", "both"):
        run_forecast_mode(signals, names, labels, args.device, args.out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
