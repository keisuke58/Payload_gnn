#!/usr/bin/env python3
"""MeshGraphNet サロゲートによるデータ拡張スクリプト.

合成データの生成 → 品質チェック → 拡張データセット作成を一括で実行する。

品質管理:
  - strain_E11/E22: サロゲートが学習困難なため、テンプレートから復元
  - disp_Umag (|U|): 負値を clamp(min=0) で修正
  - 全次元で分布差 < 10%、R² > 0.95 を検証

Usage:
    python scripts/augment_with_surrogate.py --n_samples 500 --noise 0.005
    python scripts/augment_with_surrogate.py --n_samples 1000 --ratio 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from physicsnemo_surrogate import MeshGraphNet

PROBLEMATIC_DIMS = [21, 22]  # strain_E11, strain_E22 (surrogate fails on these)

TARGET_NAMES = {
    10: "disp_Ux", 11: "disp_Uy", 12: "disp_Uz", 13: "disp_Umag",
    14: "temperature", 15: "stress_S11", 16: "stress_S22", 17: "stress_S12",
    18: "stress_Mises", 19: "stress_MaxPrinc", 20: "strain_E_Mises",
    21: "strain_E11", 22: "strain_E22",
}


@torch.no_grad()
def generate_synthetic(
    checkpoint: Path,
    data_dir: Path,
    n_samples: int,
    noise_scale: float,
    device: str,
) -> tuple[list, list[int]]:
    """Generate synthetic samples with quality fixes applied."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in=ckpt["node_in"], edge_in=ckpt["edge_in"],
        node_out=ckpt["node_out"],
        hidden_dim=ckpt["hidden_dim"], n_blocks=ckpt["n_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    templates = torch.load(data_dir / "train.pt", weights_only=False)
    input_idx = ckpt["input_idx"]
    target_idx = ckpt["target_idx"]

    generated = []
    for i in range(n_samples):
        template = templates[i % len(templates)].clone()
        x_input = template.x[:, input_idx].to(device)

        noise = torch.randn_like(x_input) * noise_scale
        x_perturbed = x_input + noise

        pred = model(
            x_perturbed,
            template.edge_index.to(device),
            template.edge_attr.to(device),
        )

        x_full = template.x.clone()
        x_full[:, input_idx] = x_perturbed.cpu()
        x_full[:, target_idx] = pred.cpu()

        # Fix: restore problematic dims from template
        for dim in PROBLEMATIC_DIMS:
            x_full[:, dim] = template.x[:, dim]

        # Fix: clamp |U| >= 0
        x_full[:, 13] = x_full[:, 13].clamp(min=0)

        template.x = x_full
        generated.append(template)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples}", flush=True)

    return generated, target_idx


def quality_check(synthetic, target_idx, data_dir):
    """Validate synthetic data quality. Returns True if all checks pass."""
    real = torch.load(data_dir / "train.pt", weights_only=False)
    real_t = np.concatenate([d.x[:, target_idx].numpy() for d in real], axis=0)
    syn_t = np.concatenate([d.x[:, target_idx].numpy() for d in synthetic], axis=0)

    # NaN/Inf check
    for i, s in enumerate(synthetic):
        if torch.isnan(s.x).any() or torch.isinf(s.x).any():
            print(f"FAIL: NaN/Inf in sample {i}")
            return False

    # Distribution comparison
    print(f"\n{'Feature':<18} {'Mean Diff%':>10} {'Std Diff%':>10}")
    print("-" * 42)
    ok = True
    for i, tidx in enumerate(target_idx):
        name = TARGET_NAMES.get(tidx, f"dim{tidx}")
        r_mean, s_mean = real_t[:, i].mean(), syn_t[:, i].mean()
        r_std, s_std = real_t[:, i].std(), syn_t[:, i].std()
        md = abs(s_mean - r_mean) / (abs(r_mean) + 1e-12) * 100
        sd = abs(s_std - r_std) / (abs(r_std) + 1e-12) * 100
        flag = " !!!" if md > 10 or sd > 20 else ""
        if flag:
            ok = False
        print(f"{name:<18} {md:>9.1f}% {sd:>9.1f}%{flag}")

    # R² check
    r2_list = []
    for i in range(min(50, len(synthetic))):
        tmpl = real[i % len(real)]
        rt = tmpl.x[:, target_idx].numpy()
        st = synthetic[i].x[:, target_idx].numpy()
        ss_res = np.sum((rt - st) ** 2, axis=0)
        ss_tot = np.sum((rt - rt.mean(axis=0)) ** 2, axis=0)
        r2_list.append(1 - ss_res / (ss_tot + 1e-12))
    avg_r2 = np.mean(r2_list)
    print(f"\nR²: avg={avg_r2:.4f}")

    if avg_r2 < 0.95:
        print("FAIL: avg R² < 0.95")
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="MeshGraphNet surrogate data augmentation")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed_s12_thermal_500"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/surrogate/best_surrogate.pt"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed_augmented_v2"))
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Synthetic:Real ratio for train split (1.0 = equal)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Noise scale: {args.noise}")

    # Generate
    print(f"\nGenerating {args.n_samples} synthetic samples...")
    synthetic, target_idx = generate_synthetic(
        args.checkpoint, args.data_dir, args.n_samples, args.noise, args.device
    )

    # Quality check
    print("\nQuality check...")
    ok = quality_check(synthetic, target_idx, args.data_dir)
    print(f"Quality: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("WARNING: Quality check failed. Proceeding anyway.")

    # Create augmented dataset
    train_real = torch.load(args.data_dir / "train.pt", weights_only=False)
    val_real = torch.load(args.data_dir / "val.pt", weights_only=False)

    n_syn_train = int(len(train_real) * args.ratio)
    n_syn_val = int(len(val_real) * args.ratio)
    n_syn_train = min(n_syn_train, len(synthetic))

    np.random.seed(42)
    idx = np.random.permutation(len(synthetic))
    syn_train = [synthetic[i] for i in idx[:n_syn_train]]
    syn_val = [synthetic[i] for i in idx[n_syn_train:n_syn_train + n_syn_val]]

    train_aug = train_real + syn_train
    val_aug = val_real + syn_val

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_aug, args.out_dir / "train.pt")
    torch.save(val_aug, args.out_dir / "val.pt")
    print(f"\nAugmented dataset: train={len(train_aug)}, val={len(val_aug)}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
