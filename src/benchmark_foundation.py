# -*- coding: utf-8 -*-
"""
Foundation Model Benchmark for GW Surrogate

Compares our FNO against pretrained physics foundation models:
  - Poseidon (ETH, SwinV2-based PDE solver)
  - DPOT (Tsinghua/MS, AFNO-based operator transformer)

Both models expect 2D spatial input, so we reshape our 1D GW data
(n_sensors × T) into pseudo-2D format via adapter layers.

Strategy:
  1. Freeze pretrained backbone (spectral/attention layers)
  2. Train lightweight adapter: 1D→2D input + 2D→1D output
  3. Compare with our from-scratch FNO on same data

Usage:
  python src/benchmark_foundation.py --data_dir abaqus_work/gw_fairing_dataset \
    --model poseidon --epochs 300 --augment 10

  python src/benchmark_foundation.py --data_dir abaqus_work/gw_fairing_dataset \
    --model dpot --epochs 300 --augment 10

  # Run all benchmarks
  python src/benchmark_foundation.py --data_dir abaqus_work/gw_fairing_dataset \
    --model all --epochs 300 --augment 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'DPOT'))

from dataset_fno_gw import GWOperatorDataset
from dataset_fno_gw_augmented import AugmentedGWDataset


# =========================================================================
# 1D→2D Adapter for foundation models
# =========================================================================
class GW1Dto2DAdapter(nn.Module):
    """Reshape 1D GW waveforms to pseudo-2D for 2D foundation models.

    Input:  params (B, n_params), healthy (B, n_sensors, T)
    Output: pseudo-2D image (B, C, H, W) suitable for 2D models

    Strategy: 1D conv reduces temporal dim, then reshape to 2D grid.
    Much lighter than a full linear projection.
    """
    def __init__(self, n_sensors, T, n_params, target_size=128, n_out_channels=4):
        super().__init__()
        self.n_sensors = n_sensors
        self.T = T
        self.n_params = n_params
        self.target_size = target_size
        self.n_out_channels = n_out_channels

        # Param encoder → channel feature
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 64),
            nn.GELU(),
            nn.Linear(64, target_size),  # (B, target_size) — much smaller
        )

        # 1D conv to reduce (n_sensors, T) → (hidden, target_size)
        hidden = 16
        self.waveform_conv = nn.Sequential(
            nn.Conv1d(n_sensors, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(target_size),  # T → target_size
            nn.Conv1d(hidden, target_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Now we have (B, target_size, target_size) → reshape to (B, 1, S, S)

        # Combine: 2 channels (waveform_2d, param_map) → n_out_channels
        self.channel_expand = nn.Conv2d(2, n_out_channels, kernel_size=3, padding=1)

    def forward(self, params, healthy):
        B = params.shape[0]
        S = self.target_size

        # 1D conv pathway: (B, n_sensors, T) → (B, S, S)
        wf_2d = self.waveform_conv(healthy)  # (B, S, S)
        wf_2d = wf_2d.unsqueeze(1)  # (B, 1, S, S)

        # Param → spatial map: (B, S) → (B, 1, S, S) via broadcast
        pe = self.param_encoder(params)  # (B, S)
        param_map = pe.unsqueeze(-1).expand(-1, -1, S).unsqueeze(1)  # (B, 1, S, S)

        # Concat and expand channels
        x = torch.cat([wf_2d, param_map], dim=1)  # (B, 2, S, S)
        x = self.channel_expand(x)  # (B, n_out_channels, S, S)

        return x


class Output2Dto1DAdapter(nn.Module):
    """Convert 2D foundation model output back to 1D waveforms.

    Input:  (B, C, H, W) from foundation model
    Output: (B, n_sensors, T) waveform prediction

    Uses conv to reduce 2D → 1D, then upsample to match T.
    """
    def __init__(self, n_channels, target_size, n_sensors, T):
        super().__init__()
        self.n_sensors = n_sensors
        self.T = T

        # 2D → 1D: reduce spatial dims with conv, then reshape
        self.channel_reduce = nn.Conv2d(n_channels, n_sensors, kernel_size=1)
        # (B, n_sensors, H, W) → pool W → (B, n_sensors, H) → upsample to T
        self.spatial_pool = nn.AdaptiveAvgPool2d((target_size, 1))  # (B, n_s, S, 1)
        self.temporal_upsample = nn.Sequential(
            nn.Conv1d(n_sensors, n_sensors, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.channel_reduce(x)  # (B, n_sensors, H, W)
        x = self.spatial_pool(x).squeeze(-1)  # (B, n_sensors, H)
        # Upsample H → T
        x = F.interpolate(x, size=self.T, mode='linear', align_corners=False)
        x = self.temporal_upsample(x)  # (B, n_sensors, T)
        return x


# =========================================================================
# DPOT Wrapper
# =========================================================================
class DPOTGWWrapper(nn.Module):
    """DPOT pretrained backbone + 1D GW adapters."""

    def __init__(self, n_sensors, T, n_params, pretrained_path=None,
                 img_size=128, freeze_backbone=True):
        super().__init__()
        self.img_size = img_size

        # Input adapter: 1D → 2D
        self.input_adapter = GW1Dto2DAdapter(
            n_sensors, T, n_params,
            target_size=img_size, n_out_channels=4)

        # DPOT backbone
        from models.dpot import DPOTNet
        self.backbone = DPOTNet(
            img_size=img_size, patch_size=8, mixing_type='afno',
            in_channels=4, in_timesteps=1, out_timesteps=1,
            out_channels=4, normalize=False, embed_dim=512,
            modes=32, depth=4, n_blocks=4, mlp_ratio=1,
            out_layer_dim=32, n_cls=12)

        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location='cpu',
                              weights_only=False)
            state = ckpt['model'] if 'model' in ckpt else ckpt
            # Partial load (skip mismatched layers)
            model_state = self.backbone.state_dict()
            loaded = 0
            for k, v in state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
                    loaded += 1
            self.backbone.load_state_dict(model_state)
            print(f"  DPOT: loaded {loaded}/{len(state)} pretrained layers")

        # Freeze backbone
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'blocks' in name or 'pos_embed' in name:
                    param.requires_grad = False
            n_frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
            n_total = sum(1 for p in self.backbone.parameters())
            print(f"  DPOT: frozen {n_frozen}/{n_total} backbone params")

        # Output adapter: 2D → 1D
        self.output_adapter = Output2Dto1DAdapter(
            n_channels=4, target_size=img_size,
            n_sensors=n_sensors, T=T)

    def forward(self, params, healthy):
        x = self.input_adapter(params, healthy)  # (B, C, H, W)
        # DPOT expects (B, X, Y, T, C) — permute and add T=1 dim
        x = x.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H, W, 1, C)
        x, _ = self.backbone(x)  # (B, H, W, out_T, out_C)
        # Reshape back to (B, C, H, W) for output adapter
        x = x.squeeze(3).permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.output_adapter(x)


# =========================================================================
# Poseidon Wrapper
# =========================================================================
class PoseidonGWWrapper(nn.Module):
    """Poseidon (ScOT) pretrained backbone + 1D GW adapters."""

    def __init__(self, n_sensors, T, n_params, pretrained_path=None,
                 img_size=128, freeze_backbone=True):
        super().__init__()
        self.img_size = img_size

        # Input adapter: 1D → 2D
        self.input_adapter = GW1Dto2DAdapter(
            n_sensors, T, n_params,
            target_size=img_size, n_out_channels=4)

        # Poseidon uses SwinV2 internally — heavy dependency on transformers
        # Use a simplified approach: extract the core SwinV2 encoder
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'poseidon'))
            from scOT.model import ScOT, ScOTConfig

            config = ScOTConfig(
                image_size=img_size,
                patch_size=4,
                num_channels=4,
                num_out_channels=4,
                embed_dim=48,
                depths=[4, 4, 4, 4],
                num_heads=[3, 6, 12, 24],
                window_size=16,
                mlp_ratio=4.0,
                use_conditioning=True,
                residual_model='convnext',
                skip_connections=[2, 2, 2, 0],
            )

            if pretrained_path and os.path.exists(pretrained_path):
                # Load from local pretrained dir
                pretrained_dir = os.path.dirname(pretrained_path)
                self.backbone = ScOT.from_pretrained(
                    pretrained_dir, config=config,
                    ignore_mismatched_sizes=True)
                print(f"  Poseidon: loaded from {pretrained_dir}")
            else:
                self.backbone = ScOT(config)
                print("  Poseidon: initialized from scratch")

            self.use_scot = True
        except ImportError as e:
            print(f"  Poseidon: ScOT import failed ({e}), using fallback SwinV2")
            self.use_scot = False
            # Fallback: lightweight SwinV2-like encoder
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 48, kernel_size=4, stride=4),  # patch embed
                nn.GELU(),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(96, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(48, 4, kernel_size=4, stride=4),  # upsample
            )

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  Poseidon: all backbone params frozen")

        # Output adapter: 2D → 1D
        self.output_adapter = Output2Dto1DAdapter(
            n_channels=4, target_size=img_size,
            n_sensors=n_sensors, T=T)

    def forward(self, params, healthy):
        x = self.input_adapter(params, healthy)  # (B, 4, H, W)

        if self.use_scot:
            # ScOT requires time conditioning — pass zeros (no PDE time)
            dummy_time = torch.zeros(x.shape[0], device=x.device)
            out = self.backbone(x, time=dummy_time)
            if hasattr(out, 'output'):
                x = out.output  # ScOT output format
            else:
                x = out
        else:
            x = self.backbone(x)

        return self.output_adapter(x)


# =========================================================================
# Simple from-scratch baselines for comparison
# =========================================================================
class SimpleConvSurrogate(nn.Module):
    """Simple 1D CNN baseline (no Fourier, no pretrain)."""
    def __init__(self, n_sensors, T, n_params, hidden=64):
        super().__init__()
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.GELU(),
            nn.Linear(hidden, 16),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_sensors + 16, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, n_sensors, 1),
        )
        self.n_sensors = n_sensors
        self.T = T

    def forward(self, params, healthy):
        B = params.shape[0]
        pe = self.param_encoder(params)  # (B, 16)
        pe = pe.unsqueeze(-1).expand(-1, -1, self.T)  # (B, 16, T)
        x = torch.cat([healthy, pe], dim=1)  # (B, n_sensors+16, T)
        return healthy + self.conv(x)


# =========================================================================
# Training loop
# =========================================================================
def train_model(model, train_loader, val_loader, args, device, model_name):
    """Generic training loop."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_name}: {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)

    warmup_ep = max(1, args.epochs // 20)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_ep)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_ep, eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_ep])

    def relative_l2(pred, target):
        diff = (pred - target) ** 2
        norm = target ** 2 + 1e-8
        return torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))

    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train_log.jsonl')

    best_val = float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_b = 0, 0
        for batch in train_loader:
            p = batch['params'].to(device)
            h = batch['healthy'].to(device)
            tgt = batch['target'].to(device)

            pred = model(p, h)
            loss = relative_l2(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        scheduler.step()

        model.eval()
        val_loss, val_b = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                p = batch['params'].to(device)
                h = batch['healthy'].to(device)
                tgt = batch['target'].to(device)
                pred = model(p, h)
                val_loss += relative_l2(pred, tgt).item()
                val_b += 1

        avg_train = train_loss / max(n_b, 1)
        avg_val = val_loss / max(val_b, 1)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val,
                'model_name': model_name,
            }, os.path.join(output_dir, 'best_model.pt'))

        if epoch % args.log_interval == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  [{model_name}] Ep {epoch:3d}/{args.epochs} | "
                  f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"Best: {best_val:.4f} | LR: {lr:.2e} | {elapsed:.0f}s")

        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'epoch': epoch, 'train_loss': avg_train,
                'val_loss': avg_val, 'lr': scheduler.get_last_lr()[0],
            }) + '\n')

    elapsed = time.time() - t0
    print(f"  [{model_name}] Done. Best val: {best_val:.4f} ({elapsed:.0f}s)")
    return best_val


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Foundation Model Benchmark for GW Surrogate')
    parser.add_argument('--model', default='all',
                        choices=['poseidon', 'dpot', 'fno', 'cnn', 'all'])
    parser.add_argument('--data_dir', default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--doe', default='doe_gw_fairing.json')
    parser.add_argument('--output', default='runs/benchmark_foundation')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--augment', type=int, default=10)
    parser.add_argument('--downsample', type=int, default=4)
    parser.add_argument('--max_timesteps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=64,
                        help='Pseudo-2D image size for foundation models')

    # Foundation model paths
    parser.add_argument('--poseidon_path',
                        default='external/poseidon/pretrained/Poseidon-T/model.safetensors')
    parser.add_argument('--dpot_path',
                        default='external/DPOT/pretrained/model_Ti.pth')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=== Foundation Model Benchmark ===")
    print(f"Device: {device}")

    # Dataset
    if args.augment > 1:
        ds = AugmentedGWDataset(
            args.data_dir, args.doe,
            residual=True, downsample=args.downsample,
            max_timesteps=args.max_timesteps,
            augment_factor=args.augment)
        meta = ds.get_metadata()
        print(f"Dataset: {meta['n_original']} FEM → {len(ds)} augmented "
              f"({args.augment}x)")
    else:
        ds = GWOperatorDataset(
            args.data_dir, args.doe,
            residual=True, downsample=args.downsample,
            max_timesteps=args.max_timesteps)
        meta = ds.get_metadata()
        print(f"Dataset: {len(ds)} samples")

    print(f"  Sensors: {meta['n_sensors']}, T: {meta['T']}, "
          f"n_params: {meta.get('n_params', 3)}")

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    n_sensors = meta['n_sensors']
    T = meta['T']
    n_params = meta.get('n_params', 3)

    os.makedirs(args.output, exist_ok=True)
    results = {}

    models_to_run = []
    if args.model == 'all':
        models_to_run = ['fno', 'cnn', 'dpot', 'poseidon']
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if model_name == 'fno':
            from models_fno_gw import FNOGWSurrogate
            model = FNOGWSurrogate(
                n_sensors=n_sensors, n_params=n_params,
                modes=64, width=64, n_layers=4, residual=True,
            ).to(device)

        elif model_name == 'cnn':
            model = SimpleConvSurrogate(
                n_sensors, T, n_params, hidden=64
            ).to(device)

        elif model_name == 'dpot':
            dpot_img = min(args.img_size, 64)  # DPOT works well at 64
            model = DPOTGWWrapper(
                n_sensors, T, n_params,
                pretrained_path=os.path.join(PROJECT_ROOT, args.dpot_path),
                img_size=dpot_img,
                freeze_backbone=True,
            ).to(device)

        elif model_name == 'poseidon':
            poseidon_img = 128  # Poseidon-T pretrained at 128
            model = PoseidonGWWrapper(
                n_sensors, T, n_params,
                pretrained_path=os.path.join(PROJECT_ROOT, args.poseidon_path),
                img_size=poseidon_img,
                freeze_backbone=True,
            ).to(device)

        best_val = train_model(
            model, train_loader, val_loader, args, device, model_name)
        results[model_name] = {
            'best_val_loss': best_val,
            'n_params': sum(p.numel() for p in model.parameters()),
            'n_trainable': sum(p.numel() for p in model.parameters()
                               if p.requires_grad),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"=== BENCHMARK RESULTS ===")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Val Loss':>10} {'Params':>12} {'Trainable':>12}")
    print(f"{'-'*54}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['best_val_loss']):
        print(f"{name:<20} {r['best_val_loss']:>10.4f} "
              f"{r['n_params']:>12,} {r['n_trainable']:>12,}")

    with open(os.path.join(args.output, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}/benchmark_results.json")


if __name__ == '__main__':
    main()
