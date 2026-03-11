# -*- coding: utf-8 -*-
"""
Train FNO/DeepONet GW Surrogate

Trains neural operator surrogate:
  defect_params → sensor waveforms (via FNO1d or DeepONet)

Usage:
  python src/train_fno_gw.py --model fno --data_dir abaqus_work/gw_fairing_dataset
  python src/train_fno_gw.py --model deeponet --data_dir abaqus_work/gw_fairing_dataset
  python src/train_fno_gw.py --model fno --generate 1000  # surrogate data generation after training
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_fno_gw import GWOperatorDataset, GWDeepONetDataset
from dataset_fno_gw_augmented import AugmentedGWDataset
from models_fno_gw import FNOGWSurrogate, DualStageFNOSurrogate, DeepONetGW


def pretrain_hemew3d(args):
    """Pre-train FNO on HEMEW3D dataset (transfer learning).

    Trains spectral convolution layers on general elastic wave propagation,
    then saves checkpoint for fine-tuning on fairing-specific GW data.
    """
    from dataset_hemew3d import HEMEW3DDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== HEMEW3D Pre-training ===")
    print(f"Device: {device}")

    ds = HEMEW3DDataset(
        args.pretrain_hemew3d,
        n_sensors=9,  # match fairing sensor count
        downsample_t=max(1, 320 // 256),  # ~256 timesteps
        max_timesteps=256,
        n_params=3,
        residual=args.residual,
    )
    meta = ds.get_metadata()
    print(f"HEMEW3D: {len(ds)} samples, {meta['n_sensors']} sensors, T={meta['T']}")

    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Build model with HEMEW3D dimensions
    if args.model == 'dual_fno':
        model = DualStageFNOSurrogate(
            n_sensors=meta['n_sensors'],
            n_params=meta['n_params'],
            modes_low=max(args.modes // 4, 8),
            modes_high=args.modes,
            width=args.width,
            n_layers_low=max(args.n_layers // 2, 2),
            n_layers_high=args.n_layers,
            residual=args.residual,
        ).to(device)
    else:
        model = FNOGWSurrogate(
            n_sensors=meta['n_sensors'],
            n_params=meta['n_params'],
            modes=args.modes,
            width=args.width,
            n_layers=args.n_layers,
            residual=args.residual,
        ).to(device)

    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, {n_params_model:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs)

    def relative_l2(pred, target):
        diff = (pred - target) ** 2
        norm = target ** 2 + 1e-8
        return torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nPre-training for {args.pretrain_epochs} epochs...")
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        train_loss, n_b = 0, 0
        for batch in train_loader:
            params = batch['params'].to(device)
            healthy = batch['healthy'].to(device)
            target_key = 'defect_full' if args.residual else 'target'
            target = batch[target_key].to(device)

            pred = model(params, healthy)
            loss = relative_l2(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_b = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                healthy = batch['healthy'].to(device)
                target_key = 'defect_full' if args.residual else 'target'
                target = batch[target_key].to(device)
                pred = model(params, healthy)
                val_loss += relative_l2(pred, target).item()
                val_b += 1

        avg_val = val_loss / max(val_b, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': meta,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': avg_val,
                'pretrained': True,
                'pretrain_dataset': 'HEMEW3D',
            }, os.path.join(args.output, 'pretrained_model.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Pretrain Ep {epoch:3d}/{args.pretrain_epochs} | "
                  f"Train: {train_loss/max(n_b,1):.6f} | "
                  f"Val: {avg_val:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nPre-training done. Best val: {best_val_loss:.6f}")
    print(f"Checkpoint: {args.output}/pretrained_model.pt")
    return os.path.join(args.output, 'pretrained_model.pt')


def train_fno(args):
    """Train FNO GW surrogate."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset (augmented or plain)
    augment_factor = getattr(args, 'augment', 1)
    if augment_factor > 1:
        ds = AugmentedGWDataset(
            args.data_dir, args.doe,
            residual=args.residual,
            downsample=args.downsample,
            max_timesteps=args.max_timesteps,
            augment_factor=augment_factor,
        )
        meta = ds.get_metadata()
        print(f"Dataset: {meta['n_original']} FEM → {len(ds)} augmented samples "
              f"({augment_factor}×), {meta['n_sensors']} sensors, "
              f"T={meta['T']} (dt={meta['dt']:.2e}s)")
    else:
        ds = GWOperatorDataset(
            args.data_dir, args.doe,
            residual=args.residual,
            downsample=args.downsample,
            max_timesteps=args.max_timesteps,
        )
        meta = ds.get_metadata()
        print(f"Dataset: {len(ds)} defect samples, {meta['n_sensors']} sensors, "
              f"T={meta['T']} (dt={meta['dt']:.2e}s)")

    if len(ds) < 2:
        print("ERROR: Need at least 2 defect samples. "
              "Run Abaqus jobs first.")
        return

    # Train/val split
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    n_params = meta.get('n_params', 3)
    if args.model == 'dual_fno':
        model = DualStageFNOSurrogate(
            n_sensors=meta['n_sensors'],
            n_params=n_params,
            modes_low=max(args.modes // 4, 8),
            modes_high=args.modes,
            width=args.width,
            n_layers_low=max(args.n_layers // 2, 2),
            n_layers_high=args.n_layers,
            residual=args.residual,
        ).to(device)
    else:
        model = FNOGWSurrogate(
            n_sensors=meta['n_sensors'],
            n_params=n_params,
            modes=args.modes,
            width=args.width,
            n_layers=args.n_layers,
            residual=args.residual,
        ).to(device)

    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"FNO Parameters: {n_params_model:,}")

    # Load pre-trained / fine-tune checkpoint
    if args.finetune_from:
        ckpt = torch.load(args.finetune_from, map_location=device,
                          weights_only=False)
        # Load matching layers (skip param_encoder if n_params differs)
        state = ckpt['model_state_dict']
        model_state = model.state_dict()
        loaded, skipped = [], []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        print(f"Fine-tune: loaded {len(loaded)} layers, "
              f"skipped {len(skipped)} (shape mismatch)")
        if skipped:
            print(f"  Skipped: {skipped}")

        # Optionally freeze spectral conv layers
        if args.freeze_spectral:
            frozen = 0
            for name, param in model.named_parameters():
                if 'spectral' in name or 'weights' in name:
                    param.requires_grad = False
                    frozen += 1
            print(f"  Frozen {frozen} spectral conv parameters")

    lr = args.finetune_lr if args.finetune_from else args.lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)

    # Warmup + CosineAnnealing (Li 2021 FNO best practice)
    warmup_epochs = max(1, args.epochs // 20)  # 5% warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])

    # Loss: relative L2 (physics-informed)
    def relative_l2(pred, target):
        diff = (pred - target) ** 2
        norm = target ** 2 + 1e-8
        return torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))

    # Spectral loss: FFT domain accuracy (Gao et al. 2025 inspired)
    def spectral_loss(pred, target):
        """Penalize frequency-domain mismatch for high-freq GW modes."""
        pred_fft = torch.fft.rfft(pred, dim=-1).abs()
        tgt_fft = torch.fft.rfft(target, dim=-1).abs()
        # Relative L2 in frequency domain
        diff = (pred_fft - tgt_fft) ** 2
        norm = tgt_fft ** 2 + 1e-8
        return torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))

    def combined_loss(pred, target, alpha_spec=None):
        """Time-domain relative L2 + frequency-domain spectral loss."""
        l_time = relative_l2(pred, target)
        if alpha_spec is None or alpha_spec <= 0:
            return l_time
        l_freq = spectral_loss(pred, target)
        return l_time + alpha_spec * l_freq

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, 'train_log.jsonl')

    print(f"\nTraining FNO for {args.epochs} epochs...")
    print(f"  Modes: {args.modes}, Width: {args.width}, Layers: {args.n_layers}")
    print(f"  Residual: {args.residual}")
    print(f"  Spectral loss weight: {args.spectral_weight}")
    print(f"  Output: {args.output}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            params = batch['params'].to(device)
            healthy = batch['healthy'].to(device)
            target = batch['target'].to(device)

            if args.residual:
                # FNO predicts healthy + residual → compare with defect_full
                pred = model(params, healthy)
                loss = combined_loss(pred, batch['defect_full'].to(device),
                                     alpha_spec=args.spectral_weight)
            else:
                pred = model(params, healthy)
                loss = combined_loss(pred, target,
                                     alpha_spec=args.spectral_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                healthy = batch['healthy'].to(device)
                if args.residual:
                    pred = model(params, healthy)
                    loss = combined_loss(pred, batch['defect_full'].to(device),
                                         alpha_spec=args.spectral_weight)
                else:
                    pred = model(params, healthy)
                    loss = combined_loss(pred, batch['target'].to(device),
                                         alpha_spec=args.spectral_weight)
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': meta,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': avg_val,
            }, os.path.join(args.output, 'best_model.pt'))

        if epoch % args.log_interval == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                  f"Best: {best_val_loss:.6f} | LR: {lr:.2e}")

        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'epoch': epoch, 'train_loss': avg_train,
                'val_loss': avg_val, 'lr': scheduler.get_last_lr()[0],
            }) + '\n')

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved: {args.output}/best_model.pt")
    return model, meta


def train_deeponet(args):
    """Train DeepONet GW surrogate."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = GWDeepONetDataset(
        args.data_dir, args.doe,
        residual=args.residual,
        n_query=args.n_query,
        downsample=args.downsample,
        max_timesteps=args.max_timesteps,
    )
    meta = ds.base.get_metadata()
    print(f"Dataset: {len(ds)} samples, {meta['n_sensors']} sensors, "
          f"T={meta['T']}, n_query={args.n_query}")

    if len(ds) < 2:
        print("ERROR: Need at least 2 defect samples.")
        return

    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    n_params = meta.get('n_params', 3)
    model = DeepONetGW(
        n_params=n_params,
        coord_dim=2,
        hidden_dim=args.hidden_dim,
        basis_dim=args.basis_dim,
    ).to(device)

    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"DeepONet Parameters: {n_params_model:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    os.makedirs(args.output, exist_ok=True)

    print(f"\nTraining DeepONet for {args.epochs} epochs...")
    print(f"  Hidden: {args.hidden_dim}, Basis: {args.basis_dim}")
    print(f"  Query points/sample: {args.n_query}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        n_b = 0

        for batch in train_loader:
            params = batch['params'].to(device)
            coords = batch['coords'].to(device)
            values = batch['values'].to(device)

            pred = model(params, coords)
            loss = nn.functional.mse_loss(pred, values)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_train = train_loss / max(n_b, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_b = 0
        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                coords = batch['coords'].to(device)
                values = batch['values'].to(device)
                pred = model(params, coords)
                loss = nn.functional.mse_loss(pred, values)
                val_loss += loss.item()
                val_b += 1

        avg_val = val_loss / max(val_b, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': meta,
                'args': vars(args),
                'epoch': epoch,
                'val_loss': avg_val,
            }, os.path.join(args.output, 'best_model.pt'))

        if epoch % args.log_interval == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train MSE: {avg_train:.8f} | Val MSE: {avg_val:.8f} | "
                  f"Best: {best_val_loss:.8f} | LR: {lr:.2e}")

    print(f"\nBest val MSE: {best_val_loss:.8f}")
    print(f"Model saved: {args.output}/best_model.pt")
    return model, meta


def generate_surrogate_data(args):
    """Use trained FNO to generate synthetic waveforms for new defect configs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        os.path.join(args.output, 'best_model.pt'),
        map_location=device, weights_only=False)
    meta = checkpoint['meta']

    # Load healthy reference
    ds = GWOperatorDataset(
        args.data_dir, args.doe,
        residual=args.residual,
        downsample=args.downsample,
        max_timesteps=args.max_timesteps,
    )

    n_params = meta.get('n_params', 3)
    model = FNOGWSurrogate(
        n_sensors=meta['n_sensors'],
        n_params=n_params,
        modes=checkpoint['args']['modes'],
        width=checkpoint['args']['width'],
        n_layers=checkpoint['args']['n_layers'],
        residual=args.residual,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generate random defect configs
    n_gen = args.generate
    print(f"Generating {n_gen} synthetic waveforms...")

    healthy = torch.tensor(
        ds.healthy_waveform / ds.waveform_scale,
        dtype=torch.float32).unsqueeze(0).to(device)

    gen_dir = os.path.join(args.output, 'generated')
    os.makedirs(gen_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed + 1000)
    t0 = time.time()

    defect_types = getattr(ds, 'DEFECT_TYPES', [])
    encode_type = getattr(ds, 'encode_defect_type', False) and len(defect_types) > 0

    for i in range(n_gen):
        # Random defect params (normalized [0,1])
        z_n, theta_n, r_n = rng.rand(3).astype(np.float32)
        params_list = [z_n, theta_n, r_n]

        # Add defect type one-hot if model expects it
        if encode_type:
            type_idx = rng.randint(len(defect_types))
            one_hot = [0.0] * len(defect_types)
            one_hot[type_idx] = 1.0
            params_list.extend(one_hot)
            defect_type_name = defect_types[type_idx]
        else:
            defect_type_name = 'debonding'

        params_norm = np.array(params_list, dtype=np.float32)
        params_t = torch.tensor(params_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(params_t, healthy)  # (1, n_sensors, T)

        # Denormalize
        pred_np = pred.cpu().numpy()[0] * ds.waveform_scale

        # Denormalize spatial params
        z = z_n * (ds.z_max - ds.z_min) + ds.z_min
        theta = theta_n * (ds.theta_max - ds.theta_min) + ds.theta_min
        radius = r_n * (ds.r_max - ds.r_min) + ds.r_min

        # Save as CSV (same format as Abaqus output)
        csv_path = os.path.join(gen_dir, f'FNO-GW-{i:04d}_sensors.csv')
        header = ','.join(['time_s'] +
                          [f'sensor_{j}_Ur' for j in range(meta['n_sensors'])])
        pos_row = ','.join(['# x_mm'] +
                           [f'{p:.1f}' for p in meta['positions']])

        times = ds.times
        with open(csv_path, 'w') as f:
            f.write(header + '\n')
            f.write(pos_row + '\n')
            for t_idx in range(len(times)):
                row = [f'{times[t_idx]:.15e}']
                for s in range(meta['n_sensors']):
                    row.append(f'{pred_np[s, t_idx]:.15e}')
                f.write(','.join(row) + '\n')

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_gen} generated ({elapsed:.1f}s, "
                  f"{(i+1)/elapsed:.0f} samples/s)")

    elapsed = time.time() - t0
    print(f"\nGenerated {n_gen} waveforms in {elapsed:.1f}s "
          f"({n_gen/elapsed:.0f} samples/s)")
    print(f"Output: {gen_dir}/")

    # Save generation metadata
    gen_meta = {
        'n_generated': n_gen,
        'model_checkpoint': os.path.join(args.output, 'best_model.pt'),
        'model_val_loss': checkpoint['val_loss'],
        'generation_time_s': elapsed,
        'samples_per_second': n_gen / elapsed,
        'residual': args.residual,
        'waveform_scale': float(ds.waveform_scale),
    }
    with open(os.path.join(gen_dir, 'generation_meta.json'), 'w') as f:
        json.dump(gen_meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train FNO/DeepONet GW Surrogate')
    parser.add_argument('--model', choices=['fno', 'dual_fno', 'deeponet'], default='fno')
    parser.add_argument('--data_dir', default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--doe', default='doe_gw_fairing.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: runs/fno_gw_*)')

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10)

    # Data
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--no_residual', dest='residual', action='store_false')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Temporal downsample factor (3923→~981 steps)')
    parser.add_argument('--max_timesteps', type=int, default=None)

    # FNO
    parser.add_argument('--modes', type=int, default=64,
                        help='FNO Fourier modes')
    parser.add_argument('--width', type=int, default=64,
                        help='FNO channel width')
    parser.add_argument('--n_layers', type=int, default=4)

    # DeepONet
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--basis_dim', type=int, default=64)
    parser.add_argument('--n_query', type=int, default=512,
                        help='Query points per sample for DeepONet')

    # Loss
    parser.add_argument('--spectral_weight', type=float, default=0.1,
                        help='Weight for spectral (FFT) loss (0=disabled)')

    # Augmentation
    parser.add_argument('--augment', type=int, default=1,
                        help='Augmentation factor (1=no aug, 10=10x Mixup+noise+scale)')

    # Generation
    parser.add_argument('--generate', type=int, default=0,
                        help='Generate N synthetic waveforms (requires trained model)')

    # Pre-training / Fine-tuning
    parser.add_argument('--pretrain_hemew3d', type=str, default=None,
                        help='Pre-train on HEMEW3D data dir before fine-tuning')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Epochs for HEMEW3D pre-training phase')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Fine-tune from checkpoint (e.g., pretrained model)')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning (lower than pre-train)')
    parser.add_argument('--freeze_spectral', action='store_true',
                        help='Freeze spectral conv layers during fine-tuning')

    args = parser.parse_args()

    if args.output is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.output = f'runs/{args.model}_gw_{ts}'

    if args.pretrain_hemew3d:
        # Phase 1: Pre-train on HEMEW3D
        pretrain_ckpt = pretrain_hemew3d(args)
        # Phase 2: Fine-tune on fairing data (if data_dir exists)
        if os.path.exists(args.data_dir):
            print(f"\n=== Fine-tuning on fairing data ===")
            args.finetune_from = pretrain_ckpt
            train_fno(args)
        else:
            print(f"\nPre-training only (no fairing data at {args.data_dir})")
    elif args.generate > 0:
        generate_surrogate_data(args)
    elif args.model in ('fno', 'dual_fno'):
        train_fno(args)
    else:
        train_deeponet(args)


if __name__ == '__main__':
    main()
