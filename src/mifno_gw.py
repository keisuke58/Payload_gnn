# -*- coding: utf-8 -*-
"""
MIFNO-GW: MIFNO adapted for sparse-sensor GW-based SHM on CFRP fairing.

Architecture (2D factorized FNO, sensors x time):
  - Material branch : sensor positions + defect params  -> latent field [B, C, S, T]
  - Source branch   : excitation freq/location          -> source token  [B, C, S, T]
  - Concatenate at branching_index, then shared spectral layers
  - Decoder         : [B, C, S, T] -> anomaly logit [B, 1]

Reference: Lehmann et al. (2025) JCP 527:113813
Adaptation: 3D geology/seismic -> 2D CFRP GW (sensors x time).

Usage:
    python src/mifno_gw.py --csv_dir abaqus_work/gw_fairing_dataset \
        --epochs 100 --output_dir runs/mifno_gw_v1
"""

import os
import sys
import csv
import json
import argparse
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# =========================================================================
# Data loading
# =========================================================================

def load_sensor_csv(path, max_len=2048):
    """Load sensor CSV -> (positions [S], waveforms [S, T])."""
    with open(path) as f:
        lines = f.readlines()

    # header: time_s, sensor_0_Ur, ...
    # line 1: # x_mm, pos0, pos1, ...
    pos_line = lines[1].strip().lstrip('#').split(',')
    positions = np.array([float(x) for x in pos_line[1:]], dtype=np.float32)  # [S]

    data = []
    for line in lines[2:]:
        if line.strip() == '':
            continue
        vals = line.strip().split(',')
        data.append([float(v) for v in vals[1:]])  # skip time col

    waveforms = np.array(data, dtype=np.float32).T  # [S, T_raw]

    # Truncate or pad to max_len
    T = waveforms.shape[1]
    if T >= max_len:
        waveforms = waveforms[:, :max_len]
    else:
        pad = np.zeros((waveforms.shape[0], max_len - T), dtype=np.float32)
        waveforms = np.concatenate([waveforms, pad], axis=1)

    # z-score per sensor
    mu = waveforms.mean(axis=1, keepdims=True)
    std = waveforms.std(axis=1, keepdims=True) + 1e-8
    waveforms = (waveforms - mu) / std

    # normalize positions to [0, 1]
    p_max = positions.max() + 1e-8
    positions = positions / p_max

    return positions, waveforms  # [S], [S, T]


class GWDataset(Dataset):
    def __init__(self, samples, max_len=2048):
        self.samples = samples  # list of (path, label)
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pos, wav = load_sensor_csv(path, self.max_len)
        # pos: [S], wav: [S, T]
        return (
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(wav, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def collect_samples(csv_dir):
    samples = []
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith('.csv'):
            continue
        path = os.path.join(csv_dir, f)
        label = 0 if 'Healthy' in f else 1
        samples.append((path, label))
    return samples


# =========================================================================
# Factorized Spectral Conv 2D (sensors x time)
# =========================================================================

class SpectralConv2d(nn.Module):
    """Factorized 2D spectral conv: separate modes on S and T axes."""

    def __init__(self, in_dim, out_dim, modes_s, modes_t):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_s = modes_s
        self.modes_t = modes_t

        scale = 1.0 / (in_dim * out_dim)
        self.w_s = nn.Parameter(scale * torch.randn(in_dim, out_dim, modes_s, dtype=torch.cfloat))
        self.w_t = nn.Parameter(scale * torch.randn(in_dim, out_dim, modes_t, dtype=torch.cfloat))

    def _compl_mul1d(self, x, w):
        # x: [B, C, M], w: [Cin, Cout, M] -> [B, Cout, M]
        return torch.einsum('bim,iom->bom', x, w)

    def forward(self, x):
        # x: [B, C, S, T]
        B, C, S, T = x.shape

        # --- sensor axis: rfft over S dim ---
        xf_s = torch.fft.rfft(x, dim=2)          # [B, C, S//2+1, T]
        ms = min(self.modes_s, xf_s.shape[2])
        # reshape to [B*T, C, S//2+1], process low modes
        xf_s_bt = xf_s.permute(0, 3, 1, 2).reshape(B * T, C, xf_s.shape[2])  # [B*T, C, S//2+1]
        out_s_bt = torch.zeros(B * T, self.out_dim, xf_s.shape[2], dtype=torch.cfloat, device=x.device)
        out_s_bt[:, :, :ms] = self._compl_mul1d(xf_s_bt[:, :, :ms], self.w_s[:, :, :ms])
        out_s_f = out_s_bt.reshape(B, T, self.out_dim, xf_s.shape[2]).permute(0, 2, 3, 1)  # [B, out, S//2+1, T]
        x_s = torch.fft.irfft(out_s_f, n=S, dim=2)  # [B, out_dim, S, T]

        # --- time axis: rfft over T dim ---
        xf_t = torch.fft.rfft(x, dim=3)           # [B, C, S, T//2+1]
        mt = min(self.modes_t, xf_t.shape[3])
        xf_t_bs = xf_t.permute(0, 2, 1, 3).reshape(B * S, C, xf_t.shape[3])  # [B*S, C, T//2+1]
        out_t_bs = torch.zeros(B * S, self.out_dim, xf_t.shape[3], dtype=torch.cfloat, device=x.device)
        out_t_bs[:, :, :mt] = self._compl_mul1d(xf_t_bs[:, :, :mt], self.w_t[:, :, :mt])
        out_t_f = out_t_bs.reshape(B, S, self.out_dim, xf_t.shape[3]).permute(0, 2, 1, 3)  # [B, out, S, T//2+1]
        x_t = torch.fft.irfft(out_t_f, n=T, dim=3)  # [B, out_dim, S, T]

        return x_s + x_t


class FNOBlock2D(nn.Module):
    def __init__(self, channels, modes_s, modes_t):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_s, modes_t)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# =========================================================================
# MIFNO-GW model
# =========================================================================

class MIFNO_GW(nn.Module):
    """
    MIFNO adapted for sparse-sensor GW SHM.

    Input:
        pos  : [B, S]     sensor positions (normalized)
        wav  : [B, S, T]  sensor waveforms (z-scored)
    Output:
        logit: [B, 2]     healthy / defect logit
    """

    def __init__(self, n_sensors=9, seq_len=2048, width=32,
                 n_layers=6, branching_index=3,
                 modes_s=4, modes_t=64, dropout=0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.width = width
        self.branching_index = branching_index

        # Lift waveform [B, S, T] -> [B, width, S, T]
        self.lift = nn.Conv2d(1, width, kernel_size=1)

        # Material branch (first branching_index layers)
        self.mat_blocks = nn.ModuleList([
            FNOBlock2D(width, modes_s, modes_t)
            for _ in range(branching_index)
        ])

        # Source branch: pos [B, S] -> [B, width, S, T] token
        self.src_mlp = nn.Sequential(
            nn.Linear(n_sensors, 64),
            nn.GELU(),
            nn.Linear(64, width * n_sensors),
        )

        # Shared branch (concat mat + src -> 2*width)
        shared_ch = 2 * width
        self.shared_blocks = nn.ModuleList([
            FNOBlock2D(shared_ch, modes_s, modes_t)
            for _ in range(n_layers - branching_index)
        ])

        # Decoder: global avg pool -> MLP -> 2
        self.head = nn.Sequential(
            nn.Linear(shared_ch, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, pos, wav):
        # pos: [B, S], wav: [B, S, T]
        B, S, T = wav.shape

        # Lift to [B, width, S, T]
        x = wav.unsqueeze(1)           # [B, 1, S, T]
        x = self.lift(x)               # [B, width, S, T]

        # Material branch
        for blk in self.mat_blocks:
            x = blk(x)

        # Source branch: pos -> [B, width, S, 1] broadcast over T
        s_tok = self.src_mlp(pos)                          # [B, width*S]
        s_tok = s_tok.view(B, self.width, S, 1)            # [B, width, S, 1]
        s_tok = s_tok.expand(-1, -1, -1, T)               # [B, width, S, T]

        # Concat
        x = torch.cat([x, s_tok], dim=1)                  # [B, 2*width, S, T]

        # Shared branch
        for blk in self.shared_blocks:
            x = blk(x)

        # Decode: avg over S and T
        x = x.mean(dim=[2, 3])                            # [B, 2*width]
        return self.head(x)                                # [B, 2]


# =========================================================================
# Training
# =========================================================================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpt'), exist_ok=True)

    samples = collect_samples(args.csv_dir)
    labels = [s[1] for s in samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print("Samples: total=%d  healthy=%d  defect=%d" % (len(samples), n_neg, n_pos))

    train_s, val_s = train_test_split(samples, test_size=0.2, stratify=labels,
                                      random_state=args.seed)

    train_ds = GWDataset(train_s, args.seq_len)
    val_ds = GWDataset(val_s, args.seq_len)

    # Weighted sampler to handle 100:6 imbalance
    train_labels = [s[1] for s in train_s]
    w = [1.0 / n_neg if l == 0 else 1.0 / max(n_pos, 1) for l in train_labels]
    sampler = WeightedRandomSampler(w, num_samples=len(train_s) * 4, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # detect n_sensors from data
    pos0, wav0, _ = train_ds[0]
    n_sensors = pos0.shape[0]

    model = MIFNO_GW(
        n_sensors=n_sensors,
        seq_len=args.seq_len,
        width=args.width,
        n_layers=args.n_layers,
        branching_index=args.branching_index,
        modes_s=args.modes_s,
        modes_t=args.modes_t,
        dropout=args.dropout,
    ).to(device)
    print("Params: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Focal loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma = gamma
            self.weight = weight

        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

    class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    criterion = FocalLoss(gamma=2.0, weight=class_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    best_f1 = 0.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for pos, wav, y in train_loader:
            pos, wav, y = pos.to(device), wav.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(pos, wav)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for pos, wav, y in val_loader:
                pos, wav = pos.to(device), wav.to(device)
                logits = model(pos, wav)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                all_probs.extend(probs)

        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float('nan')

        elapsed = time.time() - t0
        avg_loss = total_loss / max(len(train_loader), 1)
        print("[%03d/%03d] loss=%.4f  val_F1=%.4f  AUC=%.4f  lr=%.2e  %.1fs" % (
            epoch, args.epochs, avg_loss, f1, auc,
            scheduler.get_last_lr()[0], elapsed))

        log_rows.append({'epoch': epoch, 'loss': avg_loss, 'f1': f1, 'auc': auc})

        if f1 > best_f1:
            best_f1 = f1
            ckpt = os.path.join(args.output_dir, 'ckpt', 'best.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'f1': f1, 'auc': auc}, ckpt)
            print("  -> saved best (F1=%.4f)" % f1)

    # Save log
    import csv as csv_mod
    log_path = os.path.join(args.output_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv_mod.DictWriter(f, fieldnames=['epoch', 'loss', 'f1', 'auc'])
        writer.writeheader()
        writer.writerows(log_rows)

    print("Done. Best F1=%.4f" % best_f1)


def main():
    parser = argparse.ArgumentParser(description='MIFNO-GW: FNO for sparse-sensor GW SHM')
    parser.add_argument('--csv_dir', type=str,
                        default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--output_dir', type=str, default='runs/mifno_gw_v1')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--branching_index', type=int, default=3)
    parser.add_argument('--modes_s', type=int, default=4,
                        help='Fourier modes on sensor axis (max 4 for 9 sensors)')
    parser.add_argument('--modes_t', type=int, default=64,
                        help='Fourier modes on time axis')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
