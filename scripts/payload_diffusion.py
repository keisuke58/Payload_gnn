"""
Temperature-Conditioned DDPM for Environmental-Robust GW-SHM
--------------------------------------------------------------
Key contribution: conditioning on temperature suppresses winter false-positives
(FPR 47.1% → near calibration level) that plague unconditional anomaly scoring.

Pipeline:
  1. Load OGW long-term pickle files (N, 8, 2000 GW + temperature + damage tag)
  2. Downsample waveform 2000→256 (4x avg pool × 2)
  3. Train DDPM on healthy (damage=0) measurements with temperature conditioning
  4. Anomaly score = partial-noise + reconstruct → MSE per path → mean over 8 paths
  5. Evaluate: AUROC / FPR@recall=0.9 for unconditioned vs conditioned

Usage:
  python scripts/payload_diffusion.py --mode train
  python scripts/payload_diffusion.py --mode eval  --ckpt results/payload_ddpm/best.pt
  python scripts/payload_diffusion.py --mode compare  # uncond vs conditioned AUROC
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

LT_DIR      = os.path.join(ROOT, "data", "external", "ogw_longterm")
RESULTS_DIR = os.path.join(ROOT, "results", "payload_ddpm")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
T_MAX     = 1000
T_PARTIAL = 300
SEQ_LEN   = 256    # downsample 2000 → 256
N_PATHS   = 8

# ─── data loading & preprocessing ───────────────────────────────────────────
def load_pickle_months(pattern="measurements_2018_*.pickle", max_months=None):
    files = sorted(glob.glob(os.path.join(LT_DIR, pattern)))
    if max_months:
        files = files[:max_months]
    gws, temps, labels = [], [], []
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
            continue
        gw    = d["guided wave"].astype(np.float32)    # (N, 8, 2000)
        tmp   = d["temperature"].astype(np.float32)    # (N,)
        dmg   = d["damage tag"].astype(np.int32)       # (N,)
        gws.append(gw); temps.append(tmp); labels.append(dmg)
    gw_all  = np.concatenate(gws,   axis=0)
    tmp_all = np.concatenate(temps, axis=0)
    lbl_all = np.concatenate(labels, axis=0)
    print(f"  loaded {len(gws)} months, {len(gw_all)} samples, dmg_frac={lbl_all.mean():.3f}")
    return gw_all, tmp_all, lbl_all


def downsample(gw: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    """(N, 8, 2000) → (N, 8, target) via adaptive avg pool."""
    t = torch.from_numpy(gw)
    N, P, L = t.shape
    t = F.adaptive_avg_pool1d(t.reshape(N * P, 1, L), target)  # (N*P, 1, target)
    return t.reshape(N, P, target).numpy()


def zscore_waveform(gw: np.ndarray, mu=None, sig=None):
    """Per-path z-score normalisation."""
    if mu is None:
        mu  = gw.mean(axis=(0, 2), keepdims=True)    # (1, 8, 1)
        sig = gw.std(axis=(0, 2), keepdims=True) + 1e-8
    return (gw - mu) / sig, mu, sig


def temp_stats(t: np.ndarray):
    return t.mean(), t.std() + 1e-8


# ─── dataset ─────────────────────────────────────────────────────────────────
class GWDataset(Dataset):
    """Healthy-only dataset for DDPM training."""
    def __init__(self, gw, temp, label, mu, sig, t_mu, t_sig, healthy_only=True):
        mask      = (label == 0) if healthy_only else np.ones(len(label), dtype=bool)
        self.gw   = torch.from_numpy(gw[mask])        # (M, 8, 256)
        self.temp = torch.from_numpy((temp[mask] - t_mu) / t_sig)  # (M,)

    def __len__(self):
        return len(self.gw)

    def __getitem__(self, i):
        return self.gw[i], self.temp[i]


# ─── temperature-conditioned UNet (1D, 8 channels = 8 sensor paths) ─────────
class SinEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half  = dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        e = t[:, None] * self.freqs[None]
        return torch.cat([e.sin(), e.cos()], dim=1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.emb   = nn.Linear(emb_dim, out_ch)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb(F.silu(emb))[:, :, None]
        h = F.silu(self.norm2(h))
        return self.conv2(h) + self.res(x)


class UNet1D(nn.Module):
    """
    1D UNet for (8 path, 256 samples) GW signals.
    emb_dim carries both diffusion timestep AND temperature.
    """
    def __init__(self, in_ch=N_PATHS, base_ch=64, emb_dim=128, levels=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinEmbed(emb_dim), nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(), nn.Linear(emb_dim * 2, emb_dim)
        )
        # temperature projects onto same embedding space (FiLM conditioning)
        self.temp_embed = nn.Sequential(
            nn.Linear(1, emb_dim // 2), nn.SiLU(), nn.Linear(emb_dim // 2, emb_dim)
        )
        chs   = [base_ch * (2 ** i) for i in range(levels)]

        self.enc_in   = nn.Conv1d(in_ch, chs[0], 3, padding=1)
        self.enc_blks = nn.ModuleList([ResBlock1D(chs[i], chs[i], emb_dim) for i in range(levels)])
        self.downs    = nn.ModuleList([nn.Conv1d(chs[i], chs[i+1], 4, stride=2, padding=1)
                                       for i in range(levels - 1)])
        self.mid      = ResBlock1D(chs[-1], chs[-1], emb_dim)
        self.ups      = nn.ModuleList([nn.ConvTranspose1d(chs[i+1], chs[i], 4, stride=2, padding=1)
                                       for i in reversed(range(levels - 1))])
        # decoder: skip concat doubles channels → reduce back to chs[i]
        self.dec_blks = nn.ModuleList([ResBlock1D(chs[i] * 2, chs[i], emb_dim)
                                       for i in reversed(range(levels - 1))])
        self.out      = nn.Conv1d(chs[0], in_ch, 1)

    def forward(self, x, t, temp=None):
        emb = self.time_embed(t.float())
        if temp is not None:
            emb = emb + self.temp_embed(temp.float().unsqueeze(1))
        h = self.enc_in(x)
        skips = []
        for blk, down in zip(self.enc_blks[:-1], self.downs):
            h = blk(h, emb); skips.append(h); h = down(h)
        h = self.enc_blks[-1](h, emb)
        h = self.mid(h, emb)
        for up, blk, skip in zip(self.ups, self.dec_blks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:   # edge case for odd lengths
                h = F.interpolate(h, size=skip.shape[-1])
            h = torch.cat([h, skip], dim=1); h = blk(h, emb)
        return self.out(h)


# ─── DDPM ────────────────────────────────────────────────────────────────────
def cosine_schedule(T, s=0.008):
    t  = np.linspace(0, T, T + 1) / T
    f  = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    b  = 1 - ab[1:] / ab[:-1]
    return np.clip(b, 0, 0.999).astype(np.float32)


class DDPM:
    def __init__(self, T=T_MAX, device=DEVICE):
        self.T = T; self.device = device
        betas  = torch.from_numpy(cosine_schedule(T)).to(device)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def loss(self, model, x0, temp=None):
        t    = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t, temp)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def partial_denoise(self, model, x0, temp=None, t_start=T_PARTIAL):
        model.eval()
        tb = torch.full((x0.shape[0],), t_start - 1, device=self.device, dtype=torch.long)
        x  = self.q_sample(x0, tb)[0]
        for t in range(t_start - 1, -1, -1):
            tb2  = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            ab   = self.alpha_bar[t]
            ab_p = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            eps  = model(x, tb2, temp)
            x0h  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-3, 3)
            mean = ab_p.sqrt() * x0h + (1 - ab_p).sqrt() * eps
            x    = mean + (1 - ab_p).sqrt() * torch.randn_like(x) if t > 0 else mean
        return x


# ─── training ────────────────────────────────────────────────────────────────
def train(args):
    print(f"[train] device={DEVICE}")
    gw_raw, tmp_raw, lbl = load_pickle_months()
    gw_ds = downsample(gw_raw)                             # (N, 8, 256)
    gw_zs, mu, sig = zscore_waveform(gw_ds[lbl == 0])     # fit on healthy only
    # apply to all (needed for eval later)
    gw_zs_all, _, _ = zscore_waveform(gw_ds, mu, sig)
    t_mu, t_sig = temp_stats(tmp_raw[lbl == 0])

    ds     = GWDataset(gw_zs_all, tmp_raw, lbl, mu, sig, t_mu, t_sig)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"  healthy training samples: {len(ds)}")

    use_temp = not args.no_temp
    model    = UNet1D(base_ch=args.base_ch).to(DEVICE)
    ddpm     = DDPM(device=DEVICE)
    optim    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # save normalisation stats alongside checkpoint
    np.save(f"{RESULTS_DIR}/norm_stats.npz", {
        "mu": mu, "sig": sig, "t_mu": t_mu, "t_sig": t_sig
    })
    np.savez(f"{RESULTS_DIR}/norm_stats.npz", mu=mu, sig=sig, t_mu=t_mu, t_sig=t_sig)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0
        for gw_b, tmp_b in loader:
            gw_b  = gw_b.to(DEVICE)
            tmp_b = tmp_b.to(DEVICE) if use_temp else None
            loss  = ddpm.loss(model, gw_b, tmp_b)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        sched.step()
        avg = total / len(loader)
        if epoch % 10 == 0:
            print(f"  epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg,
                        "use_temp": use_temp},
                       f"{RESULTS_DIR}/best.pt")
    print(f"[train] best loss={best_loss:.4f}")


# ─── evaluation ──────────────────────────────────────────────────────────────
def anomaly_scores(model, ddpm, gw_zs, tmp_norm, use_temp, batch_size=64):
    """Returns per-sample anomaly score = mean MSE across 8 paths."""
    scores = []
    for i in range(0, len(gw_zs), batch_size):
        x0  = torch.from_numpy(gw_zs[i:i+batch_size]).to(DEVICE)
        tmp = torch.from_numpy(tmp_norm[i:i+batch_size]).to(DEVICE) if use_temp else None
        xr  = ddpm.partial_denoise(model, x0, tmp)
        sc  = (x0 - xr).pow(2).mean(dim=(1, 2))   # (B,)
        scores.append(sc.cpu().numpy())
    return np.concatenate(scores)


def eval_fpr_at_recall(scores, labels, recall_target=0.90):
    """FPR when recall = recall_target (threshold by defect percentile)."""
    thresh = np.percentile(scores[labels == 1], (1 - recall_target) * 100)
    pred   = (scores >= thresh).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    fpr     = fp / (fp + tn + 1e-9)
    rec     = tp / (tp + fn + 1e-9)
    return float(fpr), float(rec)


def evaluate(args):
    ckpt     = torch.load(args.ckpt, map_location=DEVICE)
    use_temp = ckpt.get("use_temp", True) and not args.no_temp
    model    = UNet1D(base_ch=args.base_ch).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    ddpm     = DDPM(device=DEVICE)

    stats = np.load(f"{RESULTS_DIR}/norm_stats.npz")
    mu, sig, t_mu, t_sig = stats["mu"], stats["sig"], float(stats["t_mu"]), float(stats["t_sig"])

    gw_raw, tmp_raw, lbl = load_pickle_months()
    gw_zs, _, _ = zscore_waveform(downsample(gw_raw), mu, sig)
    tmp_norm    = (tmp_raw - t_mu) / (t_sig if t_sig > 0 else 1)

    sc = anomaly_scores(model, ddpm, gw_zs, tmp_norm, use_temp)

    auroc = roc_auc_score(lbl, sc)
    fpr, rec = eval_fpr_at_recall(sc, lbl)
    print(f"[eval] conditioned={use_temp}  AUROC={auroc:.4f}  FPR@rec{rec:.2f}={fpr:.4f}")

    # Per-temperature breakdown (5°C bins)
    bins = np.arange(0, 35, 5)
    print("  temp-bin breakdown (healthy only, FPR proxy):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (tmp_raw >= lo) & (tmp_raw < hi) & (lbl == 0)
        if mask.sum() < 20:
            continue
        thresh = np.percentile(sc[lbl == 1], 10)   # recall≈0.9 threshold
        fpr_b  = (sc[mask] >= thresh).mean()
        print(f"    [{lo:.0f}-{hi:.0f}°C]  n={mask.sum():<5d}  FPR={fpr_b:.3f}")

    # Score distribution plot
    plt.figure(figsize=(8, 4))
    plt.scatter(tmp_raw[lbl == 0], sc[lbl == 0], s=1, alpha=0.3, label="healthy")
    plt.scatter(tmp_raw[lbl == 1], sc[lbl == 1], s=4, alpha=0.6, c="red", label="damage")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Anomaly score")
    plt.title(f"DDPM anomaly score vs temperature  [cond={use_temp}, AUROC={auroc:.3f}]")
    plt.legend(markerscale=5); plt.tight_layout()
    tag = "cond" if use_temp else "uncond"
    out = f"{RESULTS_DIR}/score_vs_temp_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved → {out}")

    return auroc, fpr


def compare(args):
    """Train / load both conditioned and unconditioned, then compare."""
    print("\n=== Conditioned ===")
    args.no_temp = False
    args.ckpt    = f"{RESULTS_DIR}/best.pt"
    auroc_c, fpr_c = evaluate(args)

    print("\n=== Unconditioned ===")
    args.no_temp = True
    args.ckpt    = f"{RESULTS_DIR}/best_uncond.pt"
    if os.path.exists(args.ckpt):
        auroc_u, fpr_u = evaluate(args)
    else:
        print("  (no unconditioned checkpoint found — run with --mode train --no_temp first)")
        return

    print(f"\n{'':20s} {'AUROC':>8} {'FPR@r=0.9':>10}")
    print(f"  Conditioned       {auroc_c:>8.4f} {fpr_c:>10.4f}")
    print(f"  Unconditioned     {auroc_u:>8.4f} {fpr_u:>10.4f}")
    print(f"  Δ FPR (improvement): {fpr_u - fpr_c:+.4f}")


# ─── entrypoint ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="train", choices=["train", "eval", "compare"])
    p.add_argument("--ckpt",       default=f"{RESULTS_DIR}/best.pt")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--no_temp",    action="store_true", help="disable temperature conditioning")
    args = p.parse_args()

    if   args.mode == "train":   train(args)
    elif args.mode == "eval":    evaluate(args)
    elif args.mode == "compare": compare(args)


if __name__ == "__main__":
    main()
