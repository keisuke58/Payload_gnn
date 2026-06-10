"""
Temperature + Damage Conditioned DDPM for GW-SHM
--------------------------------------------------
Two roles:
  1. Environmental-robust anomaly detection: temperature conditioning suppresses
     winter false-positives (FPR 47.1% uncond) that plague unconditional scoring.
  2. Data augmentation: damage-tag conditioning + classifier-free guidance lets us
     SYNTHESIZE damaged GW waveforms at arbitrary temperatures — directly attacking
     the n-too-small problem (damaged measurements are ~5% of the long-term data,
     OGW stringer has effectively 3 specimens).

Pipeline:
  - Load OGW long-term pickles (N, 8, 2000 GW + temperature + damage tag)
  - Downsample 2000→256, per-path z-score (healthy stats)
  - Train DDPM on ALL data, conditioned on (temperature, damage tag) with CFG dropout
  - Anomaly score  = partial-noise + reconstruct under dmg=0 condition → MSE
  - Augmentation   = full ancestral sampling under dmg=1 at chosen temperatures

Usage:
  python scripts/payload_diffusion.py --mode train   --months 4
  python scripts/payload_diffusion.py --mode eval    --ckpt results/payload_ddpm/best.pt
  python scripts/payload_diffusion.py --mode augment --n_synth 2000
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
NULL_DMG  = 2      # CFG null token for damage class

# ─── data loading & preprocessing ───────────────────────────────────────────
# Months reserved for the downstream augmentation test (augment_eval.py).
# The generator must NEVER see these, or the augmentation result is circular.
HELDOUT_MONTHS = ("2020_12", "2022_02")


def load_pickle_months(pattern="measurements_*.pickle", max_months=None):
    files = sorted(glob.glob(os.path.join(LT_DIR, pattern)))
    files = [f for f in files if not any(m in f for m in HELDOUT_MONTHS)]
    if max_months:
        files = files[:max_months]
    gws, temps, labels = [], [], []
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
            continue
        gws.append(d["guided wave"].astype(np.float32))   # (N, 8, 2000)
        temps.append(d["temperature"].astype(np.float32)) # (N,)
        # damage tag is a damage-state ID (0..13), NOT binary — binarize for conditioning
        labels.append((np.asarray(d["damage tag"]) > 0).astype(np.int64))
    gw_all  = np.concatenate(gws,   axis=0)
    tmp_all = np.concatenate(temps, axis=0)
    lbl_all = np.concatenate(labels, axis=0)
    print(f"  loaded {len(gws)} months, {len(gw_all)} samples, "
          f"dmg_frac={lbl_all.mean():.3f}, temp {tmp_all.min():.1f}–{tmp_all.max():.1f}°C")
    return gw_all, tmp_all, lbl_all


def downsample(gw: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    """(N, 8, 2000) → (N, 8, target) via adaptive avg pool."""
    t = torch.from_numpy(gw)
    N, P, L = t.shape
    t = F.adaptive_avg_pool1d(t.reshape(N * P, 1, L), target)
    return t.reshape(N, P, target).numpy()


def zscore_waveform(gw: np.ndarray, mu=None, sig=None):
    """Per-path z-score normalisation."""
    if mu is None:
        mu  = gw.mean(axis=(0, 2), keepdims=True)    # (1, 8, 1)
        sig = gw.std(axis=(0, 2), keepdims=True) + 1e-8
    return (gw - mu) / sig, mu, sig


# ─── dataset ─────────────────────────────────────────────────────────────────
class GWDataset(Dataset):
    """All measurements; DDPM learns conditional distribution p(x | temp, dmg)."""
    def __init__(self, gw, temp_norm, label):
        self.gw   = torch.from_numpy(gw)                       # (M, 8, 256)
        self.temp = torch.from_numpy(temp_norm.astype(np.float32))
        self.dmg  = torch.from_numpy(label.astype(np.int64))

    def __len__(self):
        return len(self.gw)

    def __getitem__(self, i):
        return self.gw[i], self.temp[i], self.dmg[i]


# ─── conditioned 1D UNet ─────────────────────────────────────────────────────
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
    1D UNet for (8 path, 256 sample) GW signals.
    Conditioning: diffusion timestep + temperature (FiLM) + damage tag (embedding).
    Damage tag 2 = null token (classifier-free guidance).
    """
    def __init__(self, in_ch=N_PATHS, base_ch=64, emb_dim=128, levels=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinEmbed(emb_dim), nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(), nn.Linear(emb_dim * 2, emb_dim)
        )
        self.temp_embed = nn.Sequential(
            nn.Linear(1, emb_dim // 2), nn.SiLU(), nn.Linear(emb_dim // 2, emb_dim)
        )
        self.dmg_embed = nn.Embedding(3, emb_dim)   # 0=healthy, 1=damaged, 2=null
        chs = [base_ch * (2 ** i) for i in range(levels)]

        self.enc_in   = nn.Conv1d(in_ch, chs[0], 3, padding=1)
        self.enc_blks = nn.ModuleList([ResBlock1D(chs[i], chs[i], emb_dim) for i in range(levels)])
        self.downs    = nn.ModuleList([nn.Conv1d(chs[i], chs[i+1], 4, stride=2, padding=1)
                                       for i in range(levels - 1)])
        self.mid      = ResBlock1D(chs[-1], chs[-1], emb_dim)
        self.ups      = nn.ModuleList([nn.ConvTranspose1d(chs[i+1], chs[i], 4, stride=2, padding=1)
                                       for i in reversed(range(levels - 1))])
        self.dec_blks = nn.ModuleList([ResBlock1D(chs[i] * 2, chs[i], emb_dim)
                                       for i in reversed(range(levels - 1))])
        self.out      = nn.Conv1d(chs[0], in_ch, 1)

    def forward(self, x, t, temp=None, dmg=None):
        emb = self.time_embed(t.float())
        if temp is not None:
            emb = emb + self.temp_embed(temp.float().unsqueeze(1))
        if dmg is not None:
            emb = emb + self.dmg_embed(dmg)
        h = self.enc_in(x)
        skips = []
        for blk, down in zip(self.enc_blks[:-1], self.downs):
            h = blk(h, emb); skips.append(h); h = down(h)
        h = self.enc_blks[-1](h, emb)
        h = self.mid(h, emb)
        for up, blk, skip in zip(self.ups, self.dec_blks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
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

    def loss(self, model, x0, temp=None, dmg=None, p_uncond=0.1):
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, noise = self.q_sample(x0, t)
        # classifier-free guidance dropout on the damage tag
        if dmg is not None and p_uncond > 0:
            drop = torch.rand(dmg.shape[0], device=self.device) < p_uncond
            dmg  = torch.where(drop, torch.full_like(dmg, NULL_DMG), dmg)
        pred = model(xt, t, temp, dmg)
        return F.mse_loss(pred, noise)

    def _eps_cfg(self, model, x, t, temp, dmg, guidance):
        """Classifier-free guided noise prediction."""
        if guidance <= 0 or dmg is None:
            return model(x, t, temp, dmg)
        eps_c = model(x, t, temp, dmg)
        eps_u = model(x, t, temp, torch.full_like(dmg, NULL_DMG))
        return (1 + guidance) * eps_c - guidance * eps_u

    @torch.no_grad()
    def partial_denoise(self, model, x0, temp=None, dmg=None,
                        t_start=T_PARTIAL, guidance=0.0):
        """Anomaly scoring: noise to t_start, denoise back under given condition."""
        model.eval()
        t_start = min(t_start, self.T)
        tb = torch.full((x0.shape[0],), t_start - 1, device=self.device, dtype=torch.long)
        x  = self.q_sample(x0, tb)[0]
        return self._ancestral(model, x, temp, dmg, t_start, guidance)

    @torch.no_grad()
    def sample(self, model, n, temp=None, dmg=None, guidance=2.0):
        """Full generation from pure noise (augmentation)."""
        model.eval()
        x = torch.randn(n, N_PATHS, SEQ_LEN, device=self.device)
        return self._ancestral(model, x, temp, dmg, self.T, guidance)

    def _ancestral(self, model, x, temp, dmg, t_start, guidance):
        for t in range(t_start - 1, -1, -1):
            tb   = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            ab   = self.alpha_bar[t]
            ab_p = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            eps  = self._eps_cfg(model, x, tb, temp, dmg, guidance)
            x0h  = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-4, 4)
            mean = ab_p.sqrt() * x0h + (1 - ab_p).sqrt() * eps
            x    = mean + (1 - ab_p).sqrt() * torch.randn_like(x) if t > 0 else mean
        return x


# ─── Flow Matching (rectified flow) ──────────────────────────────────────────
class FlowMatching:
    """
    Conditional OT flow matching / rectified flow on the same UNet.
      x_t = (1-t)·x0 + t·ε ,  target velocity v = ε − x0 ,  t ~ U[0,1]
    The UNet's sinusoidal embedding expects large t, so t is scaled by T_MAX.
    Sampling = Euler ODE from t=1 (noise) → t=0 (data); deterministic, ~10x
    fewer steps than DDPM ancestral sampling.
    """
    def __init__(self, device=DEVICE, n_steps=100):
        self.device  = device
        self.n_steps = n_steps

    def loss(self, model, x0, temp=None, dmg=None, p_uncond=0.1):
        B   = x0.shape[0]
        t   = torch.rand(B, device=self.device)
        eps = torch.randn_like(x0)
        xt  = (1 - t)[:, None, None] * x0 + t[:, None, None] * eps
        if dmg is not None and p_uncond > 0:
            drop = torch.rand(B, device=self.device) < p_uncond
            dmg  = torch.where(drop, torch.full_like(dmg, NULL_DMG), dmg)
        pred = model(xt, t * T_MAX, temp, dmg)
        return F.mse_loss(pred, eps - x0)

    def _v_cfg(self, model, x, t, temp, dmg, guidance):
        if guidance <= 0 or dmg is None:
            return model(x, t * T_MAX, temp, dmg)
        v_c = model(x, t * T_MAX, temp, dmg)
        v_u = model(x, t * T_MAX, temp, torch.full_like(dmg, NULL_DMG))
        return (1 + guidance) * v_c - guidance * v_u

    @torch.no_grad()
    def _integrate(self, model, x, temp, dmg, t_from, guidance):
        """Euler ODE: dx/dt = v, integrate t_from → 0."""
        model.eval()
        n  = max(2, int(self.n_steps * t_from))
        dt = t_from / n
        for i in range(n):
            t  = t_from - i * dt
            tb = torch.full((x.shape[0],), t, device=self.device)
            v  = self._v_cfg(model, x, tb, temp, dmg, guidance)
            x  = x - v * dt
        return x

    @torch.no_grad()
    def partial_denoise(self, model, x0, temp=None, dmg=None,
                        t_start=0.3, guidance=0.0):
        eps = torch.randn_like(x0)
        x   = (1 - t_start) * x0 + t_start * eps
        return self._integrate(model, x, temp, dmg, t_start, guidance)

    @torch.no_grad()
    def sample(self, model, n, temp=None, dmg=None, guidance=2.0):
        x = torch.randn(n, N_PATHS, SEQ_LEN, device=self.device)
        return self._integrate(model, x, temp, dmg, 1.0, guidance)


def make_sampler(objective: str):
    return FlowMatching(device=DEVICE) if objective == "fm" else DDPM(device=DEVICE)


def ckpt_name(args):
    base = "best" if not args.no_temp else "best_uncond"
    return f"{base}_fm.pt" if args.objective == "fm" else f"{base}.pt"


# ─── shared prep ─────────────────────────────────────────────────────────────
def prepare_data(args):
    gw_raw, tmp_raw, lbl = load_pickle_months(max_months=args.months)
    gw_ds = downsample(gw_raw)
    healthy = lbl == 0
    _, mu, sig = zscore_waveform(gw_ds[healthy])          # stats from healthy only
    gw_zs, _, _ = zscore_waveform(gw_ds, mu, sig)
    t_mu  = float(tmp_raw[healthy].mean())
    t_sig = float(tmp_raw[healthy].std() + 1e-8)
    tmp_norm = (tmp_raw - t_mu) / t_sig
    return gw_zs, tmp_norm, tmp_raw, lbl, dict(mu=mu, sig=sig, t_mu=t_mu, t_sig=t_sig)


# ─── training ────────────────────────────────────────────────────────────────
def train(args):
    print(f"[train] device={DEVICE}")
    gw_zs, tmp_norm, tmp_raw, lbl, stats = prepare_data(args)
    np.savez(f"{RESULTS_DIR}/norm_stats.npz", **stats)

    ds     = GWDataset(gw_zs, tmp_norm, lbl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    print(f"  training samples: {len(ds)} (dmg {lbl.sum()}, healthy {(lbl==0).sum()})")

    use_temp = not args.no_temp
    model    = UNet1D(base_ch=args.base_ch).to(DEVICE)
    sampler  = make_sampler(args.objective)
    optim    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    cname     = ckpt_name(args)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0
        for gw_b, tmp_b, dmg_b in loader:
            gw_b  = gw_b.to(DEVICE)
            tmp_b = tmp_b.to(DEVICE) if use_temp else None
            dmg_b = dmg_b.to(DEVICE)
            loss  = sampler.loss(model, gw_b, tmp_b, dmg_b)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item()
        sched.step()
        avg = total / len(loader)
        if epoch % 5 == 0:
            print(f"  epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}", flush=True)
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg,
                        "use_temp": use_temp, "base_ch": args.base_ch,
                        "objective": args.objective},
                       f"{RESULTS_DIR}/{cname}")
    print(f"[train] best loss={best_loss:.4f} → {RESULTS_DIR}/{cname}")


# ─── evaluation (anomaly detection) ──────────────────────────────────────────
def load_model(args):
    ckpt  = torch.load(args.ckpt, map_location=DEVICE)
    model = UNet1D(base_ch=ckpt.get("base_ch", args.base_ch)).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    sampler = make_sampler(ckpt.get("objective", args.objective))
    return model, ckpt.get("use_temp", True), sampler


def anomaly_scores(model, sampler, gw_zs, tmp_norm, use_temp, batch_size=256):
    """Score = MSE between input and its reconstruction under the HEALTHY condition."""
    scores = []
    for i in range(0, len(gw_zs), batch_size):
        x0  = torch.from_numpy(gw_zs[i:i+batch_size]).to(DEVICE)
        tmp = (torch.from_numpy(tmp_norm[i:i+batch_size].astype(np.float32)).to(DEVICE)
               if use_temp else None)
        dmg = torch.zeros(x0.shape[0], dtype=torch.long, device=DEVICE)  # condition: healthy
        xr  = sampler.partial_denoise(model, x0, tmp, dmg)
        scores.append((x0 - xr).pow(2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


def evaluate(args):
    model, use_temp, sampler = load_model(args)
    if args.no_temp:
        use_temp = False

    gw_zs, tmp_norm, tmp_raw, lbl, _ = prepare_data(args)
    sc = anomaly_scores(model, sampler, gw_zs, tmp_norm, use_temp)

    auroc  = roc_auc_score(lbl, sc)
    thresh = np.percentile(sc[lbl == 1], 10)   # recall ≈ 0.9
    fpr    = (sc[lbl == 0] >= thresh).mean()
    print(f"[eval] conditioned={use_temp}  AUROC={auroc:.4f}  FPR@rec0.9={fpr:.4f}")

    print("  temp-bin FPR breakdown (healthy only):")
    for lo in range(0, 35, 5):
        mask = (tmp_raw >= lo) & (tmp_raw < lo + 5) & (lbl == 0)
        if mask.sum() < 20:
            continue
        print(f"    [{lo:2d}-{lo+5:2d}°C]  n={mask.sum():<6d}  FPR={(sc[mask] >= thresh).mean():.3f}")

    plt.figure(figsize=(8, 4))
    plt.scatter(tmp_raw[lbl == 0], sc[lbl == 0], s=1, alpha=0.3, label="healthy")
    plt.scatter(tmp_raw[lbl == 1], sc[lbl == 1], s=4, alpha=0.6, c="red", label="damage")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Anomaly score")
    tag = "cond" if use_temp else "uncond"
    plt.title(f"DDPM anomaly score vs temperature  [{tag}, AUROC={auroc:.3f}]")
    plt.legend(markerscale=5); plt.tight_layout()
    out = f"{RESULTS_DIR}/score_vs_temp_{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved → {out}")
    return auroc, fpr


# ─── augmentation (synthetic damaged data) ───────────────────────────────────
def augment(args):
    """
    Generate synthetic DAMAGED waveforms across the real temperature range.
    Output: results/payload_ddpm/synthetic_damaged.npz
      gw   (n, 8, 256)  z-scored synthetic waveforms (same scale as training)
      temp (n,)         temperature in °C
      also synthetic_healthy.npz for distribution sanity checks
    """
    model, use_temp, sampler = load_model(args)
    stats = np.load(f"{RESULTS_DIR}/norm_stats.npz")
    t_mu, t_sig = float(stats["t_mu"]), float(stats["t_sig"])

    # sample temperatures uniformly over the observed range
    rng    = np.random.default_rng(0)
    temps  = rng.uniform(args.temp_lo, args.temp_hi, args.n_synth).astype(np.float32)
    t_norm = (temps - t_mu) / t_sig

    for dmg_class, fname in [(1, "synthetic_damaged"), (0, "synthetic_healthy")]:
        outs = []
        bs   = args.batch_size
        for i in range(0, args.n_synth, bs):
            n   = min(bs, args.n_synth - i)
            tmp = (torch.from_numpy(t_norm[i:i+n]).to(DEVICE) if use_temp else None)
            dmg = torch.full((n,), dmg_class, dtype=torch.long, device=DEVICE)
            x   = sampler.sample(model, n, tmp, dmg, guidance=args.guidance)
            outs.append(x.cpu().numpy())
            print(f"  [{fname}] {i+n}/{args.n_synth}", flush=True)
        gw_synth = np.concatenate(outs)
        np.savez(f"{RESULTS_DIR}/{fname}.npz",
                 gw=gw_synth, temp=temps,
                 mu=stats["mu"], sig=stats["sig"], guidance=args.guidance)
        print(f"  saved → {RESULTS_DIR}/{fname}.npz  shape={gw_synth.shape}")


# ─── entrypoint ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       default="train", choices=["train", "eval", "augment"])
    p.add_argument("--objective",  default="ddpm",  choices=["ddpm", "fm"],
                   help="ddpm = ancestral diffusion, fm = rectified flow matching")
    p.add_argument("--ckpt",       default=None)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--months",     type=int,   default=None, help="limit number of months loaded")
    p.add_argument("--no_temp",    action="store_true", help="disable temperature conditioning")
    p.add_argument("--n_synth",    type=int,   default=2000, help="synthetic samples per class")
    p.add_argument("--guidance",   type=float, default=2.0,  help="CFG scale for augmentation")
    p.add_argument("--temp_lo",    type=float, default=0.0)
    p.add_argument("--temp_hi",    type=float, default=30.0)
    args = p.parse_args()
    if args.ckpt is None:
        args.ckpt = f"{RESULTS_DIR}/{ckpt_name(args)}"

    if   args.mode == "train":   train(args)
    elif args.mode == "eval":    evaluate(args)
    elif args.mode == "augment": augment(args)


if __name__ == "__main__":
    main()
