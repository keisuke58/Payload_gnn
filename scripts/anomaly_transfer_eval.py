#!/usr/bin/env python3
"""
Experiment E: does ANOMALY DETECTION transfer where supervised classification
collapses?

Diagnosis (A-D) showed cross-month supervised transfer is broken at task level
(real B: AUC 0.485 ≈ synth C: 0.479; within-month ceiling A: 0.999) because
damage states change between months. Anomaly detection depends only on the
HEALTHY manifold — which the conditional FM/DDPM learned across seasons — so
it should survive damage-state changes.

Score = reconstruction error under (temperature, dmg=0) condition.
Test months: 2019_10 / 2021_05 (mixed within-month, unseen by generator).

Run (frontale, CPU): python3 scripts/anomaly_transfer_eval.py --objective fm
"""
import os, sys, argparse, pickle
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from payload_diffusion import (UNet1D, DDPM, FlowMatching, downsample,
                               zscore_waveform, RESULTS_DIR, LT_DIR, DEVICE)
from sklearn.metrics import roc_auc_score, average_precision_score

TEST_MONTHS = ["2019_10", "2021_05"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", default="fm", choices=["fm", "ddpm"])
    p.add_argument("--batch",     type=int, default=256)
    p.add_argument("--t_frac",    type=float, default=0.3, help="FM noise depth fraction")
    p.add_argument("--n_per_month", type=int, default=4000, help="subsample per month (CPU budget)")
    args = p.parse_args()

    ckpt_file = f"{RESULTS_DIR}/best_fm.pt" if args.objective == "fm" else f"{RESULTS_DIR}/best.pt"
    ckpt  = torch.load(ckpt_file, map_location=DEVICE)
    model = UNet1D(base_ch=ckpt.get("base_ch", 32)).to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    sampler = (FlowMatching(device=DEVICE) if args.objective == "fm"
               else DDPM(device=DEVICE))
    print(f"[E] objective={args.objective} ckpt={ckpt_file} (loss {ckpt['loss']:.4f})")

    stats = np.load(f"{RESULTS_DIR}/norm_stats.npz")
    mu, sig = stats["mu"], stats["sig"]
    t_mu, t_sig = float(stats["t_mu"]), float(stats["t_sig"])
    rng = np.random.default_rng(0)

    for month in TEST_MONTHS:
        d   = pickle.load(open(os.path.join(LT_DIR, f"measurements_{month}.pickle"), "rb"))
        y   = (np.asarray(d["damage tag"]) > 0).astype(int)
        idx = rng.choice(len(y), min(args.n_per_month, len(y)), replace=False)
        gw  = downsample(d["guided wave"][idx].astype(np.float32))
        gw  = (gw - mu) / sig
        tmp = (d["temperature"][idx].astype(np.float32) - t_mu) / t_sig
        y   = y[idx]

        scores = []
        with torch.no_grad():
            for i in range(0, len(gw), args.batch):
                x0 = torch.from_numpy(gw[i:i+args.batch]).to(DEVICE)
                tt = torch.from_numpy(tmp[i:i+args.batch]).to(DEVICE)
                dz = torch.zeros(x0.shape[0], dtype=torch.long, device=DEVICE)
                if args.objective == "fm":
                    xr = sampler.partial_denoise(model, x0, tt, dz, t_start=args.t_frac)
                else:
                    xr = sampler.partial_denoise(model, x0, tt, dz)
                scores.append((x0 - xr).pow(2).mean(dim=(1, 2)).cpu().numpy())
        sc = np.concatenate(scores)

        auc = roc_auc_score(y, sc)
        aup = average_precision_score(y, sc)
        thr = np.percentile(sc[y == 1], 10)            # recall ≈ 0.9
        fpr = (sc[y == 0] >= thr).mean()
        print(f"  [{month}] n={len(y)} dmg={y.mean():.2f}  "
              f"AUC={auc:.4f}  AUPRC={aup:.4f}  FPR@rec0.9={fpr:.4f}")

    print("\nCompare: supervised cross-month B=0.485 / synth C=0.479. "
          "Anomaly AUC >> 0.5 ⇒ healthy-manifold scoring transfers.")


if __name__ == "__main__":
    main()
