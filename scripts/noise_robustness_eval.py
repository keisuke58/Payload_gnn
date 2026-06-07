#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_robustness_eval.py — Evaluate GNN models under additive Gaussian feature noise.

Hypothesis: PINN-regularised models degrade more gracefully than unregularised
            baselines, because physics constraints anchor predictions to the
            physically plausible solution manifold even with noisy inputs.

Protocol:
  - Load trained checkpoint + ID validation split (norm_stats from training)
  - Add zero-mean Gaussian noise at σ_rel ∈ {0, 1%, 3%, 10%, 30%} of feature std
  - Report OptF1, AUC, macro_F1 for each noise level
  - Repeat with N_SEEDS independent noise realisations for error bars

Usage:
  python scripts/noise_robustness_eval.py \\
    --runs runs/pinn_l12_lgsta_nodecls runs/pinn_l12_gcn_mt runs/pinn_l12_gin_mt \\
    --data_dir data/processed_s12_thermal_700_5class \\
    --out results/noise_robustness.json

  python scripts/noise_robustness_eval.py --all_runs  # auto-discover pinn_l12_* runs
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

NOISE_LEVELS = [0.0, 0.01, 0.03, 0.10, 0.30]   # σ_rel = σ_noise / σ_feature
N_SEEDS      = 5                                  # realisations per noise level


# ── helpers ──────────────────────────────────────────────────────────────────

def threshold_opt_f1(labels, probs):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 37):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def find_best_ckpt(run_dir):
    for cand in [
        os.path.join(run_dir, 'best_model.pt'),
        *sorted(glob.glob(os.path.join(run_dir, '*', 'best_model.pt'))),
    ]:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'No best_model.pt in {run_dir}')


def load_model(run_dir, device):
    from models import build_model
    ckpt_path = find_best_ckpt(run_dir)
    ckpt      = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state     = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    a         = ckpt.get('args', {})

    model = build_model(
        a.get('arch', 'lgsta'),
        ckpt.get('in_channels', 34),
        edge_attr_dim  = ckpt.get('edge_attr_dim', 0),
        task           = a.get('task', 'multitask'),
        hidden_channels= a.get('hidden', 128),
        num_layers     = a.get('layers', 4),
        dropout        = a.get('dropout', 0.1),
        num_classes    = 5,
        use_residual   = a.get('residual', False),
        reg_out_dim    = a.get('reg_out_dim', 3),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    run_name = os.path.basename(run_dir)
    arch     = a.get('arch', run_name)
    task     = a.get('task', '?')
    ls       = a.get('physics_lambda_stress', 0.0)
    le       = a.get('physics_lambda_eq', 0.0)
    label    = f"{arch}({'PINN' if (ls>0 or le>0) else 'no-PINN'})"
    print(f'  Loaded: {run_name} → {label}  ({sum(p.numel() for p in model.parameters())} params)')
    return model, label, ckpt


@torch.no_grad()
def evaluate_with_noise(model, val_list, norm_mean, norm_std, device,
                        sigma_rel=0.0, seed=0):
    """Evaluate on val_list with additive Gaussian noise σ = sigma_rel * norm_std."""
    rng = np.random.default_rng(seed)

    all_labels, all_probs = [], []

    loader = DataLoader(val_list, batch_size=1, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        # Normalise
        x = (batch.x - norm_mean) / norm_std
        # Add noise in normalised space (σ_rel * 1 because std=1 after normalisation)
        if sigma_rel > 0.0:
            noise = torch.tensor(
                rng.normal(0.0, sigma_rel, x.shape).astype(np.float32),
                device=device,
            )
            x = x + noise
        batch.x = x

        try:
            out = model(batch)
        except TypeError:
            out = model(batch.x, batch.edge_index,
                        getattr(batch, 'edge_attr', None),
                        getattr(batch, 'batch', None))

        logits = out[0] if isinstance(out, tuple) else out
        probs  = F.softmax(logits, dim=1).cpu().numpy()

        all_labels.extend(batch.y.cpu().numpy())
        all_probs.append(probs)

    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)
    def_labels = (all_labels > 0).astype(int)
    prob_def   = 1.0 - all_probs[:, 0]

    opt_f1, _ = threshold_opt_f1(def_labels, prob_def)
    try:
        auc = float(roc_auc_score(def_labels, prob_def))
    except ValueError:
        auc = float('nan')
    macro_f1 = float(f1_score(all_labels, all_probs.argmax(1), average='macro', zero_division=0))

    return {'opt_f1': opt_f1, 'auc': auc, 'macro_f1': macro_f1}


def evaluate_run(run_dir, val_list, device):
    model, label, ckpt = load_model(run_dir, device)

    # Load norm_stats from original data dir (stored in checkpoint)
    a        = ckpt.get('args', {})
    data_dir = a.get('data_dir', 'data/processed_s12_thermal_700_5class')
    ns_path  = os.path.join(PROJECT_ROOT, data_dir, 'norm_stats.pt')
    if not os.path.exists(ns_path):
        # Fallback: compute from val_list
        x_cat = torch.cat([d.x for d in val_list], dim=0)
        norm_mean = x_cat.mean(0).to(device)
        norm_std  = x_cat.std(0).clamp(min=1e-8).to(device)
    else:
        ns = torch.load(ns_path, map_location='cpu', weights_only=False)
        norm_mean = ns['mean'].to(device)
        norm_std  = ns['std'].to(device)

    results = {'run': os.path.basename(run_dir), 'label': label, 'noise_curve': []}

    for sigma in NOISE_LEVELS:
        seed_results = []
        n_seeds = 1 if sigma == 0.0 else N_SEEDS
        for seed in range(n_seeds):
            m = evaluate_with_noise(model, val_list, norm_mean, norm_std,
                                    device, sigma_rel=sigma, seed=seed)
            seed_results.append(m)

        entry = {
            'sigma_rel': sigma,
            'sigma_pct': sigma * 100,
            'opt_f1_mean':  float(np.mean([r['opt_f1']    for r in seed_results])),
            'opt_f1_std':   float(np.std( [r['opt_f1']    for r in seed_results])),
            'auc_mean':     float(np.mean([r['auc']        for r in seed_results])),
            'macro_f1_mean':float(np.mean([r['macro_f1']  for r in seed_results])),
        }
        results['noise_curve'].append(entry)
        print(f'    σ={sigma*100:4.0f}%  OptF1={entry["opt_f1_mean"]:.4f}±{entry["opt_f1_std"]:.4f}'
              f'  AUC={entry["auc_mean"]:.4f}')

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+',
                        help='Run directories to evaluate')
    parser.add_argument('--all_runs', action='store_true',
                        help='Auto-discover all runs/pinn_l12_* directories')
    parser.add_argument('--data_dir',
                        default='data/processed_s12_thermal_700_5class',
                        help='Original dataset directory (for val.pt)')
    parser.add_argument('--out', default='results/noise_robustness.json')
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Discover runs
    if args.all_runs:
        run_dirs = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'runs', 'pinn_l12_*')))
    else:
        run_dirs = [os.path.join(PROJECT_ROOT, r) if not os.path.isabs(r) else r
                    for r in (args.runs or [])]
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    print(f'Evaluating {len(run_dirs)} runs.')

    # Load validation data (original ID split)
    data_dir  = os.path.join(PROJECT_ROOT, args.data_dir)
    val_list  = torch.load(os.path.join(data_dir, 'val.pt'),
                           map_location='cpu', weights_only=False)
    print(f'Val set: {len(val_list)} samples')

    all_results = []
    for run_dir in run_dirs:
        print(f'\n[{os.path.basename(run_dir)}]')
        try:
            r = evaluate_run(run_dir, val_list, device)
            all_results.append(r)
        except Exception as e:
            print(f'  SKIP: {e}')

    # Save JSON
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Print summary table
    print('\n' + '='*80)
    print(f'{"Run":<35} {"σ=0%":>8} {"σ=1%":>8} {"σ=3%":>8} {"σ=10%":>9} {"σ=30%":>9}')
    print('-'*80)
    for r in all_results:
        curve = r['noise_curve']
        vals  = [f'{e["opt_f1_mean"]:.4f}' for e in curve]
        print(f'{r["label"]:<35} ' + '  '.join(f'{v:>7}' for v in vals))


if __name__ == '__main__':
    main()
