# -*- coding: utf-8 -*-
"""
MAML (Model-Agnostic Meta-Learning) for FNO GW Surrogate

Few-shot adaptation to new defect types/configurations using MAML.
No external dependency (learn2learn/higher not required).

Meta-training strategy:
  - Tasks = defect types (7 types) or spatial regions
  - Support set: k samples per task (k-shot)
  - Query set: remaining samples per task
  - Inner loop: adapt model to support set (1-5 gradient steps)
  - Outer loop: optimize for generalization across tasks

Reference: Sawant et al. "Few-Shot Learning for GW Damage Detection"
           (NDT&E International, 2023)

Usage:
  python src/train_fno_maml.py --data_dir abaqus_work/gw_fairing_dataset \\
    --doe doe_gw_fairing.json --k_shot 3 --n_inner 3

  # With pre-trained backbone:
  python src/train_fno_maml.py --finetune_from runs/fno_gw/pretrained_model.pt \\
    --k_shot 3
"""

import argparse
import copy
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset_fno_gw import GWOperatorDataset
from dataset_fno_gw_augmented import AugmentedGWDataset
from models_fno_gw import FNOGWSurrogate, DualStageFNOSurrogate


# =========================================================================
# Lightweight MAML (no external dependency)
# =========================================================================
def clone_parameters(model):
    """Clone model parameters for inner loop (no computation graph)."""
    return {name: p.clone() for name, p in model.named_parameters()}


def functional_forward(model, params, x_params, x_healthy):
    """Forward pass using external parameter dict (for MAML inner loop).

    Instead of modifying model weights, we manually apply params
    via torch.nn.utils.stateless.functional_call (PyTorch 2.0+).
    """
    return torch.func.functional_call(model, params, (x_params, x_healthy))


def inner_loop(model, support_data, n_inner_steps, inner_lr, device):
    """MAML inner loop: adapt model to support set.

    Args:
        model: FNO model
        support_data: list of (params, healthy, target) tuples
        n_inner_steps: number of gradient steps
        inner_lr: learning rate for inner adaptation

    Returns:
        adapted_params: dict of adapted parameters
    """
    # Clone parameters
    adapted = {name: p.clone().requires_grad_(True)
                for name, p in model.named_parameters()}

    for step in range(n_inner_steps):
        total_loss = 0
        n = 0
        for batch in support_data:
            params_in = batch['params'].to(device)
            healthy = batch['healthy'].to(device)
            target = batch['target'].to(device)

            pred = functional_forward(model, adapted, params_in, healthy)

            # Relative L2 loss
            diff = (pred - target) ** 2
            norm = target ** 2 + 1e-8
            loss = torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))
            total_loss = total_loss + loss
            n += 1

        avg_loss = total_loss / max(n, 1)

        # Compute gradients w.r.t. adapted params
        grads = torch.autograd.grad(avg_loss, adapted.values(),
                                     create_graph=True)

        # SGD update
        adapted = {name: p - inner_lr * g
                   for (name, p), g in zip(adapted.items(), grads)}

    return adapted


def maml_outer_step(model, tasks, n_inner_steps, inner_lr,
                    outer_optimizer, device):
    """MAML outer loop: one meta-training step.

    Args:
        model: FNO model
        tasks: list of (support_data, query_data) task tuples
        n_inner_steps: inner loop steps per task
        inner_lr: inner loop learning rate
        outer_optimizer: optimizer for outer loop

    Returns:
        meta_loss: average loss across tasks
    """
    outer_optimizer.zero_grad()
    meta_loss = 0
    n_tasks = 0

    for support_data, query_data in tasks:
        # Inner loop: adapt to support set
        adapted_params = inner_loop(
            model, support_data, n_inner_steps, inner_lr, device)

        # Evaluate on query set with adapted parameters
        task_loss = 0
        n_q = 0
        for batch in query_data:
            params_in = batch['params'].to(device)
            healthy = batch['healthy'].to(device)
            target = batch['target'].to(device)

            pred = functional_forward(model, adapted_params, params_in, healthy)

            diff = (pred - target) ** 2
            norm = target ** 2 + 1e-8
            loss = torch.mean(diff.sum(dim=-1) / norm.sum(dim=-1))
            task_loss = task_loss + loss
            n_q += 1

        meta_loss = meta_loss + task_loss / max(n_q, 1)
        n_tasks += 1

    meta_loss = meta_loss / max(n_tasks, 1)
    meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    outer_optimizer.step()

    return meta_loss.item()


# =========================================================================
# Task construction
# =========================================================================
def build_tasks_by_defect_type(dataset, k_shot=3, batch_size=4):
    """Create meta-learning tasks split by defect type.

    Each defect type becomes one task. Within each task:
    - Support set: k_shot random samples
    - Query set: remaining samples
    """
    # Group samples by defect type
    type_indices = defaultdict(list)
    defect_types = getattr(dataset, 'DEFECT_TYPES', [])

    for idx in range(len(dataset)):
        sample = dataset[idx]
        params = sample['params'].numpy()

        # Determine defect type from one-hot encoding
        if len(params) > 3 and len(defect_types) > 0:
            type_oh = params[3:]
            type_idx = int(type_oh.argmax()) if type_oh.max() > 0 else 0
            type_name = defect_types[type_idx] if type_idx < len(defect_types) else f'type_{type_idx}'
        else:
            type_name = 'unknown'

        type_indices[type_name].append(idx)

    tasks = []
    for type_name, indices in type_indices.items():
        if len(indices) < k_shot + 1:
            continue  # need at least k_shot + 1 samples

        rng = np.random.RandomState(hash(type_name) % 2**31)
        rng.shuffle(indices)

        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:]

        support_ds = Subset(dataset, support_idx)
        query_ds = Subset(dataset, query_idx)

        support_loader = [
            {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
             for k, v in support_ds[i].items()}
            for i in range(len(support_ds))
        ]
        query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=True)

        tasks.append((support_loader, query_loader))
        print(f"  Task '{type_name}': {len(support_idx)} support, "
              f"{len(query_idx)} query")

    return tasks


def build_tasks_by_region(dataset, k_shot=3, n_regions=4, batch_size=4):
    """Create tasks by spatial region (z-axis quadrants)."""
    region_indices = defaultdict(list)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        z_norm = sample['params'][0].item()  # normalized z
        region = min(int(z_norm * n_regions), n_regions - 1)
        region_indices[f'region_{region}'].append(idx)

    tasks = []
    for region_name, indices in region_indices.items():
        if len(indices) < k_shot + 1:
            continue

        rng = np.random.RandomState(hash(region_name) % 2**31)
        rng.shuffle(indices)

        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:]

        support_ds = Subset(dataset, support_idx)
        query_ds = Subset(dataset, query_idx)

        support_loader = [
            {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
             for k, v in support_ds[i].items()}
            for i in range(len(support_ds))
        ]
        query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=True)

        tasks.append((support_loader, query_loader))
        print(f"  Task '{region_name}': {len(support_idx)} support, "
              f"{len(query_idx)} query")

    return tasks


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='MAML Meta-Learning for FNO GW Surrogate')
    parser.add_argument('--data_dir', default='abaqus_work/gw_fairing_dataset')
    parser.add_argument('--doe', default='doe_gw_fairing.json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--model', choices=['fno', 'dual_fno'], default='fno')

    # MAML
    parser.add_argument('--k_shot', type=int, default=3,
                        help='Support set size per task')
    parser.add_argument('--n_inner', type=int, default=3,
                        help='Inner loop gradient steps')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-4,
                        help='Outer loop (meta) learning rate')
    parser.add_argument('--meta_epochs', type=int, default=200,
                        help='Meta-training epochs')
    parser.add_argument('--task_split', choices=['defect_type', 'region'],
                        default='defect_type',
                        help='How to split data into tasks')

    # Model
    parser.add_argument('--modes', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--no_residual', dest='residual', action='store_false')
    parser.add_argument('--downsample', type=int, default=4)
    parser.add_argument('--max_timesteps', type=int, default=None)

    # Transfer
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Initialize from pre-trained checkpoint')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== MAML Meta-Learning for FNO GW ===")
    print(f"Device: {device}")

    # Dataset
    ds = GWOperatorDataset(
        args.data_dir, args.doe,
        residual=args.residual,
        downsample=args.downsample,
        max_timesteps=args.max_timesteps,
    )
    meta = ds.get_metadata()
    print(f"Dataset: {len(ds)} samples, {meta['n_sensors']} sensors, T={meta['T']}")

    if len(ds) < 4:
        print("ERROR: Need at least 4 samples for meta-learning.")
        return

    # Build tasks
    print(f"\nBuilding tasks ({args.task_split})...")
    if args.task_split == 'defect_type':
        tasks = build_tasks_by_defect_type(ds, k_shot=args.k_shot,
                                            batch_size=args.batch_size)
    else:
        tasks = build_tasks_by_region(ds, k_shot=args.k_shot,
                                       batch_size=args.batch_size)

    if len(tasks) < 2:
        print(f"WARNING: Only {len(tasks)} tasks. "
              "Meta-learning works best with 3+ tasks.")
        if len(tasks) == 0:
            print("ERROR: No valid tasks. Need more data or different split.")
            return

    # Model
    n_params = meta.get('n_params', 3)
    if args.model == 'dual_fno':
        model = DualStageFNOSurrogate(
            n_sensors=meta['n_sensors'], n_params=n_params,
            modes_low=max(args.modes // 4, 8), modes_high=args.modes,
            width=args.width,
            n_layers_low=max(args.n_layers // 2, 2),
            n_layers_high=args.n_layers,
            residual=args.residual,
        ).to(device)
    else:
        model = FNOGWSurrogate(
            n_sensors=meta['n_sensors'], n_params=n_params,
            modes=args.modes, width=args.width,
            n_layers=args.n_layers, residual=args.residual,
        ).to(device)

    # Load pre-trained weights if available
    if args.finetune_from:
        ckpt = torch.load(args.finetune_from, map_location=device,
                          weights_only=False)
        state = ckpt['model_state_dict']
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"Loaded {loaded} layers from {args.finetune_from}")

    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, {n_params_model:,} params")

    # Outer optimizer
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

    if args.output is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.output = f'runs/maml_{args.model}_{ts}'
    os.makedirs(args.output, exist_ok=True)

    # Meta-training
    print(f"\nMAML training: {args.meta_epochs} epochs, "
          f"{args.k_shot}-shot, {args.n_inner} inner steps")
    print(f"  Inner LR: {args.inner_lr}, Outer LR: {args.outer_lr}")
    print(f"  Tasks: {len(tasks)}")
    print()

    best_meta_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, args.meta_epochs + 1):
        # Shuffle task order
        task_order = np.random.permutation(len(tasks))
        epoch_tasks = [tasks[i] for i in task_order]

        meta_loss = maml_outer_step(
            model, epoch_tasks, args.n_inner, args.inner_lr,
            outer_optimizer, device)

        if meta_loss < best_meta_loss:
            best_meta_loss = meta_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'meta': meta,
                'args': vars(args),
                'epoch': epoch,
                'meta_loss': meta_loss,
                'maml': True,
            }, os.path.join(args.output, 'best_maml_model.pt'))

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Ep {epoch:4d}/{args.meta_epochs} | "
                  f"Meta Loss: {meta_loss:.6f} | "
                  f"Best: {best_meta_loss:.6f} | {elapsed:.0f}s")

    print(f"\nBest meta loss: {best_meta_loss:.6f}")
    print(f"Model saved: {args.output}/best_maml_model.pt")

    # Save results
    results = {
        'args': vars(args),
        'best_meta_loss': best_meta_loss,
        'n_tasks': len(tasks),
        'n_params': n_params_model,
        'total_time_s': time.time() - t0,
    }
    with open(os.path.join(args.output, 'maml_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
