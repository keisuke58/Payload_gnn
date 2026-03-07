# -*- coding: utf-8 -*-
"""
Foundation Model vs From-Scratch 比較実験

Step 1: 500件全データで自己教師あり事前学習
Step 2: {50, 100, 200, 400} 件で fine-tune (事前学習あり / なし)
Step 3: 結果を比較して可視化

Usage:
    python scripts/run_foundation_experiment.py --data_dir data/processed_s12_thermal_500
    python scripts/run_foundation_experiment.py --data_dir data/processed_s12_thermal_500 --gpu 0
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print("\n" + "=" * 60)
    print("[CMD] %s" % desc)
    print("  %s" % cmd)
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    print("[DONE] %s (%.1f sec, exit=%d)" % (desc, elapsed, result.returncode))
    if result.returncode != 0:
        print("[ERROR] Command failed!")
        sys.exit(1)
    return result


def subsample_data(train_data, val_data, n_train):
    """Subsample training data to n_train graphs, keep val intact."""
    if n_train >= len(train_data):
        return train_data, val_data
    np.random.seed(42)
    indices = np.random.permutation(len(train_data))[:n_train]
    sub_train = [train_data[i] for i in indices]
    return sub_train, val_data


def save_subset(train_sub, val_data, out_dir):
    """Save subsampled dataset for train.py."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train_sub, os.path.join(out_dir, 'train.pt'))
    torch.save(val_data, os.path.join(out_dir, 'val.pt'))
    # Copy norm_stats if exists
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Foundation model comparison experiment')
    parser.add_argument('--data_dir', type=str,
                        default='data/processed_s12_thermal_500')
    parser.add_argument('--output_dir', type=str, default='runs/foundation_experiment')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')

    # Pre-training config
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--mask_ratio', type=float, default=0.20)

    # Fine-tuning config
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--train_sizes', type=str, default='50,100,200,400',
                        help='Comma-separated list of training set sizes')

    # Model config (shared)
    parser.add_argument('--arch', type=str, default='gat')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)

    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pre-training if encoder already exists')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    exp_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(exp_dir, exist_ok=True)

    train_sizes = [int(s) for s in args.train_sizes.split(',')]

    # Save experiment config
    config = vars(args)
    config['train_sizes'] = train_sizes
    config['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(exp_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Foundation Model Comparison Experiment")
    print("=" * 60)
    print("Data: %s" % args.data_dir)
    print("Train sizes: %s" % train_sizes)
    print("Pre-train epochs: %d" % args.pretrain_epochs)
    print("Fine-tune epochs: %d" % args.finetune_epochs)
    print("Output: %s" % exp_dir)

    # ================================================================
    # Step 0: Prepare subsampled datasets
    # ================================================================
    print("\n\n>>> Step 0: Preparing subsampled datasets...")
    data_dir_abs = (os.path.join(PROJECT_ROOT, args.data_dir)
                    if not os.path.isabs(args.data_dir) else args.data_dir)
    train_data = torch.load(os.path.join(data_dir_abs, 'train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(data_dir_abs, 'val.pt'), weights_only=False)
    print("Full dataset: %d train, %d val" % (len(train_data), len(val_data)))

    # Copy norm_stats to experiment dir
    norm_path = os.path.join(data_dir_abs, 'norm_stats.pt')
    subset_dirs = {}
    for n in train_sizes:
        sub_dir = os.path.join(exp_dir, 'data_n%d' % n)
        sub_train, sub_val = subsample_data(train_data, val_data, n)
        save_subset(sub_train, sub_val, sub_dir)
        if os.path.exists(norm_path):
            import shutil
            shutil.copy2(norm_path, os.path.join(sub_dir, 'norm_stats.pt'))
        subset_dirs[n] = sub_dir
        print("  n=%d: saved %d train + %d val to %s" % (
            n, len(sub_train), len(sub_val), sub_dir))

    # ================================================================
    # Step 1: Pre-train on ALL data
    # ================================================================
    pretrain_dir = os.path.join(exp_dir, 'pretrain')
    encoder_path = os.path.join(pretrain_dir, 'best_encoder.pt')

    if args.skip_pretrain and os.path.exists(encoder_path):
        print("\n\n>>> Step 1: SKIP (encoder exists: %s)" % encoder_path)
    else:
        print("\n\n>>> Step 1: Self-supervised pre-training on ALL %d+%d graphs..." % (
            len(train_data), len(val_data)))

        cmd = (
            "python src/pretrain_foundation.py"
            " --data_dir %s"
            " --output_dir %s"
            " --arch %s --hidden %d --layers %d"
            " --mask_ratio %.2f"
            " --epochs %d"
            " --batch_size 4"
            " --patience 20"
        ) % (args.data_dir, pretrain_dir, args.arch, args.hidden, args.layers,
             args.mask_ratio, args.pretrain_epochs)
        run_cmd(cmd, "Pre-train (mask_ratio=%.0f%%)" % (args.mask_ratio * 100))

        # Find the actual run directory (pretrain_gat_YYYYMMDD_HHMMSS)
        subdirs = sorted([d for d in os.listdir(pretrain_dir)
                          if d.startswith('pretrain_')],
                         key=lambda x: os.path.getmtime(os.path.join(pretrain_dir, x)))
        if subdirs:
            actual_dir = os.path.join(pretrain_dir, subdirs[-1])
            encoder_path = os.path.join(actual_dir, 'best_encoder.pt')
            print("Encoder: %s" % encoder_path)

    if not os.path.exists(encoder_path):
        print("[ERROR] Encoder not found: %s" % encoder_path)
        sys.exit(1)

    # ================================================================
    # Step 2: Fine-tune vs From-Scratch for each train size
    # ================================================================
    results = {}

    for n in train_sizes:
        print("\n\n>>> Step 2: Training with n=%d..." % n)
        sub_dir = subset_dirs[n]

        # --- (a) From scratch ---
        scratch_output = os.path.join(exp_dir, 'scratch_n%d' % n)
        os.makedirs(scratch_output, exist_ok=True)
        cmd_scratch = (
            "python src/train.py"
            " --data_dir %s"
            " --output_dir %s"
            " --arch %s --hidden %d --layers %d"
            " --epochs %d"
            " --batch_size 4"
            " --loss focal --focal_gamma 2.0"
            " --patience 30"
        ) % (sub_dir, scratch_output, args.arch, args.hidden, args.layers,
             args.finetune_epochs)
        run_cmd(cmd_scratch, "From-scratch n=%d" % n)

        # --- (b) Fine-tune from pre-trained ---
        finetune_output = os.path.join(exp_dir, 'finetune_n%d' % n)
        os.makedirs(finetune_output, exist_ok=True)
        cmd_finetune = (
            "python src/train.py"
            " --data_dir %s"
            " --output_dir %s"
            " --arch %s --hidden %d --layers %d"
            " --pretrained %s"
            " --freeze_layers 2"
            " --epochs %d"
            " --batch_size 4"
            " --loss focal --focal_gamma 2.0"
            " --patience 30"
        ) % (sub_dir, finetune_output, args.arch, args.hidden, args.layers,
             encoder_path, args.finetune_epochs)
        run_cmd(cmd_finetune, "Fine-tune n=%d (freeze 2 layers)" % n)

        # Collect results
        scratch_f1 = find_best_f1(scratch_output)
        finetune_f1 = find_best_f1(finetune_output)
        results[n] = {
            'scratch_f1': scratch_f1,
            'finetune_f1': finetune_f1,
            'improvement': finetune_f1 - scratch_f1,
        }
        print("\n  [n=%d] Scratch F1=%.4f | Fine-tune F1=%.4f | Δ=%+.4f" % (
            n, scratch_f1, finetune_f1, finetune_f1 - scratch_f1))

    # ================================================================
    # Step 3: Summary
    # ================================================================
    print("\n\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print("%-8s  %-12s  %-12s  %-10s" % ("N_train", "Scratch F1", "Finetune F1", "Δ F1"))
    print("-" * 45)
    for n in train_sizes:
        r = results[n]
        print("%-8d  %-12.4f  %-12.4f  %+.4f" % (
            n, r['scratch_f1'], r['finetune_f1'], r['improvement']))

    # Save results
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: %s" % results_path)

    # Generate comparison plot
    try:
        plot_results(results, train_sizes, exp_dir)
        print("Plot saved: %s" % os.path.join(exp_dir, 'foundation_comparison.png'))
    except Exception as e:
        print("Plot generation failed: %s" % e)


def find_best_f1(output_dir):
    """Find best F1 from any run in the output directory."""
    best_f1 = 0.0
    for entry in os.listdir(output_dir):
        ckpt_path = os.path.join(output_dir, entry, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            f1 = ckpt.get('val_f1', 0.0)
            if f1 > best_f1:
                best_f1 = f1
    return best_f1


def plot_results(results, train_sizes, exp_dir):
    """Generate comparison bar chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scratch_f1s = [results[n]['scratch_f1'] for n in train_sizes]
    finetune_f1s = [results[n]['finetune_f1'] for n in train_sizes]

    x = np.arange(len(train_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, scratch_f1s, width, label='From Scratch',
                   color='#4A90D9', alpha=0.85)
    bars2 = ax.bar(x + width / 2, finetune_f1s, width, label='Foundation + Fine-tune',
                   color='#E74C3C', alpha=0.85)

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Validation F1 Score', fontsize=12)
    ax.set_title('Foundation Model Pre-training Effect on Data Efficiency', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(train_sizes)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                '%.3f' % h, ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                '%.3f' % h, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'foundation_comparison.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
