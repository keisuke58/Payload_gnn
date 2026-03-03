# -*- coding: utf-8 -*-
"""run_two_stage_pipeline.py — End-to-end FNO screening + GNN integration test.

Validates the two-stage screening concept:
  Phase 1: Train FNO surrogate (or load existing checkpoint)
  Phase 2: MC-Dropout screening → needs_fem / can_skip
  Phase 3: Validate screening safety (no missed severe defects)
  Phase 4: Train GNN on screened subset vs. full dataset
  Phase 5: Comparison report

Usage:
  # Full pipeline (GPU recommended):
  python src/run_two_stage_pipeline.py --fno_epochs 500

  # Skip FNO training (use existing checkpoint):
  python src/run_two_stage_pipeline.py --fno_checkpoint runs/fno_production/best_model.pt

  # Quick test (CPU, fewer epochs):
  python src/run_two_stage_pipeline.py --fno_epochs 10 --gnn_epochs 20 --quick
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print("\n" + "=" * 60)
    print("PHASE: %s" % desc)
    print("CMD: %s" % cmd)
    print("=" * 60)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    print("[%s] Done in %.1fs (exit=%d)" % (desc, elapsed, ret.returncode))
    if ret.returncode != 0:
        print("ERROR: %s failed" % desc)
    return ret.returncode, elapsed


def recover_split_mapping(n_samples=200, seed=42, val_ratio=0.2):
    """Recover the train/val split mapping used by prepare_ml_data.py.

    Returns:
        sample_to_split: dict mapping sample_index (0-199) to ('train', local_idx)
        train_global_indices: list of global indices that went to train
        val_global_indices: list of global indices that went to val
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    n_val = int(n_samples * val_ratio)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    sample_to_split = {}
    for local_i, global_i in enumerate(train_idx):
        sample_to_split[global_i] = ("train", local_i)
    for local_i, global_i in enumerate(val_idx):
        sample_to_split[global_i] = ("val", local_i)

    return sample_to_split, train_idx, val_idx


def validate_screening(screening_path, fno_meta_path, pyg_dir,
                       split_seed=42, val_ratio=0.2):
    """Validate screening decisions against actual FEM data.

    Checks that "can_skip" cases truly have low defect severity.
    """
    with open(screening_path) as f:
        screening = json.load(f)
    with open(fno_meta_path) as f:
        fno_meta = json.load(f)

    n_samples = fno_meta["n_samples"]
    sample_names = [s["sample"] for s in fno_meta["samples"]]

    # Recover train/val mapping
    mapping, train_idx, val_idx = recover_split_mapping(
        n_samples, split_seed, val_ratio)

    # Load actual PyG data
    train_data = torch.load(os.path.join(pyg_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(pyg_dir, "val.pt"), weights_only=False)

    # Analyze each case
    results = []
    skipped_severe = []
    max_defect_pct_skipped = 0.0

    for i, case in enumerate(screening["cases"]):
        needs_fem = case["needs_fem"]
        split, local_idx = mapping[i]
        graph = train_data[local_idx] if split == "train" else val_data[local_idx]

        n_nodes = graph.y.shape[0]
        n_defect = (graph.y > 0).sum().item()
        defect_pct = 100.0 * n_defect / max(n_nodes, 1)

        # Defect severity based on stress (smises = dim 18)
        if n_defect > 0:
            defect_mask = graph.y > 0
            stress = graph.x[:, 18]  # smises
            defect_stress_mean = stress[defect_mask].mean().item()
            healthy_stress_mean = stress[~defect_mask].mean().item()
            stress_diff = abs(defect_stress_mean - healthy_stress_mean)
        else:
            stress_diff = 0.0

        info = {
            "sample": case["sample"],
            "needs_fem": needs_fem,
            "split": split,
            "n_defect": n_defect,
            "defect_pct": defect_pct,
            "fno_score": case["score"],
            "fno_unc": case["uncertainty"],
            "actual_stress_diff": stress_diff,
        }
        results.append(info)

        if not needs_fem:
            max_defect_pct_skipped = max(max_defect_pct_skipped, defect_pct)
            # Flag if skipped case has significant defects
            if defect_pct > 1.0 or stress_diff > 10.0:
                skipped_severe.append(info)

    # Summary statistics
    fem_cases = [r for r in results if r["needs_fem"]]
    skip_cases = [r for r in results if not r["needs_fem"]]

    summary = {
        "total": len(results),
        "needs_fem": len(fem_cases),
        "can_skip": len(skip_cases),
        "reduction_pct": 100.0 * len(skip_cases) / max(len(results), 1),
        "skipped_severe_count": len(skipped_severe),
        "max_defect_pct_skipped": max_defect_pct_skipped,
        "avg_defect_pct_fem": np.mean([r["defect_pct"] for r in fem_cases]) if fem_cases else 0,
        "avg_defect_pct_skip": np.mean([r["defect_pct"] for r in skip_cases]) if skip_cases else 0,
        "avg_stress_diff_fem": np.mean([r["actual_stress_diff"] for r in fem_cases]) if fem_cases else 0,
        "avg_stress_diff_skip": np.mean([r["actual_stress_diff"] for r in skip_cases]) if skip_cases else 0,
    }

    # Per-split breakdown
    for split_name in ["train", "val"]:
        split_fem = [r for r in results if r["split"] == split_name and r["needs_fem"]]
        split_skip = [r for r in results if r["split"] == split_name and not r["needs_fem"]]
        summary["%s_needs_fem" % split_name] = len(split_fem)
        summary["%s_can_skip" % split_name] = len(split_skip)

    return summary, results, skipped_severe, mapping, train_idx, val_idx


def create_filtered_dataset(pyg_dir, screening_path, fno_meta_path, output_dir,
                            split_seed=42, val_ratio=0.2):
    """Create filtered PyG dataset containing only 'needs_fem' cases."""
    with open(screening_path) as f:
        screening = json.load(f)

    mapping, train_idx, val_idx = recover_split_mapping(
        len(screening["cases"]), split_seed, val_ratio)

    # Determine which global indices are kept (needs_fem=True)
    kept_global = set()
    for i, case in enumerate(screening["cases"]):
        if case["needs_fem"]:
            kept_global.add(i)

    # Load original data
    train_data = torch.load(os.path.join(pyg_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(pyg_dir, "val.pt"), weights_only=False)

    # Filter: keep only graphs whose global index is in kept_global
    filtered_train = []
    for local_i, global_i in enumerate(sorted(train_idx)):
        # train_idx is in permuted order; local_i maps to position in train_data
        pass

    # Correct approach: train_idx[local_i] = global_i
    filtered_train = []
    for local_i in range(len(train_data)):
        global_i = train_idx[local_i]
        if global_i in kept_global:
            filtered_train.append(train_data[local_i])

    filtered_val = []
    for local_i in range(len(val_data)):
        global_i = val_idx[local_i]
        if global_i in kept_global:
            filtered_val.append(val_data[local_i])

    os.makedirs(output_dir, exist_ok=True)
    torch.save(filtered_train, os.path.join(output_dir, "train.pt"))
    torch.save(filtered_val, os.path.join(output_dir, "val.pt"))

    # Copy norm_stats from original
    norm_src = os.path.join(pyg_dir, "norm_stats.pt")
    if os.path.exists(norm_src):
        import shutil
        shutil.copy2(norm_src, os.path.join(output_dir, "norm_stats.pt"))

    print("Filtered dataset: train=%d (was %d), val=%d (was %d)" % (
        len(filtered_train), len(train_data),
        len(filtered_val), len(val_data)))

    return len(filtered_train), len(filtered_val)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage FNO screening + GNN integration pipeline")

    # FNO args
    parser.add_argument("--fno_data", default="data/fno_grids_200")
    parser.add_argument("--fno_epochs", type=int, default=500)
    parser.add_argument("--fno_checkpoint", type=str, default=None,
                        help="Skip FNO training, use existing checkpoint")
    parser.add_argument("--fno_run_name", default="fno_production")

    # Screening args
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--k_margin", type=float, default=1.5)
    parser.add_argument("--unc_percentile", type=float, default=70)

    # GNN args
    parser.add_argument("--pyg_dir", default="data/processed_s12_czm_thermal_200_binary")
    parser.add_argument("--gnn_arch", default="gat")
    parser.add_argument("--gnn_epochs", type=int, default=200)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=4)
    parser.add_argument("--gnn_batch", type=int, default=2)
    parser.add_argument("--gnn_lr", type=float, default=1e-3)
    parser.add_argument("--defect_weight", type=float, default=3.0)

    # General
    parser.add_argument("--output_dir", default="runs/two_stage_pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (fewer epochs)")
    parser.add_argument("--skip_gnn", action="store_true",
                        help="Skip GNN training (only FNO + screening)")
    args = parser.parse_args()

    if args.quick:
        args.fno_epochs = min(args.fno_epochs, 20)
        args.gnn_epochs = min(args.gnn_epochs, 20)
        args.mc_samples = min(args.mc_samples, 5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    timings = {}
    t_start = time.time()

    # =====================================================================
    # Phase 1: FNO Training
    # =====================================================================
    if args.fno_checkpoint:
        fno_ckpt = args.fno_checkpoint
        print("\n[Phase 1] Using existing FNO checkpoint: %s" % fno_ckpt)
        timings["fno_train"] = 0.0
    else:
        fno_run = "runs/%s" % args.fno_run_name
        fno_ckpt = os.path.join(fno_run, "best_model.pt")

        cmd = (
            "python src/train_fno.py"
            " --data_dir %s"
            " --epochs %d"
            " --run_name %s"
            " --batch_size 16 --lr 1e-3 --modes 12 --width 32"
            " --patience 50 --seed %d"
        ) % (args.fno_data, args.fno_epochs, args.fno_run_name, args.seed)

        rc, elapsed = run_cmd(cmd, "FNO Training (%d epochs)" % args.fno_epochs)
        timings["fno_train"] = elapsed
        if rc != 0:
            print("FATAL: FNO training failed")
            return

    if not os.path.exists(fno_ckpt):
        print("FATAL: FNO checkpoint not found: %s" % fno_ckpt)
        return

    # =====================================================================
    # Phase 2: FNO Screening
    # =====================================================================
    screening_output = os.path.join(run_dir, "screening_results.json")

    cmd = (
        "python src/screen_cases.py"
        " --checkpoint %s"
        " --data_dir %s"
        " --mc_samples %d --dropout_p %.2f"
        " --k_margin %.2f --unc_percentile %.0f"
        " --output %s"
    ) % (fno_ckpt, args.fno_data, args.mc_samples, args.dropout_p,
         args.k_margin, args.unc_percentile, screening_output)

    rc, elapsed = run_cmd(cmd, "FNO Screening (MC-Dropout T=%d)" % args.mc_samples)
    timings["screening"] = elapsed
    if rc != 0:
        print("FATAL: Screening failed")
        return

    # =====================================================================
    # Phase 3: Screening Validation
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE: Screening Validation")
    print("=" * 60)

    fno_meta = os.path.join(args.fno_data, "meta.json")
    summary, case_details, skipped_severe, mapping, train_idx, val_idx = \
        validate_screening(screening_output, fno_meta, args.pyg_dir,
                           split_seed=args.seed)

    print("\n--- Screening Validation ---")
    print("Total cases:       %d" % summary["total"])
    print("Needs FEM:         %d (%.0f%%)" % (
        summary["needs_fem"], 100 - summary["reduction_pct"]))
    print("Can skip:          %d (%.0f%% cost reduction)" % (
        summary["can_skip"], summary["reduction_pct"]))
    print("  train: needs=%d, skip=%d" % (
        summary["train_needs_fem"], summary["train_can_skip"]))
    print("  val:   needs=%d, skip=%d" % (
        summary["val_needs_fem"], summary["val_can_skip"]))
    print("")
    print("Avg defect %% (FEM cases):  %.3f%%" % summary["avg_defect_pct_fem"])
    print("Avg defect %% (skip cases): %.3f%%" % summary["avg_defect_pct_skip"])
    print("Avg stress diff (FEM):      %.2f" % summary["avg_stress_diff_fem"])
    print("Avg stress diff (skip):     %.2f" % summary["avg_stress_diff_skip"])
    print("Max defect %% in skipped:   %.3f%%" % summary["max_defect_pct_skipped"])
    print("")

    if skipped_severe:
        print("WARNING: %d skipped cases have significant defects:" % len(skipped_severe))
        for s in skipped_severe:
            print("  %s: defect=%.2f%%, stress_diff=%.2f, fno_score=%.2f" % (
                s["sample"], s["defect_pct"], s["actual_stress_diff"], s["fno_score"]))
    else:
        print("SAFE: No severe defects missed by screening")

    # Save validation results
    validation_report = {
        "summary": summary,
        "skipped_severe": skipped_severe,
    }
    with open(os.path.join(run_dir, "validation_report.json"), "w") as f:
        json.dump(validation_report, f, indent=2)

    timings["validation"] = 0.0  # Negligible

    if args.skip_gnn:
        print("\n[Skipping GNN training (--skip_gnn)]")
    else:
        # =================================================================
        # Phase 4: GNN Training — Full vs. Screened
        # =================================================================

        # 4a: Create filtered dataset
        filtered_dir = os.path.join(run_dir, "filtered_data")
        n_train_filtered, n_val_filtered = create_filtered_dataset(
            args.pyg_dir, screening_output, fno_meta, filtered_dir,
            split_seed=args.seed)

        # Common GNN args
        gnn_base = (
            " --arch %s --epochs %d --hidden %d --layers %d"
            " --batch_size %d --lr %s --loss focal --focal_gamma 2.0"
            " --defect_weight %.1f --residual --patience 30 --seed %d"
        ) % (args.gnn_arch, args.gnn_epochs, args.gnn_hidden, args.gnn_layers,
             args.gnn_batch, args.gnn_lr, args.defect_weight, args.seed)

        # 4b: Train GNN on FULL dataset (baseline)
        gnn_full_dir = os.path.join(run_dir, "gnn_full")
        cmd_full = (
            "python src/train.py --data_dir %s --output_dir %s%s"
        ) % (args.pyg_dir, gnn_full_dir, gnn_base)

        rc, elapsed = run_cmd(cmd_full, "GNN Training — FULL dataset (%d graphs)" % 160)
        timings["gnn_full"] = elapsed

        # 4c: Train GNN on SCREENED dataset
        gnn_screened_dir = os.path.join(run_dir, "gnn_screened")
        cmd_screened = (
            "python src/train.py --data_dir %s --output_dir %s%s"
        ) % (filtered_dir, gnn_screened_dir, gnn_base)

        rc, elapsed = run_cmd(cmd_screened,
                              "GNN Training — SCREENED dataset (%d graphs)" % n_train_filtered)
        timings["gnn_screened"] = elapsed

        # =================================================================
        # Phase 5: Compare Results
        # =================================================================
        print("\n" + "=" * 60)
        print("PHASE: Results Comparison")
        print("=" * 60)

        # Find best model results from each GNN run
        gnn_results = {}
        for label, gnn_dir in [("full", gnn_full_dir), ("screened", gnn_screened_dir)]:
            # Look for best_model.pt in the latest run subdirectory
            best_f1 = 0.0
            best_metrics = {}
            for root, dirs, files in os.walk(gnn_dir):
                if "best_model.pt" in files:
                    ckpt = torch.load(os.path.join(root, "best_model.pt"),
                                      map_location="cpu", weights_only=False)
                    metrics = ckpt.get("val_metrics", {})
                    f1 = metrics.get("f1", 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_metrics = metrics
            gnn_results[label] = {"f1": best_f1, **best_metrics}

        print("\n%-15s  %8s  %8s  %8s  %8s  %8s" % (
            "Dataset", "F1", "AUC", "Prec", "Recall", "Train"))
        print("-" * 65)
        for label, m in gnn_results.items():
            n_train = 160 if label == "full" else n_train_filtered
            print("%-15s  %8.4f  %8.4f  %8.4f  %8.4f  %5d gr" % (
                label, m.get("f1", 0), m.get("auc", 0),
                m.get("precision", 0), m.get("recall", 0), n_train))
        print("-" * 65)

        if gnn_results.get("full", {}).get("f1", 0) > 0:
            f1_drop = gnn_results["full"]["f1"] - gnn_results.get("screened", {}).get("f1", 0)
            print("F1 drop from screening: %.4f" % f1_drop)
            if f1_drop < 0.02:
                print("PASS: Screening preserves GNN accuracy (drop < 2%%)")
            else:
                print("WARN: Significant accuracy drop from screening")

    # =====================================================================
    # Final Report
    # =====================================================================
    total_time = time.time() - t_start
    timings["total"] = total_time

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Total time: %.0fs (%.1f min)" % (total_time, total_time / 60))
    for phase, t in timings.items():
        if phase != "total":
            print("  %-20s: %6.0fs" % (phase, t))
    print("Results: %s" % run_dir)

    # Save final report
    final_report = {
        "timestamp": timestamp,
        "args": vars(args),
        "timings": timings,
        "screening": summary,
        "skipped_severe": len(skipped_severe),
    }
    if not args.skip_gnn:
        final_report["gnn_results"] = {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in gnn_results.items()
        }

    with open(os.path.join(run_dir, "pipeline_report.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    print("Report saved: %s" % os.path.join(run_dir, "pipeline_report.json"))


if __name__ == "__main__":
    main()
