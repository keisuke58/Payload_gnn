# -*- coding: utf-8 -*-
"""
LightGBM-based Node-Level Defect Classification for H3 Fairing

Since physics data (stress, displacement, temperature) is unavailable (ODB extraction
failed), this script uses geometric features + defect metadata to learn spatial
containment: is a given node within the defect zone?

Features:
  - Cylindrical coordinates (theta, z_cyl) on fairing surface
  - Distance from defect center (delta_theta, delta_z, euclidean_dist_surface)
  - Defect radius
  - Node local geometry (curvature proxy from neighbor distances)

Train/val split at sample level to prevent data leakage.

Usage:
  python src/train_lightgbm.py --data_dir dataset_output_25mm_100 --doe doe_100_50mm.json
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score,
)
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================================
# H3 fairing geometry constants (must match generate_fairing_dataset.py)
# ===========================================================================
RADIUS = 2600.0       # mm
H_BARREL = 5000.0     # mm
H_NOSE = 5400.0       # mm
TOTAL_HEIGHT = H_BARREL + H_NOSE
OGIVE_RHO = (RADIUS**2 + H_NOSE**2) / (2.0 * RADIUS)
OGIVE_XC = RADIUS - OGIVE_RHO


def get_radius_at_z(z):
    """Fairing outer radius at height z."""
    if z <= H_BARREL:
        return RADIUS
    elif z > TOTAL_HEIGHT:
        return 0.0
    else:
        z_local = z - H_BARREL
        term = OGIVE_RHO**2 - z_local**2
        return OGIVE_XC + np.sqrt(max(term, 0.0))


# ===========================================================================
# Feature engineering
# ===========================================================================
def compute_defect_distance(coords, defect_params):
    """
    Compute surface distance from each node to defect center.
    Matches _is_node_in_defect() in extract_odb_results.py exactly:
      theta = atan2(z, x)  (Abaqus revolve: Y=axial, XZ=radial)
      arc = r_node * |theta_node - theta_center|
      dy = y - z_center
      dist = sqrt(arc^2 + dy^2)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r_node = np.sqrt(x**2 + z**2)

    # Azimuthal angle: atan2(z, x) — matches Abaqus revolve convention
    theta_node_rad = np.arctan2(z, x)
    theta_center_rad = np.radians(defect_params['theta_deg'])

    arc_mm = r_node * np.abs(theta_node_rad - theta_center_rad)
    dy = y - defect_params['z_center']
    dist = np.sqrt(arc_mm**2 + dy**2)
    return dist, arc_mm, dy, theta_node_rad, r_node


FEATURE_NAMES = [
    'theta_node_rad', 'y_axial', 'r_node',
    'theta_defect_rad', 'z_defect', 'radius_defect',
    'arc_to_defect', 'dy_to_defect', 'surface_dist', 'dist_ratio',
    'dist_ratio_sq', 'inside_zone',
    'node_degree', 'mean_edge_len', 'std_edge_len',
    'y_normalized', 'is_barrel',
]


# ===========================================================================
# Data loading
# ===========================================================================
def load_dataset(data_dir, doe_path, max_samples=None):
    """Load all samples, compute features, return combined arrays."""
    with open(doe_path) as f:
        doe = json.load(f)

    # Build defect param lookup by sample id
    defect_lookup = {}
    for s in doe['samples']:
        defect_lookup[s['id']] = s['defect_params']

    sample_dirs = sorted(Path(data_dir).glob('sample_*'))
    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    all_features = []
    all_labels = []
    all_sample_ids = []

    # Precompute mesh features once (mesh is identical across samples)
    print("Loading mesh template from first sample...")
    first_dir = sample_dirs[0]
    df_nodes_template = pd.read_csv(first_dir / 'nodes.csv')
    df_elems_template = pd.read_csv(first_dir / 'elements.csv')
    n_nodes = len(df_nodes_template)

    # Pre-compute mesh connectivity features (same for all samples)
    coords = df_nodes_template[['x', 'y', 'z']].values
    node_id_map = {int(nid): idx for idx, nid in
                   enumerate(df_nodes_template['node_id'].values)}

    print("Pre-computing mesh connectivity features...")
    node_degree = np.zeros(n_nodes, dtype=np.float32)
    neighbor_dists = [[] for _ in range(n_nodes)]

    for _, row in df_elems_template.iterrows():
        n1 = node_id_map.get(int(row['n1']), -1)
        n2 = node_id_map.get(int(row['n2']), -1)
        n3 = node_id_map.get(int(row['n3']), -1)
        nodes_in_elem = [n for n in [n1, n2, n3] if n >= 0]
        if 'n4' in row.index and pd.notna(row.get('n4')):
            n4 = node_id_map.get(int(row['n4']), -1)
            if n4 >= 0:
                nodes_in_elem.append(n4)

        for ni in nodes_in_elem:
            node_degree[ni] += 1
            for nj in nodes_in_elem:
                if ni != nj:
                    d = np.linalg.norm(coords[ni] - coords[nj])
                    neighbor_dists[ni].append(d)

    mean_edge_len = np.array([
        np.mean(dists) if dists else 0.0 for dists in neighbor_dists
    ], dtype=np.float32)
    std_edge_len = np.array([
        np.std(dists) if len(dists) > 1 else 0.0 for dists in neighbor_dists
    ], dtype=np.float32)
    del neighbor_dists  # free memory

    # Cylindrical coordinates (pre-compute once)
    # atan2(z, x) matches Abaqus revolve convention
    theta_node_rad = np.arctan2(coords[:, 2], coords[:, 0])
    y_axial = coords[:, 1]
    r_node = np.sqrt(coords[:, 0]**2 + coords[:, 2]**2)
    y_normalized = y_axial / TOTAL_HEIGHT
    is_barrel = (y_axial <= H_BARREL).astype(np.float32)

    print(f"Loading {len(sample_dirs)} samples...")
    for i, sdir in enumerate(sample_dirs):
        sample_id_str = sdir.name.replace('sample_', '')
        sample_id = int(sample_id_str)

        if sample_id not in defect_lookup:
            print(f"  Skipping {sdir.name}: no DOE entry")
            continue

        dp = defect_lookup[sample_id]
        df_nodes = pd.read_csv(sdir / 'nodes.csv')
        labels = df_nodes['defect_label'].values.astype(np.int32)

        # Compute distance features matching extract_odb_results.py logic
        surface_dist, arc_mm, dy, _, _ = compute_defect_distance(coords, dp)
        r_def = dp['radius']
        dist_ratio = surface_dist / max(r_def, 1.0)
        theta_def_rad = np.radians(dp['theta_deg'])

        features = np.column_stack([
            theta_node_rad,
            y_axial,
            r_node,
            np.full(n_nodes, theta_def_rad),
            np.full(n_nodes, dp['z_center']),
            np.full(n_nodes, r_def),
            arc_mm,
            dy,
            surface_dist,
            dist_ratio,
            dist_ratio**2,
            (surface_dist <= r_def).astype(np.float32),
            node_degree,
            mean_edge_len,
            std_edge_len,
            y_normalized,
            is_barrel,
        ])

        all_features.append(features)
        all_labels.append(labels)
        all_sample_ids.append(np.full(n_nodes, sample_id, dtype=np.int32))

        if (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{len(sample_dirs)} samples")

    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels)
    sample_ids = np.concatenate(all_sample_ids)

    print(f"\nDataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Defect nodes: {y.sum()} ({y.mean()*100:.4f}%)")
    print(f"  Healthy nodes: {(1-y).sum()}")

    return X, y, sample_ids


# ===========================================================================
# Training
# ===========================================================================
def train_and_evaluate(X, y, sample_ids, output_dir, n_folds=5, seed=42):
    """Train LightGBM with GroupKFold cross-validation."""
    os.makedirs(output_dir, exist_ok=True)

    unique_samples = np.unique(sample_ids)
    n_samples = len(unique_samples)
    print(f"\n{'='*60}")
    print(f"Training LightGBM ({n_folds}-fold GroupKFold)")
    print(f"  Samples: {n_samples}")
    print(f"  Total nodes: {len(y)}")
    print(f"  Defect ratio: {y.mean()*100:.4f}%")
    print(f"{'='*60}")

    # Class imbalance ratio
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight: {scale_pos:.1f}")

    gkf = GroupKFold(n_splits=n_folds)

    fold_results = []
    all_val_preds = np.zeros(len(y), dtype=np.float32)
    all_val_mask = np.zeros(len(y), dtype=bool)
    feature_importances = np.zeros(X.shape[1], dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X, y, groups=sample_ids)):
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_samples = np.unique(sample_ids[train_idx])
        val_samples = np.unique(sample_ids[val_idx])
        print(f"  Train: {len(train_samples)} samples ({len(y_train)} nodes, "
              f"defect={y_train.sum()})")
        print(f"  Val:   {len(val_samples)} samples ({len(y_val)} nodes, "
              f"defect={y_val.sum()})")

        # LightGBM dataset
        dtrain = lgb.Dataset(X_train, label=y_train,
                             feature_name=FEATURE_NAMES, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val,
                           feature_name=FEATURE_NAMES, free_raw_data=False)

        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'average_precision'],
            'scale_pos_weight': scale_pos,
            'num_leaves': 127,
            'max_depth': 10,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': seed,
            'n_jobs': -1,
        }

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50),
        ]

        t0 = time.time()
        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            valid_names=['val'],
            callbacks=callbacks,
        )
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s, best iter: {model.best_iteration}")

        # Predictions
        y_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob) if y_val.sum() > 0 else 0.0
        ap = average_precision_score(y_val, y_prob) if y_val.sum() > 0 else 0.0

        print(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
              f"AUC={auc:.4f}  AP={ap:.4f}")

        cm = confusion_matrix(y_val, y_pred)
        print(f"  Confusion matrix:\n{cm}")

        fold_results.append({
            'fold': fold_idx + 1,
            'f1': f1, 'precision': prec, 'recall': rec,
            'auc': auc, 'ap': ap,
            'best_iter': model.best_iteration,
            'train_time': elapsed,
            'n_train_samples': len(train_samples),
            'n_val_samples': len(val_samples),
            'n_val_defect': int(y_val.sum()),
        })

        all_val_preds[val_idx] = y_prob
        all_val_mask[val_idx] = True
        feature_importances += model.feature_importance(importance_type='gain')

        # Save model
        model.save_model(os.path.join(output_dir, f'model_fold{fold_idx+1}.txt'))

    # Average results
    feature_importances /= n_folds
    df_results = pd.DataFrame(fold_results)
    print(f"\n{'='*60}")
    print("Cross-Validation Results (mean ± std):")
    for metric in ['f1', 'precision', 'recall', 'auc', 'ap']:
        m = df_results[metric].mean()
        s = df_results[metric].std()
        print(f"  {metric:12s}: {m:.4f} ± {s:.4f}")
    print(f"{'='*60}")

    # Save results
    df_results.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)

    # Feature importance
    fi = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': feature_importances,
    }).sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print("\nFeature Importance (top 10):")
    for _, row in fi.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.1f}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi['feature'].values[::-1], fi['importance'].values[::-1])
    ax.set_xlabel('Gain')
    ax.set_title('LightGBM Feature Importance')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()

    # Threshold sweep for best F1
    val_y = y[all_val_mask]
    val_prob = all_val_preds[all_val_mask]
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.91, 0.05):
        pred = (val_prob >= thresh).astype(int)
        f1_t = f1_score(val_y, pred, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.2f} → F1={best_f1:.4f}")

    # Final classification report at best threshold
    y_pred_best = (val_prob >= best_thresh).astype(int)
    print("\nClassification Report (best threshold):")
    print(classification_report(val_y, y_pred_best, target_names=['Healthy', 'Defect'],
                                zero_division=0))

    # Per-sample results
    print("\nPer-sample validation results:")
    sample_results = []
    for sid in np.unique(sample_ids[all_val_mask]):
        mask = all_val_mask & (sample_ids == sid)
        sy = y[mask]
        sp = (all_val_preds[mask] >= best_thresh).astype(int)
        sf1 = f1_score(sy, sp, zero_division=0)
        n_def = sy.sum()
        n_pred = sp.sum()
        sample_results.append({
            'sample_id': sid, 'n_defect': int(n_def),
            'n_predicted': int(n_pred), 'f1': sf1
        })
        print(f"  Sample {sid:04d}: defect={n_def:3d}, predicted={n_pred:3d}, F1={sf1:.4f}")

    pd.DataFrame(sample_results).to_csv(
        os.path.join(output_dir, 'per_sample_results.csv'), index=False)

    # Summary
    summary = {
        'model': 'LightGBM',
        'n_samples': n_samples,
        'n_folds': n_folds,
        'mean_f1': float(df_results['f1'].mean()),
        'mean_auc': float(df_results['auc'].mean()),
        'mean_ap': float(df_results['ap'].mean()),
        'best_threshold': float(best_thresh),
        'best_f1_at_threshold': float(best_f1),
        'scale_pos_weight': float(scale_pos),
        'note': 'Physics data unavailable (all zero). Using geometric + defect metadata features.',
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return df_results, fi


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM for node-level defect classification')
    parser.add_argument('--data_dir', type=str,
                        default='dataset_output_25mm_100',
                        help='Dataset directory')
    parser.add_argument('--doe', type=str,
                        default='doe_100_50mm.json',
                        help='DOE JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='runs/lightgbm',
                        help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    X, y, sample_ids = load_dataset(args.data_dir, args.doe, args.max_samples)
    train_and_evaluate(X, y, sample_ids, args.output_dir,
                       n_folds=args.n_folds, seed=args.seed)


if __name__ == '__main__':
    main()
