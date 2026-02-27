#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健全ベースライン (Healthy Baseline) 全方面検証

欠陥挿入前に、healthy データの完全性・一貫性・物理妥当性をチェック。
全項目 PASS であることを確認してからデボンディングサンプル生成を行う。

Usage:
  python scripts/validate_healthy_baseline.py
  python scripts/validate_healthy_baseline.py --data dataset_output/healthy_baseline --strict
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Model geometry (match generate_fairing_dataset.py)
RADIUS = 2600.0
H_BARREL = 5000.0
H_NOSE = 5400.0
HEIGHT = H_BARREL + H_NOSE
ANGLE_DEG = 60.0
FACE_T = 1.0
CORE_T = 38.0
R_OUTER = RADIUS
R_CORE_O = RADIUS - FACE_T / 2.0
R_INNER = RADIUS - FACE_T - CORE_T

LABEL_OFFSETS = {'Skin_Outer': 0, 'Skin_Inner': 100000, 'Core': 200000}


def _part_from_node_id(nid):
    nid = int(nid)
    if nid < 100000:
        return 'Skin_Outer'
    if nid < 200000:
        return 'Skin_Inner'
    return 'Core'


def check_geometry(df, results):
    """Geometry: z, r, theta ranges"""
    r = np.sqrt(df['x']**2 + df['y']**2)
    theta = np.degrees(np.arctan2(df['y'].values, df['x'].values))
    theta = np.where(theta < 0, theta + 360, theta)

    z_min, z_max = df['z'].min(), df['z'].max()
    r_min, r_max = r.min(), r.max()
    theta_min, theta_max = theta.min(), theta.max()

    # Z: 0 to HEIGHT. Accept Barrel-only (0-5000) or Full (0-10400)
    ok_z = abs(z_min) <= 1.0 and 0 < z_max <= HEIGHT + 10
    mode = 'Barrel-only' if z_max < 6000 else 'Full (Barrel+Ogive)'
    results.append(('Geometry.Z_range', ok_z, f'z=[{z_min:.2f}, {z_max:.2f}] mm ({mode})'))

    # R: R_INNER - tol to R_OUTER + tol
    tol_r = 50.0
    ok_r = R_INNER - tol_r <= r_min and r_max <= R_OUTER + tol_r
    results.append(('Geometry.R_range', ok_r, f'r=[{r_min:.1f}, {r_max:.1f}] mm (expect {R_INNER:.0f}-{R_OUTER:.0f})'))

    # Theta: 0 to ANGLE (1/6 sector)
    ok_theta = -5 <= theta_min <= 5 and ANGLE_DEG - 5 <= theta_max <= ANGLE_DEG + 5
    results.append(('Geometry.Theta_sector', ok_theta, f'theta=[{theta_min:.1f}, {theta_max:.1f}] deg (expect 0-{ANGLE_DEG})'))

    return results


def check_data_integrity(df_nodes, df_elems, results):
    """No NaN, Inf, duplicates; element-node consistency"""
    # NaN / Inf
    has_nan = df_nodes.isna().any().any()
    has_inf = np.isinf(df_nodes.select_dtypes(include=[np.number])).any().any()
    results.append(('Integrity.No_NaN', not has_nan, 'No NaN in nodes'))
    results.append(('Integrity.No_Inf', not has_inf, 'No Inf in nodes'))

    # Duplicate node_ids
    dup = df_nodes['node_id'].duplicated().any()
    results.append(('Integrity.No_duplicate_nodes', not dup, 'Unique node_id'))

    # Element nodes exist
    node_set = set(df_nodes['node_id'].astype(int))
    missing = []
    for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']:
        if col not in df_elems.columns:
            continue
        for nid in df_elems[col].dropna().astype(int).unique():
            if nid >= 0 and nid not in node_set:
                missing.append(nid)
    missing = list(set(missing))[:10]
    ok_conn = len(missing) == 0
    results.append(('Integrity.Element_nodes_exist', ok_conn,
                    f'All element nodes in nodes.csv' + (f' (missing: {missing})' if missing else '')))

    return results


def check_defect_labels(df, results):
    """Healthy: all defect_label must be 0"""
    unique = df['defect_label'].unique()
    all_zero = len(unique) == 1 and unique[0] == 0
    n_defect = (df['defect_label'] == 1).sum()
    results.append(('Labels.All_zero', all_zero, f'defect_label all 0 (n_defect={n_defect})'))
    return results


def check_physics(df, results):
    """Stress and displacement magnitude sanity"""
    s11, s22, s12 = df['s11'], df['s22'], df['s12']
    dspss = df['dspss']

    # Stress: expect order ~0.01-100 MPa for 50 kPa pressure
    s_max = max(abs(s11).max(), abs(s22).max(), abs(s12).max())
    ok_stress = s_max < 1000 and s_max > 1e-6
    results.append(('Physics.Stress_magnitude', ok_stress, f'max|stress|={s_max:.2f} MPa'))

    # dspss: displacement, expect positive small (mm)
    dspss_ok = dspss.min() >= -0.1 and dspss.max() < 1000
    results.append(('Physics.DSPSS_range', dspss_ok, f'dspss=[{dspss.min():.4f}, {dspss.max():.4f}]'))

    return results


def check_metadata(meta, df_nodes, df_elems, results):
    """metadata.csv consistency"""
    if meta is None:
        results.append(('Metadata.Exists', False, 'metadata.csv not found'))
        return results

    meta_dict = dict(zip(meta['key'], meta['value']))
    n_nodes = int(meta_dict.get('n_nodes', -1))
    n_elements = int(meta_dict.get('n_elements', -1))
    n_defect = int(meta_dict.get('n_defect_nodes', -1))

    ok_nodes = n_nodes == len(df_nodes)
    ok_elems = n_elements == len(df_elems)
    ok_defect = n_defect == 0

    results.append(('Metadata.n_nodes', ok_nodes, f'n_nodes: meta={n_nodes} vs actual={len(df_nodes)}'))
    results.append(('Metadata.n_elements', ok_elems, f'n_elements: meta={n_elements} vs actual={len(df_elems)}'))
    results.append(('Metadata.n_defect_zero', ok_defect, f'n_defect_nodes={n_defect} (expect 0)'))
    results.append(('Metadata.defect_type', meta_dict.get('defect_type') == 'healthy', 'defect_type=healthy'))

    return results


def check_part_distribution(df, results):
    """Node count per part"""
    df = df.copy()
    df['part'] = df['node_id'].apply(_part_from_node_id)
    counts = df['part'].value_counts()
    total = len(df)
    for part in ['Skin_Outer', 'Skin_Inner', 'Core']:
        c = counts.get(part, 0)
        ok = c > 0
        results.append((f'Parts.{part}', ok, f'{c} nodes'))
    return results


def check_preprocessing(sample_dir, results):
    """preprocess_fairing_data runs without error"""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
        from preprocess_fairing_data import process_single_sample, load_baseline_dspss
        baseline_dspss = load_baseline_dspss(sample_dir)
        data = process_single_sample(sample_dir, baseline_dspss, mesh_size=50.0, height=10400.0)
        ok = data is not None and hasattr(data, 'x') and data.x is not None
        n_feat = data.x.shape[1] if ok else 0
        results.append(('Preprocess.PyG_convert', ok, f'PyG Data OK, {n_feat} features'))
    except Exception as e:
        results.append(('Preprocess.PyG_convert', False, str(e)))
    return results


def run_validation(data_dir, strict=False):
    """Run all checks and return (results, all_pass)."""
    nodes_path = os.path.join(data_dir, 'nodes.csv')
    elems_path = os.path.join(data_dir, 'elements.csv')
    meta_path = os.path.join(data_dir, 'metadata.csv')

    results = []

    if not os.path.exists(nodes_path):
        results.append(('Files.nodes_csv', False, 'nodes.csv not found'))
        return results, False
    results.append(('Files.nodes_csv', True, 'OK'))

    if not os.path.exists(elems_path):
        results.append(('Files.elements_csv', False, 'elements.csv not found'))
        return results, False
    results.append(('Files.elements_csv', True, 'OK'))

    df_nodes = pd.read_csv(nodes_path)
    df_elems = pd.read_csv(elems_path)
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    check_geometry(df_nodes, results)
    check_data_integrity(df_nodes, df_elems, results)
    check_defect_labels(df_nodes, results)
    check_physics(df_nodes, results)
    check_metadata(meta, df_nodes, df_elems, results)
    check_part_distribution(df_nodes, results)
    check_preprocessing(data_dir, results)

    all_pass = all(r[1] for r in results)
    if strict and not all_pass:
        return results, False
    return results, all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(PROJECT_ROOT, 'dataset_output', 'healthy_baseline'))
    parser.add_argument('--strict', action='store_true', help='Exit 1 if any check fails')
    parser.add_argument('--report', type=str, default='', help='Save report to file')
    args = parser.parse_args()

    results, all_pass = run_validation(args.data, args.strict)

    # Print
    print("=" * 70)
    print("Healthy Baseline Validation Report")
    print("=" * 70)
    print(f"Data: {args.data}")
    print()

    passed = sum(1 for r in results if r[1])
    total = len(results)
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    print()
    print(f"Result: {passed}/{total} passed" + (" - ALL PASS" if all_pass else " - FAILED"))
    print("=" * 70)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("Healthy Baseline Validation Report\n")
            f.write("=" * 70 + "\n")
            for name, ok, msg in results:
                f.write(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}\n")
            f.write(f"\nResult: {passed}/{total} passed\n")
        print(f"Report saved: {args.report}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
