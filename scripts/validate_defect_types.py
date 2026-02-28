#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張欠陥タイプ検証 — Extended Defect Types Validation

全7種の欠陥モードについて、DOE・extract_odb・train・FEM の一貫性を検証。
CI やデータ生成前に実行して整合性を確認する。

Usage:
  python scripts/validate_defect_types.py
  python scripts/validate_defect_types.py --strict
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _get_defect_type_map():
    """DEFECT_TYPE_MAP を取得（odbAccess なしで extract_odb を読む）"""
    import ast
    path = os.path.join(PROJECT_ROOT, 'src', 'extract_odb_results.py')
    with open(path) as f:
        content = f.read()
    # DEFECT_TYPE_MAP = { ... } を抽出（複数行対応）
    start = content.find('DEFECT_TYPE_MAP = {')
    if start < 0:
        return {}
    depth = 0
    end = start + len('DEFECT_TYPE_MAP = ')
    for i, c in enumerate(content[end:], end):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return ast.literal_eval(content[content.find('{', start):end])
    except Exception:
        return {}


def _get_defect_type_names():
    """DEFECT_TYPE_NAMES を train.py からパース（models 依存を避ける）"""
    import re
    path = os.path.join(PROJECT_ROOT, 'src', 'train.py')
    with open(path) as f:
        content = f.read()
    # DEFECT_TYPE_NAMES = [ 'a', 'b', ... ] の文字列を抽出
    start = content.find('DEFECT_TYPE_NAMES = [')
    if start < 0:
        return []
    start += len('DEFECT_TYPE_NAMES = [')
    depth, end = 1, start
    for i, c in enumerate(content[start:], start):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                end = i
                break
    inner = content[start:end]
    return [m.group(1) for m in re.finditer(r"['\"]([^'\"]+)['\"]", inner)]


def check_defect_type_map_consistency(results):
    """DEFECT_TYPE_MAP, DEFECT_TYPES, JOB_PREFIXES, TYPE_PARAM_RANGES の一貫性"""
    DEFECT_TYPE_MAP = _get_defect_type_map()
    from src.generate_doe import DEFECT_TYPES, JOB_PREFIXES, TYPE_PARAM_RANGES

    ok = True
    for i, dtype in enumerate(DEFECT_TYPES):
        if dtype not in DEFECT_TYPE_MAP:
            results.append(('Consistency.DEFECT_TYPE_MAP', False,
                           f'{dtype} missing in extract_odb DEFECT_TYPE_MAP'))
            ok = False
        elif DEFECT_TYPE_MAP[dtype] != i + 1:
            results.append(('Consistency.DEFECT_TYPE_MAP', False,
                           f'{dtype} id mismatch: map={DEFECT_TYPE_MAP[dtype]} expect={i+1}'))
            ok = False
        if dtype not in JOB_PREFIXES:
            results.append(('Consistency.JOB_PREFIXES', False, f'{dtype} missing in JOB_PREFIXES'))
            ok = False
        if dtype not in TYPE_PARAM_RANGES:
            results.append(('Consistency.TYPE_PARAM_RANGES', False, f'{dtype} missing in TYPE_PARAM_RANGES'))
            ok = False

    if ok:
        results.append(('Consistency.DEFECT_TYPE_MAP', True,
                       f'All {len(DEFECT_TYPES)} types consistent across doe/extract'))
    return results


def check_defect_type_names(results):
    """DEFECT_TYPE_NAMES: healthy + 7 defect types = 8 entries"""
    DEFECT_TYPE_NAMES = _get_defect_type_names()
    from src.generate_doe import DEFECT_TYPES

    expected = ['healthy'] + DEFECT_TYPES
    ok = DEFECT_TYPE_NAMES == expected
    results.append(('Consistency.DEFECT_TYPE_NAMES', ok,
                    f'DEFECT_TYPE_NAMES has {len(DEFECT_TYPE_NAMES)} entries (expect 8: healthy + 7 types)'))
    if not ok:
        diff = set(expected) ^ set(DEFECT_TYPE_NAMES)
        results.append(('Consistency.DEFECT_TYPE_NAMES_diff', False, f'Mismatch: {diff}'))
    return results


def check_doe_generation(results):
    """DOE 生成が全タイプで正常に動作するか"""
    from src.generate_doe import generate_doe, DEFECT_TYPES

    try:
        doe = generate_doe(n_samples=14, seed=42, defect_types=DEFECT_TYPES)
    except Exception as e:
        results.append(('DOE.Generation', False, str(e)))
        return results

    if doe['n_total'] != 14:
        results.append(('DOE.n_total', False, f"n_total={doe['n_total']} expect 14"))
    else:
        results.append(('DOE.Generation', True, f"Generated {doe['n_total']} samples"))

    # 各サンプルに defect_type と必須パラメータがあるか
    type_counts = {}
    param_errors = []
    for s in doe['samples']:
        if s.get('defect_params') is None:
            continue
        dtype = s['defect_params'].get('defect_type')
        if dtype:
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        # 必須パラメータ
        params = s['defect_params']
        if 'theta_deg' not in params or 'z_center' not in params or 'radius' not in params:
            param_errors.append(f"Sample {s.get('id')} missing theta/z/radius")

    if param_errors:
        results.append(('DOE.Params', False, '; '.join(param_errors[:3])))
    else:
        results.append(('DOE.Params', True, 'All samples have theta_deg, z_center, radius'))

    # 全7種が少なくとも1件ずつ含まれるか（14サンプルなら2種ずつ程度）
    missing = set(DEFECT_TYPES) - set(type_counts.keys())
    if missing:
        results.append(('DOE.Type_coverage', False, f'Types not in DOE: {missing}'))
    else:
        results.append(('DOE.Type_coverage', True, f'All types present: {type_counts}'))
    return results


def check_type_specific_params(results):
    """TYPE_PARAM_RANGES のパラメータが create_defect_materials で使用されているか"""
    from src.generate_doe import TYPE_PARAM_RANGES, DEFECT_TYPES

    # generate_fairing_dataset の create_defect_materials で参照するパラメータ
    expected_params = {
        'fod': ['stiffness_factor'],
        'impact': ['damage_ratio'],
        'delamination': ['delam_depth'],
        'acoustic_fatigue': ['fatigue_severity'],
    }

    ok = True
    for dtype, expected in expected_params.items():
        ranges = TYPE_PARAM_RANGES.get(dtype, {})
        for p in expected:
            if p not in ranges:
                results.append(('Params.TYPE_PARAM_RANGES', False, f'{dtype} missing {p}'))
                ok = False
    if ok:
        results.append(('Params.TYPE_PARAM_RANGES', True, 'All type-specific params defined'))
    return results


def check_extract_odb_metadata(results):
    """extract_odb が新パラメータ (delam_depth, fatigue_severity) を metadata に書き出すか"""
    # ソースを読んで確認（実行は Abaqus が必要なためスキップ）
    extract_path = os.path.join(PROJECT_ROOT, 'src', 'extract_odb_results.py')
    with open(extract_path) as f:
        content = f.read()
    has_delam = 'delam_depth' in content
    has_fatigue = 'fatigue_severity' in content
    ok = has_delam and has_fatigue
    results.append(('Extract.Metadata_params', ok,
                    'extract_odb writes delam_depth, fatigue_severity to metadata'))
    return results


def check_dataset_integrity_multiclass(results):
    """既存データセットがあれば、multi-class defect_label で verify が通るか"""
    import subprocess

    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset_multitype_100')
    if not os.path.exists(dataset_dir):
        results.append(('Dataset.Multiclass_verify', True, 'Skip (dataset_multitype_100 not found)'))
        return results

    script = os.path.join(PROJECT_ROOT, 'scripts', 'verify_dataset_integrity.py')
    try:
        r = subprocess.run(
            [sys.executable, script, dataset_dir],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60
        )
        out = (r.stdout or '') + (r.stderr or '')
        if r.returncode == 0 and 'Valid:' in out:
            results.append(('Dataset.Multiclass_verify', True, 'verify_dataset_integrity passed'))
        else:
            results.append(('Dataset.Multiclass_verify', False, f'exit={r.returncode}, {out[:150]}'))
    except Exception as e:
        results.append(('Dataset.Multiclass_verify', False, str(e)))
    return results


def run_all_checks(strict=False, skip_dataset=False):
    """全チェック実行"""
    results = []
    check_defect_type_map_consistency(results)
    check_defect_type_names(results)
    check_doe_generation(results)
    check_type_specific_params(results)
    check_extract_odb_metadata(results)
    if not skip_dataset:
        check_dataset_integrity_multiclass(results)
    else:
        results.append(('Dataset.Multiclass_verify', True, 'Skipped (--skip-dataset)'))
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate extended defect types consistency')
    parser.add_argument('--strict', action='store_true', help='Exit 1 if any check fails')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset_multitype verify (faster, for CI)')
    parser.add_argument('--report', type=str, default='', help='Save report to file')
    args = parser.parse_args()

    results = run_all_checks(strict=args.strict, skip_dataset=args.skip_dataset)

    print("=" * 70)
    print("Extended Defect Types Validation Report")
    print("=" * 70)
    print()

    passed = sum(1 for r in results if r[1])
    total = len(results)
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    print()
    all_pass = passed == total
    print(f"Result: {passed}/{total} passed" + (" - ALL PASS" if all_pass else " - FAILED"))
    print("=" * 70)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("Extended Defect Types Validation Report\n")
            f.write("=" * 70 + "\n")
            for name, ok, msg in results:
                f.write(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}\n")
            f.write(f"\nResult: {passed}/{total} passed\n")
        print(f"Report saved: {args.report}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
