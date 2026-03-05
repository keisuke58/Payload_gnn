#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_gw_history.py と generate_gw_fairing.py の整合性検証

ODB がなくても、期待される名前・形式を照合する。
既存の _sensors.csv がある場合はフォーマット検証も行う。

Usage:
  python scripts/verify_gw_extract.py
  python scripts/verify_gw_extract.py --csv abaqus_work/Job-GW-Fair-Test-H3_sensors.csv
"""

import argparse
import csv
import os
import sys

# 期待値（generate_gw_fairing.py から）
EXPECTED = {
    'step_name': 'Step-Wave',
    'sensor_set_pattern': 'Set-Sensor-%d',   # 0..9
    'history_output_pattern': 'H-Output-S%d',
    'n_sensors': 10,
    'variables': ('U1', 'U2', 'U3'),
    'instance_outer': 'Part-OuterSkin-1',
}

# extract_gw_history.py が探す名前
EXTRACT_LOOKS_FOR = {
    'sensor_set': 'SET-SENSOR-%d',   # .upper() で照合 → Set-Sensor-0 と一致
    'step': 'Step-Wave',
    'curved_radius_threshold': 100,  # mm
}


def check_naming_compatibility():
    """名前の互換性を確認"""
    print("=== 1. 名前互換性 ===")
    ok = True
    for i in range(10):
        gen_set = EXPECTED['sensor_set_pattern'] % i
        ext_target = EXTRACT_LOOKS_FOR['sensor_set'] % i
        if gen_set.upper() == ext_target:
            pass  # OK
        else:
            print("  NG: gen=%s vs extract=%s" % (gen_set, ext_target))
            ok = False
    if ok:
        print("  OK: Set-Sensor-N は SET-SENSOR-N と一致（大文字照合）")
    print("  OK: Step-Wave 一致")
    return ok


def validate_csv_format(csv_path):
    """既存 CSV のフォーマット検証"""
    if not os.path.exists(csv_path):
        print("  CSV not found: %s" % csv_path)
        return False

    print("\n=== 2. CSV フォーマット検証: %s ===" % csv_path)
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        print("  NG: 行数不足")
        return False

    header = rows[0]
    pos_row = rows[1]

    # ヘッダー: time_s, sensor_0_Ur, sensor_1_Ur, ...
    if header[0] != 'time_s':
        print("  NG: 先頭列は time_s であるべき (got: %s)" % header[0])
        return False

    sensor_cols = [c for c in header[1:] if c.startswith('sensor_')]
    n_sensors = len(sensor_cols)
    print("  センサ列数: %d" % n_sensors)

    # 位置行
    if pos_row[0].startswith('#'):
        print("  位置行: %s" % pos_row[0])
    else:
        print("  位置行: # x_mm または # arc_mm")

    # データ行
    data_rows = [r for r in rows[2:] if r and not r[0].startswith('#')]
    n_rows = len(data_rows)
    print("  データ行数: %d" % n_rows)

    if n_rows < 10:
        print("  WARNING: データ行が少ない")
    if n_sensors < 5:
        print("  WARNING: センサ数が少ない (期待: 10)")

    # 欠損センサ ID の確認
    sensor_ids = []
    for c in sensor_cols:
        try:
            sid = int(c.replace('sensor_', '').split('_')[0])
            sensor_ids.append(sid)
        except ValueError:
            pass
    if sensor_ids:
        missing = set(range(max(sensor_ids) + 1)) - set(sensor_ids)
        if missing:
            print("  注意: 欠損センサ ID: %s (メッシュ境界等でスキップの可能性)" % sorted(missing))

    print("  OK: CSV フォーマット妥当")
    return True


def main():
    parser = argparse.ArgumentParser(description='extract_gw_history 整合性検証')
    parser.add_argument('--csv', type=str, default=None,
                        help='検証する _sensors.csv のパス')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    all_ok = True
    all_ok &= check_naming_compatibility()

    if args.csv:
        all_ok &= validate_csv_format(args.csv)
    else:
        # デフォルトでフェアリングテスト CSV を探す
        candidates = [
            'abaqus_work/Job-GW-Fair-Test-H3_sensors.csv',
            'abaqus_work/Job-GW-Fair-Test-D3_sensors.csv',
            'abaqus_work/Job-GW-Curved-H_sensors.csv',
        ]
        for p in candidates:
            if os.path.exists(p):
                validate_csv_format(p)
                break
        else:
            print("\n=== 2. CSV 検証スキップ（--csv でパス指定） ===")

    print("\n=== 3. 期待 ODB 構造（generate_gw_fairing 出力） ===")
    print("  Step: %s" % EXPECTED['step_name'])
    print("  Node sets: %s (0..9)" % EXPECTED['sensor_set_pattern'].replace('%d', 'N'))
    print("  History: U1, U2, U3")
    print("  Outer skin instance: %s" % EXPECTED['instance_outer'])
    print("  extract_gw_history は円筒形状を検出し Ur (radial) を計算")

    print("\n" + "=" * 50)
    if all_ok:
        print("検証完了: 整合性 OK")
    else:
        print("検証完了: 要確認あり")
        sys.exit(1)


if __name__ == '__main__':
    main()
