#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動解析 (GW) データセット品質検証

静解析 (verify_dataset_quality.py) とは別基準で評価。
センサ CSV の存在・センサ数・時刻歴長・信号品質をチェック。

Usage:
  # 本番前チェック（フェアリングのみ、100点で本番データ生成OK）
  python scripts/verify_gw_dataset_quality.py --data_dir abaqus_work --filter fairing

  # 全 GW データ（旧モデル含む）
  python scripts/verify_gw_dataset_quality.py --data_dir abaqus_work
"""
import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_gw_csv(csv_path):
    """Single CSV quality check. Returns dict or None if invalid."""
    if not os.path.exists(csv_path) or not csv_path.endswith('_sensors.csv'):
        return None
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception:
        return None
    if len(rows) < 3:
        return None

    header = rows[0]
    sensor_cols = [c for c in header[1:] if c.startswith('sensor_')]
    n_sensors = len(sensor_cols)
    data_rows = [r for r in rows[2:] if r and not r[0].startswith('#')]
    n_rows = len(data_rows)

    # Signal quality: RMS of first sensor (exclude time col)
    rms = 0.0
    if data_rows and len(header) > 1:
        vals = []
        for r in data_rows:
            if len(r) > 1 and r[1]:
                try:
                    vals.append(float(r[1]))
                except ValueError:
                    pass
        if vals:
            import math
            rms = math.sqrt(sum(v * v for v in vals) / len(vals))

    return {
        'path': csv_path,
        'n_sensors': n_sensors,
        'n_rows': n_rows,
        'rms': rms,
    }


def main():
    parser = argparse.ArgumentParser(description='GW (動解析) データセット品質検証')
    parser.add_argument('--data_dir', type=str, default='abaqus_work/gw_fairing_dataset',
                        help='CSV が格納されたディレクトリ')
    parser.add_argument('--scan_parent', action='store_true',
                        help='親ディレクトリ内の全 *_sensors.csv をスキャン')
    parser.add_argument('--filter', type=str, choices=['fairing', 'all'], default='all',
                        help='fairing: Job-GW-Fair-* のみ評価（本番前チェック用）')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    csv_files = []

    if args.scan_parent:
        parent = os.path.dirname(data_dir.rstrip('/'))
        if os.path.isdir(parent):
            for f in os.listdir(parent):
                if f.endswith('_sensors.csv'):
                    csv_files.append(os.path.join(parent, f))
    elif os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('_sensors.csv'):
                csv_files.append(os.path.join(data_dir, f))

    csv_files = sorted(csv_files)
    if not csv_files:
        print("GW dataset quality: %s" % data_dir)
        print("  No *_sensors.csv found")
        if not args.scan_parent:
            print("  Tip: --scan_parent to scan parent directory")
        return 0

    results = []
    for p in csv_files:
        r = verify_gw_csv(p)
        if r:
            if args.filter == 'fairing' and 'Job-GW-Fair-' not in os.path.basename(p):
                continue
            results.append(r)

    n = len(results)
    if n == 0:
        print("GW dataset quality: %s" % data_dir)
        print("  No valid CSV found")
        return 1

    n_sensors_avg = sum(r['n_sensors'] for r in results) / n
    n_rows_avg = sum(r['n_rows'] for r in results) / n
    n_signal_ok = sum(1 for r in results if r['rms'] > 1e-12)
    n_sensors_ok = sum(1 for r in results if r['n_sensors'] >= 5)
    n_rows_ok = sum(1 for r in results if r['n_rows'] >= 100)

    # Score (0-100)
    score = 0
    if n > 0:
        score += 25 * min(1.0, n_sensors_ok / n)   # 25pt: センサ数
        score += 25 * min(1.0, n_rows_ok / n)      # 25pt: 時刻歴長
        score += 25 * min(1.0, n_signal_ok / n)    # 25pt: 信号
        score += 25 * (1.0 if n_sensors_avg >= 7 else n_sensors_avg / 7)  # 25pt: 平均センサ数
    score = int(min(100, score))

    print("=" * 50)
    print("GW (動解析) データセット品質")
    print("=" * 50)
    print("  Data dir: %s" % data_dir)
    print("  Total CSVs: %d" % n)
    print("  Sensors/sample (avg): %.1f" % n_sensors_avg)
    print("  Rows/sample (avg): %.0f" % n_rows_avg)
    print("  With signal (rms>0): %d / %d" % (n_signal_ok, n))
    print("  Sensors>=5: %d / %d" % (n_sensors_ok, n))
    print("  Rows>=100: %d / %d" % (n_rows_ok, n))
    print("  Estimated score: %d/100" % score)
    print()
    for r in results[:5]:
        print("  %s: %d sensors, %d rows, rms=%.2e" % (
            os.path.basename(r['path']), r['n_sensors'], r['n_rows'], r['rms']))
    if len(results) > 5:
        print("  ... and %d more" % (len(results) - 5))

    return 0


if __name__ == '__main__':
    sys.exit(main())
