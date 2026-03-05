# -*- coding: utf-8 -*-
"""augment_gw_healthy.py — GW フェアリング Healthy センサ時刻歴の augmentation（かさ増し）

1 つの Healthy センサ CSV から、ノイズ付加により N 個の健全バリアントを生成。
クラス不均衡（欠陥 100 vs 健全 1）の緩和と、実測ノイズ（PZT SNR 30–40 dB）のシミュレーション。

Strategy:
  1. 元の Healthy CSV を読み込み
  2. 乗法ノイズ: Ur *= (1 + η), η ~ N(0, mult_std²) — 較正ドリフト
  3. 加法ノイズ: Ur += ε, ε ~ N(0, (add_ratio * std)²) — センサノイズ
  4. N 個の CSV を出力

Usage:
  python src/augment_gw_healthy.py \
    --input abaqus_work/Job-GW-Fair-Healthy_sensors.csv \
    --output abaqus_work/gw_fairing_dataset \
    --n_healthy 100 --seed 42
"""

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gw_csv(csv_path):
    """Load GW sensor CSV. Returns (times, sensor_dict, x_positions)."""
    times = []
    sensors = {}
    x_positions = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_sensors = len(header) - 1
        for i in range(n_sensors):
            sensors[i] = []

        for row in reader:
            if row[0].startswith('#'):
                x_positions = [float(row[j]) if j + 1 < len(row) else 0.0
                               for j in range(1, n_sensors + 1)]
                continue
            try:
                times.append(float(row[0]))
                for i in range(n_sensors):
                    val = float(row[i + 1]) if row[i + 1] else 0.0
                    sensors[i].append(val)
            except (ValueError, IndexError):
                pass

    return (np.array(times),
            {k: np.array(v) for k, v in sensors.items()},
            x_positions)


def augment_sensor_data(sensor_dict, rng, mult_std=0.02, add_ratio=0.02):
    """Add multiplicative + additive noise to sensor time series.

    mult_std: calibration drift (2% typical)
    add_ratio: sensor noise relative to signal std (2% typical)
    """
    out = {}
    for sid, arr in sensor_dict.items():
        arr = np.array(arr, dtype=np.float64)
        std = np.std(arr)
        if std < 1e-12:
            std = 1e-12
        eta = rng.normal(0, mult_std, size=arr.shape)
        eps = rng.normal(0, add_ratio * std, size=arr.shape)
        out[sid] = arr * (1.0 + eta) + eps
    return out


def write_gw_csv(out_path, times, sensor_dict, x_positions, disp_label='Ur'):
    """Write augmented CSV in extract_gw_history format."""
    sensor_ids = sorted(sensor_dict.keys())
    n_rows = len(times)

    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        header = ['time_s'] + ['sensor_%d_%s' % (sid, disp_label) for sid in sensor_ids]
        writer.writerow(header)
        pos_row = ['# x_mm'] + [('%.1f' % x_positions[i]) if i < len(x_positions) else '0'
                                for i in range(len(sensor_ids))]
        writer.writerow(pos_row)

        for ti in range(n_rows):
            row = ['%.9e' % times[ti]]
            for sid in sensor_ids:
                row.append('%.9e' % sensor_dict[sid][ti])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='GW Healthy センサ時刻歴 augmentation（かさ増し）')
    parser.add_argument('--input', type=str,
                        default='abaqus_work/Job-GW-Fair-Healthy_sensors.csv',
                        help='元の Healthy CSV')
    parser.add_argument('--output', type=str,
                        default='abaqus_work/gw_fairing_dataset',
                        help='出力ディレクトリ')
    parser.add_argument('--n_healthy', type=int, default=100,
                        help='生成する健全サンプル数')
    parser.add_argument('--mult_std', type=float, default=0.02,
                        help='乗法ノイズ std (較正ドリフト, 2%%)')
    parser.add_argument('--add_ratio', type=float, default=0.02,
                        help='加法ノイズ比 (信号 std に対する割合, 2%%)')
    parser.add_argument('--prefix', type=str, default='Job-GW-Fair-Healthy-A',
                        help='出力ファイル接頭辞')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_path = (os.path.join(PROJECT_ROOT, args.input)
                  if not os.path.isabs(args.input) else args.input)
    output_dir = (os.path.join(PROJECT_ROOT, args.output)
                  if not os.path.isabs(args.output) else args.output)

    if not os.path.exists(input_path):
        print("ERROR: Input not found: %s" % input_path)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading: %s" % input_path)
    times, sensors, x_positions = load_gw_csv(input_path)
    n_sensors = len(sensors)
    n_pts = len(times)
    print("  %d time steps, %d sensors" % (n_pts, n_sensors))

    rng = np.random.RandomState(args.seed)

    print("\nGenerating %d healthy variants (mult_std=%.3f, add_ratio=%.3f)..." % (
        args.n_healthy, args.mult_std, args.add_ratio))

    for i in range(args.n_healthy):
        sensor_aug = augment_sensor_data(sensors, rng,
                                        mult_std=args.mult_std,
                                        add_ratio=args.add_ratio)
        out_name = '%s%03d_sensors.csv' % (args.prefix, i)
        out_path = os.path.join(output_dir, out_name)
        write_gw_csv(out_path, times, sensor_aug, x_positions)
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, args.n_healthy))

    print("\nDone. %d files in %s" % (args.n_healthy, output_dir))


if __name__ == '__main__':
    main()
