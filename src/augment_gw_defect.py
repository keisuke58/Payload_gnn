# -*- coding: utf-8 -*-
"""augment_gw_defect.py — GW 欠陥センサ時刻歴の augmentation（かさ増し）

各欠陥 CSV にノイズ付加で N バリアントを生成。
データ量増加と実測ノイズ（PZT SNR 30–40 dB）のシミュレーション。

Strategy: augment_gw_healthy と同様の乗法・加法ノイズ

Usage:
  python src/augment_gw_defect.py \
    --input abaqus_work/gw_fairing_dataset \
    --doe doe_gw_fairing.json \
    --n_per_defect 5 --output abaqus_work/gw_fairing_dataset
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from augment_gw_healthy import load_gw_csv, augment_sensor_data, write_gw_csv


def main():
    parser = argparse.ArgumentParser(
        description='GW 欠陥センサ時刻歴 augmentation')
    parser.add_argument('--input', type=str, default='abaqus_work/gw_fairing_dataset',
                        help='欠陥 CSV があるディレクトリ')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ディレクトリ（省略時は input に上書き）')
    parser.add_argument('--doe', type=str, default='doe_gw_fairing.json',
                        help='DOE JSON（欠陥リスト）')
    parser.add_argument('--n_per_defect', type=int, default=5,
                        help='各欠陥から生成するバリアント数')
    parser.add_argument('--mult_std', type=float, default=0.02)
    parser.add_argument('--add_ratio', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_dir = os.path.join(PROJECT_ROOT, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(PROJECT_ROOT, args.output) if args.output and not os.path.isabs(args.output) else (args.output or input_dir)
    if not args.output:
        output_dir = input_dir

    if not os.path.isdir(input_dir):
        print("ERROR: Input dir not found: %s" % input_dir)
        sys.exit(1)

    doe_path = os.path.join(PROJECT_ROOT, args.doe) if not os.path.isabs(args.doe) else args.doe
    if not os.path.exists(doe_path):
        print("ERROR: DOE not found: %s" % doe_path)
        sys.exit(1)

    with open(doe_path) as f:
        doe = json.load(f)
    n_defect = doe.get('n_samples', 0)
    defect_names = ['Job-GW-Fair-%04d' % i for i in range(n_defect)]

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    total = 0
    for base_name in defect_names:
        csv_path = os.path.join(input_dir, base_name + '_sensors.csv')
        if not os.path.exists(csv_path):
            continue
        times, sensors, x_positions = load_gw_csv(csv_path)
        for j in range(args.n_per_defect):
            sensor_aug = augment_sensor_data(sensors, rng,
                                            mult_std=args.mult_std,
                                            add_ratio=args.add_ratio)
            out_name = '%s-A%03d_sensors.csv' % (base_name, j)
            out_path = os.path.join(output_dir, out_name)
            write_gw_csv(out_path, times, sensor_aug, x_positions)
            total += 1
        if total % 50 == 0 and total > 0:
            print("  %d variants..." % total)

    print("Done. %d defect variants in %s" % (total, output_dir))


if __name__ == '__main__':
    main()
