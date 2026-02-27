#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyG Data インスペクション — 前処理済み .pt ファイルの構造確認

Usage:
  python scripts/inspect_pyg_data.py path/to/data.pt
  python scripts/inspect_pyg_data.py  # デフォルト: dataset/processed/*.pt を検索
"""

import argparse
import glob
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def inspect_data(file_path):
    """Load and print PyG Data structure."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    try:
        try:
            data = torch.load(file_path, weights_only=False)
        except TypeError:
            data = torch.load(file_path)
        print(f"\n=== {os.path.basename(file_path)} ===")
        print(f"Type: {type(data).__name__}")

        if hasattr(data, 'keys'):
            keys = list(data.keys()) if callable(data.keys) else list(data.keys)
            print(f"Keys: {keys}")

        for attr in ['x', 'y', 'edge_index', 'pos']:
            if hasattr(data, attr) and getattr(data, attr) is not None:
                t = getattr(data, attr)
                print(f"  {attr}: shape={t.shape}, dtype={t.dtype}")
                if attr == 'y':
                    print(f"    unique: {torch.unique(t).tolist()}")

        for attr in ['num_node_features', 'num_classes', 'num_nodes', 'num_edges']:
            if hasattr(data, attr):
                print(f"  {attr}: {getattr(data, attr)}")

        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect PyG .pt data files")
    parser.add_argument('path', nargs='?', default=None,
                        help='Path to .pt file or directory')
    args = parser.parse_args()

    if args.path:
        if os.path.isfile(args.path):
            inspect_data(args.path)
        elif os.path.isdir(args.path):
            for p in sorted(glob.glob(os.path.join(args.path, '*.pt'))):
                inspect_data(p)
        else:
            print(f"Not found: {args.path}")
    else:
        default_dir = os.path.join(PROJECT_ROOT, 'dataset', 'processed')
        if os.path.isdir(default_dir):
            files = glob.glob(os.path.join(default_dir, '*.pt'))
            if files:
                for p in sorted(files)[:5]:
                    inspect_data(p)
            else:
                print(f"No .pt files in {default_dir}")
        else:
            print("Usage: python scripts/inspect_pyg_data.py <path_to.pt>")
            print("  Or run preprocess_fairing_data.py first to generate data.")


if __name__ == '__main__':
    main()
