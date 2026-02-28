#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract from existing ODBs in abaqus_work to dataset_output.
Use when ODBs exist and you want to update nodes.csv without re-running Abaqus.

Usage:
  python scripts/extract_existing_odbs.py --doe doe_100.json [--start 0] [--end 100]
"""
import os
import sys
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'extract_odb_results.py')
WORK_DIR = os.path.join(PROJECT_ROOT, 'abaqus_work')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doe', type=str, default='doe_100.json')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='dataset_output')
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.doe)) as f:
        doe = json.load(f)
    samples = doe['samples'][args.start:args.end]

    n_ok = 0
    for i, s in enumerate(samples):
        job_name = s['job_name']
        sample_id = s['id']
        odb_path = os.path.join(WORK_DIR, job_name + '.odb')
        sample_dir = os.path.join(PROJECT_ROOT, args.output_dir, 'sample_%04d' % sample_id)
        param_file = os.path.join(WORK_DIR, 'defect_params_%04d.json' % sample_id)

        if not os.path.exists(odb_path):
            print("[%d] Skip %s: ODB not found" % (i + 1, job_name))
            continue

        if s.get('defect_params') and not os.path.exists(param_file):
            with open(param_file, 'w') as f:
                json.dump(s['defect_params'], f)

        os.makedirs(sample_dir, exist_ok=True)
        cmd = ['abaqus', 'python', EXTRACT_SCRIPT, '--odb', os.path.abspath(odb_path), '--output', sample_dir]
        if os.path.exists(param_file):
            cmd.extend(['--defect_json', param_file])
        if args.strict:
            cmd.append('--strict')
        r = subprocess.call(cmd, cwd=WORK_DIR)
        if r == 0:
            n_ok += 1
            print("[%d] OK: %s" % (i + 1, job_name))
        else:
            print("[%d] Fail: %s" % (i + 1, job_name))

    print("\nExtracted %d / %d samples" % (n_ok, len(samples)))


if __name__ == '__main__':
    main()
