#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run Abaqus analysis with thermal patch on existing INPs.
Use when INPs exist in abaqus_work but ODBs were generated without thermal.

Usage:
  python scripts/re_run_thermal_only.py --doe doe_100.json [--start 0] [--end 100]
"""
import os
import sys
import json
import argparse
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'patch_inp_thermal.py')
EXTRACT_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'extract_odb_results.py')
WORK_DIR = os.path.join(PROJECT_ROOT, 'abaqus_work')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doe', type=str, default='doe_100.json')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='dataset_output')
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.doe)) as f:
        doe = json.load(f)
    samples = doe['samples'][args.start:args.end]

    for i, s in enumerate(samples):
        job_name = s['job_name']
        sample_id = s['id']
        inp_path = os.path.join(WORK_DIR, job_name + '.inp')
        odb_path = os.path.join(WORK_DIR, job_name + '.odb')
        sample_dir = os.path.join(PROJECT_ROOT, args.output_dir, 'sample_%04d' % sample_id)
        param_file = os.path.join(WORK_DIR, 'defect_params_%04d.json' % sample_id)
        # Ensure defect_params exists for extraction
        if s.get('defect_params') and not os.path.exists(param_file):
            with open(param_file, 'w') as f:
                json.dump(s['defect_params'], f)

        if not os.path.exists(inp_path):
            print("[%d] Skip %s: INP not found" % (i + 1, job_name))
            continue

        # 1. Patch INP
        r = subprocess.call([sys.executable, PATCH_SCRIPT, inp_path], cwd=WORK_DIR)
        if r != 0:
            print("[%d] Patch failed: %s" % (i + 1, job_name))
            continue

        # 2. Run Abaqus (returns when queued; we must wait for ODB)
        subprocess.Popen(['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp', 'cpus=4'],
                         cwd=WORK_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for ODB (job runs in background, poll until complete)
        sta_path = odb_path.replace('.odb', '.sta')
        for wait in range(900):  # up to 15 min per job
            time.sleep(1)
            if os.path.exists(odb_path) and os.path.getsize(odb_path) > 100000:
                lck = odb_path.replace('.odb', '.lck')
                if os.path.exists(lck):
                    time.sleep(2)
                    try:
                        os.remove(lck)
                    except OSError:
                        pass
                break
            if wait > 0 and wait % 60 == 0:
                print("  ... waiting %d min for %s" % (wait // 60, job_name))
        if not os.path.exists(odb_path):
            print("[%d] ODB not found after run: %s" % (i + 1, job_name))
            continue

        # 3. Extract
        os.makedirs(sample_dir, exist_ok=True)
        cmd = ['abaqus', 'python', EXTRACT_SCRIPT, '--odb', os.path.abspath(odb_path), '--output', sample_dir]
        if os.path.exists(param_file):
            cmd.extend(['--defect_json', param_file])
        r = subprocess.call(cmd, cwd=WORK_DIR)
        if r != 0:
            print("[%d] Extract failed: %s" % (i + 1, job_name))
        else:
            print("[%d] OK: %s" % (i + 1, job_name))


if __name__ == '__main__':
    main()
