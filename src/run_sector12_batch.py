#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sector12_batch.py — 1/12セクター バッチ実行オーケストレーション

DOE JSON → INPパッチ → ソルバー実行 → ODB抽出 を一括管理。
直列・並列実行に対応。

Usage:
    # ローカル（直列）
    python src/run_sector12_batch.py \
        --doe doe_sector12.json \
        --template abaqus_work/Job-CZM-S12-Test.inp \
        --workdir abaqus_work/batch_s12 \
        --cpus 4 --memory "8 gb"

    # frontale（並列4ジョブ）
    python src/run_sector12_batch.py \
        --doe doe_sector12.json \
        --template abaqus_work/Job-CZM-S12-Test.inp \
        --workdir abaqus_work/batch_s12 \
        --cpus 4 --memory "16 gb" --parallel 4

    # サンプル範囲指定（リトライ用）
    python src/run_sector12_batch.py \
        --doe doe_sector12.json \
        --template abaqus_work/Job-CZM-S12-Test.inp \
        --workdir abaqus_work/batch_s12 \
        --cpus 4 --memory "8 gb" --range 5-10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def _patch_inp(template, defect_params, output_path):
    """Run patch_inp_defects.py to create defected INP."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'scripts', 'patch_inp_defects.py')

    # Write temp defect params JSON
    defect_json = output_path.replace('.inp', '_defect_tmp.json')
    with open(defect_json, 'w') as f:
        json.dump(defect_params, f)

    cmd = [sys.executable, script,
           '--template', template,
           '--defect_json', defect_json,
           '--output', output_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Clean up temp file
    if os.path.exists(defect_json):
        os.remove(defect_json)

    if result.returncode != 0:
        print("  PATCH FAILED: %s" % result.stderr[-500:])
        return False
    # Print patcher output (indented)
    for line in result.stdout.strip().split('\n'):
        print("  " + line)
    return True


def _check_solver_success(job_dir, job_name):
    """Check if solver completed successfully by reading .sta file."""
    sta_path = os.path.join(job_dir, job_name + '.sta')
    if not os.path.exists(sta_path):
        return False
    with open(sta_path) as f:
        return 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in f.read()


def _run_solver(job_name, inp_path, job_dir, cpus, memory):
    """Run Abaqus solver interactively."""
    cmd = 'abaqus job=%s input=%s cpus=%d memory="%s" interactive' % (
        job_name, os.path.abspath(inp_path), cpus, memory)

    try:
        result = subprocess.run(cmd, shell=True, cwd=job_dir,
                                capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print("  SOLVER TIMEOUT (1h)")
        return False

    return _check_solver_success(job_dir, job_name)


def _run_extract(odb_path, output_dir, defect_json):
    """Run ODB extraction via abaqus python."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'extract_odb_results.py')
    cmd = 'abaqus python %s --odb %s --output %s --defect_json %s' % (
        script, odb_path, output_dir, defect_json)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  EXTRACT TIMEOUT")
        return False

    if result.returncode != 0:
        print("  EXTRACT FAILED: %s" % result.stderr[-500:])
        return False
    return True


def _process_sample(sample, template, workdir, cpus, memory, do_extract):
    """Process one sample: patch → solve → extract.

    Returns: status string ('completed', 'patch_failed', 'solver_failed', 'extract_failed')
    """
    job_name = sample['job_name']
    defect_params = sample['defect_params']
    job_dir = os.path.join(workdir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    inp_path = os.path.join(job_dir, job_name + '.inp')
    defect_json = os.path.join(job_dir, 'defect_params.json')

    # Save defect params for later extraction
    with open(defect_json, 'w') as f:
        json.dump(defect_params, f, indent=2)

    # Handle healthy samples (no patching needed)
    is_healthy = (defect_params is None or
                  defect_params.get('defect_type') == 'healthy')

    # Step 1: Patch INP
    if is_healthy:
        print("  [%s] Copying template (healthy)" % job_name)
        shutil.copy2(template, inp_path)
    else:
        print("  [%s] Patching INP..." % job_name)
        if not _patch_inp(template, defect_params, inp_path):
            return 'patch_failed'

    # Step 2: Run solver
    print("  [%s] Running solver..." % job_name)
    if not _run_solver(job_name, inp_path, job_dir, cpus, memory):
        return 'solver_failed'
    print("  [%s] Solver OK" % job_name)

    # Step 3: Extract ODB
    if do_extract:
        odb_path = os.path.join(job_dir, job_name + '.odb')
        results_dir = os.path.join(job_dir, 'results')
        print("  [%s] Extracting ODB..." % job_name)
        if not _run_extract(odb_path, results_dir, defect_json):
            return 'extract_failed'
        print("  [%s] Extract OK" % job_name)

    return 'completed'


def run_serial(samples, template, workdir, cpus, memory, do_extract):
    """Run all samples sequentially."""
    results = {}
    for i, sample in enumerate(samples):
        job = sample['job_name']
        print("\n=== [%d/%d] %s ===" % (i + 1, len(samples), job))
        t0 = time.time()
        status = _process_sample(sample, template, workdir, cpus, memory,
                                 do_extract)
        elapsed = time.time() - t0
        results[job] = status
        print("  [%s] -> %s (%.1fs)" % (job, status, elapsed))
    return results


def run_parallel(samples, template, workdir, cpus, memory, n_parallel,
                 do_extract):
    """Run solver phase with N parallel Abaqus jobs.

    Phase 1: Patch all INPs (fast, serial)
    Phase 2: Run solvers in parallel
    Phase 3: Extract ODBs (serial)
    """
    results = {}

    # Phase 1: Patch all INPs
    print("=== Phase 1: Patching %d INPs ===" % len(samples))
    patch_ok = []
    for sample in samples:
        job_name = sample['job_name']
        defect_params = sample['defect_params']
        job_dir = os.path.join(workdir, job_name)
        os.makedirs(job_dir, exist_ok=True)

        inp_path = os.path.join(job_dir, job_name + '.inp')
        defect_json = os.path.join(job_dir, 'defect_params.json')

        with open(defect_json, 'w') as f:
            json.dump(defect_params, f, indent=2)

        is_healthy = (defect_params is None or
                      defect_params.get('defect_type') == 'healthy')

        if is_healthy:
            shutil.copy2(template, inp_path)
            patch_ok.append(sample)
            print("  [%s] Copied (healthy)" % job_name)
        elif _patch_inp(template, defect_params, inp_path):
            patch_ok.append(sample)
        else:
            results[job_name] = 'patch_failed'
            print("  [%s] PATCH FAILED" % job_name)

    # Phase 2: Parallel solver execution
    print("\n=== Phase 2: Running %d solvers (parallel=%d) ===" % (
        len(patch_ok), n_parallel))
    active = {}   # job_name -> (Popen, sample)
    queue = list(patch_ok)

    while queue or active:
        # Launch jobs up to limit
        while queue and len(active) < n_parallel:
            sample = queue.pop(0)
            job_name = sample['job_name']
            job_dir = os.path.join(workdir, job_name)
            inp_path = os.path.join(job_dir, job_name + '.inp')

            cmd = 'abaqus job=%s input=%s cpus=%d memory="%s" interactive' % (
                job_name, os.path.abspath(inp_path), cpus, memory)

            print("  [%s] Starting solver..." % job_name)
            proc = subprocess.Popen(cmd, shell=True, cwd=job_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            active[job_name] = (proc, sample)

        # Poll for completion
        completed = []
        for job_name, (proc, sample) in active.items():
            if proc.poll() is not None:
                completed.append(job_name)
                job_dir = os.path.join(workdir, job_name)
                if _check_solver_success(job_dir, job_name):
                    results[job_name] = 'solver_done'
                    print("  [%s] Solver OK" % job_name)
                else:
                    results[job_name] = 'solver_failed'
                    print("  [%s] Solver FAILED" % job_name)

        for job_name in completed:
            del active[job_name]

        if active and not completed:
            time.sleep(2)

    # Phase 3: Extract ODBs
    if do_extract:
        extract_targets = [s for s in patch_ok
                           if results.get(s['job_name']) == 'solver_done']
        print("\n=== Phase 3: Extracting %d ODBs ===" % len(extract_targets))
        for sample in extract_targets:
            job_name = sample['job_name']
            job_dir = os.path.join(workdir, job_name)
            odb_path = os.path.join(job_dir, job_name + '.odb')
            results_dir = os.path.join(job_dir, 'results')
            defect_json = os.path.join(job_dir, 'defect_params.json')

            if _run_extract(odb_path, results_dir, defect_json):
                results[job_name] = 'completed'
                print("  [%s] Extract OK" % job_name)
            else:
                results[job_name] = 'extract_failed'
                print("  [%s] Extract FAILED" % job_name)
    else:
        for job_name in list(results.keys()):
            if results[job_name] == 'solver_done':
                results[job_name] = 'completed'

    return results


def _save_status(workdir, samples, results, elapsed):
    """Save batch status JSON."""
    status = {
        'total': len(samples),
        'completed': sum(1 for v in results.values() if v == 'completed'),
        'failed': {k: v for k, v in results.items() if v != 'completed'},
        'elapsed_sec': round(elapsed, 1),
        'results': results,
    }
    path = os.path.join(workdir, 'batch_status.json')
    with open(path, 'w') as f:
        json.dump(status, f, indent=2)
    return status, path


def main():
    parser = argparse.ArgumentParser(
        description='Sector12 batch: patch -> solve -> extract')
    parser.add_argument('--doe', required=True,
                        help='DOE JSON file')
    parser.add_argument('--template', required=True,
                        help='Template INP file')
    parser.add_argument('--workdir', required=True,
                        help='Working directory for outputs')
    parser.add_argument('--cpus', type=int, default=4,
                        help='CPUs per job (default: 4)')
    parser.add_argument('--memory', type=str, default='8 gb',
                        help='Memory per job (default: "8 gb")')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel solver jobs (default: 1)')
    parser.add_argument('--no-extract', action='store_true',
                        help='Skip ODB extraction step')
    parser.add_argument('--range', type=str, default=None,
                        help='Sample range "start-end" (1-indexed) or single index')
    args = parser.parse_args()

    with open(args.doe) as f:
        doe = json.load(f)

    samples = doe['samples']

    # Filter by range
    if args.range:
        if '-' in args.range:
            start, end = args.range.split('-')
            samples = samples[int(start) - 1:int(end)]
        else:
            idx = int(args.range) - 1
            samples = [samples[idx]]

    os.makedirs(args.workdir, exist_ok=True)

    print("Batch: %d samples, parallel=%d, cpus=%d/job, memory=%s" % (
        len(samples), args.parallel, args.cpus, args.memory))
    print("Template: %s" % args.template)
    print("Workdir: %s" % args.workdir)

    t0 = time.time()

    do_extract = not args.no_extract
    if args.parallel <= 1:
        results = run_serial(samples, args.template, args.workdir,
                             args.cpus, args.memory, do_extract)
    else:
        results = run_parallel(samples, args.template, args.workdir,
                               args.cpus, args.memory, args.parallel,
                               do_extract)

    elapsed = time.time() - t0
    status, status_path = _save_status(args.workdir, samples, results, elapsed)

    print("\n=== Batch Complete ===")
    print("  Completed: %d/%d" % (status['completed'], status['total']))
    n_failed = len(status['failed'])
    print("  Failed: %d" % n_failed)
    print("  Time: %.1fs (%.1fs/sample avg)" % (
        elapsed, elapsed / max(len(samples), 1)))
    print("  Status: %s" % status_path)

    if status['failed']:
        print("\nFailed jobs:")
        for job, reason in sorted(status['failed'].items()):
            print("  %s -> %s" % (job, reason))

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == '__main__':
    main()
