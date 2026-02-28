#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch FEM Generation — Multi-Defect-Type Dataset

Orchestrates batch execution of Abaqus FEM simulations for generating
GNN training data with multiple defect types (debonding, fod, impact).

Workflow per sample:
  1. Write defect parameters to temp JSON
  2. Run Abaqus CAE (generate_fairing_dataset.py) with defect params
  3. Run Abaqus Python (extract_odb_results.py) to export CSV
  4. Move results to dataset output directory
  5. Clean up intermediate files

Usage:
  # Generate DOE first
  python src/generate_doe.py --n_samples 500 --output doe_params.json

  # Run batch
  python src/run_batch.py --doe doe_params.json --output_dir dataset_output

  # Resume from sample 100 (skip completed)
  python src/run_batch.py --doe doe_params.json --output_dir dataset_output --resume

  # Dry run (print commands without executing)
  python src/run_batch.py --doe doe_params.json --dry_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import logging

# Paths (relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GEN_SCRIPT_SIMPLE = os.path.join(SCRIPT_DIR, 'generate_fairing_dataset.py')
GEN_SCRIPT_REALISTIC = os.path.join(SCRIPT_DIR, 'generate_realistic_dataset.py')
GEN_SCRIPT = GEN_SCRIPT_SIMPLE  # overridden by --gen_script
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, 'extract_odb_results.py')
PATCH_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'patch_inp_thermal.py')
WORK_DIR = os.path.join(PROJECT_ROOT, 'abaqus_work')  # default, overridden by --work_dir


def setup_logging(log_file):
    """Configure logging to file and console."""
    logger = logging.getLogger('batch')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_sample(sample, output_dir, n_cpus=4, logger=None, dry_run=False, keep_inp=False, keep_odb=False,
              force=False, strict_extract=False, global_seed=None, defect_seed=None, opening_params=None,
              opening_seed=None):
    """
    Run a single FEM sample (Abaqus model generation + ODB extraction).

    Args:
        sample: dict with 'id', 'job_name', 'defect_params'
        output_dir: base output directory
        n_cpus: number of CPUs for Abaqus
        logger: logging.Logger instance
        dry_run: if True, print commands without executing

    Returns:
        True if successful, False otherwise
    """
    sample_id = sample['id']
    job_name = sample['job_name']
    defect_params = sample.get('defect_params')

    # Output directory for this sample
    if defect_params is None:
        sample_dir = os.path.join(output_dir, 'healthy_baseline')
    else:
        sample_dir = os.path.join(output_dir, 'sample_%04d' % sample_id)

    # Skip if already completed (unless --force)
    if not force and os.path.exists(os.path.join(sample_dir, 'nodes.csv')):
        if logger:
            logger.info("Sample %04d: already completed, skipping" % sample_id)
        return True

    if logger:
        if defect_params:
            defect_type = defect_params.get('defect_type', 'debonding')
            logger.info("Sample %04d [%s]: theta=%.1f z=%.0f r=%.0f" % (
                sample_id, defect_type, defect_params['theta_deg'],
                defect_params['z_center'], defect_params['radius']))
        else:
            logger.info("Sample %04d: healthy baseline" % sample_id)

    os.makedirs(sample_dir, exist_ok=True)

    # --- Step 1: Write defect params to temp JSON ---
    param_file = None
    if defect_params:
        param_file = os.path.join(WORK_DIR, 'defect_params_%04d.json' % sample_id)
        with open(param_file, 'w') as f:
            json.dump(defect_params, f)

    # --- Step 2: Run Abaqus CAE ---
    abaqus_args = []
    if defect_params:
        abaqus_args = ['--param_file', param_file, '--job_name', job_name]
    else:
        abaqus_args = ['--job_name', job_name]
    abaqus_args.extend(['--project_root', PROJECT_ROOT])
    abaqus_args.append('--no_run')  # run_batch patches and runs Abaqus for reliable thermal
    if global_seed is not None:
        abaqus_args.extend(['--global_seed', str(global_seed)])
    if defect_seed is not None:
        abaqus_args.extend(['--defect_seed', str(defect_seed)])
    if opening_seed is not None:
        abaqus_args.extend(['--opening_seed', str(opening_seed)])

    env = os.environ.copy()
    env['PROJECT_ROOT'] = PROJECT_ROOT

    cmd_gen = [
        'abaqus', 'cae', 'noGUI=%s' % GEN_SCRIPT,
        '--', *abaqus_args
    ]

    if dry_run:
        if logger:
            logger.info("  [DRY RUN] %s" % ' '.join(cmd_gen))
        return True

    t0 = time.time()
    timeout_gen = 7200 if (global_seed is not None and global_seed < 25) else 1800  # 2h for fine mesh
    try:
        result = subprocess.run(
            cmd_gen, cwd=WORK_DIR, env=env,
            capture_output=True, text=True, timeout=timeout_gen)

        if result.returncode != 0:
            if logger:
                logger.error("Sample %04d: Abaqus CAE failed (rc=%d)" %
                             (sample_id, result.returncode))
                logger.error("  stderr: %s" % result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        if logger:
            logger.error("Sample %04d: Abaqus CAE timed out" % sample_id)
        return False
    except Exception as e:
        if logger:
            logger.error("Sample %04d: Abaqus CAE error: %s" %
                         (sample_id, str(e)))
        return False

    t_gen = time.time() - t0
    if logger:
        logger.info("  FEM generation: %.1f sec" % t_gen)

    # --- Step 2b: Patch INP for thermal load (run_batch ensures this runs) ---
    inp_path = os.path.join(WORK_DIR, '%s.inp' % job_name)
    if os.path.exists(inp_path) and os.path.exists(PATCH_SCRIPT):
        patch_result = subprocess.run(
            [sys.executable, PATCH_SCRIPT, inp_path],
            cwd=WORK_DIR, capture_output=True, text=True, timeout=30)
        if patch_result.returncode == 0 and logger:
            if 'Patched' in (patch_result.stdout or ''):
                logger.info("  INP patched for thermal load")

    # --- Step 2c: Run Abaqus job ---
    try:
        run_result = subprocess.run(
            ['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp', 'cpus=%d' % n_cpus],
            cwd=WORK_DIR, capture_output=True, text=True, timeout=7200)
        if run_result.returncode != 0 and logger:
            logger.warning("  Abaqus job exit %d (check .sta, .msg)" % run_result.returncode)
    except Exception as e:
        if logger:
            logger.error("Sample %04d: Abaqus job error: %s" % (sample_id, str(e)))
        return False

    # --- Step 3: Extract ODB results ---
    odb_path = os.path.join(WORK_DIR, '%s.odb' % job_name)
    lck_path = os.path.join(WORK_DIR, '%s.lck' % job_name)
    sta_path = os.path.join(WORK_DIR, '%s.sta' % job_name)

    # Wait for solver completion via .sta file (abaqus job=... runs in background)
    solver_timeout = 7200  # 2 hours max
    poll_interval = 5  # check every 5 sec
    solver_ok = False
    for _ in range(solver_timeout // poll_interval):
        if os.path.exists(sta_path):
            try:
                with open(sta_path, 'r') as f_sta:
                    sta_content = f_sta.read()
                if 'COMPLETED SUCCESSFULLY' in sta_content:
                    solver_ok = True
                    break
                if 'HAS NOT BEEN COMPLETED' in sta_content or 'ABORTED' in sta_content:
                    if logger:
                        logger.error("Sample %04d: Solver failed (see .sta)" % sample_id)
                    return False
            except (IOError, OSError):
                pass
        time.sleep(poll_interval)

    if not solver_ok:
        if logger:
            logger.error("Sample %04d: Solver timed out (%d sec)" %
                         (sample_id, solver_timeout))
        return False

    if logger:
        logger.info("  Solver completed successfully")

    # Wait for ODB file to appear
    for _ in range(30):
        if os.path.exists(odb_path):
            break
        time.sleep(1)
    if not os.path.exists(odb_path):
        if logger:
            logger.error("Sample %04d: ODB not found: %s" %
                         (sample_id, odb_path))
        return False

    # Wait for lock file release
    for _ in range(60):
        if not os.path.exists(lck_path):
            break
        time.sleep(1)
    if os.path.exists(lck_path):
        try:
            os.remove(lck_path)
        except OSError:
            pass

    # File size stabilization
    prev_size = -1
    for _ in range(10):
        try:
            sz = os.path.getsize(odb_path)
            if sz == prev_size and sz > 0:
                break
            prev_size = sz
        except OSError:
            pass
        time.sleep(0.5)
    time.sleep(2)  # Extra buffer for filesystem sync

    # Use absolute paths to avoid cwd ambiguity with abaqus python
    script_rel = os.path.relpath(EXTRACT_SCRIPT, PROJECT_ROOT)
    cmd_extract = ['abaqus', 'python', script_rel, '--odb', os.path.abspath(odb_path),
                   '--output', os.path.abspath(sample_dir)]
    if defect_params and param_file:
        cmd_extract.extend(['--defect_json', os.path.abspath(param_file)])

    # Retry extraction (large ODBs may need extra time to be readable)
    max_retries = 3
    retry_delay = 15
    result = None
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd_extract, cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                break
            if attempt < max_retries - 1:
                if logger:
                    logger.warning("Sample %04d: extraction failed (rc=%d), retry %d/%d in %ds" %
                                  (sample_id, result.returncode, attempt + 1, max_retries - 1, retry_delay))
                time.sleep(retry_delay)
        except Exception as e:
            if logger:
                logger.error("Sample %04d: extraction error: %s" %
                             (sample_id, str(e)))
            return False

    if result and result.returncode != 0:
        if logger:
            logger.error("Sample %04d: ODB extraction failed (rc=%d)" %
                         (sample_id, result.returncode))
            # extract_odb_results.py prints errors to stdout
            err_msg = (result.stdout or '') + (result.stderr or '')
            if err_msg.strip():
                logger.error("  output: %s" % err_msg[:800])
        return False

    t_total = time.time() - t0
    if logger:
        logger.info("  Total: %.1f sec" % t_total)

    # --- Step 4: Optionally copy INP to sample dir (for archival) ---
    inp_path = os.path.join(WORK_DIR, '%s.inp' % job_name)
    if os.path.exists(inp_path) and keep_inp:
        shutil.copy2(inp_path, os.path.join(sample_dir, 'model.inp'))

    # --- Step 5: Clean up intermediate files ---
    exts_to_clean = ['.dat', '.msg', '.com', '.prt', '.sim',
                     '.sta', '.lck', '.023', '.mdl', '.stt', '.res']
    if not keep_odb:
        exts_to_clean.insert(0, '.odb')
    for ext in exts_to_clean:
        fpath = os.path.join(WORK_DIR, '%s%s' % (job_name, ext))
        if os.path.exists(fpath):
            os.remove(fpath)

    if param_file and os.path.exists(param_file):
        os.remove(param_file)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Batch FEM generation for multi-type defect dataset')
    parser.add_argument('--doe', type=str, required=True,
                        help='DOE parameters JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: dataset_output_<mesh>mm_<n>)')
    parser.add_argument('--n_cpus', type=int, default=4,
                        help='CPUs per Abaqus job (default: 4)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip completed samples')
    parser.add_argument('--start', type=int, default=0,
                        help='Start from sample index')
    parser.add_argument('--end', type=int, default=None,
                        help='End at sample index (exclusive)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--keep_inp', action='store_true',
                        help='Copy .inp to each sample dir')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing samples (re-generate)')
    parser.add_argument('--keep_odb', action='store_true',
                        help='Keep ODB files after extraction (for debugging)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail extraction if physics are all zero (thermal/load check)')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Override mesh GLOBAL_SEED (mm). Fine: 12–15, ideal: 10–12.')
    parser.add_argument('--defect_seed', type=float, default=None,
                        help='Override mesh DEFECT_SEED (mm). Fine: 5–8.')
    parser.add_argument('--opening_seed', type=float, default=None,
                        help='Override mesh OPENING_SEED (mm). Realistic model only.')
    parser.add_argument('--gen_script', type=str, default='simple',
                        choices=['simple', 'realistic'],
                        help='Generation script: simple (default) or realistic (openings+frames+Tie)')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Override Abaqus work directory (for parallel execution on multiple machines)')
    args = parser.parse_args()

    # Switch generation script if realistic mode
    global GEN_SCRIPT, WORK_DIR
    if args.gen_script == 'realistic':
        GEN_SCRIPT = GEN_SCRIPT_REALISTIC
    if args.work_dir:
        WORK_DIR = os.path.join(PROJECT_ROOT, args.work_dir)

    # Load DOE
    with open(args.doe, 'r') as f:
        doe = json.load(f)

    samples = doe['samples']

    # Default output_dir: dataset_output_<mesh>mm_<n> (mesh from generate_fairing_dataset GLOBAL_SEED)
    if args.output_dir is None:
        mesh_mm = 50  # match generate_fairing_dataset.GLOBAL_SEED
        n = doe.get('n_defective', doe.get('n_total', len(samples)))
        args.output_dir = 'dataset_output_%dmm_%d' % (mesh_mm, n)
    if args.end is not None:
        samples = samples[args.start:args.end]
    else:
        samples = samples[args.start:]

    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    # Logging
    log_file = os.path.join(output_dir, 'batch_log.txt')
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Batch FEM Generation")
    logger.info("  DOE: %s (%d samples)" % (args.doe, len(samples)))
    logger.info("  Output: %s" % output_dir)
    logger.info("  CPUs: %d" % args.n_cpus)
    logger.info("  Gen script: %s (%s)" % (args.gen_script, GEN_SCRIPT))
    logger.info("=" * 60)

    # Run samples
    n_success = 0
    n_fail = 0
    n_skip = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        logger.info("[%d/%d] Processing sample %04d..." %
                    (i + 1, len(samples), sample['id']))

        success = run_sample(
            sample, output_dir,
            n_cpus=args.n_cpus,
            logger=logger,
            dry_run=args.dry_run,
            keep_inp=getattr(args, 'keep_inp', False),
            keep_odb=getattr(args, 'keep_odb', False),
            force=getattr(args, 'force', False),
            strict_extract=getattr(args, 'strict', False),
            global_seed=args.global_seed,
            defect_seed=args.defect_seed,
            opening_params=doe.get('opening_params'),
            opening_seed=args.opening_seed)

        if success:
            n_success += 1
        else:
            n_fail += 1

    t_total = time.time() - t_start

    logger.info("=" * 60)
    logger.info("Batch complete: %d success, %d failed (%.1f min total)" %
                (n_success, n_fail, t_total / 60.0))
    if n_fail > 0:
        logger.warning("Check %s for error details" % log_file)


if __name__ == '__main__':
    main()
