#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch FEM Generation — Debonding Dataset

Orchestrates batch execution of Abaqus FEM simulations for generating
GNN training data with debonding defects.

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
GEN_SCRIPT = os.path.join(SCRIPT_DIR, 'generate_fairing_dataset.py')
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, 'extract_odb_results.py')
WORK_DIR = os.path.join(PROJECT_ROOT, 'abaqus_work')


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


def run_sample(sample, output_dir, n_cpus=4, logger=None, dry_run=False):
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

    # Skip if already completed
    if os.path.exists(os.path.join(sample_dir, 'nodes.csv')):
        if logger:
            logger.info("Sample %04d: already completed, skipping" % sample_id)
        return True

    if logger:
        if defect_params:
            logger.info("Sample %04d: theta=%.1f z=%.0f r=%.0f" % (
                sample_id, defect_params['theta_deg'],
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

    cmd_gen = [
        'abaqus', 'cae', 'noGUI=%s' % GEN_SCRIPT,
        '--', *abaqus_args
    ]

    if dry_run:
        if logger:
            logger.info("  [DRY RUN] %s" % ' '.join(cmd_gen))
        return True

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd_gen, cwd=WORK_DIR,
            capture_output=True, text=True, timeout=1800)  # 30 min timeout

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

    # --- Step 3: Extract ODB results ---
    odb_path = os.path.join(WORK_DIR, '%s.odb' % job_name)
    if not os.path.exists(odb_path):
        if logger:
            logger.error("Sample %04d: ODB not found: %s" %
                         (sample_id, odb_path))
        return False

    cmd_extract = [
        'abaqus', 'python', EXTRACT_SCRIPT,
        odb_path, sample_dir,
    ]
    if defect_params and param_file:
        cmd_extract.extend(['--defect_json', param_file])

    try:
        result = subprocess.run(
            cmd_extract, cwd=WORK_DIR,
            capture_output=True, text=True, timeout=600)  # 10 min timeout

        if result.returncode != 0:
            if logger:
                logger.error("Sample %04d: ODB extraction failed (rc=%d)" %
                             (sample_id, result.returncode))
                logger.error("  stderr: %s" % result.stderr[:500])
            return False

    except Exception as e:
        if logger:
            logger.error("Sample %04d: extraction error: %s" %
                         (sample_id, str(e)))
        return False

    t_total = time.time() - t0
    if logger:
        logger.info("  Total: %.1f sec" % t_total)

    # --- Step 4: Clean up intermediate files ---
    for ext in ['.odb', '.dat', '.msg', '.com', '.prt', '.sim',
                '.sta', '.lck', '.023', '.mdl', '.stt', '.res']:
        fpath = os.path.join(WORK_DIR, '%s%s' % (job_name, ext))
        if os.path.exists(fpath):
            os.remove(fpath)

    if param_file and os.path.exists(param_file):
        os.remove(param_file)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Batch FEM generation for debonding dataset')
    parser.add_argument('--doe', type=str, required=True,
                        help='DOE parameters JSON file')
    parser.add_argument('--output_dir', type=str, default='dataset_output',
                        help='Output directory for CSV samples')
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
    args = parser.parse_args()

    # Load DOE
    with open(args.doe, 'r') as f:
        doe = json.load(f)

    samples = doe['samples']
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
            dry_run=args.dry_run)

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
