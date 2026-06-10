#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N ddpm_augment
#PBS -V

# qsub_diffusion_augment.sh — FM/DDPM synthetic data generation + downstream eval
# Prereq: results/payload_ddpm/{best_fm.pt,norm_stats.npz} copied from vancouver
# Usage:
#   qsub scripts/qsub_diffusion_augment.sh                       # FM (default)
#   qsub -v OBJECTIVE=ddpm scripts/qsub_diffusion_augment.sh     # DDPM

cd ~/Payload2026
export OMP_NUM_THREADS=16
PY=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3
OBJECTIVE=${OBJECTIVE:-fm}

echo "=== augment: objective=$OBJECTIVE $(date) ==="
$PY scripts/payload_diffusion.py --mode augment --objective $OBJECTIVE \
    --n_synth 5000 --batch_size 256 --guidance 2.0

echo "=== downstream eval: baseline vs +synthetic $(date) ==="
SYNTH=results/payload_ddpm/synthetic_damaged.npz
for N in 500 2000 5000; do
    echo "--- n_synth = $N ---"
    $PY scripts/augment_eval.py --synth $SYNTH --n_synth $N --seeds 3
done

echo "=== done $(date) ==="
