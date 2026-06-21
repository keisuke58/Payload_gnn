#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N defect_confound
#PBS -V
# NET defect-type identifiability after removing temporal/temperature confounds.
# Stage scan reads ALL ~55 x 2GB OGW pickles (IO-bound, ~1-2 h); analyze is
# CPU-light. Re-runs skip scan if results/defect_type_posterior/confound_cache.npz
# already exists.
cd ~/Payload2026
export OMP_NUM_THREADS=16
PY=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3
$PY scripts/defect_type_confound.py all
