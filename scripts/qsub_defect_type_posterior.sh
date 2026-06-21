#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -N defect_type_post
#PBS -V
# Joint posterior p(defect TYPE, LOCATION | guided-wave measurement).
# Stage 1 (scan) reads ~55 x 2GB OGW pickles (IO-bound, ~30-60 min);
# stages 2-3 (train/demo) are CPU-light. Re-runs skip the scan if the
# feature cache results/defect_type_posterior/ogw_type_cache.npz exists.
cd ~/Payload2026
export OMP_NUM_THREADS=16
PY=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3
$PY scripts/defect_type_posterior.py all
