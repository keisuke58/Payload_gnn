#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -N anomaly_xfer
#PBS -V
cd ~/Payload2026
export OMP_NUM_THREADS=16
PY=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3
$PY scripts/anomaly_transfer_eval.py --objective fm
$PY scripts/anomaly_transfer_eval.py --objective ddpm
