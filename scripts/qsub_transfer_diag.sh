#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -N transfer_diag
#PBS -V
cd ~/Payload2026
export OMP_NUM_THREADS=16
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3 scripts/transfer_diagnosis.py
