#!/bin/bash
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
cd /home/nishioka/Payload2026

echo "=== GT Defect D002 generation start: $(date) ==="
echo "PWD: $(pwd)"
echo "Defect params:"
cat gt_defect_d002.json

abaqus cae noGUI=src/generate_ground_truth.py -- \
    --job Job-GT-D002 \
    --defect gt_defect_d002.json

echo "=== CAE finished: $(date) ==="
echo "=== Checking for INP ==="
ls -la abaqus_work/Job-GT-D002.*
