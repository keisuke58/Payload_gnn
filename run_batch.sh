#!/bin/bash
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
cd /home/nishioka/Payload2026

echo "Starting batch at $(date)"
echo "PWD: $(pwd)"
echo "abaqus: $(which abaqus)"

python3 src/run_sector12_batch.py \
  --doe doe_sector12_100.json \
  --template abaqus_work/Job-CZM-S12-Test.inp \
  --workdir abaqus_work/batch_s12_100 \
  --cpus 4 --memory "16 gb" \
  --parallel 4 \
  --no-extract

echo "Batch finished at $(date)"
