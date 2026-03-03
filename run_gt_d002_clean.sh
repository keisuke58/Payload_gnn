#!/bin/bash
# Run GT D002 — ensure libpython3.10 is found before system libpython3.9

echo "=== Starting with pristine environment ==="
date

cd /home/nishioka/Payload2026

# Clean old files
rm -f abaqus_work/Job-GT-D002* 2>/dev/null

# Prepend Abaqus code/bin to LD_LIBRARY_PATH so libpython3.10 is found first
ABAQUS_BIN=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin

env -i \
  HOME=/home/nishioka \
  USER=nishioka \
  DISPLAY=:99 \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 \
  LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=$ABAQUS_BIN \
  /home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus cae noGUI=src/generate_ground_truth.py -- --job Job-GT-D002 --defect gt_defect_d002.json

RET=$?

echo "=== CAE exit code: $RET ==="
date
echo "=== abaqus_work contents ==="
ls -lh abaqus_work/Job-GT-D002* 2>/dev/null || echo "No Job-GT-D002 files"

exit $RET
