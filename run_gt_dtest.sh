#!/bin/bash
# Run GT defect test — barrel region (z=1500, r=60mm)
echo "=== GT Defect Test ==="
date

cd /home/nishioka/Payload2026

# Clean old files
rm -f Job-GT-DTEST* 2>/dev/null
rm -f abaqus_work/Job-GT-DTEST* 2>/dev/null

ABAQUS_BIN=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin

env -i \
  HOME=/home/nishioka \
  USER=nishioka \
  DISPLAY=:99 \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 \
  LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=$ABAQUS_BIN \
  /home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus cae noGUI=src/generate_ground_truth.py -- --job Job-GT-DTEST --defect gt_defect_test.json

RET=$?

echo "=== CAE exit code: $RET ==="
date

# If INP exists, run solver
if [ -f "Job-GT-DTEST.inp" ]; then
    echo "=== INP found, checking for errors ==="
    ls -lh Job-GT-DTEST.inp Job-GT-DTEST.cae 2>/dev/null
    
    echo "=== Waiting for solver ==="
    # Solver launched by CAE, wait for completion
    sleep 5
    while ps aux | grep "Job-GT-DTEST" | grep -v grep > /dev/null 2>&1; do
        sleep 10
    done
    
    echo "=== Solver done ==="
    date
    
    if [ -f "Job-GT-DTEST.dat" ]; then
        grep "ERROR\|WARNING" Job-GT-DTEST.dat | tail -20
    fi
    
    ls -lh Job-GT-DTEST.odb 2>/dev/null || echo "No ODB"
fi

echo "=== Done ==="
exit $RET
