#!/bin/bash
# Run GT D002 on frontale with Xvfb
set -e

echo "=== Starting Xvfb ==="
~/.local/bin/Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2
echo "Xvfb PID: $XVFB_PID"

export DISPLAY=:99
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH

cd /home/nishioka/Payload2026

# Clean old files
rm -f abaqus_work/Job-GT-D002* 2>/dev/null || true

echo "=== Running GT D002 ==="
abaqus cae noGUI=src/generate_ground_truth.py -- --job Job-GT-D002 --defect gt_defect_d002.json
RET=$?

echo "=== CAE exit code: $RET ==="

# Kill Xvfb
kill $XVFB_PID 2>/dev/null || true

# Show results
echo "=== abaqus_work contents ==="
ls -lh abaqus_work/Job-GT-D002* 2>/dev/null || echo "No Job-GT-D002 files"

exit $RET
