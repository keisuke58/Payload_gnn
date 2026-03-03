#!/bin/bash
# Run GT D002 on frontale with Xvfb — clean environment (no conda)

# Deactivate conda completely
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE
unset PYTHONPATH PYTHONHOME
# Remove conda/pyenv from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'conda|pyenv|miniconda' | tr '\n' ':' | sed 's/:$//')

echo "=== Clean PATH ==="
echo "$PATH" | tr ':' '\n' | head -10

echo "=== Starting Xvfb ==="
~/.local/bin/Xvfb :99 -screen 0 1024x768x24 2>/dev/null &
XVFB_PID=$!
sleep 2

export DISPLAY=:99
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH

cd /home/nishioka/Payload2026

# Clean old files
rm -f abaqus_work/Job-GT-D002* 2>/dev/null

echo "=== Running GT D002 ==="
echo "which abaqus: $(which abaqus)"
date
abaqus cae noGUI=src/generate_ground_truth.py -- --job Job-GT-D002 --defect gt_defect_d002.json
RET=$?
date

echo "=== CAE exit code: $RET ==="

# Kill Xvfb
kill $XVFB_PID 2>/dev/null

# Show results
echo "=== abaqus_work contents ==="
ls -lh abaqus_work/Job-GT-D002* 2>/dev/null || echo "No Job-GT-D002 files"

exit $RET
