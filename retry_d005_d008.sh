#!/bin/bash
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
cd /home/nishioka/Payload2026

for i in 005 006 007 008; do
  JOB="Job-S12-D${i}"
  DIR="abaqus_work/batch_s12_100/${JOB}"
  ABS_DIR="/home/nishioka/Payload2026/${DIR}"
  INP="${ABS_DIR}/${JOB}.inp"

  echo "=== Retrying $JOB ==="

  # Clean up stale files from killed process
  rm -f ${ABS_DIR}/${JOB}.lck ${ABS_DIR}/${JOB}.sta ${ABS_DIR}/${JOB}.odb \
        ${ABS_DIR}/${JOB}.sim ${ABS_DIR}/${JOB}.msg ${ABS_DIR}/${JOB}.dat \
        ${ABS_DIR}/${JOB}.com ${ABS_DIR}/${JOB}.prt ${ABS_DIR}/${JOB}.mdl \
        ${ABS_DIR}/${JOB}.stt ${ABS_DIR}/${JOB}.res ${ABS_DIR}/${JOB}.log

  # Run solver from job directory
  cd "${ABS_DIR}"
  abaqus job=${JOB} input=${INP} cpus=4 memory="16 gb" interactive

  if grep -q "COMPLETED SUCCESSFULLY" ${ABS_DIR}/${JOB}.sta 2>/dev/null; then
    echo "$JOB: SUCCESS"
  else
    echo "$JOB: FAILED"
  fi
  cd /home/nishioka/Payload2026
done

echo "Retry complete at $(date)"
