#!/bin/bash
# run_gw_sweep_sequential.sh
# Run guided wave sweep jobs sequentially on frontale04.
# Each job waits for the previous to finish (QXT license constraint).
#
# Usage: nohup bash scripts/run_gw_sweep_sequential.sh > gw_sweep.log 2>&1 &

set -e
cd ~/Payload2026/abaqus_work

ENV="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin"

JOBS="Job-GW-D3-r40 Job-GW-D4-near Job-GW-D5-edge"

# Wait for D2 to finish first
echo "Waiting for Job-GW-D2-r15 to complete..."
while [ -f Job-GW-D2-r15.lck ]; do
    sleep 30
done
echo "Job-GW-D2-r15 done."

for JOB in $JOBS; do
    echo "$(date): Starting $JOB"
    $ENV abaqus job=$JOB input=$JOB.inp cpus=4
    echo "$(date): $JOB completed."
    echo "---"
done

echo "$(date): All sweep jobs complete."
