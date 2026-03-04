#!/bin/bash
# batch_gw_sweep.sh
# Generate and run guided wave defect sweep on frontale.
#
# Usage: bash scripts/batch_gw_sweep.sh [generate|run|extract|all]
#
# Prerequisite: generate_guided_wave.py already copied to frontale04

set -e

REMOTE=frontale04
WORK_DIR=~/Payload2026/abaqus_work
SRC_DIR=~/Payload2026/src
SCRIPTS_DIR=~/Payload2026/scripts
CPUS=4

# Abaqus environment
ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

# DOE cases: job_name defect_json
# Healthy-v2 and Debond-v2 (r=25, x=80) already exist
CASES=(
  "Job-GW-D2-r15  {\"x_center\":80,\"y_center\":0,\"radius\":15}"
  "Job-GW-D3-r40  {\"x_center\":80,\"y_center\":0,\"radius\":40}"
  "Job-GW-D4-near {\"x_center\":40,\"y_center\":0,\"radius\":25}"
  "Job-GW-D5-edge {\"x_center\":120,\"y_center\":0,\"radius\":25}"
)

generate_models() {
    echo "=== Generating INP files ==="
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        defect=$(echo "$case_str" | awk '{$1=""; print $0}' | sed 's/^ //')
        echo "Generating: $job (defect: $defect)"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_guided_wave.py \
            -- --job $job --defect '$defect' --no_run" 2>&1 | tail -5
        echo "---"
    done
    echo "INP generation complete."
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "ls -la $WORK_DIR/Job-GW-D*.inp"
}

run_jobs() {
    echo "=== Submitting solver jobs ==="
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        echo "Submitting: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS" &
        # Wait between submissions for license
        sleep 5
    done
    echo "All jobs submitted. Monitor with: ssh frontale04 'tail -2 ~/Payload2026/abaqus_work/Job-GW-D*.sta'"
    wait
}

run_sequential() {
    echo "=== Running jobs sequentially ==="
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        echo "Running: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS" 2>&1 | tail -3
        echo "Completed: $job"
        echo "---"
    done
    echo "All jobs complete."
}

extract_all() {
    echo "=== Extracting sensor histories ==="
    ODB_LIST=""
    # Include existing v2 results
    ODB_LIST="Job-GW-Healthy-v2.odb Job-GW-Debond-v2.odb"
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        ODB_LIST="$ODB_LIST $job.odb"
    done
    echo "ODBs: $ODB_LIST"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $ODB_LIST" 2>&1
    echo "Extraction complete."
}

case "${1:-all}" in
    generate) generate_models ;;
    run)      run_sequential ;;
    extract)  extract_all ;;
    all)
        generate_models
        run_sequential
        extract_all
        ;;
    *) echo "Usage: $0 [generate|run|extract|all]" ;;
esac
