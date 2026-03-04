#!/bin/bash
# batch_gw_freq_sweep.sh
# Frequency sweep for guided wave detection band identification.
#
# Runs healthy + D1 defect at 25 / 50 / 75 / 100 kHz.
# Mesh seed auto-adjusts per frequency (lambda/8 rule).
#
# Usage: bash scripts/batch_gw_freq_sweep.sh [generate|run|extract|all]

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

# Defect D1 (baseline): r=25mm, x=80mm
DEFECT='{"x_center":80,"y_center":0,"radius":25}'

# Frequencies to sweep (kHz)
FREQS=(25 50 75 100)

generate_models() {
    echo "=== Generating frequency sweep INP files ==="
    for freq in "${FREQS[@]}"; do
        # Healthy
        job="Job-GW-Freq${freq}k-H"
        echo "Generating: $job (freq=${freq}kHz, healthy)"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_guided_wave.py \
            -- --job $job --freq $freq --no_run" 2>&1 | tail -5
        echo "---"

        # Defect D1
        job="Job-GW-Freq${freq}k-D"
        echo "Generating: $job (freq=${freq}kHz, defect D1)"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_guided_wave.py \
            -- --job $job --freq $freq --defect '$DEFECT' --no_run" 2>&1 | tail -5
        echo "---"
    done
    echo "INP generation complete."
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "ls -la $WORK_DIR/Job-GW-Freq*.inp"
}

run_jobs() {
    echo "=== Running frequency sweep jobs sequentially ==="
    for freq in "${FREQS[@]}"; do
        for suffix in H D; do
            job="Job-GW-Freq${freq}k-${suffix}"
            echo "Running: $job"
            LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
                $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS interactive" 2>&1 | tail -3
            echo "Completed: $job"
            echo "---"
        done
    done
    echo "All jobs complete."
}

extract_all() {
    echo "=== Extracting sensor histories ==="
    ODB_LIST=""
    for freq in "${FREQS[@]}"; do
        ODB_LIST="$ODB_LIST Job-GW-Freq${freq}k-H.odb Job-GW-Freq${freq}k-D.odb"
    done
    echo "ODBs: $ODB_LIST"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $ODB_LIST" 2>&1
    echo "Extraction complete."
    # Copy CSVs back
    echo "=== Copying CSVs to local ==="
    for freq in "${FREQS[@]}"; do
        for suffix in H D; do
            scp_cmd="LD_LIBRARY_PATH='' /usr/bin/scp $REMOTE:$WORK_DIR/Job-GW-Freq${freq}k-${suffix}_sensors.csv abaqus_work/"
            eval $scp_cmd 2>/dev/null && echo "  Got Freq${freq}k-${suffix}" || echo "  SKIP Freq${freq}k-${suffix}"
        done
    done
}

case "${1:-all}" in
    generate) generate_models ;;
    run)      run_jobs ;;
    extract)  extract_all ;;
    all)
        generate_models
        run_jobs
        extract_all
        ;;
    *) echo "Usage: $0 [generate|run|extract|all]" ;;
esac
