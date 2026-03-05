#!/bin/bash
# batch_generate_gw_dataset.sh
# Generate guided wave GNN dataset on fairing sector: INP → Abaqus/Explicit → CSV
#
# Reads doe_gw_fairing.json and runs healthy + all defect samples sequentially.
# QAE license constraint: only 1 job at a time.
#
# Usage:
#   bash scripts/batch_generate_gw_dataset.sh [generate|run|extract|all]
#   bash scripts/batch_generate_gw_dataset.sh all     # full pipeline
#   bash scripts/batch_generate_gw_dataset.sh run 10   # run from sample 10 onward (resume)

set -e

# Run from project root (where doe_gw_fairing.json lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE=frontale04
WORK_DIR=~/Payload2026/abaqus_work
SRC_DIR=~/Payload2026/src
SCRIPTS_DIR=~/Payload2026/scripts
DOE_FILE="$PROJECT_ROOT/doe_gw_fairing.json"
CPUS=4
MESH_SEED=5.0   # mm — λ/6 at 50kHz, good balance of accuracy and speed

# Abaqus environment (headless, fake X11)
ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

START_IDX=${2:-0}
GENERATE_LIMIT=${3:-999999}   # 3rd arg: max defect samples for generate (default: all)

# Parse DOE JSON
get_n_samples() {
    python3 -c "import json; d=json.load(open('$DOE_FILE')); print(d['n_samples'])"
}

get_defect_json() {
    local idx=$1
    python3 -c "
import json
d = json.load(open('$DOE_FILE'))
s = d['samples'][$idx]
p = s['defect_params']
print(json.dumps({'z_center': p['z_center'], 'theta_deg': p['theta_deg'], 'radius': p['radius']}))
"
}

get_freq() {
    python3 -c "import json; d=json.load(open('$DOE_FILE')); print(d['freq_khz'])"
}

generate_models() {
    local n=$(get_n_samples)
    local freq=$(get_freq)
    echo "=== Generating $n + 1 (healthy) INP files ==="

    # Healthy baseline
    local job="Job-GW-Fair-Healthy"
    echo "Generating: $job (healthy baseline)"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
        -- --job $job --freq $freq --mesh_seed $MESH_SEED --no_run" 2>&1 | tail -3
    echo "---"

    # Defect samples
    n_limit=$((START_IDX + GENERATE_LIMIT))
    for ((i=START_IDX; i<n && i<n_limit; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        local defect=$(get_defect_json $i)
        echo "Generating [$i/$n]: $job (defect: $defect)"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
            -- --job $job --freq $freq --mesh_seed $MESH_SEED --defect '$defect' --no_run" 2>&1 | tail -3
        echo "---"
    done
    echo "INP generation complete."
}

run_jobs() {
    local n=$(get_n_samples)
    echo "=== Running $n + 1 jobs sequentially ==="

    # Healthy baseline (skip if ODB exists)
    local job="Job-GW-Fair-Healthy"
    local odb_check=$(LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "test -f $WORK_DIR/$job.odb && echo exists || echo missing" 2>/dev/null)
    if [ "$odb_check" = "exists" ] && [ "$START_IDX" -gt 0 ]; then
        echo "Skip: $job (ODB exists)"
    else
        echo "Running: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS interactive" 2>&1 | tail -3
        echo "Completed: $job"
    fi
    echo "---"

    # Defect samples
    for ((i=START_IDX; i<n; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        # Skip if ODB already exists
        local odb_check=$(LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "test -f $WORK_DIR/$job.odb && echo exists || echo missing" 2>/dev/null)
        if [ "$odb_check" = "exists" ]; then
            echo "Skip [$i/$n]: $job (ODB exists)"
            continue
        fi
        echo "Running [$i/$n]: $job  $(date '+%H:%M:%S')"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS interactive" 2>&1 | tail -3
        echo "Completed: $job"
        echo "---"
    done
    echo "All jobs complete."
}

extract_all() {
    local n=$(get_n_samples)
    echo "=== Extracting sensor histories ==="

    # Build ODB list
    local odb_list="Job-GW-Fair-Healthy.odb"
    for ((i=0; i<n; i++)); do
        odb_list="$odb_list Job-GW-Fair-$(printf '%04d' $i).odb"
    done

    # Run extraction in batches of 20
    local batch_size=20
    local total=$((n + 1))
    local odbs=($odb_list)
    for ((start=0; start<total; start+=batch_size)); do
        local end=$((start + batch_size))
        if [ $end -gt $total ]; then end=$total; fi
        local batch="${odbs[@]:$start:$((end-start))}"
        echo "  Extracting batch $start-$((end-1))..."
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $batch" 2>&1 | grep -E "^(CSV written|WARNING|ERROR)" || true
    done
    echo "Extraction complete."

    # Copy CSVs to local
    echo "=== Copying CSVs to local ==="
    local csv_dir="abaqus_work/gw_fairing_dataset"
    mkdir -p "$csv_dir"

    # Healthy
    LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/Job-GW-Fair-Healthy_sensors.csv "$csv_dir/" 2>/dev/null \
        && echo "  Got Healthy" || echo "  SKIP Healthy"

    # Defect samples
    for ((i=0; i<n; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/${job}_sensors.csv "$csv_dir/" 2>/dev/null \
            && echo "  Got $job" || echo "  SKIP $job"
    done
    echo "CSVs copied to $csv_dir/"
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
    *) echo "Usage: $0 [generate|run|extract|all] [start_idx] [generate_limit]"
       echo "  generate_limit: for generate only, max defect samples (e.g. 1 = Healthy + 0000)" ;;
esac
