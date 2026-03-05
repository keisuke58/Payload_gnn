#!/bin/bash
# batch_generate_gw_dataset.sh
# Generate guided wave GNN dataset on fairing sector: INP → Abaqus/Explicit → CSV
#
# Reads doe_gw_fairing.json and runs healthy + all defect samples sequentially.
# QAE license constraint: only 1 job at a time.
#
# Usage:
#   bash scripts/batch_generate_gw_dataset.sh all     # 100センサで1回解析 → s10/s20/s30/s50 を抽出
#   bash scripts/batch_generate_gw_dataset.sh healthy # Healthy 1個のみ: generate→run→extract→copy
#   bash scripts/batch_generate_gw_dataset.sh subset  # 既存100センサCSVから抽出のみ
#
# 100センサで1回解析し、extract_gw_sensor_subset で 10/20/30/50 を抽出。再解析不要。

set -e

# Run from project root (where doe_gw_fairing.json lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE=frontale04
N_SENSORS=100
WORK_DIR=~/Payload2026/abaqus_work
CSV_DIR="abaqus_work/gw_fairing_dataset"
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
MIN_ODB_SIZE=500000           # 500KB — 1解析終了後の ODB 最小サイズ

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

# 1解析終了後の適切性チェック（ODB 存在・サイズ）
check_odb() {
    local job=$1
    local out
    out=$(LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "
        p=$WORK_DIR/$job.odb
        if [ ! -f \"\$p\" ]; then echo MISSING; exit 0; fi
        s=\$(stat -c%s \"\$p\" 2>/dev/null || echo 0)
        if [ \"\$s\" -lt $MIN_ODB_SIZE ]; then echo SIZE:\$s; exit 0; fi
        echo OK:\$s
    " 2>/dev/null)
    case "$out" in
        OK:*)
            echo "  Check OK: $job (${out#OK:} bytes)"
            return 0 ;;
        SIZE:*)
            echo "  CHECK FAIL: $job ODB too small (${out#SIZE:} bytes)"
            return 1 ;;
        *)
            echo "  CHECK FAIL: $job ODB missing"
            return 1 ;;
    esac
}

generate_models() {
    local n=$(get_n_samples)
    local freq=$(get_freq)
    echo "=== Generating $n + 1 (healthy) INP files ==="

    # Healthy baseline
    local job="Job-GW-Fair-Healthy"
    echo "Generating: $job (healthy baseline, n_sensors=$N_SENSORS)"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "mkdir -p $WORK_DIR && cd $WORK_DIR && \
        $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
        -- --job $job --freq $freq --mesh_seed $MESH_SEED --n_sensors $N_SENSORS --no_run" 2>&1 | tail -3
    echo "---"

    # Defect samples
    n_limit=$((START_IDX + GENERATE_LIMIT))
    for ((i=START_IDX; i<n && i<n_limit; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        local defect=$(get_defect_json $i)
        echo "Generating [$i/$n]: $job (defect: $defect)"
        LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
            -- --job $job --freq $freq --mesh_seed $MESH_SEED --n_sensors $N_SENSORS --defect '$defect' --no_run" 2>&1 | tail -3
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
        check_odb "$job" || true
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
        check_odb "$job" || true
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
    mkdir -p "$CSV_DIR"

    # Healthy
    LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/Job-GW-Fair-Healthy_sensors.csv "$CSV_DIR/" 2>/dev/null \
        && echo "  Got Healthy" || echo "  SKIP Healthy"

    # Defect samples
    for ((i=0; i<n; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/${job}_sensors.csv "$CSV_DIR/" 2>/dev/null \
            && echo "  Got $job" || echo "  SKIP $job"
    done
    echo "CSVs copied to $CSV_DIR/"

    # CSV 品質チェック
    if command -v python3 >/dev/null 2>&1; then
        python3 scripts/verify_gw_dataset_quality.py --data_dir "$CSV_DIR" --filter fairing 2>&1 | tail -15
    fi
}

extract_subset() {
    echo "=== Extracting sensor subsets (10, 20, 30, 50) ==="
    python scripts/extract_gw_sensor_subset.py --input "$CSV_DIR" --output "$CSV_DIR" --k 10 20 30 50
    echo "  -> $CSV_DIR/s10/, s20/, s30/, s50/"
}

# Healthy 1個のみ: generate → run → extract → copy
run_healthy_only() {
    local job="Job-GW-Fair-Healthy"
    local odb_check=$(LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "test -f $WORK_DIR/$job.odb && echo exists || echo missing" 2>/dev/null)
    if [ "$odb_check" = "exists" ]; then
        echo "Skip: $job (ODB exists)"
        return 0
    fi
    echo "Running: $job (100 sensors)"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS interactive" 2>&1
    check_odb "$job" || true
}

extract_healthy_only() {
    echo "=== Extracting Healthy only ==="
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py Job-GW-Fair-Healthy.odb" 2>&1
    echo "Extraction done."
}

copy_healthy_only() {
    echo "=== Copying Healthy CSV to local ==="
    mkdir -p "$CSV_DIR"
    LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/Job-GW-Fair-Healthy_sensors.csv "$CSV_DIR/" 2>/dev/null \
        && echo "  Got Job-GW-Fair-Healthy_sensors.csv -> $CSV_DIR/" || echo "  SKIP (file not found)"
}

case "${1:-all}" in
    generate) generate_models ;;
    run)      run_jobs ;;
    extract)  extract_all ;;
    subset)   extract_subset ;;
    healthy)
        echo "=== Healthy 1個のみ (100 sensors): generate → run → extract → copy → prepare_gw_1000 ==="
        GENERATE_LIMIT=0 generate_models
        run_healthy_only
        extract_healthy_only
        copy_healthy_only
        echo "Done. $CSV_DIR/Job-GW-Fair-Healthy_sensors.csv"
        echo ""
        echo "=== Running prepare_gw_1000.sh --healthy-only ==="
        bash scripts/prepare_gw_1000.sh --healthy-only
        ;;
    all)
        generate_models
        run_jobs
        extract_all
        extract_subset
        ;;
    *) echo "Usage: $0 [generate|run|extract|subset|all|healthy] [start_idx] [generate_limit]"
       echo "  all: 100センサで解析 → 10/20/30/50 を抽出（再解析不要）"
       echo "  healthy: Healthy 1個のみ generate→run→extract→copy"
       echo "  subset: 既存CSVから抽出のみ" ;;
esac
