#!/bin/bash
# auto_gw_pipeline.sh — INP生成 → Abaqusジョブ → CSV抽出 → FNO学習 → 合成 → GNN 全自動
#
# nohup で実行し、各ステップ完了後に自動的に次へ進む。
# ログは abaqus_work/logs/auto_pipeline.log に出力。
#
# Usage:
#   nohup bash scripts/auto_gw_pipeline.sh > abaqus_work/logs/auto_pipeline.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

WORK_DIR=~/Payload2026/abaqus_work
LOG_DIR="$WORK_DIR/logs"
CSV_DIR="abaqus_work/gw_fairing_dataset"
DOE="doe_gw_fairing.json"
CPUS=4

MACHINES=(frontale01 frontale02 frontale04)
ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

N_SAMPLES=$(python3 -c "import json; d=json.load(open('$DOE')); print(d['n_samples'])")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ==============================================================================
# Phase 1: INP 生成 (frontale04, 1件ずつ)
# ==============================================================================
phase1_generate() {
    log "===== Phase 1: INP Generation ====="
    local freq=$(python3 -c "import json; d=json.load(open('$DOE')); print(d['freq_khz'])")
    local generated=0

    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        if [ -f "$WORK_DIR/$job.inp" ]; then
            continue
        fi

        local defect=$(python3 -c "
import json; d=json.load(open('$DOE'))
s=d['samples'][$i]; p=s['defect_params']
print(json.dumps({'z_center':p['z_center'],'theta_deg':p['theta_deg'],'radius':p['radius']}))")

        log "  Generate [$i/$N_SAMPLES]: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 frontale04 \
            "cd $WORK_DIR && $ENV_CMD abaqus cae noGUI=~/Payload2026/src/generate_gw_fairing.py \
            -- --job $job --freq $freq --mesh_seed 5.0 --n_sensors 100 --defect '$defect' --no_run" \
            2>&1 | tail -2
        ((generated++))
    done

    log "Phase 1 complete: $generated new INPs generated"
    log "Total INPs: $(ls $WORK_DIR/Job-GW-Fair-0*.inp 2>/dev/null | wc -l) / $N_SAMPLES"
}

# ==============================================================================
# Phase 2: Abaqus ジョブ並列実行 (3台 frontale)
# ==============================================================================
phase2_run_jobs() {
    log "===== Phase 2: Run Abaqus Jobs (3 machines) ====="

    # Collect pending jobs
    local pending=()
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        if [ -f "$WORK_DIR/$job.inp" ] && [ ! -f "$WORK_DIR/$job.odb" ]; then
            pending+=($i)
        fi
    done

    local n_pending=${#pending[@]}
    log "Pending jobs: $n_pending"
    if [ $n_pending -eq 0 ]; then
        log "No pending jobs."
        return
    fi

    # Write per-machine runner scripts
    local n_machines=${#MACHINES[@]}
    local per_machine=$(( (n_pending + n_machines - 1) / n_machines ))
    local pids=()

    for ((m=0; m<n_machines; m++)); do
        local host=${MACHINES[$m]}
        local start=$((m * per_machine))
        local end=$((start + per_machine))
        [ $end -gt $n_pending ] && end=$n_pending
        [ $start -ge $n_pending ] && break

        local job_indices=""
        for ((j=start; j<end; j++)); do
            job_indices="$job_indices ${pending[$j]}"
        done

        local runner="$LOG_DIR/runner_${host}.sh"
        cat > "$runner" << RUNNER_HEREDOC
#!/bin/bash
WORK_DIR=$WORK_DIR
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
ENV_CMD="$ENV_CMD"

for idx in $job_indices; do
    job=\$(printf "Job-GW-Fair-%04d" \$idx)
    [ -f "\$WORK_DIR/\$job.odb" ] && echo "[\$(date '+%H:%M')] SKIP \$job" && continue
    [ ! -f "\$WORK_DIR/\$job.inp" ] && echo "[\$(date '+%H:%M')] NO_INP \$job" && continue
    echo "[\$(date '+%H:%M')] START \$job on \$(hostname)"
    cd "\$WORK_DIR"
    \$ENV_CMD \$ABAQUS job=\$job input=\$job.inp cpus=$CPUS interactive 2>&1 | tail -3
    if grep -q "THE ANALYSIS HAS BEEN COMPLETED" "\$WORK_DIR/\$job.sta" 2>/dev/null; then
        echo "[\$(date '+%H:%M')] DONE \$job"
    else
        echo "[\$(date '+%H:%M')] FAIL \$job"
    fi
done
echo "[\$(date '+%H:%M')] ALL DONE on \$(hostname)"
RUNNER_HEREDOC
        chmod +x "$runner"

        local logfile="$LOG_DIR/run_${host}.log"
        log "  $host: ${#pending[@]} total, indices [$start-$((end-1))]"

        # Launch via SSH with persistent connection
        LD_LIBRARY_PATH="" /usr/bin/ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=1440 $host \
            "bash $runner" > "$logfile" 2>&1 &
        pids+=($!)
    done

    # Wait for all machines to finish
    log "Waiting for ${#pids[@]} machines..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            ((failed++))
        fi
    done

    local n_completed=$(ls $WORK_DIR/Job-GW-Fair-0*.odb 2>/dev/null | wc -l)
    log "Phase 2 complete: $n_completed ODBs, $failed machines had errors"

    # Show per-machine results
    for host in "${MACHINES[@]}"; do
        local logfile="$LOG_DIR/run_${host}.log"
        if [ -f "$logfile" ]; then
            local done_count=$(grep -c "^.*DONE" "$logfile" 2>/dev/null || echo 0)
            local fail_count=$(grep -c "^.*FAIL" "$logfile" 2>/dev/null || echo 0)
            log "  $host: $done_count completed, $fail_count failed"
        fi
    done
}

# ==============================================================================
# Phase 3: ODB → CSV 抽出
# ==============================================================================
phase3_extract() {
    log "===== Phase 3: Extract Sensor CSVs ====="
    mkdir -p "$CSV_DIR"

    local extracted=0
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        [ -f "$CSV_DIR/${job}_sensors.csv" ] && continue
        [ ! -f "$WORK_DIR/$job.odb" ] && continue

        if ! grep -q "THE ANALYSIS HAS BEEN COMPLETED" "$WORK_DIR/$job.sta" 2>/dev/null; then
            log "  SKIP $job (incomplete)"
            continue
        fi

        log "  Extract: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh -o ConnectTimeout=30 frontale04 \
            "cd $WORK_DIR && $ENV_CMD abaqus python ~/Payload2026/scripts/extract_gw_history.py $job.odb" \
            2>&1 | tail -2

        if [ -f "$WORK_DIR/${job}_sensors.csv" ]; then
            cp "$WORK_DIR/${job}_sensors.csv" "$CSV_DIR/"
            ((extracted++))
        fi
    done

    log "Phase 3 complete: $extracted CSVs extracted"
    log "Total defect CSVs: $(ls $CSV_DIR/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)"
}

# ==============================================================================
# Phase 4: FNO 学習 → 合成 → GNN 学習
# ==============================================================================
phase4_fno_gnn() {
    local n_csv=$(ls "$CSV_DIR"/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)
    log "===== Phase 4: FNO→GNN Pipeline (defect CSVs: $n_csv) ====="

    if [ "$n_csv" -lt 5 ]; then
        log "ERROR: Need at least 5 defect CSVs, have $n_csv. Skipping."
        return 1
    fi

    log "--- Step 4a: Train FNO ---"
    bash scripts/pipeline_fno_gnn.sh train_fno

    log "--- Step 4b: Generate 5000 synthetic waveforms ---"
    bash scripts/pipeline_fno_gnn.sh generate 5000

    log "--- Step 4c: Prepare GNN dataset ---"
    bash scripts/pipeline_fno_gnn.sh prepare

    log "--- Step 4d: Train GNN ---"
    bash scripts/pipeline_fno_gnn.sh train_gnn

    log "Phase 4 complete."
    bash scripts/pipeline_fno_gnn.sh status
}

# ==============================================================================
# Main: 全フェーズ順次実行
# ==============================================================================
log "========================================"
log "GW Auto Pipeline Start"
log "  Samples: $N_SAMPLES"
log "  Machines: ${MACHINES[*]}"
log "========================================"

phase1_generate
phase2_run_jobs
phase3_extract
phase4_fno_gnn

log "========================================"
log "ALL PHASES COMPLETE"
log "========================================"
