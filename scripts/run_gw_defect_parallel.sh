#!/bin/bash
# run_gw_defect_parallel.sh — 3台のfrontaleで欠陥ジョブを並列実行
#
# Usage:
#   bash scripts/run_gw_defect_parallel.sh generate   # INP生成のみ (1台、QAE license)
#   bash scripts/run_gw_defect_parallel.sh run         # 3台並列でジョブ実行
#   bash scripts/run_gw_defect_parallel.sh extract      # ODB→CSV抽出
#   bash scripts/run_gw_defect_parallel.sh status       # 進捗確認
#   bash scripts/run_gw_defect_parallel.sh all          # generate→run→extract

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DOE_FILE="$PROJECT_ROOT/doe_gw_fairing.json"
WORK_DIR=~/Payload2026/abaqus_work
CSV_DIR="abaqus_work/gw_fairing_dataset"
SRC_DIR=~/Payload2026/src
SCRIPTS_DIR=~/Payload2026/scripts
LOG_DIR="$WORK_DIR/logs"
CPUS=4
MESH_SEED=5.0
N_SENSORS=100

# Machines: 3台 frontale
MACHINES=(frontale01 frontale02 frontale04)

ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

N_SAMPLES=$(python3 -c "import json; d=json.load(open('$DOE_FILE')); print(d['n_samples'])")

get_defect_json() {
    python3 -c "
import json
d = json.load(open('$DOE_FILE'))
s = d['samples'][$1]
p = s['defect_params']
print(json.dumps({'z_center': p['z_center'], 'theta_deg': p['theta_deg'], 'radius': p['radius']}))
"
}

get_freq() {
    python3 -c "import json; d=json.load(open('$DOE_FILE')); print(d['freq_khz'])"
}

# ==============================================================================
# generate: 未生成のINPを生成 (frontale04 で1つずつ)
# ==============================================================================
do_generate() {
    local freq=$(get_freq)
    local gen_host=frontale04
    echo "=== Generating missing INPs (0000-$((N_SAMPLES-1))) on $gen_host ==="

    local generated=0
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        # Skip if INP already exists
        local exists=$(LD_LIBRARY_PATH="" /usr/bin/ssh $gen_host \
            "test -f $WORK_DIR/$job.inp && echo yes || echo no" 2>/dev/null)
        if [ "$exists" = "yes" ]; then
            continue
        fi
        local defect=$(get_defect_json $i)
        echo "  Generating [$i/$N_SAMPLES]: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh $gen_host "cd $WORK_DIR && \
            $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
            -- --job $job --freq $freq --mesh_seed $MESH_SEED --n_sensors $N_SENSORS --defect '$defect' --no_run" 2>&1 | tail -2
        ((generated++))
    done
    echo "Generated $generated new INP files."
}

# ==============================================================================
# run: 3台に分散してジョブ実行 (nohup)
# ==============================================================================
do_run() {
    mkdir -p "$LOG_DIR"
    echo "=== Distributing jobs across ${#MACHINES[@]} machines ==="

    # Collect jobs that need running (no ODB yet)
    local pending_jobs=()
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        if [ ! -f "$WORK_DIR/$job.odb" ] && [ -f "$WORK_DIR/$job.inp" ]; then
            pending_jobs+=($i)
        fi
    done

    local n_pending=${#pending_jobs[@]}
    echo "Pending jobs: $n_pending"
    if [ $n_pending -eq 0 ]; then
        echo "No pending jobs."
        return
    fi

    # Split across machines
    local n_machines=${#MACHINES[@]}
    local per_machine=$(( (n_pending + n_machines - 1) / n_machines ))

    for ((m=0; m<n_machines; m++)); do
        local host=${MACHINES[$m]}
        local start=$((m * per_machine))
        local end=$((start + per_machine))
        if [ $end -gt $n_pending ]; then end=$n_pending; fi
        if [ $start -ge $n_pending ]; then break; fi

        # Build job list for this machine
        local job_indices=""
        for ((j=start; j<end; j++)); do
            job_indices="$job_indices ${pending_jobs[$j]}"
        done

        local logfile="$LOG_DIR/run_${host}.log"
        echo "  $host: indices [$start-$((end-1))] → ${pending_jobs[$start]}..${pending_jobs[$((end-1))]}"

        # Write per-machine runner script
        local runner="$LOG_DIR/runner_${host}.sh"
        cat > "$runner" << 'RUNNER_EOF'
#!/bin/bash
WORK_DIR=~/Payload2026/abaqus_work
CPUS=4
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

for idx in JOB_INDICES_PLACEHOLDER; do
    job=$(printf "Job-GW-Fair-%04d" $idx)
    if [ -f "$WORK_DIR/$job.odb" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $job (ODB exists)"
        continue
    fi
    if [ ! -f "$WORK_DIR/$job.inp" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $job (INP missing)"
        continue
    fi
    echo "[$(date '+%H:%M:%S')] START $job on $(hostname)"
    cd "$WORK_DIR"
    $ENV_CMD $ABAQUS job=$job input=$job.inp cpus=$CPUS interactive 2>&1 | tail -3
    # Check completion
    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] DONE $job"
    else
        echo "[$(date '+%H:%M:%S')] FAIL $job (check .sta/.msg)"
    fi
done
echo "[$(date '+%H:%M:%S')] ALL DONE on $(hostname)"
RUNNER_EOF
        # Replace placeholder with actual indices
        sed -i "s/JOB_INDICES_PLACEHOLDER/$job_indices/" "$runner"
        chmod +x "$runner"

        # Launch via nohup on remote machine
        echo "  Launching on $host (log: $logfile)"
        LD_LIBRARY_PATH="" /usr/bin/ssh -o ServerAliveInterval=60 $host \
            "nohup bash $runner > $logfile 2>&1 &" 2>/dev/null
    done

    echo ""
    echo "Jobs launched in background. Monitor with:"
    echo "  bash scripts/run_gw_defect_parallel.sh status"
}

# ==============================================================================
# status: 進捗確認
# ==============================================================================
do_status() {
    echo "=== GW Defect Job Status ==="

    # Count completed ODBs
    local n_odb=0
    local n_inp=0
    local running_list=""
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        [ -f "$WORK_DIR/$job.inp" ] && ((n_inp++))
        if [ -f "$WORK_DIR/$job.odb" ]; then
            # Check if completed
            if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
                ((n_odb++))
            else
                # In progress or failed
                local pct=""
                if [ -f "$WORK_DIR/$job.sta" ]; then
                    pct=$(tail -1 "$WORK_DIR/$job.sta" 2>/dev/null | awk '{print $2}')
                fi
                running_list="$running_list  $job: step_time=$pct/3.920E-03\n"
            fi
        fi
    done

    echo "INP files: $n_inp / $N_SAMPLES"
    echo "Completed ODB: $n_odb / $N_SAMPLES"
    echo "CSV extracted: $(ls $CSV_DIR/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l) / $N_SAMPLES"

    if [ -n "$running_list" ]; then
        echo ""
        echo "In progress / incomplete:"
        echo -e "$running_list"
    fi

    # Check running processes on each machine
    echo ""
    echo "Running Abaqus processes:"
    for host in "${MACHINES[@]}"; do
        local procs=$(LD_LIBRARY_PATH="" /usr/bin/ssh $host \
            "ps aux | grep -E 'abaqus|ABQcaeK|standard|explicit' | grep -v grep | wc -l" 2>/dev/null || echo "?")
        local log_tail=""
        if [ -f "$LOG_DIR/run_${host}.log" ]; then
            log_tail=$(tail -1 "$LOG_DIR/run_${host}.log" 2>/dev/null)
        fi
        echo "  $host: $procs processes  | $log_tail"
    done
}

# ==============================================================================
# extract: ODB → CSV
# ==============================================================================
do_extract() {
    echo "=== Extracting sensor histories from completed ODBs ==="
    mkdir -p "$CSV_DIR"

    local extracted=0
    for ((i=0; i<N_SAMPLES; i++)); do
        local job="Job-GW-Fair-$(printf '%04d' $i)"
        # Skip if CSV exists
        if [ -f "$CSV_DIR/${job}_sensors.csv" ]; then
            continue
        fi
        # Check ODB exists and completed
        if [ ! -f "$WORK_DIR/$job.odb" ]; then
            continue
        fi
        if ! grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            echo "  SKIP $job (not completed)"
            continue
        fi
        echo "  Extracting: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh frontale04 "cd $WORK_DIR && \
            $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $job.odb" 2>&1 | tail -2

        # Copy CSV from work dir
        if [ -f "$WORK_DIR/${job}_sensors.csv" ]; then
            cp "$WORK_DIR/${job}_sensors.csv" "$CSV_DIR/"
            ((extracted++))
        fi
    done
    echo "Extracted $extracted new CSVs."
}

# ==============================================================================
case "${1:-status}" in
    generate) do_generate ;;
    run)      do_run ;;
    extract)  do_extract ;;
    status)   do_status ;;
    all)
        do_generate
        do_run
        ;;
    *) echo "Usage: $0 [generate|run|extract|status|all]" ;;
esac
