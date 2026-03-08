#!/bin/bash
# launch_stgnn_pipeline.sh — GW ST-GNN パイプライン
#
# 1. 完了済みODB → CSV抽出 (frontale)
# 2. データをvancouver02に転送
# 3. ST-GNN + DI baseline 学習 (vancouver02)
#
# Usage:
#   bash scripts/launch_stgnn_pipeline.sh extract   # ODB→CSV抽出のみ
#   bash scripts/launch_stgnn_pipeline.sh transfer   # CSV転送のみ
#   bash scripts/launch_stgnn_pipeline.sh train      # 学習のみ
#   bash scripts/launch_stgnn_pipeline.sh all        # 全部

set -e

PROJECT_ROOT=~/Payload2026
WORK_DIR=$PROJECT_ROOT/abaqus_work
CSV_DIR=$WORK_DIR/gw_fairing_dataset
SCRIPTS_DIR=$PROJECT_ROOT/scripts
SRC_DIR=$PROJECT_ROOT/src
DOE_FILE=$PROJECT_ROOT/doe_gw_fairing.json

EXTRACT_HOST=frontale04
TRAIN_HOST=vancouver02

ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

N_SAMPLES=$(python3 -c "import json; d=json.load(open('$DOE_FILE')); print(d['n_samples'])")

# ==============================================================================
# extract: 完了済みODB → CSV
# ==============================================================================
do_extract() {
    echo "=== Extracting sensor CSVs ==="
    mkdir -p "$CSV_DIR"

    local extracted=0
    local skipped=0

    # Healthy
    local job="Job-GW-Fair-Healthy"
    if [ -f "$WORK_DIR/$job.odb" ] && [ ! -f "$CSV_DIR/${job}_sensors.csv" ]; then
        if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            echo "  Extracting: $job"
            LD_LIBRARY_PATH="" /usr/bin/ssh $EXTRACT_HOST "cd $WORK_DIR && \
                $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $job.odb" 2>&1 | tail -2
            if [ -f "$WORK_DIR/${job}_sensors.csv" ]; then
                cp "$WORK_DIR/${job}_sensors.csv" "$CSV_DIR/"
                ((extracted++))
            fi
        else
            echo "  SKIP $job (not completed)"
            ((skipped++))
        fi
    fi

    # Defect samples
    for ((i=0; i<N_SAMPLES; i++)); do
        job=$(printf "Job-GW-Fair-%04d" $i)
        if [ -f "$CSV_DIR/${job}_sensors.csv" ]; then
            continue
        fi
        if [ ! -f "$WORK_DIR/$job.odb" ]; then
            continue
        fi
        if ! grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            echo "  SKIP $job (not completed)"
            ((skipped++))
            continue
        fi
        echo "  Extracting: $job"
        LD_LIBRARY_PATH="" /usr/bin/ssh $EXTRACT_HOST "cd $WORK_DIR && \
            $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $job.odb" 2>&1 | tail -2
        if [ -f "$WORK_DIR/${job}_sensors.csv" ]; then
            cp "$WORK_DIR/${job}_sensors.csv" "$CSV_DIR/"
            ((extracted++))
        fi
    done

    echo "Extracted: $extracted new, Skipped: $skipped"
    echo "Total CSVs: $(ls $CSV_DIR/Job-GW-Fair-0*_sensors.csv $CSV_DIR/Job-GW-Fair-Healthy_sensors.csv 2>/dev/null | wc -l)"
}

# ==============================================================================
# transfer: CSV + ソースコードを vancouver02 に転送
# ==============================================================================
do_transfer() {
    echo "=== Transferring to $TRAIN_HOST ==="

    # Sync CSV directory
    rsync -avz --progress "$CSV_DIR/" $TRAIN_HOST:$CSV_DIR/

    # Sync source code
    rsync -avz "$SRC_DIR/train_gw_stgnn.py" $TRAIN_HOST:$SRC_DIR/
    rsync -avz "$DOE_FILE" $TRAIN_HOST:$DOE_FILE

    echo "Transfer complete."
}

# ==============================================================================
# train: vancouver02 で ST-GNN + DI baseline
# ==============================================================================
do_train() {
    echo "=== Launching training on $TRAIN_HOST ==="

    local log_file="$PROJECT_ROOT/runs/stgnn_train.log"

    LD_LIBRARY_PATH="" /usr/bin/ssh $TRAIN_HOST "cd $PROJECT_ROOT && \
        nohup bash -l -c 'python3 src/train_gw_stgnn.py \
            --mode both \
            --csv_dir abaqus_work/gw_fairing_dataset \
            --doe doe_gw_fairing.json \
            --epochs 200 \
            --d_temporal 64 \
            --d_hidden 64 \
            --n_gat_layers 3 \
            --batch_size 16 \
            --lr 1e-3 \
            --patience 30' > $log_file 2>&1 &"

    echo "Training launched in background."
    echo "Monitor: ssh $TRAIN_HOST 'tail -f $log_file'"
}

# ==============================================================================
# status: 進捗確認
# ==============================================================================
do_status() {
    echo "=== ST-GNN Pipeline Status ==="

    # CSVs
    local n_defect=$(ls $CSV_DIR/Job-GW-Fair-0*_sensors.csv 2>/dev/null | wc -l)
    local n_healthy=0
    [ -f "$CSV_DIR/Job-GW-Fair-Healthy_sensors.csv" ] && n_healthy=1
    echo "CSVs: $n_defect defect + $n_healthy healthy (99-sensor)"

    # Completed ODBs
    local n_completed=0
    for ((i=0; i<N_SAMPLES; i++)); do
        local job=$(printf "Job-GW-Fair-%04d" $i)
        if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            ((n_completed++))
        fi
    done
    echo "Completed ODBs: $n_completed / $N_SAMPLES"

    # Healthy status
    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/Job-GW-Fair-Healthy.sta" 2>/dev/null; then
        echo "Healthy ODB: COMPLETED"
    else
        local pct=$(tail -1 "$WORK_DIR/Job-GW-Fair-Healthy.sta" 2>/dev/null | awk '{print $2}')
        echo "Healthy ODB: in progress (step_time=$pct/3.920E-03)"
    fi

    # Training status
    if LD_LIBRARY_PATH="" /usr/bin/ssh $TRAIN_HOST "ps aux | grep train_gw_stgnn | grep -v grep" 2>/dev/null | grep -q python; then
        echo "Training: RUNNING on $TRAIN_HOST"
        LD_LIBRARY_PATH="" /usr/bin/ssh $TRAIN_HOST "tail -3 $PROJECT_ROOT/runs/stgnn_train.log" 2>/dev/null
    else
        echo "Training: NOT RUNNING"
        if [ -f "$PROJECT_ROOT/runs/stgnn_train.log" ]; then
            echo "Last log:"
            tail -5 "$PROJECT_ROOT/runs/stgnn_train.log"
        fi
    fi
}

# ==============================================================================
case "${1:-status}" in
    extract)  do_extract ;;
    transfer) do_transfer ;;
    train)    do_train ;;
    status)   do_status ;;
    all)
        do_extract
        do_transfer
        do_train
        ;;
    *) echo "Usage: $0 [extract|transfer|train|status|all]" ;;
esac
