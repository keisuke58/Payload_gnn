#!/bin/bash
# post_batch_pipeline.sh
# バッチ完了 → 失敗リトライ → PyG 前処理 → 学習コマンド案内
#
# Usage:
#   nohup bash scripts/post_batch_pipeline.sh &
#   # or chain from retry_failed_samples.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

DOE="doe_realistic_25mm_100.json"
DATASET_DIR="dataset_realistic_25mm_100"
OUTPUT_DIR="data/processed_realistic_25mm"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [pipeline] $*"; }

# --- 1. バッチ完了を待つ ---
log "Waiting for run_batch.py to finish..."
while pgrep -f "run_batch.py.*${DOE}" > /dev/null 2>&1; do
    sleep 60
done
log "Batch finished."

# --- 2. 失敗サンプルのリトライ ---
log "Retrying failed samples..."
FAILED_IDS=()
for d in "${PROJECT_ROOT}/${DATASET_DIR}"/sample_*/; do
    if [ ! -f "${d}/nodes.csv" ]; then
        sid=$(basename "$d" | sed 's/sample_0*//')
        [ -z "$sid" ] && sid=0
        FAILED_IDS+=("$sid")
    fi
done

if [ ${#FAILED_IDS[@]} -gt 0 ]; then
    log "Found ${#FAILED_IDS[@]} failed samples: ${FAILED_IDS[*]}"
    for sid in "${FAILED_IDS[@]}"; do
        sample_dir="${PROJECT_ROOT}/${DATASET_DIR}/sample_$(printf '%04d' $sid)"
        if [ -d "${sample_dir}" ]; then
            rm -rf "${sample_dir}"
        fi
        log "Retrying sample $(printf '%04d' $sid)..."
        python src/run_batch.py \
            --doe "${DOE}" \
            --output_dir "${DATASET_DIR}" \
            --gen_script realistic \
            --global_seed 25 \
            --defect_seed 10 \
            --work_dir abaqus_work \
            --n_cpus 8 \
            --start "${sid}" \
            --end "$((sid + 1))" || true
    done
else
    log "No failed samples."
fi

# --- 3. データセット統計 ---
n_total=0; n_ok=0; n_fail=0
for d in "${PROJECT_ROOT}/${DATASET_DIR}"/sample_*/; do
    ((n_total++)) || true
    if [ -f "${d}/nodes.csv" ]; then
        ((n_ok++)) || true
    else
        ((n_fail++)) || true
    fi
done
log "Dataset: ${n_ok}/${n_total} completed (${n_fail} failed)"

# --- 4. PyG 前処理 ---
log "Running prepare_ml_data.py..."
python src/prepare_ml_data.py \
    --input "${DATASET_DIR}" \
    --output "${OUTPUT_DIR}" \
    --val_ratio 0.2

log "PyG data saved to ${OUTPUT_DIR}/"
ls -lh "${PROJECT_ROOT}/${OUTPUT_DIR}/"

# --- 5. 学習コマンドの案内 ---
log "============================================================"
log "Pipeline complete. Ready for training."
log ""
log "Recommended commands (run on GPU server):"
log ""
log "  # Multi-class + Focal Loss + Boundary-aware"
log "  python src/train.py \\"
log "    --data_dir ${OUTPUT_DIR} \\"
log "    --arch gat --loss focal --focal_gamma 2.0 \\"
log "    --sampler defect_centric --subgraph_hops 4 \\"
log "    --boundary_weight 3.0 --epochs 200"
log ""
log "  # Transfer learning (pretrain on simple → fine-tune)"
log "  python src/train.py \\"
log "    --data_dir ${OUTPUT_DIR} \\"
log "    --pretrained runs/<pretrain_run>/best_model.pt \\"
log "    --freeze_layers 2 --lr 1e-4 --epochs 100"
log "============================================================"
