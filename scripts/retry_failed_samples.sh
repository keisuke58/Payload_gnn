#!/bin/bash
# retry_failed_samples.sh
# 現在のバッチ完了を待ってから、失敗サンプル (0004, 0009) を再実行する
#
# Usage:
#   nohup bash scripts/retry_failed_samples.sh &

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

FAILED_IDS=(4 9)
DOE="doe_realistic_25mm_100.json"
OUTPUT_DIR="dataset_realistic_25mm_100"
WORK_DIR="abaqus_work"
N_CPUS=8

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [retry] $*"; }

# --- 1. バッチ (run_batch.py) の完了を待つ ---
log "Waiting for run_batch.py to finish..."
while pgrep -f "run_batch.py.*${DOE}" > /dev/null 2>&1; do
    sleep 60
done
log "Batch finished."

# --- 2. 失敗サンプルの再実行 ---
for sid in "${FAILED_IDS[@]}"; do
    sample_dir="${PROJECT_ROOT}/${OUTPUT_DIR}/sample_$(printf '%04d' $sid)"

    # nodes.csv があれば既に成功済み
    if [ -f "${sample_dir}/nodes.csv" ]; then
        log "Sample $(printf '%04d' $sid): already completed, skipping"
        continue
    fi

    log "Retrying sample $(printf '%04d' $sid)..."

    # 空ディレクトリを削除して --force 不要にする
    if [ -d "${sample_dir}" ]; then
        rm -rf "${sample_dir}"
    fi

    python src/run_batch.py \
        --doe "${DOE}" \
        --output_dir "${OUTPUT_DIR}" \
        --gen_script realistic \
        --global_seed 25 \
        --defect_seed 10 \
        --work_dir "${WORK_DIR}" \
        --n_cpus "${N_CPUS}" \
        --start "${sid}" \
        --end "$((sid + 1))"

    # 結果チェック
    if [ -f "${sample_dir}/nodes.csv" ]; then
        log "Sample $(printf '%04d' $sid): SUCCESS"
    else
        log "Sample $(printf '%04d' $sid): FAILED again"
    fi
done

log "All retries done."
