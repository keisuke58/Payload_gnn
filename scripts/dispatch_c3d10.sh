#!/bin/bash
# dispatch_c3d10.sh — C3D10+機械荷重 105サンプル 6サーバー並列生成
#
# 使い方:
#   bash scripts/dispatch_c3d10.sh          # 全マシン起動
#   bash scripts/dispatch_c3d10.sh status   # 進捗確認
#   bash scripts/dispatch_c3d10.sh kill     # 全マシン停止
#
set -euo pipefail

PROJECT_ROOT="/home/nishioka/Payload2026"
DOE="doe_c3d10_mech_100.json"
OUTPUT_DIR="dataset_c3d10_mech_105"
GEN_SCRIPT="realistic"
GLOBAL_SEED=25
DEFECT_SEED=10
N_CPUS=8
MEMORY="40gb"
N_TOTAL=105

# マシン定義 (6サーバー, localhost除外 — 62GB RAM では C3D10 OOM リスク)
MACHINES=(frontale01 frontale02 frontale04 marinos03)
WORK_DIRS=(abaqus_work_f01 abaqus_work_f02 abaqus_work_f04 abaqus_work_m03)

# 105 サンプルを 4 マシンで分割
# frontale03, marinos01, marinos02 は接続不可のためスキップ (MEMORY.md 参照)
STARTS=(0 27 54 80)
ENDS=(27 54 80 105)

# SSH コマンド (OpenSSL 不整合対策)
SSH_CMD='LD_LIBRARY_PATH="" /usr/bin/ssh'

CMD="dispatch"
if [ $# -ge 1 ]; then
    CMD="$1"
fi

case "$CMD" in
    status)
        echo "=== C3D10+機械荷重 並列バッチ進捗 ==="
        echo "  DOE: ${DOE} (${N_TOTAL} samples)"
        echo "  Output: ${OUTPUT_DIR}"
        echo ""
        total_done=0
        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            s="${STARTS[$i]}"
            e="${ENDS[$i]}"
            range_size=$((e - s))

            running=$(eval ${SSH_CMD} "$host" "ps aux | grep 'run_batch.*work_dir.*${wdir}' | grep -v grep | wc -l" 2>/dev/null || echo "?")

            # サンプル範囲内の完了数
            done_count=0
            for sid in $(seq "$s" $((e - 1))); do
                sid_fmt=$(printf "%04d" "$sid")
                if [ -f "${PROJECT_ROOT}/${OUTPUT_DIR}/sample_${sid_fmt}/nodes.csv" ]; then
                    done_count=$((done_count + 1))
                fi
            done
            total_done=$((total_done + done_count))

            status_icon="."
            if [ "$running" -gt 0 ] 2>/dev/null; then
                status_icon="RUN"
            elif [ "$done_count" -eq "$range_size" ]; then
                status_icon="DONE"
            fi

            printf "  %-12s [%s] %04d-%04d  %d/%d done  (work: %s)\n" \
                "$host" "$status_icon" "$s" $((e-1)) "$done_count" "$range_size" "$wdir"
        done

        # 全体
        all_done=$(ls -d "${PROJECT_ROOT}/${OUTPUT_DIR}"/sample_*/nodes.csv 2>/dev/null | wc -l)
        echo ""
        echo "全体: ${all_done}/${N_TOTAL} 完了 (並列範囲: ${total_done}/${N_TOTAL})"
        ;;

    kill)
        echo "=== 全マシンのバッチを停止 ==="
        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            eval ${SSH_CMD} "$host" "pkill -f 'run_batch.*work_dir.*${wdir}'" 2>/dev/null && echo "  ${host}: stopped" || echo "  ${host}: not running"
        done
        ;;

    dispatch|start)
        echo "=== C3D10+機械荷重 並列バッチ起動 ==="
        echo "  DOE: ${DOE} (${N_TOTAL} samples)"
        echo "  Output: ${OUTPUT_DIR}"
        echo "  CPUs/job: ${N_CPUS}"
        echo "  Memory: ${MEMORY}"
        echo ""

        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            s="${STARTS[$i]}"
            e="${ENDS[$i]}"

            # 作業ディレクトリ作成
            eval ${SSH_CMD} "$host" "mkdir -p ${PROJECT_ROOT}/${wdir}"

            BATCH_CMD="cd ${PROJECT_ROOT} && python src/run_batch.py \
                --doe ${DOE} \
                --output_dir ${OUTPUT_DIR} \
                --gen_script ${GEN_SCRIPT} \
                --global_seed ${GLOBAL_SEED} \
                --defect_seed ${DEFECT_SEED} \
                --work_dir ${wdir} \
                --n_cpus ${N_CPUS} \
                --memory ${MEMORY} \
                --start ${s} \
                --end ${e} \
                --keep_inp"

            LOG="${PROJECT_ROOT}/${OUTPUT_DIR}/batch_${host}.log"

            echo "  ${host}: samples ${s}-$((e-1)) → ${wdir}"
            eval ${SSH_CMD} "$host" "mkdir -p ${PROJECT_ROOT}/${OUTPUT_DIR} && nohup bash -c '${BATCH_CMD}' > ${LOG} 2>&1 &"
            echo "    launched via SSH"
        done

        echo ""
        echo "全マシン起動完了。進捗確認: bash scripts/dispatch_c3d10.sh status"
        ;;

    *)
        echo "Usage: bash scripts/dispatch_c3d10.sh [dispatch|status|kill]"
        ;;
esac
