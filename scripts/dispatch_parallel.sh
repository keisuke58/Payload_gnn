#!/bin/bash
# dispatch_parallel.sh — 複数マシンで並列バッチ実行
#
# 使い方:
#   bash scripts/dispatch_parallel.sh          # 全マシン起動
#   bash scripts/dispatch_parallel.sh status   # 進捗確認
#   bash scripts/dispatch_parallel.sh kill     # 全マシン停止
#
set -euo pipefail

PROJECT_ROOT="/home/nishioka/Payload2026"
DOE="doe_realistic_25mm_100.json"
OUTPUT_DIR="dataset_realistic_25mm_100"
GEN_SCRIPT="realistic"
GLOBAL_SEED=25
DEFECT_SEED=10
N_CPUS=8

# マシン定義 (frontale01 は負荷高のためスキップ)
MACHINES=(localhost frontale02 frontale03 marinos01 marinos02)
WORK_DIRS=(abaqus_work abaqus_work_f02 abaqus_work_f03 abaqus_work_m01 abaqus_work_m02)

# 完了: 0000-0003,0005-0008,0082 (9個)  失敗: 0004
# 残り: 0004,0009-0081,0083-0099 = 91個 → 5マシンで分割
# run_batch.py は完了済みサンプルを自動スキップするので、連続レンジで分割
STARTS=(0 20 40 60 80)
ENDS=(20 40 60 80 100)

CMD="dispatch"
if [ $# -ge 1 ]; then
    CMD="$1"
fi

case "$CMD" in
    status)
        echo "=== 並列バッチ進捗 ==="
        total_done=0
        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            s="${STARTS[$i]}"
            e="${ENDS[$i]}"
            range_size=$((e - s))

            if [ "$host" = "localhost" ]; then
                running=$(ps aux | grep "run_batch.*work_dir.*${wdir}" | grep -v grep | wc -l)
            else
                running=$(ssh "$host" "ps aux | grep 'run_batch.*work_dir.*${wdir}' | grep -v grep | wc -l" 2>/dev/null || echo "?")
            fi

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
        echo "全体: ${all_done}/100 完了 (並列範囲: ${total_done}/90)"
        ;;

    kill)
        echo "=== 全マシンのバッチを停止 ==="
        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            if [ "$host" = "localhost" ]; then
                pkill -f "run_batch.*work_dir.*${wdir}" 2>/dev/null && echo "  ${host}: stopped" || echo "  ${host}: not running"
            else
                ssh "$host" "pkill -f 'run_batch.*work_dir.*${wdir}'" 2>/dev/null && echo "  ${host}: stopped" || echo "  ${host}: not running"
            fi
        done
        ;;

    dispatch|start)
        echo "=== 並列バッチ起動 ==="
        echo "  DOE: ${DOE}"
        echo "  Output: ${OUTPUT_DIR}"
        echo "  CPUs/job: ${N_CPUS}"
        echo ""

        for i in "${!MACHINES[@]}"; do
            host="${MACHINES[$i]}"
            wdir="${WORK_DIRS[$i]}"
            s="${STARTS[$i]}"
            e="${ENDS[$i]}"

            # 作業ディレクトリ作成
            if [ "$host" = "localhost" ]; then
                mkdir -p "${PROJECT_ROOT}/${wdir}"
            else
                ssh "$host" "mkdir -p ${PROJECT_ROOT}/${wdir}"
            fi

            BATCH_CMD="cd ${PROJECT_ROOT} && python src/run_batch.py \
                --doe ${DOE} \
                --output_dir ${OUTPUT_DIR} \
                --gen_script ${GEN_SCRIPT} \
                --global_seed ${GLOBAL_SEED} \
                --defect_seed ${DEFECT_SEED} \
                --work_dir ${wdir} \
                --n_cpus ${N_CPUS} \
                --start ${s} \
                --end ${e}"

            LOG="${PROJECT_ROOT}/${OUTPUT_DIR}/batch_${host}.log"

            if [ "$host" = "localhost" ]; then
                echo "  ${host}: samples ${s}-$((e-1)) → ${wdir}"
                nohup bash -c "${BATCH_CMD}" > "$LOG" 2>&1 &
                echo "    PID: $!"
            else
                echo "  ${host}: samples ${s}-$((e-1)) → ${wdir}"
                ssh "$host" "nohup bash -c '${BATCH_CMD}' > ${LOG} 2>&1 &"
                echo "    launched via SSH"
            fi
        done

        echo ""
        echo "全マシン起動完了。進捗確認: bash scripts/dispatch_parallel.sh status"
        ;;

    *)
        echo "Usage: bash scripts/dispatch_parallel.sh [dispatch|status|kill]"
        ;;
esac
