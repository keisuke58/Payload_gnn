#!/bin/bash
# gpu_status.sh — 研究室 GPU サーバーの空き状況を一覧表示
#
# Usage: ./scripts/gpu_status.sh
#
# 各サーバーの GPU 使用量・利用者を表示し、空き GPU を推薦する

SSH_CMD="LD_LIBRARY_PATH= /usr/bin/ssh"
SERVERS="vancouver01 vancouver02 stuttgart01 stuttgart02 stuttgart03 celtic01 celtic02"

declare -A GPU_SPECS
GPU_SPECS[vancouver01]="RTX 4090 x4 (24GB)"
GPU_SPECS[vancouver02]="RTX 4090 x4 (24GB)"
GPU_SPECS[stuttgart01]="RTX 3090 x4 (24GB)"
GPU_SPECS[stuttgart02]="RTX 3090 x4 (24GB)"
GPU_SPECS[stuttgart03]="RTX 3090 x4 (24GB)"
GPU_SPECS[celtic01]="RTX 2080Ti x4 (11GB)"
GPU_SPECS[celtic02]="RTX 2080Ti x4 (11GB)"

FREE_THRESHOLD=8000

printf "\n"
printf "=%.0s" {1..72}
printf "\n"
printf "  GPU Server Status  (%s)\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "=%.0s" {1..72}
printf "\n\n"

RECOMMENDATIONS=()

for SERVER in $SERVERS; do
    SPEC="${GPU_SPECS[$SERVER]}"
    printf "■ %-14s  %s\n" "$SERVER" "$SPEC"

    # 1回の SSH で全情報を取得
    ALL_INFO=$(eval $SSH_CMD -o ConnectTimeout=5 -o BatchMode=yes "$SERVER" 'bash -c '"'"'
echo "===GPU_INFO==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
echo "===PROC_INFO==="
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader 2>/dev/null
echo "===GPU_BUS==="
nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader 2>/dev/null
echo "===USERS==="
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " " | while read pid; do
  [ -z "$pid" ] && continue
  user=$(ps -o user= -p "$pid" 2>/dev/null)
  echo "${pid}:${user}"
done
echo "===END==="
'"'" 2>/dev/null)

    if [ $? -ne 0 ]; then
        printf "  → 接続不可\n\n"
        continue
    fi

    # セクション分割
    GPU_INFO=$(echo "$ALL_INFO" | sed -n '/===GPU_INFO===/,/===PROC_INFO===/p' | grep -v '===')
    PROC_INFO=$(echo "$ALL_INFO" | sed -n '/===PROC_INFO===/,/===GPU_BUS===/p' | grep -v '===')
    GPU_BUS=$(echo "$ALL_INFO" | sed -n '/===GPU_BUS===/,/===USERS===/p' | grep -v '===')
    USERS_INFO=$(echo "$ALL_INFO" | sed -n '/===USERS===/,/===END===/p' | grep -v '===')

    if echo "$GPU_INFO" | grep -qi "error\|unable\|failed"; then
        printf "  → ドライバ異常\n\n"
        continue
    fi

    # GPU index → bus_id マッピング
    declare -A IDX_TO_BUS
    while IFS=',' read -r idx bus; do
        idx=$(echo "$idx" | tr -d ' ')
        bus=$(echo "$bus" | tr -d ' ')
        [ -z "$idx" ] && continue
        IDX_TO_BUS[$idx]="$bus"
    done <<< "$GPU_BUS"

    # GPU 使用状況を表示
    while IFS=',' read -r idx mem_used mem_total util; do
        idx=$(echo "$idx" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' MiB')
        mem_total=$(echo "$mem_total" | tr -d ' MiB')
        util=$(echo "$util" | tr -d ' %')

        [ -z "$idx" ] && continue

        mem_free=$((mem_total - mem_used))
        bus_id="${IDX_TO_BUS[$idx]:-}"

        # この GPU のプロセスを bus_id で特定
        USER_STR=""
        if [ -n "$bus_id" ] && [ -n "$PROC_INFO" ]; then
            while IFS=',' read -r pbus ppid pmem; do
                pbus=$(echo "$pbus" | tr -d ' ')
                ppid=$(echo "$ppid" | tr -d ' ')
                pmem=$(echo "$pmem" | tr -d ' ')
                [ -z "$pbus" ] && continue
                if [ "$pbus" = "$bus_id" ]; then
                    UNAME=$(echo "$USERS_INFO" | grep "^${ppid}:" | head -1 | cut -d: -f2)
                    [ -z "$UNAME" ] && UNAME="?"
                    USER_STR="${USER_STR} ${UNAME}(${pmem})"
                fi
            done <<< "$PROC_INFO"
        fi

        if [ "$mem_used" -le 10 ]; then
            STATUS="◯ 空き"
            RECOMMENDATIONS+=("${SERVER}:GPU${idx}  ${SPEC%% *} ${mem_total}MiB free")
        elif [ "$mem_free" -ge "$FREE_THRESHOLD" ]; then
            STATUS="△ 一部使用"
            RECOMMENDATIONS+=("${SERVER}:GPU${idx}  ${SPEC%% *} ${mem_free}MiB free")
        else
            STATUS="✕ 使用中"
        fi

        if [ -n "$USER_STR" ]; then
            printf "  GPU%s: %5s/%s MiB  util=%3s%%  %s %s\n" \
                "$idx" "$mem_used" "$mem_total" "$util" "$STATUS" "$USER_STR"
        else
            printf "  GPU%s: %5s/%s MiB  util=%3s%%  %s\n" \
                "$idx" "$mem_used" "$mem_total" "$util" "$STATUS"
        fi
    done <<< "$GPU_INFO"

    unset IDX_TO_BUS
    printf "\n"
done

# 推薦
printf -- "-%.0s" {1..72}
printf "\n"
if [ ${#RECOMMENDATIONS[@]} -gt 0 ]; then
    printf "  推薦 (空き GPU):\n"
    for rec in "${RECOMMENDATIONS[@]}"; do
        printf "    → %s\n" "$rec"
    done
else
    printf "  ⚠ 空き GPU がありません\n"
fi
printf "\n"
