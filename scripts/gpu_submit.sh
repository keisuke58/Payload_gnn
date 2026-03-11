#!/bin/bash
# gpu_submit.sh — GPU サーバーにジョブを投入（tmux セッションで実行）
#
# Usage:
#   ./scripts/gpu_submit.sh <server> <gpu_id> "<command>"
#   ./scripts/gpu_submit.sh auto "<command>"
#
# Examples:
#   ./scripts/gpu_submit.sh vancouver02 0 "python src/train.py --config config.yaml"
#   ./scripts/gpu_submit.sh auto "python src/train.py --config config.yaml"
#
# 特徴:
#   - tmux セッションで実行 → SSH 切断しても継続
#   - CUDA_VISIBLE_DEVICES で指定 GPU のみ使用
#   - nice で CPU 優先度を下げる
#   - 実行前に対象 GPU の空きを確認

set -euo pipefail

SSH_CMD="LD_LIBRARY_PATH= /usr/bin/ssh"
PROJECT_DIR="Payload2026"
SERVERS="vancouver01 vancouver02 stuttgart02 celtic01 celtic02"
FREE_THRESHOLD=8000  # MiB

usage() {
    echo "Usage:"
    echo "  $0 <server> <gpu_id> \"<command>\""
    echo "  $0 auto \"<command>\""
    echo ""
    echo "Examples:"
    echo "  $0 vancouver02 0 \"python src/train.py\""
    echo "  $0 auto \"python src/train.py\""
    exit 1
}

# GPU が空いているか確認 (戻り値: 0=空き, 1=使用中)
check_gpu_free() {
    local server=$1
    local gpu_id=$2
    local mem_used
    mem_used=$(eval $SSH_CMD -o ConnectTimeout=5 -o BatchMode=yes "$server" \
        "bash -c 'nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null'" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$mem_used" ]; then
        return 1
    fi
    mem_used=$(echo "$mem_used" | tr -d ' ')
    [ "$mem_used" -le 100 ]
}

# 自動で空き GPU を探す（優先順: vancouver > stuttgart > celtic）
find_free_gpu() {
    for server in $SERVERS; do
        local gpu_info
        gpu_info=$(eval $SSH_CMD -o ConnectTimeout=5 -o BatchMode=yes "$server" \
            "bash -c 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>&1'" 2>/dev/null)

        if [ $? -ne 0 ] || echo "$gpu_info" | grep -qi "error\|unable\|failed"; then
            continue
        fi

        while IFS=',' read -r idx mem_used mem_total; do
            idx=$(echo "$idx" | tr -d ' ')
            mem_used=$(echo "$mem_used" | tr -d ' ')
            [ -z "$idx" ] && continue
            if [ "$mem_used" -le 100 ]; then
                echo "${server}:${idx}"
                return 0
            fi
        done <<< "$gpu_info"
    done
    return 1
}

# セッション名生成
make_session_name() {
    local cmd_short
    cmd_short=$(echo "$1" | sed 's/.*\///' | cut -d' ' -f1 | sed 's/\.py//')
    echo "gpu_${cmd_short}_$(date +%m%d_%H%M)"
}

# --- メイン ---

if [ $# -lt 2 ]; then
    usage
fi

if [ "$1" = "auto" ]; then
    COMMAND="${*:2}"
    echo "空き GPU を検索中..."
    RESULT=$(find_free_gpu) || { echo "エラー: 空き GPU が見つかりません。./scripts/gpu_status.sh で確認してください。"; exit 1; }
    SERVER=$(echo "$RESULT" | cut -d: -f1)
    GPU_ID=$(echo "$RESULT" | cut -d: -f2)
    echo "→ ${SERVER} GPU${GPU_ID} が空いています"
elif [ $# -lt 3 ]; then
    usage
else
    SERVER="$1"
    GPU_ID="$2"
    COMMAND="${*:3}"
fi

SESSION_NAME=$(make_session_name "$COMMAND")

echo ""
echo "=========================================="
echo "  GPU ジョブ投入"
echo "=========================================="
echo "  サーバー: ${SERVER}"
echo "  GPU:      ${GPU_ID}"
echo "  セッション: ${SESSION_NAME}"
echo "  コマンド: ${COMMAND}"
echo "=========================================="
echo ""

# GPU の空きを確認
echo "GPU${GPU_ID} の空きを確認中..."
if ! check_gpu_free "$SERVER" "$GPU_ID"; then
    echo "⚠ 警告: ${SERVER} GPU${GPU_ID} は使用中の可能性があります"
    read -p "続行しますか？ [y/N]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "中止しました"
        exit 1
    fi
fi
echo "✓ GPU${GPU_ID} は空いています"

# リモートでプロジェクトディレクトリの存在確認
echo "リモート環境を確認中..."
REMOTE_HOME=$(eval $SSH_CMD -o ConnectTimeout=5 "$SERVER" 'echo $HOME' 2>/dev/null)
REMOTE_PROJECT="${REMOTE_HOME}/${PROJECT_DIR}"

HAS_PROJECT=$(eval $SSH_CMD -o ConnectTimeout=5 "$SERVER" \
    "test -d ${REMOTE_PROJECT} && echo yes || echo no" 2>/dev/null)

if [ "$HAS_PROJECT" != "yes" ]; then
    echo "⚠ ${SERVER} に ${PROJECT_DIR} がありません"
    echo "  先に git clone するか scp してください"
    exit 1
fi

# tmux セッションでジョブ投入
echo "ジョブを投入中..."
eval $SSH_CMD "$SERVER" "tmux new-session -d -s '${SESSION_NAME}' \
    'cd ${REMOTE_PROJECT} && \
     export CUDA_VISIBLE_DEVICES=${GPU_ID} && \
     echo \"=== Job started: \$(date) ===\" && \
     echo \"Server: \$(hostname), GPU: ${GPU_ID}\" && \
     echo \"Command: ${COMMAND}\" && \
     echo \"=========================================\" && \
     nice -n 10 bash -l -c \"CUDA_VISIBLE_DEVICES=${GPU_ID} ${COMMAND}\" ; \
     echo \"\" && \
     echo \"=== Job finished: \$(date) ===\"  && \
     echo \"Press Enter to close...\" && \
     read'" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ジョブ投入完了!"
    echo ""
    echo "  確認: $SSH_CMD $SERVER tmux attach -t $SESSION_NAME"
    echo "  一覧: $SSH_CMD $SERVER tmux ls"
    echo "  切断: tmux 内で Ctrl+b → d"
    echo ""
else
    echo "エラー: ジョブ投入に失敗しました"
    exit 1
fi
