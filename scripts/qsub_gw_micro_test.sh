#!/bin/bash
# qsub_gw_micro_test.sh — 微小欠陥テスト3ケース: 5mm, 10mm, 15mm
#
# 1. Abaqus CAE で INP 生成 (frontale)
# 2. qsub で FEM 実行
#
# Usage:
#   bash scripts/qsub_gw_micro_test.sh           # INP生成 + qsub投入
#   bash scripts/qsub_gw_micro_test.sh status     # 進捗確認
#   bash scripts/qsub_gw_micro_test.sh inp_only   # INP生成のみ

set -e

PROJECT_ROOT=~/Payload2026
WORK_DIR=$PROJECT_ROOT/abaqus_work
SRC_DIR=$PROJECT_ROOT/src
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QSUB_SCRIPT="$SCRIPT_DIR/qsub_gw_single.sh"
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus

export LD_PRELOAD=/home/nishioka/libfake_x11.so

# 3テストケース: ジョブ名 + DOEファイル
declare -A JOBS
JOBS[Job-GW-Fair-5mm-Test]=$PROJECT_ROOT/doe_gw_5mm_test1.json
JOBS[Job-GW-Fair-Micro-Test]=$PROJECT_ROOT/doe_gw_micro_test1.json
JOBS[Job-GW-Fair-15mm-Test]=$PROJECT_ROOT/doe_gw_15mm_test1.json

MODE="${1:-run}"

# ==============================================================================
# status
# ==============================================================================
if [ "$MODE" = "status" ]; then
    echo "=== Micro Defect Test Status ==="
    for job in "Job-GW-Fair-5mm-Test" "Job-GW-Fair-Micro-Test" "Job-GW-Fair-15mm-Test"; do
        doe="${JOBS[$job]}"
        radius=$(python3 -c "import json; d=json.load(open('$doe')); print(d['samples'][0]['defect_params']['radius'])")
        printf "  %-30s (r=%smm): " "$job" "$radius"
        if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
            echo "DONE"
        elif [ -f "$WORK_DIR/$job.lck" ]; then
            pct=$(tail -1 "$WORK_DIR/$job.sta" 2>/dev/null | awk '{print $2}')
            echo "RUNNING (step_time=$pct)"
        elif [ -f "$WORK_DIR/$job.inp" ]; then
            echo "INP ready (not started)"
        else
            echo "NO INP"
        fi
    done
    echo ""
    echo "PBS queue:"
    qstat -u nishioka 2>/dev/null || echo "  (empty)"
    exit 0
fi

# ==============================================================================
# INP 生成 + qsub
# ==============================================================================
cd "$WORK_DIR"

for job in "Job-GW-Fair-5mm-Test" "Job-GW-Fair-Micro-Test" "Job-GW-Fair-15mm-Test"; do
    doe="${JOBS[$job]}"
    defect_json=$(python3 -c "
import json
d = json.load(open('$doe'))
p = d['samples'][0]['defect_params']
print(json.dumps(p))
")
    radius=$(python3 -c "import json; print(json.loads('$defect_json')['radius'])")

    echo "=== $job (radius=${radius}mm) ==="

    # すでに完了ならスキップ
    if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$job.sta" 2>/dev/null; then
        echo "  Already completed, skipping."
        continue
    fi

    # INP 生成
    if [ ! -f "$WORK_DIR/$job.inp" ]; then
        echo "  Generating INP..."
        $ABAQUS cae noGUI=$SRC_DIR/generate_gw_fairing.py -- \
            --job "$job" \
            --defect "$defect_json" \
            --no_run 2>&1 | tail -5
        if [ -f "$WORK_DIR/$job.inp" ]; then
            echo "  INP generated: $job.inp"
        else
            echo "  ERROR: INP generation failed!"
            continue
        fi
    else
        echo "  INP already exists."
    fi

    # qsub 投入
    if [ "$MODE" != "inp_only" ]; then
        echo "  Submitting to PBS..."
        qsub -v "JOB_NAME=$job" -N "$job" "$QSUB_SCRIPT" 2>&1
    fi

    echo ""
done

if [ "$MODE" != "inp_only" ]; then
    echo "確認: bash scripts/qsub_gw_micro_test.sh status"
    echo "PBS:  qstat -u nishioka"
fi
