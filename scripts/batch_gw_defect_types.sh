#!/bin/bash
# batch_gw_defect_types.sh
# 平板ガイド波: FOD / Impact / Delamination 等の波伝播アニメーションを一括生成
#
# 各欠陥タイプで generate → run → extract_field → plot_gw_animation
#
# Usage:
#   bash scripts/batch_gw_defect_types.sh generate   # INP 生成のみ
#   bash scripts/batch_gw_defect_types.sh run         # Abaqus 実行
#   bash scripts/batch_gw_defect_types.sh extract     # フィールド抽出
#   bash scripts/batch_gw_defect_types.sh animate     # GIF 生成
#   bash scripts/batch_gw_defect_types.sh all         # 全工程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE="${REMOTE:-frontale04}"
WORK_DIR="${WORK_DIR:-$HOME/Payload2026/abaqus_work}"
SRC_DIR="$PROJECT_ROOT/src"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
OUT_DIR="$PROJECT_ROOT/wiki_repo/images/guided_wave"
CPUS=4
MESH_SEED=3.0

ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

# 欠陥ケース: job_name defect_json
# defect_type を指定して FOD / Impact / Delamination を追加
CASES=(
  "Job-GW-Flat-Healthy  "
  "Job-GW-Flat-Debond   {\"defect_type\":\"debonding\",\"x_center\":80,\"y_center\":0,\"radius\":25}"
  "Job-GW-Flat-FOD      {\"defect_type\":\"fod\",\"x_center\":80,\"y_center\":0,\"radius\":25,\"stiffness_factor\":10}"
  "Job-GW-Flat-Impact   {\"defect_type\":\"impact\",\"x_center\":80,\"y_center\":0,\"radius\":25,\"damage_ratio\":0.3}"
  "Job-GW-Flat-Delam    {\"defect_type\":\"delamination\",\"x_center\":80,\"y_center\":0,\"radius\":25,\"delam_depth\":0.5}"
)

generate_models() {
    echo "=== Generating flat panel INP files (defect types) ==="
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "mkdir -p $WORK_DIR" 2>/dev/null || true

    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        defect=$(echo "$case_str" | awk '{$1=""; print $0}' | sed 's/^ *//')
        if [[ -z "$defect" ]]; then
            echo "Generating: $job (healthy)"
            LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
                $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_guided_wave.py \
                -- --job $job --mesh_seed $MESH_SEED --no_run" 2>&1 | tail -3
        else
            echo "Generating: $job (defect: $defect)"
            LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
                $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_guided_wave.py \
                -- --job $job --mesh_seed $MESH_SEED --defect '$defect' --no_run" 2>&1 | tail -3
        fi
        echo "---"
    done
    echo "INP generation complete."
}

run_jobs() {
    echo "=== Running Abaqus jobs sequentially ==="
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        odb_check=$(LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "test -f $WORK_DIR/$job.odb && echo exists || echo missing" 2>/dev/null)
        if [[ "$odb_check" == "exists" ]]; then
            echo "Skip: $job (ODB exists)"
        else
            echo "Running: $job"
            LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
                $ENV_CMD abaqus job=$job input=$job.inp cpus=$CPUS interactive" 2>&1 | tail -3
            echo "Completed: $job"
        fi
        echo "---"
    done
    echo "All jobs complete."
}

extract_field() {
    echo "=== Extracting U3 field for animation ==="
    ODB_LIST=""
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        ODB_LIST="$ODB_LIST $job.odb"
    done
    echo "ODBs:$ODB_LIST"
    LD_LIBRARY_PATH="" /usr/bin/ssh $REMOTE "cd $WORK_DIR && \
        $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_field.py $ODB_LIST" 2>&1
    echo "Field extraction complete."

    echo "=== Copying CSVs to local ==="
    mkdir -p "$PROJECT_ROOT/abaqus_work"
    for case_str in "${CASES[@]}"; do
        job=$(echo "$case_str" | awk '{print $1}')
        for suf in _coords.csv _frames.csv; do
            LD_LIBRARY_PATH="" /usr/bin/scp $REMOTE:$WORK_DIR/${job}${suf} "$PROJECT_ROOT/abaqus_work/" 2>/dev/null \
                && echo "  Got ${job}${suf}" || true
        done
    done
}

animate_all() {
    echo "=== Generating wave comparison GIFs ==="
    WORK_LOCAL="${WORK_LOCAL:-$PROJECT_ROOT/abaqus_work}"
    PREFIX_HEALTHY="$WORK_LOCAL/Job-GW-Flat-Healthy"
    if [[ ! -f "${PREFIX_HEALTHY}_coords.csv" ]]; then
        echo "ERROR: ${PREFIX_HEALTHY}_coords.csv not found."
        echo "Run 'extract' first, or copy from remote."
        exit 1
    fi

    mkdir -p "$OUT_DIR"

    # 各欠陥タイプ vs Healthy の比較 GIF
    DEFECT_JOBS=("Job-GW-Flat-Debond" "Job-GW-Flat-FOD" "Job-GW-Flat-Impact" "Job-GW-Flat-Delam")
    DEFECT_LABELS=("Debonding r=25mm" "FOD r=25mm" "Impact r=25mm" "Delamination r=25mm")
    for i in "${!DEFECT_JOBS[@]}"; do
        job="${DEFECT_JOBS[$i]}"
        label="${DEFECT_LABELS[$i]}"
        prefix_d="$WORK_LOCAL/$job"
        if [[ ! -f "${prefix_d}_coords.csv" ]]; then
            echo "  SKIP $job (no _coords.csv)"
            continue
        fi
        out_name="gw_wave_comparison_${job#Job-GW-Flat-}.gif"
        out_path="$OUT_DIR/$out_name"
        echo "--- $label ---"
        python scripts/plot_gw_animation.py "$PREFIX_HEALTHY" "$prefix_d" \
            --output "$out_path" --defect 80,0,25 --defect_label "$label" --no_snapshot
    done

    # 全欠陥種を横並びにした1枚のGIF
    echo ""
    echo "--- All defect types (1 GIF) ---"
    python scripts/plot_gw_animation.py "$PREFIX_HEALTHY" --multi_defects --out_dir "$OUT_DIR"
    echo "GIFs saved to $OUT_DIR"
}

case "${1:-all}" in
    generate) generate_models ;;
    run)      run_jobs ;;
    extract)  extract_field ;;
    animate)  animate_all ;;
    all)
        generate_models
        run_jobs
        extract_field
        animate_all
        ;;
    *) echo "Usage: $0 [generate|run|extract|animate|all]"
       echo "  generate: INP 生成"
       echo "  run:      Abaqus 実行"
       echo "  extract:  フィールド抽出 + ローカルへコピー"
       echo "  animate:  GIF 生成 (extract 済み CSV 使用)"
       echo "  all:      全工程" ;;
esac
