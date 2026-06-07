#!/bin/bash
# batch_healthy_99s.sh
# One-shot: generate 100-sensor Healthy baseline on frontale04 (cpus=1),
# then augment to 100 Healthy-B CSVs, create gw_dataset_99s/ with consistent
# 99-sensor format (100H + 97D), and launch MIFNO-GW retraining on vancouver01.
#
# Usage:
#   bash scripts/batch_healthy_99s.sh            # full pipeline
#   bash scripts/batch_healthy_99s.sh --augment-only  # skip Abaqus, just augment+prep
#   bash scripts/batch_healthy_99s.sh --train-only    # skip Abaqus+augment, just launch training

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE=frontale04
WORK_DIR=~/Payload2026/abaqus_work
SRC_DIR=~/Payload2026/src
SCRIPTS_DIR=~/Payload2026/scripts
N_SENSORS=100          # Abaqus output: 100 sensors (99 data + 1 actuator skipped)
CPUS=1                 # cpus=1: only option due to lost parallel Abaqus license
FREQ_KHZ=50
MESH_SEED=5.0
JOB=Job-GW-Fair-Healthy-99s
LOCAL_CSV_DIR=abaqus_work/gw_fairing_dataset
DATASET_99S_DIR=abaqus_work/gw_dataset_99s
MIN_ODB_SIZE=500000

ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin"

MODE="${1:-full}"

# ---------------------------------------------------------------
step_abaqus() {
    echo "=== [1/4] Generating INP on $REMOTE ==="
    # generate_gw_fairing.py with no defect = healthy baseline, 100 sensors
    ssh "$REMOTE" "cd $WORK_DIR && \
        $ENV_CMD abaqus cae noGUI=$SRC_DIR/generate_gw_fairing.py \
        -- --job $JOB --freq $FREQ_KHZ --mesh_seed $MESH_SEED \
           --n_sensors $N_SENSORS --no_run" 2>&1 | tail -5
    echo "INP generated."

    echo "=== [2/4] Running Abaqus/Explicit on $REMOTE (cpus=1, ~90min) ==="
    local odb_ok
    odb_ok=$(ssh "$REMOTE" "test -f $WORK_DIR/$JOB.odb && \
        s=\$(stat -c%s $WORK_DIR/$JOB.odb); \
        [ \"\$s\" -gt $MIN_ODB_SIZE ] && echo OK || echo SMALL" 2>/dev/null || echo MISSING)
    if [ "$odb_ok" = "OK" ]; then
        echo "ODB already exists and looks valid — skipping Abaqus run."
    else
        ssh "$REMOTE" "cd $WORK_DIR && \
            $ENV_CMD abaqus job=$JOB input=$JOB.inp cpus=$CPUS interactive" 2>&1 | tail -5
        echo "Abaqus run complete."
    fi

    echo "=== [2b] Extracting sensor history ==="
    ssh "$REMOTE" "cd $WORK_DIR && \
        $ENV_CMD abaqus python $SCRIPTS_DIR/extract_gw_history.py $JOB.odb" 2>&1 | grep -E "CSV|ERROR|WARNING" || true

    echo "=== [2c] Copying CSV to local ==="
    mkdir -p "$LOCAL_CSV_DIR"
    scp "$REMOTE:$WORK_DIR/${JOB}_sensors.csv" "$LOCAL_CSV_DIR/" \
        && echo "  Copied ${JOB}_sensors.csv" \
        || { echo "ERROR: copy failed"; exit 1; }
}

step_augment() {
    local src="$LOCAL_CSV_DIR/${JOB}_sensors.csv"
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found. Run without --augment-only first."
        exit 1
    fi

    echo "=== [3/4] Augmenting Healthy 99-sensor: 1 → 100 (prefix Healthy-B) ==="
    python3 src/augment_gw_healthy.py \
        --input "$src" \
        --output "$LOCAL_CSV_DIR" \
        --n_healthy 100 \
        --prefix "Job-GW-Fair-Healthy-B" \
        --seed 99
    echo "  -> $LOCAL_CSV_DIR/Job-GW-Fair-Healthy-B000..B099_sensors.csv"

    echo "=== [3b] Building gw_dataset_99s/ (100H-B + 97D) ==="
    mkdir -p "$DATASET_99S_DIR"

    # Copy 100-sensor Healthy-B files
    for f in "$LOCAL_CSV_DIR"/Job-GW-Fair-Healthy-B*_sensors.csv; do
        [ -f "$f" ] && cp "$f" "$DATASET_99S_DIR/"
    done
    echo "  Healthy-B copied: $(ls "$DATASET_99S_DIR"/Job-GW-Fair-Healthy-B*_sensors.csv 2>/dev/null | wc -l) files"

    # Copy 99-sensor defect files (main dataset, non-_9s_, non-Healthy, 99s)
    python3 - <<'PYEOF'
import os, sys
src = 'abaqus_work/gw_fairing_dataset'
dst = 'abaqus_work/gw_dataset_99s'
os.makedirs(dst, exist_ok=True)
copied = 0
for f in sorted(os.listdir(src)):
    if not f.endswith('_sensors.csv'): continue
    if 'Healthy' in f or '_9s_' in f or 'Multi' in f or 'Valid' in f or 'Test' in f: continue
    path = os.path.join(src, f)
    lines = open(path).readlines()
    if len(lines) < 2: continue
    n = len(lines[1].strip().lstrip('#').split(',')) - 1
    if n != 99: continue
    import shutil
    shutil.copy2(path, os.path.join(dst, f))
    copied += 1
print('  Defect (99s) copied:', copied, 'files')
PYEOF

    # Summary
    python3 - <<'PYEOF'
import os
d = 'abaqus_work/gw_dataset_99s'
files = [f for f in os.listdir(d) if f.endswith('.csv')]
healthy = [f for f in files if 'Healthy' in f]
defect  = [f for f in files if 'Healthy' not in f]
print('gw_dataset_99s summary: total=%d  healthy=%d  defect=%d' % (len(files), len(healthy), len(defect)))
PYEOF
}

step_train() {
    echo "=== [4/4] Launching MIFNO-GW (99-sensor) on vancouver01 ==="
    ssh -J copaam vancouver01 "cd ~/Payload2026 && git pull && \
        nohup python src/mifno_gw.py \
          --csv_dir abaqus_work/gw_dataset_99s \
          --n_sensors 99 \
          --output_dir runs/mifno_gw_99s \
          --epochs 200 \
          --batch_size 4 \
          --width 32 \
          --modes_s 8 --modes_t 64 \
          --lambda_disp 0.05 \
          > logs/mifno_gw_99s.log 2>&1 &
        echo 'MIFNO-GW 99s PID' \$!" 2>/dev/null
    echo "Training job queued on vancouver01."
}

# ---------------------------------------------------------------
case "$MODE" in
    --augment-only)
        step_augment
        step_train
        ;;
    --train-only)
        step_train
        ;;
    full|*)
        step_abaqus
        step_augment
        step_train
        echo ""
        echo "=== All done ==="
        echo "  ODB:      $REMOTE:$WORK_DIR/${JOB}.odb"
        echo "  CSV src:  $LOCAL_CSV_DIR/${JOB}_sensors.csv"
        echo "  Dataset:  $DATASET_99S_DIR/ (100H-B + 97D, 99-sensor consistent)"
        echo "  Training: see logs/mifno_gw_99s.log on vancouver01"
        ;;
esac
