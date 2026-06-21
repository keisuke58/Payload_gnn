#!/bin/bash
# EXP4: 8-class from-scratch on czm_96_8class (stratified split)
# Usage: bash scripts/run_s12_multiclass.sh [GPU_ID]

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
PYTHON="conda run --no-capture-output -n gnn_final_env python3"
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

DATA="data/processed_s12_czm_100_8class"
LOG_ROOT="runs/s12_multiclass"
RESULTS_DIR="results/s12_multiclass"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 123 7)

echo "============================================================"
echo "EXP4: 8-class multiclass  |  GPU=$GPU  |  $(date)"
echo "  data: $DATA"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
    OUT="$LOG_ROOT/exp4_scratch8class/seed${SEED}"
    echo ""
    echo "  seed=$SEED → $OUT"
    $PYTHON src/train.py \
        --data_dir "$DATA" \
        --arch sage \
        --hidden 128 \
        --layers 4 \
        --epochs 300 \
        --batch_size 4 \
        --lr 1e-3 \
        --loss focal \
        --focal_gamma 2.0 \
        --patience 50 \
        --seed "$SEED" \
        --output_dir "$OUT"
done

# ── collect results ───────────────────────────────────────────────────────────
echo ""
echo ">>> Collecting results..."
$PYTHON - <<'PYEOF'
import os, csv, statistics

base = "runs/s12_multiclass"

def best_val_f1(run_dir):
    log = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log):
        return None, None
    best, best_ep = 0.0, 0
    with open(log) as f:
        for row in csv.DictReader(f):
            f1 = float(row.get("val_f1", 0))
            if f1 > best:
                best = f1
                best_ep = int(row["epoch"])
    return best, best_ep

vals = []
for seed in [42, 123, 7]:
    run = os.path.join(base, f"exp4_scratch8class/seed{seed}")
    f1, ep = best_val_f1(run)
    if f1 is not None:
        vals.append(f1)
        print(f"  seed={seed}: best_val_f1={f1:.4f} @ ep{ep}")

if vals:
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n  EXP4 (8-class): {mean:.4f} ± {std:.4f}")

    import json
    out = "results/s12_multiclass/multiclass_summary.json"
    with open(out, "w") as f:
        json.dump({"exp4_8class": {"mean": mean, "std": std, "seeds": vals}}, f, indent=2)
    print(f"Summary → {out}")
PYEOF

echo ""
echo "Done: $(date)"
