#!/bin/bash
# s12 Transfer Learning: proxy(mixed_400) → faithful(czm_96_binary)
# 3 experiments × 3 seeds
# Usage: bash scripts/run_s12_transfer.sh [GPU_ID]
#
# Experiments:
#   exp1_scratch    -- from-scratch on czm_96_binary
#   exp2_proxy      -- train on mixed_400 (material-swap proxy)
#   exp2_zeroshot   -- eval proxy model on czm_96_binary val (no fine-tune)
#   exp3_finetune   -- proxy pretrain → czm_96_binary fine-tune

set -e
cd "$(dirname "$0")/.."

PYTHON="conda run --no-capture-output -n gnn_final_env python3"
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

LOG_ROOT="runs/s12_transfer"
PROXY_DATA="data/processed_s12_mixed_400"
FAITHFUL_DATA="data/processed_s12_czm_96_binary"
RESULTS_DIR="results/s12_transfer"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 123 7)

echo "============================================================"
echo "S6 Transfer Learning  |  GPU=$GPU  |  $(date)"
echo "  proxy  : $PROXY_DATA"
echo "  faithful: $FAITHFUL_DATA"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# EXP 1: From-scratch on faithful czm_96_binary
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> EXP 1: from-scratch on faithful (czm_96_binary)"
for SEED in "${SEEDS[@]}"; do
    OUT="$LOG_ROOT/exp1_scratch/seed${SEED}"
    echo "  seed=$SEED → $OUT"
    $PYTHON src/train.py \
        --data_dir "$FAITHFUL_DATA" \
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

# ─────────────────────────────────────────────────────────────
# EXP 2a: Proxy-only training on mixed_400
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> EXP 2a: proxy-only training (mixed_400)"
for SEED in "${SEEDS[@]}"; do
    OUT="$LOG_ROOT/exp2_proxy/seed${SEED}"
    echo "  seed=$SEED → $OUT"
    $PYTHON src/train.py \
        --data_dir "$PROXY_DATA" \
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

# ─────────────────────────────────────────────────────────────
# EXP 2b: Zero-shot eval (proxy model → faithful val, no adaptation)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> EXP 2b: zero-shot eval (proxy → faithful val)"
for SEED in "${SEEDS[@]}"; do
    PROXY_CKPT="$LOG_ROOT/exp2_proxy/seed${SEED}/best_model.pt"
    OUT_JSON="$RESULTS_DIR/exp2_zeroshot_seed${SEED}.json"
    echo "  seed=$SEED → $OUT_JSON"
    $PYTHON src/eval_cross_dataset.py \
        --model "$PROXY_CKPT" \
        --datasets "$FAITHFUL_DATA/val.pt" \
        --output "$OUT_JSON"
done

# ─────────────────────────────────────────────────────────────
# EXP 3: Fine-tune from proxy checkpoint on faithful
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> EXP 3: fine-tune proxy → faithful (czm_96_binary)"
for SEED in "${SEEDS[@]}"; do
    PROXY_CKPT="$LOG_ROOT/exp2_proxy/seed${SEED}/best_model.pt"
    OUT="$LOG_ROOT/exp3_finetune/seed${SEED}"
    echo "  seed=$SEED → $OUT"
    $PYTHON src/train.py \
        --data_dir "$FAITHFUL_DATA" \
        --pretrained "$PROXY_CKPT" \
        --arch sage \
        --hidden 128 \
        --layers 4 \
        --epochs 150 \
        --batch_size 4 \
        --lr 3e-4 \
        --loss focal \
        --focal_gamma 2.0 \
        --freeze_layers 2 \
        --patience 30 \
        --seed "$SEED" \
        --output_dir "$OUT"
done

# ─────────────────────────────────────────────────────────────
# Collect results
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Collecting results..."
$PYTHON - <<'PYEOF'
import os, json, csv, glob, statistics

base = "runs/s12_transfer"
results_dir = "results/s12_transfer"

def best_val_f1(run_dir):
    log = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log):
        return None, None
    best = 0.0
    best_ep = 0
    with open(log) as f:
        for row in csv.DictReader(f):
            f1 = float(row.get("val_f1", 0))
            if f1 > best:
                best = f1
                best_ep = int(row["epoch"])
    return best, best_ep

summary = {}
for exp in ["exp1_scratch", "exp3_finetune"]:
    vals = []
    for seed in [42, 123, 7]:
        run = os.path.join(base, exp, f"seed{seed}")
        f1, ep = best_val_f1(run)
        if f1 is not None:
            vals.append(f1)
            print(f"  {exp}/seed{seed}: best_val_f1={f1:.4f} @ ep{ep}")
    if vals:
        summary[exp] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals)>1 else 0}

# Zero-shot results
zs_vals = []
for seed in [42, 123, 7]:
    jf = os.path.join(results_dir, f"exp2_zeroshot_seed{seed}.json")
    if not os.path.exists(jf):
        continue
    with open(jf) as f:
        d = json.load(f)
    agg = d.get("aggregate", {})
    f1 = agg.get("f1")
    if f1 is not None:
        zs_vals.append(f1)
        print(f"  exp2_zeroshot/seed{seed}: F1={f1:.4f}")
if zs_vals:
    summary["exp2_zeroshot"] = {"mean": statistics.mean(zs_vals), "std": statistics.stdev(zs_vals) if len(zs_vals)>1 else 0}

print("\n=== SUMMARY (val F1, mean±std) ===")
for exp, stats in summary.items():
    print(f"  {exp:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

out = os.path.join(results_dir, "transfer_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary → {out}")
PYEOF

echo ""
echo "Done: $(date)"
