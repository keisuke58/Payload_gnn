#!/bin/bash
# Grid search: smooth × connected (3×3=9), stress=0.05 fixed
# No CV (single split) → fast screening
# 4 GPU parallel, round-robin assignment

PYTHON=~/miniconda3/bin/python
cd ~/Payload2026

STRESS=0.05
SMOOTHS=(0.01 0.1 0.5)
CONNECTEDS=(0.01 0.1 0.5)

COMMON="--arch gat --hidden 128 --layers 4 \
  --loss focal --focal_gamma 3.0 \
  --residual --defect_weight 5.0 \
  --feature_mask 0.1 --augment_flip 0.5 \
  --drop_edge 0.1 --feature_noise 0.01 \
  --epochs 200 --seed 42 \
  --data_dir data/processed_s12_czm_thermal_200_binary"

RESULTS_DIR=runs/grid_search_physics
mkdir -p $RESULTS_DIR

gpu=0
pids=()

for sm in "${SMOOTHS[@]}"; do
  for cn in "${CONNECTEDS[@]}"; do
    tag="sm${sm}_st${STRESS}_cn${cn}"
    log="${RESULTS_DIR}/${tag}.log"

    echo "[GPU${gpu}] smooth=${sm} stress=${STRESS} connected=${cn} → ${log}"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u src/train.py $COMMON \
      --physics_lambda_smooth $sm \
      --physics_lambda_stress $STRESS \
      --physics_lambda_connected $cn \
      > "$log" 2>&1 &

    pids+=($!)
    gpu=$(( (gpu + 1) % 4 ))

    # Wait when all 4 GPUs are busy
    if [ ${#pids[@]} -ge 4 ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

# Wait for remaining
for pid in "${pids[@]}"; do
  wait $pid
done

echo ""
echo "===== Grid Search Results ====="
echo "smooth | stress | connected | Best Val F1"
echo "-------|--------|-----------|------------"

for sm in "${SMOOTHS[@]}"; do
  for cn in "${CONNECTEDS[@]}"; do
    tag="sm${sm}_st${STRESS}_cn${cn}"
    log="${RESULTS_DIR}/${tag}.log"
    f1=$(grep "Best Val F1" "$log" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+')
    printf "%-6s | %-6s | %-9s | %s\n" "$sm" "$STRESS" "$cn" "${f1:-FAILED}"
  done
done

echo ""
echo "Done. Logs in ${RESULTS_DIR}/"
