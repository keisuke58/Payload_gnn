#!/usr/bin/env bash
# auto_eval_ood.sh — Wait for OOD training to complete, then run eval_ood.py
#
# Usage:
#   bash scripts/auto_eval_ood.sh [gpu_id]
#   bash scripts/auto_eval_ood.sh 2  # default GPU 2

PY=/home/nishioka/miniconda3/envs/gnn_final_env/bin/python
ROOT=/home/nishioka/Payload2026
GPU=${1:-2}
POLL=120   # poll every 2 minutes

cd "$ROOT"

declare -A JOBS=(
  ["ood_lgsta_p75"]="lgsta data/ood_p75"
  ["ood_lgsta_p50"]="lgsta data/ood_p50"
  ["ood_lgsta_p90"]="lgsta data/ood_p90"
  ["ood_sage_p75"]="sage data/ood_p75"
)

eval_done=()

echo "[$(date +%H:%M)] auto_eval_ood.sh started. Polling every ${POLL}s ..."

while true; do
  all_done=true

  for run_tag in "${!JOBS[@]}"; do
    # Already evaluated
    [[ " ${eval_done[*]} " == *" $run_tag "* ]] && continue

    run_dir=$(ls -td runs/${run_tag}/*/ 2>/dev/null | head -1)
    [[ -z "$run_dir" ]] && { all_done=false; continue; }

    ckpt="${run_dir}best_model.pt"
    log="${run_dir}training_log.csv"

    # Check training is complete (200 epochs in csv)
    if [[ -f "$log" ]]; then
      n_ep=$(tail -1 "$log" | cut -d, -f1)
      if [[ "$n_ep" == "200" ]] || [[ -f "${run_dir}done.flag" ]]; then
        read arch ood_dir <<< "${JOBS[$run_tag]}"
        echo "[$(date +%H:%M)] $run_tag done at ep$n_ep → running eval_ood.py"
        CUDA_VISIBLE_DEVICES=$GPU \
        $PY scripts/eval_ood.py \
          --run_dir "runs/${run_tag}" \
          --ood_dir "$ood_dir" \
          --arch "$arch" \
          2>&1 | tee "logs/eval_${run_tag}.log"
        eval_done+=("$run_tag")
      else
        all_done=false
        echo "[$(date +%H:%M)] $run_tag ep=${n_ep}/200 — waiting"
      fi
    else
      all_done=false
    fi
  done

  if $all_done; then
    echo "[$(date +%H:%M)] All OOD evaluations complete."
    break
  fi

  sleep $POLL
done

echo ""
echo "=== OOD Summary ==="
for run_tag in "${eval_done[@]}"; do
  json=$(ls runs/${run_tag}/*/ood_results.json 2>/dev/null | head -1)
  [[ -z "$json" ]] && json=$(ls runs/${run_tag}/ood_results.json 2>/dev/null)
  if [[ -f "$json" ]]; then
    id_f1=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['id_val']['opt_f1']:.4f}\")")
    ood_f1=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['ood_test']['opt_f1']:.4f}\")")
    delta=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['delta']['opt_f1']:+.4f}\")")
    echo "  $run_tag: ID_val=$id_f1  OOD_test=$ood_f1  Δ=$delta"
  fi
done
