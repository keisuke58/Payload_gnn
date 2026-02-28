#!/bin/bash
# Overnight batch: chain jobs after current batches finish
# Plan:
#   1. Wait for 25mm×100 and 12mm×20 to finish
#   2. Run 12mm×80 (samples 20-99) → total 12mm×100
#
# Expected: ~4h for 12mm×80, well within 8h budget

set -e
cd /home/nishioka/Payload2026

LOG="dataset_output_overnight.log"
echo "=== Overnight Batch Started: $(date) ===" | tee $LOG

# --- Wait for current batches ---
echo "[$(date +%H:%M)] Waiting for 25mm×100 and 12mm×20 to finish..." | tee -a $LOG

while pgrep -f "run_batch.*25mm_100" > /dev/null 2>&1 || pgrep -f "run_batch.*12mm_20" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date +%H:%M)] Current batches finished." | tee -a $LOG

# --- Report current results ---
N25=$(ls dataset_output_25mm_100/sample_*/nodes.csv 2>/dev/null | wc -l)
N12=$(ls dataset_output_12mm_20/sample_*/nodes.csv 2>/dev/null | wc -l)
echo "[$(date +%H:%M)] Results: 25mm=$N25/100, 12mm=$N12/20" | tee -a $LOG

# --- Clean stale files ---
rm -f abaqus_work/H3_Debond_*.lck 2>/dev/null

# --- Job 2: 12mm × 80 (samples 20-99) ---
echo "[$(date +%H:%M)] Starting 12mm×80 (samples 20-99)..." | tee -a $LOG
python3 src/run_batch.py \
    --doe doe_12mm_80.json \
    --output_dir dataset_output_12mm_20 \
    --global_seed 12 \
    --force 2>&1 | tee -a $LOG

N12_final=$(ls dataset_output_12mm_20/sample_*/nodes.csv 2>/dev/null | wc -l)
echo "[$(date +%H:%M)] 12mm complete: $N12_final/100 samples" | tee -a $LOG

echo "=== Overnight Batch Done: $(date) ===" | tee -a $LOG
