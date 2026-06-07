#!/bin/bash
# Wait until the OGW long-term download finishes (all 56 monthly pickles present, no *.part,
# no curl running), then run experiment A (supervised multi-year) + B' (one-class novelty)
# + the figure. Logs to /tmp/B_full.out. Launch in background.
set -u
cd /home/nishioka/Payload2026 || exit 1
DIR=data/external/ogw_longterm
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LD_LIBRARY_PATH=$HOME/miniconda3/lib
export B_EPOCHS=30 B_TRAIN_YEARS=2018,2019 B_TEST_YEARS=2020,2021,2022
LOG=/tmp/B_full.out; : > "$LOG"
echo "[wait] $(date '+%H:%M') waiting for download to finish..." | tee -a "$LOG"
done_cnt=0; prev=-1; samecnt=0
for i in $(seq 1 240); do          # up to ~8h
  n=$(ls -1 "$DIR"/measurements_*.pickle 2>/dev/null | wc -l)
  parts=$(ls -1 "$DIR"/*.part 2>/dev/null | wc -l)
  curls=$(pgrep -fc "curl .*measurements_" 2>/dev/null || echo 0)
  # complete: all 56 present, settled
  if [ "$parts" -eq 0 ] && [ "$curls" -eq 0 ] && [ "$n" -ge 56 ]; then done_cnt=$((done_cnt+1)); else done_cnt=0; fi
  # stalled: count not growing and nothing downloading (a file may have failed) -> proceed anyway
  if [ "$n" -eq "$prev" ] && [ "$parts" -eq 0 ] && [ "$curls" -eq 0 ]; then samecnt=$((samecnt+1)); else samecnt=0; fi
  prev=$n
  echo "[wait $i] $(date '+%H:%M') months=$n parts=$parts curls=$curls done=$done_cnt same=$samecnt" >> "$LOG"
  if [ "$done_cnt" -ge 2 ]; then echo "[wait] download COMPLETE (n=$n)" | tee -a "$LOG"; break; fi
  if [ "$samecnt" -ge 5 ]; then echo "[wait] download STALLED at n=$n (proceeding anyway)" | tee -a "$LOG"; break; fi
  sleep 120
done

echo "==================== EXPERIMENT A (supervised multi-year) ====================" | tee -a "$LOG"
stdbuf -oL python3 scripts/longterm_detection_over_time.py >> "$LOG" 2>&1
echo "==================== EXPERIMENT B' (one-class novelty) =======================" | tee -a "$LOG"
stdbuf -oL python3 scripts/longterm_novelty_oneclass.py >> "$LOG" 2>&1
echo "==================== FIGURE ===================================================" | tee -a "$LOG"
python3 scripts/gen_detection_over_time_fig.py >> "$LOG" 2>&1
echo "[done] $(date '+%H:%M') A+B'+figure complete" | tee -a "$LOG"
