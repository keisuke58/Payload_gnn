#!/bin/bash
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
cd /home/nishioka/Payload2026

P=abaqus_work/batch_s12_100
ok=0
fail=0

for i in $(seq -w 1 100); do
  JOB="Job-S12-D0${i}"
  # Handle 3-digit numbering
  if [ ${#i} -eq 1 ]; then JOB="Job-S12-D00${i}"; fi
  if [ ${#i} -eq 2 ]; then JOB="Job-S12-D0${i}"; fi
  if [ ${#i} -eq 3 ]; then JOB="Job-S12-D${i}"; fi

  DIR="${P}/${JOB}"
  ODB="${DIR}/${JOB}.odb"
  OUT="${DIR}/results"
  DEF="${DIR}/defect_params.json"

  if [ ! -f "$ODB" ]; then
    echo "SKIP $JOB: no ODB"
    fail=$((fail+1))
    continue
  fi

  if [ -f "${OUT}/nodes.csv" ]; then
    echo "SKIP $JOB: already extracted"
    ok=$((ok+1))
    continue
  fi

  echo "[$ok/$((ok+fail+1))] Extracting $JOB..."
  abaqus python src/extract_odb_results.py --odb "$ODB" --output "$OUT" --defect_json "$DEF" 2>&1 | tail -5

  if [ -f "${OUT}/nodes.csv" ]; then
    ok=$((ok+1))
  else
    echo "FAILED: $JOB"
    fail=$((fail+1))
  fi
done

echo ""
echo "=== ODB Extraction Complete ==="
echo "Success: $ok / 100"
echo "Failed: $fail"
echo "Done at $(date)"
