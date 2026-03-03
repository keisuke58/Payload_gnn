#!/bin/bash
# Overnight tasks on frontale01
# Wait for Abaqus batches → ODB extraction → prepare_ml_data

ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
cd ~/Payload2026

LOG=abaqus_work/overnight_$(date +%Y%m%d).log
exec > >(tee -a $LOG) 2>&1

echo "=== Overnight pipeline started: $(date) ==="

# Wait for all Abaqus jobs to finish (no .lck files)
echo "Waiting for Abaqus jobs to complete..."
while true; do
    lck_ext200=$(find abaqus_work/batch_s12_ext200/ -name "*.lck" 2>/dev/null | wc -l)
    lck_ext100b=$(find abaqus_work/batch_s12_ext100b/ -name "*.lck" 2>/dev/null | wc -l)
    odb_ext200=$(find abaqus_work/batch_s12_ext200/ -name "*.odb" 2>/dev/null | wc -l)
    odb_ext100b=$(find abaqus_work/batch_s12_ext100b/ -name "*.odb" 2>/dev/null | wc -l)
    echo "  $(date +%H:%M) ext200: ${odb_ext200}/200 odb, ${lck_ext200} running | ext100b: ${odb_ext100b}/100 odb, ${lck_ext100b} running"

    if [ "$lck_ext200" -eq 0 ] && [ "$lck_ext100b" -eq 0 ]; then
        echo "All Abaqus jobs completed!"
        break
    fi
    sleep 300  # check every 5 min
done

# Count successful ODBs
echo ""
echo "=== ODB Extraction ==="
echo "ext200 ODBs: $(find abaqus_work/batch_s12_ext200/ -name '*.odb' | wc -l)"
echo "ext100b ODBs: $(find abaqus_work/batch_s12_ext100b/ -name '*.odb' | wc -l)"

# Extract ODB results for ext200
echo ""
echo "Extracting ext200 results..."
for casedir in abaqus_work/batch_s12_ext200/case_*/; do
    casename=$(basename $casedir)
    if [ -f "$casedir/results/nodes.csv" ]; then
        continue  # already extracted
    fi
    odb=$(ls $casedir/*.odb 2>/dev/null | head -1)
    if [ -z "$odb" ]; then
        echo "  SKIP $casename (no ODB)"
        continue
    fi
    echo "  Extracting $casename..."
    cd $casedir
    $ABAQUS python ~/Payload2026/src/extract_odb_results.py $(basename $odb) 2>/dev/null
    cd ~/Payload2026
done

# Extract ODB results for ext100b
echo ""
echo "Extracting ext100b results..."
for casedir in abaqus_work/batch_s12_ext100b/case_*/; do
    casename=$(basename $casedir)
    if [ -f "$casedir/results/nodes.csv" ]; then
        continue
    fi
    odb=$(ls $casedir/*.odb 2>/dev/null | head -1)
    if [ -z "$odb" ]; then
        echo "  SKIP $casename (no ODB)"
        continue
    fi
    echo "  Extracting $casename..."
    cd $casedir
    $ABAQUS python ~/Payload2026/src/extract_odb_results.py $(basename $odb) 2>/dev/null
    cd ~/Payload2026
done

# Count results
ext200_done=$(find abaqus_work/batch_s12_ext200/ -name "nodes.csv" | wc -l)
ext100b_done=$(find abaqus_work/batch_s12_ext100b/ -name "nodes.csv" | wc -l)
thermal_done=$(find abaqus_work/batch_s12_100_thermal/ -name "nodes.csv" | wc -l)
echo ""
echo "=== Extraction Results ==="
echo "  thermal (existing): $thermal_done"
echo "  ext200: $ext200_done"
echo "  ext100b: $ext100b_done"
echo "  Total: $((thermal_done + ext200_done + ext100b_done))"

# Build unified dataset
echo ""
echo "=== Building unified dataset ==="
python3 src/prepare_ml_data.py \
  --input abaqus_work/batch_s12_100_thermal \
  --extra_inputs abaqus_work/batch_s12_ext200 abaqus_work/batch_s12_ext100b \
  --output data/processed_s12_czm_thermal_500_binary \
  2>&1

echo ""
echo "=== Dataset built ==="
ls -la data/processed_s12_czm_thermal_500_binary/ 2>/dev/null

echo ""
echo "=== Overnight pipeline completed: $(date) ==="
