#!/bin/bash
# Run 10mm and 12mm mesh for convergence check.
# Same defect (theta=30, z=2500, r=50). Compare with 25mm and 50mm.

set -e
cd "$(dirname "$0")/.."

echo "=== Convergence Check: 10mm and 12mm mesh ==="
echo "  Defect: theta=30 z=2500 r=50"
echo ""

# 12mm first (faster, ~200k nodes)
echo "--- 12mm mesh ---"
python src/run_batch.py \
  --doe doe_ideal_12mm.json \
  --output_dir dataset_output_ideal_12mm \
  --global_seed 12 \
  --defect_seed 5 \
  --force

# 10mm (finest, ~300k nodes, longest)
echo ""
echo "--- 10mm mesh ---"
python src/run_batch.py \
  --doe doe_ideal_10mm.json \
  --output_dir dataset_output_ideal_10mm \
  --global_seed 10 \
  --defect_seed 5 \
  --force

echo ""
echo "Done. Compare:"
echo "  50mm:  dataset_output/sample_0001 (or similar)"
echo "  25mm:  dataset_output_ideal/sample_0000"
echo "  12mm:  dataset_output_ideal_12mm/sample_0000"
echo "  10mm:  dataset_output_ideal_10mm/sample_0000"
