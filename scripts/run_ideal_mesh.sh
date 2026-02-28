#!/bin/bash
# Generate one fine mesh sample for mesh quality reference.
# GLOBAL_SEED=25mm, DEFECT_SEED=8mm → ~43k nodes, Small defects resolvable.
# For ultra-fine (12mm/~200k nodes): --global_seed 12 --defect_seed 5 (20–60 min).

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "=== Ideal Fine Mesh Sample ==="
echo "  GLOBAL_SEED=25 mm, DEFECT_SEED=8 mm (~43k nodes)"
echo "  Output: dataset_output_ideal/"
echo ""

OUT_DIR="$PROJECT_ROOT/dataset_output_ideal"
mkdir -p "$OUT_DIR"

python src/run_batch.py \
  --doe doe_ideal.json \
  --output_dir dataset_output_ideal \
  --global_seed 25 \
  --defect_seed 8 \
  --force

echo ""
echo "Done. Results in $OUT_DIR/sample_0000/"
echo "Visualize: python scripts/visualize_mesh_structure.py --data $OUT_DIR/sample_0000"
