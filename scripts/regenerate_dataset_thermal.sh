#!/bin/bash
# Regenerate dataset with thermal patch applied.
# Patches INPs and re-runs analysis + extraction for all samples.
#
# Usage: bash scripts/regenerate_dataset_thermal.sh [--doe doe_100.json] [--output dataset_output] [--start 0] [--end 100]

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DOE="${DOE:-doe_100.json}"
OUTPUT="${OUTPUT:-dataset_output}"
START="${START:-0}"
END="${END:-100}"

for arg in "$@"; do
    case "$arg" in
        --doe=*) DOE="${arg#*=}" ;;
        --output=*) OUTPUT="${arg#*=}" ;;
        --start=*) START="${arg#*=}" ;;
        --end=*) END="${arg#*=}" ;;
    esac
done

WORK_DIR="$PROJECT_ROOT/abaqus_work"
PATCH_SCRIPT="$PROJECT_ROOT/scripts/patch_inp_thermal.py"
EXTRACT_SCRIPT="$PROJECT_ROOT/src/extract_odb_results.py"

echo "Regenerating dataset with thermal patch"
echo "  DOE: $DOE, Output: $OUTPUT, Samples: $START..$END"
echo ""

# Run batch with --force to regenerate (uses patched INPs via project_root)
python src/run_batch.py --doe "$DOE" --output_dir "$OUTPUT" --force --start "$START" --end "$END"

echo ""
echo "Done. Verify with: abaqus python scripts/verify_odb_thermal.py --odb abaqus_work/H3_Debond_0000.odb"
