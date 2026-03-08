#!/bin/bash
# collect_gw_csv.sh — 完了済みODBからCSVを抽出し gw_fairing_dataset/ に収集
#
# 機能:
#   1. abaqus_work/ 内の Job-GW-*.odb を走査
#   2. 完了済み (.sta に COMPLETED) かつ CSV 未抽出のものを抽出
#   3. CSVを gw_fairing_dataset/ にコピー
#   4. リモートマシン (frontale01/02/04) の ODB も走査可能
#
# Usage:
#   bash scripts/collect_gw_csv.sh              # ローカルのみ
#   bash scripts/collect_gw_csv.sh --remote     # frontale01/02/04 も走査
#   bash scripts/collect_gw_csv.sh --status     # 状態確認のみ
#   bash scripts/collect_gw_csv.sh --manifest   # manifest JSON も生成

set -euo pipefail
# Note: (( )) arithmetic returning 0 triggers set -e, so use || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PROJECT_ROOT/abaqus_work"
CSV_DIR="$WORK_DIR/gw_fairing_dataset"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_gw_history.py"
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
MACHINES=(frontale01 frontale02 frontale04)

ENV_CMD="env -i HOME=/home/nishioka USER=nishioka \
  PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:/usr/local/bin:/usr/bin:/bin \
  LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
  LD_LIBRARY_PATH=/home/nishioka/SIMULIA/EstProducts/2024/linux_a64/code/bin \
  LD_PRELOAD=/home/nishioka/libfake_x11.so"

DO_REMOTE=false
DO_STATUS=false
DO_MANIFEST=false

for arg in "$@"; do
    case $arg in
        --remote)  DO_REMOTE=true ;;
        --status)  DO_STATUS=true ;;
        --manifest) DO_MANIFEST=true ;;
    esac
done

mkdir -p "$CSV_DIR"

# ============================================================
# ローカル ODB 走査
# ============================================================
collect_local() {
    echo "=== Scanning local ODBs in $WORK_DIR ==="
    local extracted=0
    local skipped=0
    local failed=0

    for odb in "$WORK_DIR"/Job-GW-*.odb; do
        [ -f "$odb" ] || continue
        local base=$(basename "$odb" .odb)
        local csv_name="${base}_sensors.csv"

        # CSV already exists?
        if [ -f "$CSV_DIR/$csv_name" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        # Check completion
        local sta="$WORK_DIR/$base.sta"
        if [ ! -f "$sta" ] || ! grep -q "COMPLETED SUCCESSFULLY" "$sta" 2>/dev/null; then
            if [ "$DO_STATUS" = true ]; then
                echo "  INCOMPLETE: $base"
            fi
            failed=$((failed + 1))
            continue
        fi

        if [ "$DO_STATUS" = true ]; then
            echo "  READY: $base (CSV missing)"
            continue
        fi

        # Extract
        echo "  Extracting: $base"
        cd "$WORK_DIR"
        $ENV_CMD $ABAQUS python "$EXTRACT_SCRIPT" "$base.odb" 2>&1 | tail -3
        cd "$PROJECT_ROOT"

        if [ -f "$WORK_DIR/$csv_name" ]; then
            cp "$WORK_DIR/$csv_name" "$CSV_DIR/"
            echo "  OK: $csv_name"
            extracted=$((extracted + 1))
        else
            echo "  FAIL: extraction produced no CSV"
            failed=$((failed + 1))
        fi
    done

    echo "Local: extracted=$extracted, skipped=$skipped (already done), incomplete=$failed"
}

# ============================================================
# リモートマシン走査
# ============================================================
collect_remote() {
    echo ""
    echo "=== Scanning remote machines ==="
    for host in "${MACHINES[@]}"; do
        echo "--- $host ---"
        # List completed ODBs without local CSV
        local remote_odbs=$(LD_LIBRARY_PATH="" /usr/bin/ssh "$host" \
            "ls $WORK_DIR/Job-GW-*.odb 2>/dev/null" 2>/dev/null || true)

        if [ -z "$remote_odbs" ]; then
            echo "  No GW ODBs found"
            continue
        fi

        while IFS= read -r odb_path; do
            local base=$(basename "$odb_path" .odb)
            local csv_name="${base}_sensors.csv"

            # Already have CSV?
            if [ -f "$CSV_DIR/$csv_name" ]; then
                continue
            fi

            # Check completion remotely
            local completed=$(LD_LIBRARY_PATH="" /usr/bin/ssh "$host" \
                "grep -q 'COMPLETED SUCCESSFULLY' '$WORK_DIR/$base.sta' 2>/dev/null && echo yes || echo no" 2>/dev/null)

            if [ "$completed" != "yes" ]; then
                if [ "$DO_STATUS" = true ]; then
                    echo "  INCOMPLETE: $base (on $host)"
                fi
                continue
            fi

            if [ "$DO_STATUS" = true ]; then
                echo "  READY: $base (on $host, CSV missing)"
                continue
            fi

            # Extract remotely
            echo "  Extracting on $host: $base"
            LD_LIBRARY_PATH="" /usr/bin/ssh "$host" \
                "cd $WORK_DIR && $ENV_CMD $ABAQUS python $EXTRACT_SCRIPT $base.odb" 2>&1 | tail -3

            # Copy CSV back
            local remote_csv="$WORK_DIR/$csv_name"
            local exists=$(LD_LIBRARY_PATH="" /usr/bin/ssh "$host" \
                "test -f '$remote_csv' && echo yes || echo no" 2>/dev/null)
            if [ "$exists" = "yes" ]; then
                LD_LIBRARY_PATH="" /usr/bin/scp "$host:$remote_csv" "$CSV_DIR/" 2>/dev/null
                echo "  OK: $csv_name (from $host)"
            else
                echo "  FAIL: no CSV produced on $host"
            fi
        done <<< "$remote_odbs"
    done
}

# ============================================================
# Manifest JSON 生成
# ============================================================
generate_manifest() {
    local manifest="$PROJECT_ROOT/manifest_gw_all.json"
    echo ""
    echo "=== Generating manifest: $manifest ==="

    python3 - "$CSV_DIR" "$manifest" << 'PYEOF'
import json, os, re, sys

csv_dir = sys.argv[1]
out_path = sys.argv[2]

healthy_re = re.compile(r'(Healthy|Valid-H|Valid-Healthy|Test-H)', re.IGNORECASE)
samples = []

for f in sorted(os.listdir(csv_dir)):
    if not f.endswith('_sensors.csv'):
        continue
    csv_path = os.path.join(csv_dir, f)
    label = 0 if healthy_re.search(f) else 1
    entry = {'csv': csv_path, 'label': label}
    # Auto-assign validation jobs to test split
    if 'Valid-' in f or 'Test-' in f:
        entry['split'] = 'test'
    samples.append(entry)

manifest = {'samples': samples}
with open(out_path, 'w') as fh:
    json.dump(manifest, fh, indent=2)

n_h = sum(1 for s in samples if s['label'] == 0)
n_d = sum(1 for s in samples if s['label'] == 1)
n_t = sum(1 for s in samples if s.get('split') == 'test')
print("  %d samples (healthy=%d, defect=%d, forced_test=%d)" % (len(samples), n_h, n_d, n_t))
print("  Written: %s" % out_path)
PYEOF
}

# ============================================================
# Summary
# ============================================================
print_summary() {
    echo ""
    echo "=== CSV Collection Summary ==="
    local n_csv=$(ls "$CSV_DIR"/*_sensors.csv 2>/dev/null | wc -l)
    local n_healthy=$(ls "$CSV_DIR"/*_sensors.csv 2>/dev/null | grep -iE "(Healthy|Valid-H|Test-H)" | wc -l)
    local n_defect=$((n_csv - n_healthy))
    echo "  Total CSVs: $n_csv (healthy=$n_healthy, defect=$n_defect)"
    echo "  Location: $CSV_DIR"
}

# ============================================================
# Main
# ============================================================
collect_local

if [ "$DO_REMOTE" = true ]; then
    collect_remote
fi

print_summary

if [ "$DO_MANIFEST" = true ]; then
    generate_manifest
fi
