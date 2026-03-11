#!/bin/bash
# qsub_junction_batch.sh — Junction DOE からジョブを一括投入 / 状態確認
#
# Usage:
#   bash scripts/qsub_junction_batch.sh status
#   bash scripts/qsub_junction_batch.sh submit [--start N] [--end M] [--max_concurrent K]
#   bash scripts/qsub_junction_batch.sh submit_healthy

DOE_FILE="${DOE_FILE:-doe_gw_junction.json}"
SCRIPT="scripts/qsub_junction_single.sh"
WORK_DIR=~/Payload2026/abaqus_work
CSV_DIR=~/Payload2026/abaqus_work/junction_dataset

if [ ! -f "$DOE_FILE" ]; then
    echo "ERROR: DOE file not found: $DOE_FILE"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq not installed"
    exit 1
fi

ACTION="${1:-status}"
shift

START_ID=0
END_ID=9999
MAX_CONCURRENT=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_ID=$2; shift 2 ;;
        --end)   END_ID=$2; shift 2 ;;
        --max_concurrent) MAX_CONCURRENT=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

count_running() {
    qstat -u "$USER" 2>/dev/null | grep "Job-Junc" | grep -c " [RQ] " || echo 0
}

case "$ACTION" in
    status)
        N_TOTAL=$(jq '.samples | length' "$DOE_FILE")
        N_HEALTHY=$(jq '.healthy_samples | length' "$DOE_FILE")
        echo "=== Junction Dataset Status ==="
        echo "DOE: $DOE_FILE ($N_TOTAL defective + $N_HEALTHY healthy)"
        echo "Model: barrel-nosecone junction (z=3500-6500, 30deg)"
        echo ""

        completed=0; failed=0; pending=0; running=0

        for i in $(seq 0 $((N_TOTAL - 1))); do
            JOB_NAME=$(jq -r ".samples[$i].job_name" "$DOE_FILE")
            DEFECT_TYPE=$(jq -r ".samples[$i].defect_params.defect_type" "$DOE_FILE")
            RADIUS=$(jq -r ".samples[$i].defect_params.radius" "$DOE_FILE")
            NEAR_JUNC=$(jq -r ".samples[$i].near_junction" "$DOE_FILE")
            STA="$WORK_DIR/$JOB_NAME/$JOB_NAME.sta"
            CSV="$CSV_DIR/${JOB_NAME}_sensors.csv"

            if [ -f "$CSV" ]; then
                status="CSV_OK"
                ((completed++))
            elif grep -q "COMPLETED SUCCESSFULLY" "$STA" 2>/dev/null; then
                status="FEM_DONE(no_csv)"
                ((completed++))
            elif [ -f "$STA" ]; then
                status="RUNNING/FAILED"
                ((running++))
            elif qstat -u "$USER" 2>/dev/null | grep -q "$JOB_NAME"; then
                status="QUEUED"
                ((running++))
            else
                status="PENDING"
                ((pending++))
            fi
            junc_flag=""
            [ "$NEAR_JUNC" = "true" ] && junc_flag="[JUNC]"
            printf "  %-18s %-18s R=%-5s %-6s %s\n" \
                "$JOB_NAME" "$DEFECT_TYPE" "$RADIUS" "$junc_flag" "$status"
        done

        echo ""
        for i in $(seq 0 $((N_HEALTHY - 1))); do
            JOB_NAME=$(jq -r ".healthy_samples[$i].job_name" "$DOE_FILE")
            STA="$WORK_DIR/$JOB_NAME/$JOB_NAME.sta"
            CSV="$CSV_DIR/${JOB_NAME}_sensors.csv"

            if [ -f "$CSV" ]; then
                status="CSV_OK"
                ((completed++))
            elif grep -q "COMPLETED SUCCESSFULLY" "$STA" 2>/dev/null; then
                status="FEM_DONE(no_csv)"
                ((completed++))
            elif [ -f "$STA" ]; then
                status="RUNNING/FAILED"
                ((running++))
            else
                status="PENDING"
                ((pending++))
            fi
            printf "  %-18s %-18s        %s\n" "$JOB_NAME" "healthy" "$status"
        done

        echo ""
        echo "Summary: completed=$completed, running/queued=$running, pending=$pending"
        echo "CSV dir: $CSV_DIR ($(ls "$CSV_DIR"/*.csv 2>/dev/null | wc -l) files)"
        ;;

    submit)
        echo "Submitting junction defective jobs (ID $START_ID to $END_ID, max=$MAX_CONCURRENT)..."
        N_TOTAL=$(jq '.samples | length' "$DOE_FILE")
        submitted=0

        for i in $(seq 0 $((N_TOTAL - 1))); do
            ID=$(jq -r ".samples[$i].id" "$DOE_FILE")
            [ "$ID" -lt "$START_ID" ] 2>/dev/null && continue
            [ "$ID" -gt "$END_ID" ] 2>/dev/null && continue

            JOB_NAME=$(jq -r ".samples[$i].job_name" "$DOE_FILE")

            if [ -f "$CSV_DIR/${JOB_NAME}_sensors.csv" ]; then
                echo "  SKIP $JOB_NAME (CSV exists)"
                continue
            fi
            if grep -q "COMPLETED SUCCESSFULLY" "$WORK_DIR/$JOB_NAME/$JOB_NAME.sta" 2>/dev/null; then
                echo "  SKIP $JOB_NAME (already completed)"
                continue
            fi

            n_running=$(count_running)
            while [ "$n_running" -ge "$MAX_CONCURRENT" ]; do
                echo "  Waiting... ($n_running jobs running, limit=$MAX_CONCURRENT)"
                sleep 60
                n_running=$(count_running)
            done

            DEFECT_JSON=$(jq -c ".samples[$i].defect_params" "$DOE_FILE")
            echo "  Submitting $JOB_NAME ($DEFECT_JSON)..."
            qsub -N "$JOB_NAME" \
                 -v "JOB_NAME=$JOB_NAME,DEFECT=$DEFECT_JSON" \
                 "$SCRIPT"
            ((submitted++))
            sleep 2
        done
        echo "Submitted $submitted jobs."
        ;;

    submit_healthy)
        echo "Submitting junction healthy jobs..."
        N_HEALTHY=$(jq '.healthy_samples | length' "$DOE_FILE")
        submitted=0

        for i in $(seq 0 $((N_HEALTHY - 1))); do
            JOB_NAME=$(jq -r ".healthy_samples[$i].job_name" "$DOE_FILE")

            if [ -f "$CSV_DIR/${JOB_NAME}_sensors.csv" ]; then
                echo "  SKIP $JOB_NAME (CSV exists)"
                continue
            fi

            echo "  Submitting $JOB_NAME (healthy)..."
            qsub -N "$JOB_NAME" \
                 -v "JOB_NAME=$JOB_NAME" \
                 "$SCRIPT"
            ((submitted++))
            sleep 2
        done
        echo "Submitted $submitted healthy jobs."
        ;;

    *)
        echo "Usage: $0 {submit|status|submit_healthy} [--start N] [--end M] [--max_concurrent K]"
        exit 1
        ;;
esac
