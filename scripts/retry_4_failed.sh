#!/bin/bash
set -e
ABAQUS=/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus
WORKDIR=/home/nishioka/Payload2026/abaqus_work/batch_s12_100_thermal

FAILED_JOBS="Job-S12-D322 Job-S12-D323 Job-S12-D324 Job-S12-D326"

echo "=== Retry 4 failed jobs: $(date) ==="

for JOB in $FAILED_JOBS; do
    JOBDIR="$WORKDIR/$JOB"
    echo ""
    echo "--- $JOB ---"

    # Clean old artifacts
    rm -f "$JOBDIR"/*.odb "$JOBDIR"/*.sta "$JOBDIR"/*.msg "$JOBDIR"/*.lck "$JOBDIR"/*.com "$JOBDIR"/*.env "$JOBDIR"/*.dat "$JOBDIR"/*.prt "$JOBDIR"/*.res "$JOBDIR"/*.sim "$JOBDIR"/*.stt "$JOBDIR"/*.mdl "$JOBDIR"/*.023

    # Run solver
    cd "$JOBDIR"
    echo "  Solver start: $(date)"
    $ABAQUS job=$JOB cpus=4 memory="16 gb" interactive 2>&1 || true
    cd /home/nishioka/Payload2026

    # Check success
    if grep -q "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" "$JOBDIR/$JOB.sta" 2>/dev/null; then
        echo "  Solver OK: $(date)"

        # Extract ODB
        echo "  Extracting ODB..."
        cd "$JOBDIR"
        $ABAQUS python /home/nishioka/Payload2026/src/extract_odb_results.py $JOB.odb 2>&1 || echo "  Extract FAILED"
        cd /home/nishioka/Payload2026

        if [ -f "$JOBDIR/results/nodes.csv" ]; then
            echo "  Extract OK"
        else
            echo "  Extract FAILED (no nodes.csv)"
        fi
    else
        echo "  Solver FAILED"
    fi
done

echo ""
echo "=== Retry complete: $(date) ==="

# Verify
for JOB in $FAILED_JOBS; do
    if [ -f "$WORKDIR/$JOB/results/nodes.csv" ]; then
        echo "  $JOB: OK"
    else
        echo "  $JOB: MISSING"
    fi
done
