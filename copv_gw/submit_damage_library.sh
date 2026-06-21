#!/bin/bash
# submit_damage_library.sh — submit COPV damage library jobs to frontale (PBS/Torque)
# Run AFTER gen_damage_library_inp.py has created damage_lib/*.inp
#
# Usage: bash submit_damage_library.sh [--dry-run]
#
# Abaqus on frontale uses QSD50 tokens (50 per analysis job).
# Jobs are submitted with SERIAL token licence (cpus=1) to respect the token budget.
# With 80 jobs × ~1h each → schedule over 2-3 days (max 4 concurrent recommended).

JOBS_DIR="$(dirname "$0")/damage_lib"
N_CONCURRENT=4   # max simultaneous Abaqus jobs (conservative for QSD50)
DRY_RUN=0

if [[ "$1" == "--dry-run" ]]; then DRY_RUN=1; fi

inps=( "$JOBS_DIR"/COPV_GW_180k_d*.inp )
echo "Found ${#inps[@]} INP files in $JOBS_DIR"

submitted=0
for inp in "${inps[@]}"; do
    job=$(basename "$inp" .inp)
    sta="$JOBS_DIR/${job}.sta"
    odb="$JOBS_DIR/${job}.odb"

    # Skip if already completed
    if [[ -f "$odb" ]]; then
        echo "SKIP (odb exists): $job"; continue
    fi

    pbsfile="$JOBS_DIR/${job}.pbs"
    cat > "$pbsfile" <<PBSEOF
#!/bin/bash
#PBS -N $job
#PBS -l nodes=1:ppn=1
#PBS -l walltime=03:00:00
#PBS -q default
#PBS -e $JOBS_DIR/${job}.e
#PBS -o $JOBS_DIR/${job}.o
cd "$JOBS_DIR"
module load abaqus/2024 2>/dev/null || true
abaqus job=$job input=$(basename "$inp") cpus=1 interactive > ${job}.log 2>&1
PBSEOF

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY: qsub $pbsfile"
    else
        # Throttle: wait if N_CONCURRENT jobs already running
        while true; do
            running=$(qstat -u "$USER" 2>/dev/null | grep "COPV_GW_180k_d" | wc -l)
            if [ "$running" -lt "$N_CONCURRENT" ]; then break; fi
            sleep 60
        done
        qid=$(qsub "$pbsfile")
        echo "SUBMITTED [$submitted]: $job  -> $qid"
        (( submitted++ ))
        sleep 5   # small delay between submissions
    fi
done

echo "Done. $submitted jobs submitted."
echo "Monitor with: qstat -u $USER | grep COPV"
echo "After all finish: abaqus python extract_damage_di.py"
