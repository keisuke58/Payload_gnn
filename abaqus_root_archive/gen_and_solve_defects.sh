#!/bin/bash
# Generate and solve all 7 defect scenarios for fairing Track B (adhesive 2.0mm, ENCASTRE BCs)
# Run ONLY after H3_healthy_a3 sta shows COMPLETED successfully
# Usage: bash gen_and_solve_defects.sh

set -e
SCRIPT_DIR=/home/nishioka/Payload2026/src
PARAM_DIR=/home/nishioka/Payload2026
WORK_DIR=/home/nishioka/Payload2026/abaqus_root_archive
PBS_LOG_DIR=/home/nishioka/render_scratch
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
export TMPDIR=/home/nishioka/render_scratch

DEFECTS=(debonding fod impact delam acoustic inner thermal)
PARAM_FILES=(params_debonding.json params_fod.json params_impact.json params_delam.json params_acoustic.json params_inner.json params_thermal.json)

mkdir -p $WORK_DIR

for i in "${!DEFECTS[@]}"; do
  D=${DEFECTS[$i]}
  P=${PARAM_FILES[$i]}
  JOB="H3_${D}_a3"

  cat > $WORK_DIR/solve_${JOB}.pbs << EOF
#!/bin/bash
#PBS -N ${JOB}
#PBS -q default
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o ${PBS_LOG_DIR}/${JOB}.pbslog
# Fairing Track B defect scenario: ${D}, adhesive 2.0mm, ENCASTRE BCs (fixed)
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:\$PATH
export TMPDIR=/home/nishioka/render_scratch

# Step 1: Generate INP via abaqus cae
cd ${SCRIPT_DIR}
echo "[gen] [\$(date '+%F %T')] generating ${JOB}"
abq2024 cae noGUI=generate_cohesive_fairing.py -- \
  --job ${JOB} \
  --param_file ${PARAM_DIR}/${P} \
  --adhesive_thickness 2.0 \
  --no_run 2>&1 | tail -20
mv ${JOB}.inp ${WORK_DIR}/${JOB}.inp 2>/dev/null || true

# Step 2: Solve
cd ${WORK_DIR}
rm -f ${JOB}.odb ${JOB}.lck ${JOB}.sta ${JOB}.msg 2>/dev/null
echo "[solve] [\$(date '+%F %T')] solving ${JOB} on \$(hostname)"
abq2024 job=${JOB} input=${JOB}.inp cpus=1 scratch=\$TMPDIR interactive
echo "===STA==="; tail -8 ${JOB}.sta 2>/dev/null
echo "${JOB}_DONE \$(date '+%F %T')"
EOF

  JID=$(qsub $WORK_DIR/solve_${JOB}.pbs)
  echo "Submitted ${JOB}: $JID"
done

echo "All 7 defect jobs submitted"
