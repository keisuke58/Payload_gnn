#!/bin/bash
# Build the 5-frequency COPV guided-wave batch (real BAM excitation freqs).
# For each freq: (1) regenerate mesh+inp via CAE (mesh size scales with A0 wavelength),
# (2) inject RX-node history output at the real 2.976 MHz sampling rate + 30 field frames,
# (3) write a PBS job chained afterany on the previous freq -> strictly serial (QEX token).
# Usage: bash make_copv_batch.sh [DEPJOB]   (DEPJOB = job id the chain waits on, e.g. 40509)
set -e
cd /home/nishioka/Payload2026/copv_gw
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:$PATH
export TMPDIR=/home/nishioka/render_scratch
DEP="${1:-}"                       # first freq waits on this job (the smoke-test solve)
FREQS="60 120 180 260 300"         # kHz, real BAM burst set

inject () {  # $1 = base inp, $2 = output _h inp
  sed -e 's/\*Output, field, variable=PRESELECT/*Output, field, number interval=30, variable=PRESELECT/' \
      -e 's@\*Output, history, variable=PRESELECT@*Output, history, time interval=3.3602e-07\n*Node Output, nset=RX\nU1, U2, U3@' \
      "$1" > "$2"
}

for f in $FREQS; do
  j="COPV_GW_${f}k"
  echo "=== [$(date '+%T')] generating $j (freq ${f} kHz) ==="
  abq2024 cae noGUI=generate_copv_gw.py -- --freq ${f}e3 --job "$j" --no_run > gen_${f}k.log 2>&1
  if [ ! -f "${j}.inp" ]; then echo "!! ${j}.inp not written, see gen_${f}k.log"; tail -5 gen_${f}k.log; exit 1; fi
  inject "${j}.inp" "${j}_h.inp"
  ne=$(grep -c '^[0-9]' "${j}.inp" 2>/dev/null || echo '?')
  echo "   wrote ${j}_h.inp  (RX history injected)"
done
echo "=== all 5 inps generated + injected ==="

# write chained PBS (each depends afterany on previous; first on $DEP if given)
prev=""
for f in $FREQS; do
  j="COPV_GW_${f}k"
  pbs="solve_${f}k.pbs"
  cat > "$pbs" <<EOF
#!/bin/bash
#PBS -N COPV${f}k
#PBS -q default
#PBS -l nodes=1:ppn=2
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/nishioka/render_scratch/${j}.pbslog
cd /home/nishioka/Payload2026/copv_gw
export PATH=/home/nishioka/DassaultSystemes/SIMULIA/Commands:\$PATH
export TMPDIR=/home/nishioka/render_scratch
j=${j}
rm -f \$j.odb \$j.lck \$j.sta \$j.msg 2>/dev/null
echo "[\$(date '+%F %T')] solving \$j on \$(hostname)"
abq2024 job=\$j input=\${j}_h.inp cpus=1 double=both scratch=\$TMPDIR
for i in \$(seq 1 2160); do
  sleep 10
  if [ ! -f \$j.lck ] && [ -f \$j.sta ]; then echo "lck gone -> finished"; break; fi
  if grep -qi 'HAS COMPLETED' \$j.sta 2>/dev/null; then echo "sta COMPLETED"; break; fi
  if grep -qiE 'ERROR|ABORTED|exited with an error' \$j.msg \$j.dat 2>/dev/null; then echo "error"; break; fi
done
echo "===STA==="; tail -8 \$j.sta 2>/dev/null
echo "${j}_DONE \$(date '+%F %T')"
EOF
  echo "wrote $pbs"
done

echo
echo "=== BATCH READY. To submit the serial chain (waits on ${DEP:-nothing}): ==="
echo "  bash submit_copv_batch.sh ${DEP}"
