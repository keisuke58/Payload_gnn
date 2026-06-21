#!/bin/bash
cd /home/nishioka/Payload2026/copv_gw
for i in $(seq 1 960); do        # up to ~8h (30s * 960)
  sleep 30
  if [ -f COPV_GW.sta ] && grep -qi 'HAS COMPLETED' COPV_GW.sta 2>/dev/null; then echo "=== COPV COMPLETED $(date '+%F %T') ==="; break; fi
  if grep -qiE 'ERROR|ABORTED|exited with an error' COPV_GW.msg COPV_GW.dat 2>/dev/null; then echo "=== COPV ERROR $(date '+%F %T') ==="; break; fi
  if ! qstat 40509 >/dev/null 2>&1; then echo "=== 40509 left queue $(date '+%F %T') ==="; break; fi
done
echo "---STA---"; tail -14 COPV_GW.sta 2>/dev/null
echo "---MSG---"; tail -20 COPV_GW.msg 2>/dev/null
echo "---ODB---"; ls -la COPV_GW.odb 2>/dev/null
echo "COPV_AUTOPILOT_DONE"
