#!/bin/bash
# Push the current EOP table into every RUNNING GNSS node kotekan, and read it back.
#
# WHY THIS EXISTS. The EOP table lives in the generated node configs and ROLLS (a 6-entry,
# ~5-day window): a table that was fresh at generation is past its edge within a week, and
# 18 days past it killed recv1 outright on 2026-08-19 (dUT1 5 ms wrong -> every N2 frame's
# time metadata rejected). The table is REST-updatable in the running process
# (kotekan_update_endpoint) -- which also means A NODE RESTART SILENTLY REVERTS IT to the
# config file's stale copy. So: run this AFTER every node bring-up, and any time the window
# is near its edge. The 2026-08-24 restart reverted Jim's Friday curls exactly this way.
#
# SOURCE. The observatory keeps the rolling file at choco:/etc/choco/configs/eop-state.json
# (refreshed daily at ~12:00). cf06 CANNOT reach choco, so the file must be staged by a host
# that reaches both:  scp choco:/etc/choco/configs/eop-state.json cf06:/tmp/eop-state.json
# This script then runs ON cf06 (or any host that reaches the cx nodes).
#
#   usage:  eop_push.sh [file]          default /tmp/eop-state.json
#   env:    GNSS_NODES                  default "cx19 cx27 cx42 cx43 cx44 cx51"
set -u
F=${1:-/tmp/eop-state.json}
NODES=${GNSS_NODES:-"cx19 cx27 cx42 cx43 cx44 cx51"}

python3 - "$F" <<'PY' || exit 1
import json, sys, datetime, time
d = json.load(open(sys.argv[1]))
t = d["earth_orientation_parameter_table"]
f = lambda ns: datetime.datetime.fromtimestamp(ns/1e9, datetime.timezone.utc)
lo, hi = f(t[0]["t_inst_ns"]), f(t[-1]["t_inst_ns"])
now = datetime.datetime.now(datetime.timezone.utc)
print("table: %d entries, %s .. %s" % (len(t), lo.date(), hi.date()))
# A push of a STALE table is worse than no push: it looks like maintenance happened.
if now > hi:
    print("REFUSING: the table's last entry is already in the past (%s < today %s)."
          % (hi.date(), now.date()))
    print("Stage a fresh one:  scp choco:/etc/choco/configs/eop-state.json %s" % sys.argv[1])
    raise SystemExit(1)
if (hi - now).days < 1:
    print("WARNING: <1 day of table left past now -- stage a fresh file soon.")
PY

fail=0
for n in $NODES; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 -X POST \
        -H 'Content-Type: application/json' --data-binary @"$F" \
        "http://$n:12048/earth_rotation_data" 2>/dev/null || echo 000)
    # READ IT BACK. A 200 proves the POST parsed, not that the table the process now
    # holds is the one we sent -- and the whole point of this script is the live copy.
    rb=$(curl -s --max-time 8 "http://$n:12048/config" 2>/dev/null | python3 -c "
import json, sys, datetime
try:
    t = json.load(sys.stdin)['earth_rotation_data']['earth_orientation_parameter_table']
    f = lambda ns: datetime.datetime.fromtimestamp(ns/1e9, datetime.timezone.utc).strftime('%m-%d')
    print(f(t[0]['t_inst_ns']) + '..' + f(t[-1]['t_inst_ns']))
except Exception:
    print('NO-READBACK')" 2>/dev/null)
    if [ "$code" = "200" ] && [ "$rb" != "NO-READBACK" ] && [ -n "$rb" ]; then
        echo "  $n  OK   live table $rb"
    else
        echo "  $n  FAIL (POST $code, readback ${rb:-none})"
        fail=1
    fi
done
[ "$fail" = 0 ] && echo "EOP push complete." || { echo "EOP push INCOMPLETE -- see above."; exit 1; }
