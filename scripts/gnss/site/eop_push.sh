#!/bin/bash
# Push the current EOP table into every running GNSS node kotekan, and read it back.
#
# The EOP table lives in the generated node configs and is a rolling window of a few days, so a
# table that was fresh at generation is past its end within a week, and a node asked for a time
# past the end shuts down (the stock telescope block sets fatal_eop_out_of_range). The table is
# REST-updatable in the running process (kotekan_update_endpoint), which also means a node
# restart reverts it to the config file's copy. So run this after every node bring-up, and
# whenever the window nears its end; eop_cron.sh runs it hourly for nodes whose live table is
# behind.
#
# Source: the observatory keeps the rolling file at choco:/var/lib/choco/eop/state.json
# (refreshed daily at ~12:00). eop_cron.sh, hourly on gnss, stages it at
# /var/tmp/gnss-logs/eop-state.json, which is the default here; run this on gnss.
#
#   usage:  eop_push.sh [file]          default /var/tmp/gnss-logs/eop-state.json
#   env:    GNSS_NODES                  default the fleet (config/gnss_fleet.py hosts)
set -u
# shellcheck source=kotekan_src.sh
. "$(dirname "${BASH_SOURCE[0]}")/kotekan_src.sh"
F=${1:-/var/tmp/gnss-logs/eop-state.json}
NODES=${GNSS_NODES:-}
[ -n "$NODES" ] || NODES=$(gnss_fleet_hosts "$GNSS_CHECKOUT") || exit 1

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
    print("Stage a fresh one:  scp choco:/var/lib/choco/eop/state.json %s" % sys.argv[1])
    raise SystemExit(1)
if (hi - now).days < 1:
    print("WARNING: <1 day of table left past now -- stage a fresh file soon.")
PY

fail=0
for n in $NODES; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 8 -X POST \
        -H 'Content-Type: application/json' --data-binary @"$F" \
        "http://$n:12048/earth_rotation_data" 2>/dev/null || echo 000)
    # Read it back: a 200 proves the POST parsed, not that the process now holds the table
    # we sent, and the live copy is the point of this script.
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
