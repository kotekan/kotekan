#!/bin/bash
# Keep the running GNSS nodes' live EOP table ahead of the clock. Cron, on the gnss VM.
#
# WHY. The EOP table is a rolling ~6-day window, and a node that asks for a time past its last
# entry does not degrade: n2_accumulate raises a FatalError and the whole kotekan shuts down.
# A one-off push is therefore a fuse, not a fix. This refreshes the live copy from the
# observatory's file every run, so the fuse never reaches the end.
#
# WHAT IT DOES, each run:
#   1. stage choco:/var/lib/choco/eop/state.json (rewritten there daily ~12:00 UTC); refuse a
#      table whose last entry is already in the past -- pushing a stale table looks like
#      maintenance happened
#   2. for each node, read the live config; a node qualifies only if it answers, is running a
#      GNSS config (gnssN_* stages present) and holds an EOP table. A down, restarting or
#      non-GNSS node is skipped and logged, never an error.
#   3. push (eop_push.sh, which reads the table back) only to qualifying nodes whose live table
#      ends EARLIER than the staged one. Re-running is a no-op.
#
#   crontab:  17 * * * *  /home/kvand/gnss/kotekan/scripts/gnss/eop_cron.sh
#   env:      GNSS_NODES (default the six), EOP_LOG (default /var/tmp/gnss-logs/eop_cron.log)
set -u
K=$(cd "$(dirname "$0")/../.." && pwd)
NODES=${GNSS_NODES:-"cx19 cx27 cx42 cx43 cx44 cx51"}
LOG=${EOP_LOG:-/var/tmp/gnss-logs/eop_cron.log}
F=/var/tmp/gnss-logs/eop-state.json
exec >>"$LOG" 2>&1
echo "== $(date -u +%FT%TZ) eop_cron"

if ! scp -q -o BatchMode=yes -o ConnectTimeout=10 choco:/var/lib/choco/eop/state.json "$F.new"; then
    echo "  choco unreachable; keeping the last staged table"
else
    mv -f "$F.new" "$F"
fi
[ -s "$F" ] || { echo "  no staged table; nothing to push"; exit 1; }

# Staged table's last entry (ns), or refuse if already past.
FILE_END=$(python3 - "$F" <<'PY'
import json, sys, time
t = json.load(open(sys.argv[1]))["earth_orientation_parameter_table"]
end = t[-1]["t_inst_ns"]
if end / 1e9 < time.time():
    sys.exit("REFUSING: staged table ended %s" % time.strftime("%F %TZ", time.gmtime(end / 1e9)))
print(end)
PY
) || { echo "  $FILE_END"; exit 1; }

PUSH=""
for n in $NODES; do
    LIVE_END=$(curl -s --max-time 10 "http://$n:12048/config" 2>/dev/null | python3 -c '
import json, re, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("DOWN"); raise SystemExit
# A node mid-shutdown still answers, with a JSON error body ({"code": 503, ...}): that is down.
if not isinstance(d, dict) or "code" in d and len(d) <= 2:
    print("DOWN"); raise SystemExit
if not any(re.match(r"gnss\d_", k) for k in d):
    print("NOT-GNSS"); raise SystemExit
try:
    print(d["earth_rotation_data"]["earth_orientation_parameter_table"][-1]["t_inst_ns"])
except Exception:
    print("NO-TABLE")' 2>/dev/null)
    case "$LIVE_END" in
        DOWN|NOT-GNSS|NO-TABLE|"") echo "  $n skip (${LIVE_END:-DOWN})" ;;
        *) if [ "$LIVE_END" -lt "$FILE_END" ]; then PUSH="$PUSH $n"
           else echo "  $n current"; fi ;;
    esac
done

if [ -n "$PUSH" ]; then
    GNSS_NODES="${PUSH# }" "$K/scripts/gnss/eop_push.sh" "$F"
else
    echo "  nothing to push"
fi
