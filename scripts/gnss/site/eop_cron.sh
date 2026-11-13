#!/bin/bash
# Keep the on-disk GNSS node configs' EOP table ahead of the clock. Cron, daily, on the gnss VM.
#
# The EOP table is a rolling window of a few days, and a node asked for a time past its last
# entry does not degrade: the stock telescope block sets fatal_eop_out_of_range, so kotekan
# shuts down. choco keeps the running nodes' live table current (it pushes its updatable values,
# the EOP table among them, to nodes in development mode), but a relaunch (the unit's
# on-failure restart, a node_up) starts from the config file, before choco's next push. This
# keeps the file's table from ending.
#
# Each run:
#   1. stage choco:/var/lib/choco/eop/state.json (rewritten there daily, ~12:00 UTC); refuse a
#      table whose last entry is already in the past
#   2. regenerate the on-disk node configs when their table ends earlier than the staged one and
#      they still match the manifest apart from live data, so a relaunch never starts on a table
#      that has ended. Nothing is pushed to a running node.
#
#   crontab:  17 12 * * *  /home/kvand/gnss/gnss-broker/scripts/gnss/site/eop_cron.sh
#   env:      EOP_LOG (default /var/tmp/gnss-logs/eop_cron.log),
#             EOP_STATE (the staged table, default /var/tmp/gnss-logs/eop-state.json)
set -u
# shellcheck source=kotekan_src.sh
. "$(dirname "${BASH_SOURCE[0]}")/kotekan_src.sh"
K=$GNSS_CHECKOUT
LOG=${EOP_LOG:-/var/tmp/gnss-logs/eop_cron.log}
F=${EOP_STATE:-/var/tmp/gnss-logs/eop-state.json}
exec >>"$LOG" 2>&1
echo "== $(date -u +%FT%TZ) eop_cron"

if ! scp -q -o BatchMode=yes -o ConnectTimeout=10 choco:/var/lib/choco/eop/state.json "$F.new"; then
    echo "  choco unreachable; keeping the last staged table"
else
    mv -f "$F.new" "$F"
fi
[ -s "$F" ] || { echo "  no staged table; nothing to compare"; exit 1; }

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

# A relaunch starts from the file, and a node started on a table that has ended shuts down.
# Regenerate the configs only when they still match the manifest apart from live
# data (gen_fleet --check leaves the EOP table and the bad inputs out of the comparison), so a
# hand-edited or half-edited manifest is never written under a node. The files are the ones
# the manifest owns (gen_fleet.py --print-path): their names carry its suffix.
PY=/home/kvand/gnss/venv/bin/python
MAN="$K/config/gnss_fleet_chord.yaml"
# gen_fleet renders kotekan's config templates and parses its headers.
KOTEKAN_SRC=$(gnss_kotekan_src "$K") && export KOTEKAN_SRC
mapfile -t OWNED < <("$PY" "$K/scripts/gnss/gen_fleet.py" "$MAN" --print-path 2>/dev/null)
DISK_END=$("$PY" - "${OWNED[@]}" <<'PYE'
import sys, yaml
ends = []
for p in sys.argv[1:]:
    try:
        ends.append(yaml.safe_load(open(p))["earth_rotation_data"]["earth_orientation_parameter_table"][-1]["t_inst_ns"])
    except Exception:
        pass
print(min(ends) if ends else "")
PYE
)
if [ -n "$DISK_END" ] && [ "$DISK_END" -lt "$FILE_END" ]; then
    rc=0
    "$PY" "$K/scripts/gnss/gen_fleet.py" "$MAN" --check >/dev/null 2>&1 || rc=$?
    if [ "$rc" = 0 ]; then
        if "$PY" "$K/scripts/gnss/gen_fleet.py" "$MAN" >/dev/null 2>&1; then
            echo "  on-disk configs regenerated: table end $DISK_END -> $FILE_END"
        else
            echo "  on-disk regen FAILED (gen_fleet write); table end still $DISK_END"
        fi
    elif [ "$rc" = 1 ]; then
        echo "  on-disk configs differ from the manifest beyond live data; NOT regenerated (table end $DISK_END)"
    else
        echo "  gen_fleet --check could not compare (rc $rc); NOT regenerated (table end $DISK_END)"
    fi
elif [ -z "$DISK_END" ]; then
    echo "  on-disk configs: no table end readable in the manifest's ${#OWNED[@]} config(s); NOT checked"
else
    echo "  on-disk configs current (table end $DISK_END)"
fi
