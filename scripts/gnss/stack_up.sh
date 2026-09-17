#!/bin/bash
# Bring the cf06 GNSS stack up, in dependency order, with the right interpreter, and prove it.
#
#   ssh cf06 /home/kvand/gnss/kotekan/scripts/gnss/stack_up.sh          # everything
#   ssh cf06 /home/kvand/gnss/kotekan/scripts/gnss/stack_up.sh --list   # show the plan, change nothing
#   ssh cf06 /home/kvand/gnss/kotekan/scripts/gnss/stack_up.sh --no-preflight
#
# The counterpart of stack_down.sh. Until 2026-09-14 the bring-up was eight scripts run by hand
# from a list in a doc, and that day it cost two hours: the list omitted the broker's
# interpreter (GNSS_PY), so the broker came up under the GIL and every chain but L5 was dark
# while every port answered. This script exists so the order, the environment and the checks
# are one committed artifact.
#
# ORDER: producers before consumers, and the archiver before ANY node (buglist #110: a cube
# frame arriving with no archiver segfaulted six nodes). The nodes are NOT started here --
# they need sudo on each node and are `node_up.sh <node> restart`, run after this succeeds.
#
# PREFLIGHT (refuses, --no-preflight overrides):
#   * this is cf06 (every *_up.sh below refuses elsewhere anyway)
#   * /mnt/cs00/data is a mountpoint (the archiver writes there; the stub dir exists regardless)
#   * chive:54321/get-frame0-time answers with frame0_nano (the nodes will need it; a stack
#     brought up against a dead F-engine controller is a stack waiting for nothing)
# WARNS (does not refuse): the six node configs' baked EOP headroom -- node_up.sh refuses on
# that itself, but hearing it here saves a round trip.
set -u

# ⚠️⚠️ SUPERSEDED 2026-09-17 -- THIS WOULD START A SECOND LIVE STACK.
# The broker, gather, aggregator, obs writers and viewer moved to the gnss VM and run there as
# systemd user units. This script still starts all five. Run on cf06 today it gives the fleet
# TWO BROKERS commanding the same nodes, and two gathers competing for the same telemetry --
# which is the failure broker_restart.sh's own host guard exists to prevent, arriving by a
# different door.
#
#     ssh gnss systemctl --user start gnss-stack.target      <- what you want
#     docs/CHORD_GNSS_RUNBOOK.md                             <- why
#
# Kept, not deleted: it records the dependency order and the preflight conditions, and cf06
# still runs the cube leg (cubearch_up.sh, cubecompact_up.sh, beamview_up.sh), which this
# script never started anyway.
if [ "${GNSS_ALLOW_LEGACY_STACK_UP:-0}" != "1" ]; then
    echo "REFUSING: the live GNSS stack moved to the gnss VM on 2026-09-17." >&2
    echo "  This script would start a SECOND broker, gather and aggregator beside it." >&2
    echo "  Use:  ssh gnss systemctl --user start gnss-stack.target" >&2
    echo "  See:  docs/CHORD_GNSS_RUNBOOK.md" >&2
    echo "  If you genuinely mean to run a legacy stack here, set" >&2
    echo "  GNSS_ALLOW_LEGACY_STACK_UP=1 -- and stop the VM's units first." >&2
    exit 1
fi
K=/home/kvand/gnss/kotekan
S=$K/scripts/gnss
FT=/home/kvand/gnss/venv-ft/bin/python
LIST=0; PRE=1
for a in "$@"; do
    case "$a" in
        --list) LIST=1 ;;
        --no-preflight) PRE=0 ;;
        *) echo "usage: stack_up.sh [--list] [--no-preflight]" >&2; exit 2 ;;
    esac
done

# name | command | verification (a shell expression that is true when the component is up)
COMPONENTS="
cube-archiver|$S/cubearch_up.sh|ss -ltn | grep -q ':11070 '
gather|$S/gather_up.sh|ss -ltn | grep -q ':11060 ' && ss -ltn | grep -q ':12051 '
aggregator|$S/agg_up.sh|ss -ltn | grep -q ':11040 ' && ss -ltn | grep -q ':12050 '
broker|GNSS_PY=$FT $S/broker_restart.sh|ss -ltn | grep -q ':12060 '
viewer-livebeam|$S/viewer_up.sh|ss -ltn | grep -q ':8080 ' && ss -ltn | grep -q ':8539 '
viewer-static|$S/beamview_up.sh|ss -ltn | grep -q ':8877 '
cube-compactor|$S/cubecompact_up.sh|pgrep -f '[c]ubecompact_loop.sh' >/dev/null
obs-writers|$S/obs_up.sh|[ \"\$(pgrep -fc '[g]nss_observables.py')\" -ge 8 ]
"

if [ "$LIST" = 1 ]; then
    printf '%-16s %s\n' COMPONENT COMMAND
    while IFS='|' read -r name cmd _; do [ -n "$name" ] && printf '%-16s %s\n' "$name" "$cmd"; done <<< "$COMPONENTS"
    exit 0
fi

if [ "$PRE" = 1 ]; then
    fail=0
    [ "$(hostname -s)" = cf06 ] || { echo "PREFLIGHT: this is $(hostname -s), the stack runs on cf06" >&2; fail=1; }
    mountpoint -q /mnt/cs00/data || { echo "PREFLIGHT: /mnt/cs00/data is not mounted (the archiver would write to the local stub)" >&2; fail=1; }
    if ! curl -sf -m 20 http://chive:54321/get-frame0-time 2>/dev/null | grep -q frame0_nano; then
        echo "PREFLIGHT: chive:54321/get-frame0-time does not answer with frame0_nano -- the F-engine" >&2
        echo "  controller is down or restarting; the nodes cannot start until it does." >&2; fail=1
    fi
    _h=$(python3 - "$K"/config/generated/chord_gnss_cx*_multi.yaml <<'PYEOP'
import re, sys, time
hs = []
for f in sys.argv[1:]:
    ts = [int(x) for x in re.findall(r"t_inst_ns:\s*(\d+)", open(f).read())]
    if ts: hs.append((max(ts) / 1e9 - time.time()) / 3600)
print("%.1f" % min(hs) if hs else "nan")
PYEOP
)
    if [ "$_h" = nan ] || [ "$(python3 -c "print(int(float('$_h') < 12))")" = 1 ]; then
        echo "WARNING: the node configs' baked EOP table has ${_h} h of headroom; node_up.sh will refuse." >&2
        echo "  python3 $K/scripts/gnss/gen_fleet.py $K/config/gnss_fleet_chord.yaml && ... --check, then commit." >&2
    fi
    [ $fail = 0 ] || { echo "PREFLIGHT FAILED (--no-preflight to override deliberately)" >&2; exit 1; }
fi

rc=0
while IFS='|' read -r name cmd verify; do
    [ -z "$name" ] && continue
    printf '=== %-16s %s\n' "$name" "$(date -u +%H:%M:%S)"
    if eval "$verify" 2>/dev/null; then
        printf '%-16s already up -- left alone\n' "$name"; continue
    fi
    eval "$cmd" 2>&1 | sed 's/^/    | /' | tail -4
    ok=0
    for _ in $(seq 1 30); do sleep 1; if eval "$verify" 2>/dev/null; then ok=1; break; fi; done
    if [ $ok = 1 ]; then printf '%-16s UP\n' "$name"; else printf '%-16s FAILED to verify within 30 s\n' "$name" >&2; rc=1; fi
done <<< "$COMPONENTS"

echo
echo "listeners: $(ss -ltn 2>/dev/null | grep -oE ':(11070|11060|12051|11040|12050|12060|8080|8539|8877) ' | tr -d ' :' | sort -n | tr '\n' ' ')"
echo "procs: kotekan=$(pgrep -c -x kotekan) broker=$(pgrep -fc '[b]roker_multi') obs=$(pgrep -fc '[g]nss_observables') compact=$(pgrep -fc '[c]ubecompact_loop') http8877=$(pgrep -fc '[h]ttp.server 8877') livebeam=$(pgrep -fc '[l]ivebeam_server')"
p=$(pgrep -f '[b]roker_multi' | head -1); [ -n "$p" ] && echo "broker interpreter: $(readlink -f /proc/$p/exe)"
echo
echo "next: the nodes.  for n in cx19 cx27 cx42 cx43 cx44 cx51; do $S/node_up.sh \$n restart; done"
echo "      then, once they answer:  $S/eop_push.sh   (from a host that reaches the nodes; cf06 often cannot)"
exit $rc
