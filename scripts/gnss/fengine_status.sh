#!/bin/bash
# Is the F-engine actually delivering, and is our reference time still valid?
#
# usage: fengine_status.sh [node ...]        (default: the six GNSS nodes)
#
# THREE QUESTIONS, in the order that matters. Written after 2026-08-07, when a single
# snapshot of /gnss0_combine/get_status showed 30 PRNs at deep_snr 11-14 on five nodes and
# was reported as "fleet healthy" -- while the F-engine was in its dev mode, the links were
# up, and every one of those numbers was FROZEN. Two nodes were even reporting adr_lock_s
# identical to 12 significant figures. Quality metrics look perfectly plausible when nothing
# is moving; counters do not.
#
#   1. IS DATA FLOWING? Poll kotekan_dpdk_distributor_packets_total TWICE and require it to
#      MOVE. Link-up is not data. A large total is not data. Only a rising total is data.
#   2. IS chive BACK? The timing service is a STARTUP-only dependency, so this can be down
#      while everything runs -- and up while nothing does. It is a separate question.
#   3. HAS FRAME 0 MOVED? Nodes CACHE time0 at startup. If the F-engine restarted or changed
#      mode, frame 0 is re-established and every cached copy (including --frame0-nano configs
#      generated from one) is silently wrong: wrong record stamps, wrong predicted code
#      phase, every PRN despreading noise while it reads as a clock bug.
set -u
NODES=${*:-"cx19 cx27 cx42 cx43 cx44 cx51"}
PORT=12049
GAP=${GAP:-20}
CHIVE=${CHIVE:-10.222.0.54:54321}

pkts() { timeout 12 curl -sS -m 8 "http://$1:$PORT/metrics" 2>/dev/null \
         | awk '/^kotekan_dpdk_distributor_packets_total/ {s += $2} END {printf "%.0f", s+0}'; }
t0()   { timeout 12 curl -sS -m 8 "http://$1:$PORT/telescope/time0_ns" 2>/dev/null \
         | sed -n 's/.*"time0_ns"[: ]*\([0-9]\+\).*/\1/p' | head -1; }

echo "== $(date -u '+%Y-%m-%d %H:%M:%S UTC')  /  $(TZ=America/Vancouver date '+%H:%M %Z') at the telescope"

printf "\n-- chive %s: " "$CHIVE"
if timeout 8 curl -sS -m 6 "http://$CHIVE/get-frame0-time" >/dev/null 2>&1; then
    echo "UP"
    echo "   $(timeout 8 curl -sS -m 6 "http://$CHIVE/get-frame0-time" 2>/dev/null | head -c 200)"
else
    echo "DOWN (startup-only dependency: running nodes are unaffected)"
fi

declare -A P1
for n in $NODES; do P1[$n]=$(pkts "$n"); done
sleep "$GAP"

echo
printf "%-6s %18s %18s  %-9s %s\n" node "CRS pkts t0" "t0+${GAP}s" flowing "frame0_ns (cached at ITS startup)"
flowing=0; total=0
for n in $NODES; do
    a=${P1[$n]:-}; b=$(pkts "$n"); z=$(t0 "$n")
    [ -z "$a$b" ] && { printf "%-6s %18s %18s  %-9s %s\n" "$n" - - "NO REST" -; continue; }
    total=$((total+1))
    if [ -n "$a" ] && [ -n "$b" ] && [ "$a" != "$b" ]; then s=YES; flowing=$((flowing+1)); else s=no; fi
    printf "%-6s %18s %18s  %-9s %s\n" "$n" "${a:-?}" "${b:-?}" "$s" "${z:-?}"
done

echo
if [ "$flowing" -eq 0 ]; then
    echo "VERDICT: NO DATA anywhere ($total node(s) polled). The F-engine is not delivering."
    echo "  Frozen status endpoints will still serve plausible deep_snr/lock values -- ignore them."
else
    echo "VERDICT: data flowing on $flowing/$total node(s)."
    echo "  NOW CHECK FRAME 0 before trusting any science number: the frame0_ns column above is"
    echo "  what each node cached AT ITS OWN STARTUP. If the F-engine restarted during the"
    echo "  outage, a node started BEFORE it is carrying a stale epoch. Prefer chive; else read"
    echo "  time0 from a node started SINCE data resumed, and regenerate any --frame0-nano"
    echo "  config whose baked value no longer matches."
fi
