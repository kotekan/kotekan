#!/bin/bash
# Is the F-engine actually delivering, and is our reference time still valid?
#
# usage: fengine_status.sh [node ...]        (default: the six GNSS nodes)
#
# Three questions, in the order that matters. Status endpoints keep serving their last values
# when no data arrives, so quality metrics (deep_snr, lock times) look plausible when nothing
# is moving; counters do not.
#
#   1. Is data flowing? Poll kotekan_dpdk_distributor_packets_total twice and require it to
#      move. Link-up is not data, and neither is a large total; only a rising total is.
#   2. Is the F-engine controller (crs-control) up? The timing service is a startup-only dependency, so it can be down while
#      everything runs, and up while nothing does. It is a separate question.
#   3. Has frame 0 moved? Nodes cache time0 at startup. If the F-engine restarted or changed
#      mode, frame 0 is re-established and every cached copy (including configs generated with
#      --frame0-nano) is silently wrong: wrong record stamps, wrong predicted code phase, every
#      PRN despreading noise while it reads as a clock bug.
set -u
NODES=${*:-"cx19 cx27 cx42 cx43 cx44 cx51"}
# The nodes serve REST on 0.0.0.0:12048 (node_up.sh --bind-address). A node that cannot be
# polled reads as "?", never as zero packets, so a wrong port says so instead of reporting a
# silent F-engine.
PORT=${GNSS_PORT:-12048}
GAP=${GAP:-20}
CHIVE=${CHIVE:-crs-control:54321}  # the F-engine controller (fpga_controller)

# The packet count, or the literal "?" when the endpoint could not be polled at all, so the
# caller can say so instead of folding "could not ask" into "the answer is none".
pkts() { local out
         out=$(timeout 12 curl -sS -m 8 "http://$1:$PORT/metrics" 2>/dev/null) || { echo "?"; return; }
         [ -z "$out" ] && { echo "?"; return; }
         printf "%s" "$out" \
         | awk '/^kotekan_dpdk_distributor_packets_total/ {s += $2; n++}
                END {if (n) printf "%.0f", s; else printf "?"}'; }
t0()   { timeout 12 curl -sS -m 8 "http://$1:$PORT/telescope/time0_ns" 2>/dev/null \
         | sed -n 's/.*"time0_ns"[: ]*\([0-9]\+\).*/\1/p' | head -1; }

echo "== $(date -u '+%Y-%m-%d %H:%M:%S UTC')  /  $(TZ=America/Vancouver date '+%H:%M %Z') at the telescope"

printf "\n-- F-engine controller %s: " "$CHIVE"
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
printf "%-6s %18s %18s  %-9s %s\n" node "CRS pkts t0" "t0+${GAP}s" flowing "frame0_ns (cached at its startup)"
flowing=0; total=0; unpolled=0
for n in $NODES; do
    a=${P1[$n]:-}; b=$(pkts "$n"); z=$(t0 "$n")
    [ -z "$a$b" ] && { printf "%-6s %18s %18s  %-9s %s\n" "$n" - - "NO REST" -; continue; }
    total=$((total+1))
    # Three outcomes, not two: flowing, stalled, or not polled. Folding the third into "no"
    # would report a silent F-engine whenever the nodes cannot be reached.
    if [ "$a" = "?" ] || [ "$b" = "?" ]; then s="?"; unpolled=$((unpolled+1))
    elif [ -n "$a" ] && [ -n "$b" ] && [ "$a" != "$b" ]; then s=YES; flowing=$((flowing+1))
    else s=no; fi
    printf "%-6s %18s %18s  %-9s %s\n" "$n" "${a:-?}" "${b:-?}" "$s" "${z:-?}"
done

echo
if [ "$unpolled" -eq "$total" ] && [ "$total" -gt 0 ]; then
    echo "VERDICT: could not poll any node on :$PORT -- this says nothing about the F-engine."
    echo "  Check the port (nodes bind :12048; GNSS_PORT overrides) before reading anything into it."
elif [ "$flowing" -eq 0 ]; then
    echo "VERDICT: NO DATA anywhere ($total polled, $unpolled unreachable). The F-engine is not delivering."
    echo "  Frozen status endpoints will still serve plausible deep_snr/lock values -- ignore them."
else
    echo "VERDICT: data flowing on $flowing/$total node(s)$([ "$unpolled" -gt 0 ] && echo \", $unpolled UNREACHABLE\")."
    echo "  Now check frame 0 before trusting any science number: the frame0_ns column above is"
    echo "  what each node cached at its own startup. If the F-engine restarted during the"
    echo "  outage, a node started before it is carrying a stale epoch. Prefer the controller; else read"
    echo "  time0 from a node started since data resumed, and regenerate any --frame0-nano"
    echo "  config whose baked value no longer matches."
fi
