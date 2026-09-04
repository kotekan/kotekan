#!/bin/bash
# WHY IS THIS NODE'S GPU-n PIPELINE DEAD? One command, no restart, no reboot (task #60).
#
# THE WEDGE, as measured 2026-08-15 on cx19/cx42/cx43: every stage on GPU 0 -- the five GNSS
# combiners AND the main N2 accumulator -- stopped within 16 ms of each other and stayed
# stopped. It cost three node reboots and a day of confusion before anyone looked at the
# packet counters, because the symptom surfaces far downstream (a beam-amplitude column
# scattering 1-9) and every obvious suspect is innocent:
#
#   the GPU     0% util, 31 C, memory still allocated, no Xid, no ECC -- IDLE, not hung
#   the NIC     link up, zero rx_errors, packets arriving at full rate
#   the chains  all five stop at the IDENTICAL hop, so it is upstream of every combiner
#
# What is actually broken is one step further up: the distributor ring for that port is
# never drained, so 99.5% of arriving packets are dropped at the ring and the whole GPU
# pipeline starves. The tell is a SLEEPING dpdk-worker: those are poll-mode threads that
# should spin forever, and a blocked one means it is waiting on a full output buffer, i.e.
# back-pressure from a downstream stage that stopped consuming.
#
# THE DISCRIMINATOR, measured 3 wedged vs 3 healthy nodes, no overlap whatsoever:
#     wedged   dpdk-worker3/4 state S, ~3100 ticks (frozen); ring drops 276-281 MILLION
#     healthy  all six workers state R, ~257000 ticks;        ring drops 0
#
# usage:  node_wedge_probe.sh [node ...]        (default: the whole fleet)
#         node_wedge_probe.sh --bt cx19         ⚠️ needs sudo; dumps thread backtraces so
#                                                the stage that stopped consuming can be
#                                                NAMED rather than inferred.
set -u
NODES_DEFAULT="cx19 cx27 cx42 cx43 cx44 cx51"
BT=0
if [ "${1:-}" = "--bt" ]; then BT=1; shift; fi
NODES="${*:-$NODES_DEFAULT}"

probe_one() {
    local n=$1
    ssh "$n" 'PID=$(pgrep -f "build/kotekan" | head -1)
if [ -z "$PID" ]; then echo "  NO KOTEKAN PROCESS"; exit 0; fi
# --- dpdk workers: the discriminator. A sleeping poll-mode worker is the whole finding.
bad=0
for t in /proc/$PID/task/*; do
    c=$(cat $t/comm 2>/dev/null)
    case "$c" in dpdk-worker*)
        s=$(awk "{print \$3}" $t/stat 2>/dev/null); u=$(awk "{print \$14+\$15}" $t/stat 2>/dev/null)
        printf "  %-14s %s %8s ticks%s\n" "$c" "$s" "$u" "$([ "$s" = "S" ] && echo "   <-- BLOCKED (should spin)")"
        [ "$s" = "S" ] && bad=$((bad+1)) ;;
    esac
done
# --- ring drops per port: the consequence, in packets thrown away
curl -s --max-time 5 http://localhost:12048/metrics 2>/dev/null \
 | awk -F"[{}]" "/ring_full_dropped/ && !/^#/ {split(\$2,a,\"port=\"); gsub(/[^0-9]/,\"\",a[2]); n=\$3+0; printf \"  port %s ring-drop %.0f%s\n\", a[2], n, (n>0 ? \"   <-- STARVING ITS PIPELINE\" : \"\")}"
# --- GPU state: proves the device is idle rather than hung
nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null \
 | awk "{printf \"  gpu%s util %s temp %s\n\", \$1, \$2\$3, \$4}"
# --- per-stage last-activity: which side of the node stopped, and when
curl -s --max-time 5 http://localhost:12048/metrics 2>/dev/null \
 | awk "/gnss_combiner_last_emit_time_us\{/ {split(\$0,a,\"\\\"\"); ts=\$NF; print a[2], ts}" \
 | sort | awk "{n=split(\$1,p,\"/\"); print \"  \" \$1, \$2}" | head -4
[ "$bad" -gt 0 ] && echo "  ==> WEDGED: $bad dpdk worker(s) blocked; this node is dropping packets at the ring."
exit 0' 2>/dev/null
}

for n in $NODES; do
    echo "=== $n ==="
    probe_one "$n"
done

if [ "$BT" = "1" ]; then
    n=$(echo $NODES | awk '{print $1}')
    echo
    echo "=== $n thread backtraces (needs sudo on the node) ==="
    echo "    Naming the stage that stopped consuming is the ONE thing the counters above"
    echo "    cannot do: they prove back-pressure, not its origin. Run:"
    echo
    echo "      ssh $n 'sudo gdb -p \$(pgrep -f build/kotekan | head -1) --batch \\"
    echo "         -ex \"set pagination off\" -ex \"thread apply all bt 12\"' > /tmp/kotekan_bt_$n.txt"
    echo
    echo "    Then look for the dpdk-worker3/4 frames and any stage sitting in"
    echo "    wait_for_empty_frame (blocked on a FULL OUTPUT = the back-pressure origin)"
    echo "    versus wait_for_full_frame (merely starved = a victim, not the cause)."
fi
