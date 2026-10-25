#!/bin/bash
# Why is this node's GPU-n pipeline dead? One command, no restart, no reboot.
#
# In a wedge every stage on one GPU (the GNSS combiners and the main N2 accumulator) stops at
# the same hop and stays stopped. The symptom surfaces far downstream, and every obvious
# suspect is innocent:
#
#   the GPU     0% util, memory still allocated, no Xid, no ECC: idle, not hung
#   the NIC     link up, zero rx_errors, packets arriving at full rate
#   the chains  all stop at the identical hop, so the cause is upstream of every combiner
#
# The break is one step further up: the distributor ring for that port is never drained, so
# nearly every arriving packet is dropped at the ring and the whole GPU pipeline starves. The
# tell is a sleeping dpdk-worker: these are poll-mode threads that should spin forever, and a
# blocked one is waiting on a full output buffer, i.e. back-pressure from a downstream stage
# that stopped consuming.
#
# The discriminator, with no overlap between wedged and healthy nodes:
#     wedged   a dpdk-worker in state S, CPU ticks frozen; ring drops climbing
#     healthy  every dpdk-worker in state R;                ring drops 0
#
# usage:  node_wedge_probe.sh [node ...]        (default: the fleet, config/gnss_fleet.py hosts)
#         node_wedge_probe.sh --bt cx19         also prints, for the first node, the gdb command
#                                                (needs sudo on the node) that dumps thread
#                                                backtraces, so the stage that stopped consuming
#                                                can be named rather than inferred.
set -u
# shellcheck source=kotekan_src.sh
. "$(dirname "${BASH_SOURCE[0]}")/kotekan_src.sh"
BT=0
if [ "${1:-}" = "--bt" ]; then BT=1; shift; fi
NODES=${*:-}
[ -n "$NODES" ] || NODES=$(gnss_fleet_hosts "$GNSS_CHECKOUT") || exit 1

# kotekan is found by process name, the oldest if more than one. "^kotekan" also matches a
# binary staged beside it (GNSS_BIN=.../kotekan.next_<tag>). A command-line pattern misses a
# binary staged outside build/, and matches the `bash -c` that ssh runs this probe in, whose
# command line carries the pattern.
probe_one() {
    local n=$1
    ssh "$n" 'PID=$(pgrep -o "^kotekan")
if [ -z "$PID" ]; then echo "  no kotekan process"; exit 0; fi
echo "  kotekan pid $PID"
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
 | awk -F"[{}]" "/ring_full_dropped/ && !/^#/ {split(\$2,a,\"port=\"); gsub(/[^0-9]/,\"\",a[2]); n=\$3+0; printf \"  port %s ring-drop %.0f%s\n\", a[2], n, (n>0 ? \"   <-- starving its pipeline\" : \"\")}"
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
    echo "    Naming the stage that stopped consuming is the one thing the counters above"
    echo "    cannot do: they prove back-pressure, not its origin. Run:"
    echo
    echo "      ssh $n 'sudo gdb -p \$(pgrep -o ^kotekan) --batch \\"
    echo "         -ex \"set pagination off\" -ex \"thread apply all bt 12\"' > /tmp/kotekan_bt_$n.txt"
    echo
    echo "    Then look for the dpdk-worker3/4 frames and any stage sitting in"
    echo "    wait_for_empty_frame (blocked on a full output = the back-pressure origin)"
    echo "    versus wait_for_full_frame (merely starved = a victim, not the cause)."
fi
