#!/bin/bash
# Bring a CHORD GNSS node's kotekan up (or down, or check it).
#
# WHY THIS EXISTS: the node service is a TRANSIENT systemd unit -- `systemd-run --unit=gnss-node`
# creates it in /run/systemd/transient/, and STOPPING IT DELETES IT. So after a
# `systemctl stop gnss-node`, `systemctl start gnss-node` fails with "Unit not found" and the
# only way back is the full systemd-run line. Six nodes were stopped overnight on 2026-08-02 and
# every one of them needs this to come back.
#
# The exec path must be ABSOLUTE: systemd-run does not resolve a relative one against
# --working-directory, and the failure mode is a unit that starts and immediately dies.
#
# usage:  node_up.sh <node> [up|down|status]      default: up
#         node_up.sh cx51
#         node_up.sh cx51 status
#         for n in cx42 cx43 cx44 cx47 cx51 cx52; do scripts/gnss/node_up.sh $n; done
#
# Needs an interactive sudo on the target, so it uses `ssh -t` and will prompt per node.
set -u
K=/home/kvand/gnss/kotekan
BIN=/home/kvand/gnss/kotekan_premerge   # NOT build/kotekan: the merge broke GnssCoherentCombiner
                                        # (segfault, still unbisected -- see CHORD_GNSS_STATE.md)
N=${1:?usage: node_up.sh <node> [up|down|status]}
ACT=${2:-up}
CFG=$K/config/generated/chord_gnss_$N.yaml

case "$ACT" in
  status)
    # `systemctl is-active` EXITS NON-ZERO for a stopped unit, so a bare `|| echo unreachable`
    # reports every healthy-but-stopped node as unreachable. Capture instead of chaining.
    printf "%-6s " "$N"
    S=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$N" systemctl is-active gnss-node 2>/dev/null)
    echo "${S:-unreachable}"
    ;;
  down)
    ssh -t "$N" "sudo systemctl stop gnss-node"
    echo "note: the unit is transient -- it is now GONE. Use '$0 $N' to bring it back."
    ;;
  up)
    # The record dir is node-local (NOT the NFS home) and rawFileWrite fails on a missing one.
    ssh -t "$N" "mkdir -p /tmp/gnss && test -r '$CFG' && sudo systemd-run --unit=gnss-node \
        --working-directory=$K \
        --property=StandardOutput=append:/tmp/gnss_node.log \
        --property=StandardError=append:/tmp/gnss_node.log \
        '$BIN' --config '$CFG' --bind-address 0.0.0.0:12049" \
      && sleep 3 && printf "%-6s " "$N" && ssh -o BatchMode=yes "$N" systemctl is-active gnss-node
    ;;
  debug)
    # Run the POST-MERGE binary under gdb so a crash leaves a full backtrace instead of just an
    # exit status. The merge broke GnssCoherentCombiner (segfault) and the node has run
    # kotekan_premerge ever since, which also blocks every combiner change from being deployed --
    # including the plain coherent rung. One node at a time: leave cx19 on premerge as a control.
    #
    # gdb --batch runs to completion and only prints on a fault, so a healthy node logs nothing
    # extra; `-ex run` starts it, and the two bt commands fire when it stops.
    DBG=/home/kvand/gnss/kotekan/build/kotekan/kotekan
    ssh -t "$N" "mkdir -p /tmp/gnss && test -x '$DBG' && sudo systemd-run --unit=gnss-node \
        --working-directory=$K \
        --property=StandardOutput=append:/tmp/gnss_node_dbg.log \
        --property=StandardError=append:/tmp/gnss_node_dbg.log \
        /usr/bin/gdb --batch -ex run -ex 'thread apply all bt' -ex 'info registers' \
          --args '$DBG' --config '$CFG' --bind-address 0.0.0.0:12049" \
      && sleep 5 && printf "%-6s " "$N" && ssh -o BatchMode=yes "$N" systemctl is-active gnss-node \
      && echo "backtrace (if it faults) lands in /tmp/gnss_node_dbg.log on $N"
    ;;
  *) echo "unknown action '$ACT' (up|down|status|debug)"; exit 2 ;;
esac
