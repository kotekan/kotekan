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
# usage:  node_up.sh <node> [up|down|restart|status]      default: up
#         node_up.sh cx51
#         node_up.sh cx51 status
#         for n in cx42 cx43 cx44 cx47 cx51 cx52; do scripts/gnss/node_up.sh $n; done
#         for n in cx19 cx27 cx42 cx43 cx44 cx51; do scripts/gnss/node_up.sh $n restart; done
#
# USE `restart` RATHER THAN down-then-up. sudo's credential cache is per-TTY and `ssh -t`
# allocates a fresh one each call, so a down loop followed by an up loop prompts for the
# password TWICE PER NODE -- twelve prompts to cycle six nodes. `restart` does the stop and the
# systemd-run inside ONE ssh session, so it is one prompt per node and one loop.
#
#   env:  GNSS_BIN  binary            (default build/kotekan/kotekan)
#         GNSS_CFG  config            (default config/generated/chord_gnss_<node>.yaml)
#         GNSS_LOG  log file          (default /tmp/gnss_node.log)
#
# Needs an interactive sudo on the target, so it uses `ssh -t` and will prompt per node.
set -u
K=/home/kvand/gnss/kotekan
# build/kotekan since 2026-08-03: the merge segfault is FIXED (c5e9e754a -- an emit-metrics loop
# indexed _navbuf past the end whenever no wipe was configured, which is every CHORD config).
# cx51 ran the fixed binary for hours without crashing. kotekan_premerge is kept as the fallback
# and is what `up-premerge` runs; it has NO deep coherent rung, so a node on it reports
# coherence_s = 0 structurally and a ~3x worse carrier residual (per-emit squared fit rather
# than the buffered-stream fit).
# GNSS_BIN overrides (that is how up-premerge works: exec re-reads this file, so the
# choice has to travel in the ENVIRONMENT, not a shell variable).
PREMERGE=/home/kvand/gnss/kotekan_premerge
BIN=${GNSS_BIN:-/home/kvand/gnss/kotekan/build/kotekan/kotekan}
N=${1:?usage: node_up.sh <node> [up|down|status]}
ACT=${2:-up}
# GNSS_CFG / GNSS_LOG override the config and the log, the same way GNSS_BIN overrides the
# binary. Use them to bring a node up on the PROFILING config without editing anything:
#
#   GNSS_CFG=$K/config/generated/chord_gnss_cx19_prof.yaml GNSS_LOG=/tmp/gnss_prof.log \
#       scripts/gnss/node_up.sh cx19
#
# Give the profiling run its OWN log: it emits one INFO block per GPU frame per GPU (~760
# lines/s), which buries /tmp/gnss_node.log. And note the log must not already exist owned by a
# DIFFERENT user -- systemd runs the unit as root, and fs.protected_regular=2 refuses root an
# O_CREAT append onto another user's file in /tmp. That failure surfaces as status=209/STDOUT
# with nothing else to go on; `sudo rm -f` the stale file first.
CFG=${GNSS_CFG:-$K/config/generated/chord_gnss_$N.yaml}
LOG=${GNSS_LOG:-/tmp/gnss_node.log}

# --- shared by `up` and `restart` -------------------------------------------------------------
# One copy of the preflight and one copy of the systemd-run line, so the two actions can never
# drift apart -- a `restart` that started the node differently from `up` would be a nasty thing
# to debug at 2am.
preflight() {
    # FAIL LOUDLY ON A MISSING CONFIG OR BINARY. The systemd-run below is guarded by
    # `test -r '$CFG' && ...`, and a failing test short-circuits the whole chain SILENTLY:
    # the command returns instantly, prints nothing, creates no unit, and `systemctl status`
    # then says "could not be found". That is indistinguishable from a dozen other failures.
    # It cost several rounds of restarts on 2026-08-07, when a config path was pasted with a
    # literal "..." in it. Check here, where we can say which path was wrong. These are LOCAL
    # checks of an NFS-shared tree, which is the same filesystem the node will read.
    if [ ! -r "$CFG" ]; then
        echo "FAILED: config not readable: $CFG" >&2
        echo "  (a relative or elided path? GNSS_CFG must be absolute)" >&2
        exit 1
    fi
    if [ ! -x "$BIN" ]; then
        echo "FAILED: binary not executable: $BIN" >&2
        exit 1
    fi
}

# The record dir is node-local (NOT the NFS home) and rawFileWrite fails on a missing one.
# `systemctl stop` on a transient unit usually removes it, but a stopped-or-failed unit can
# stay LOADED, and then systemd-run refuses the name: "Unit gnss-node.service was already
# loaded or has a fragment file". reset-failed clears it; harmless when there is nothing to
# clear, hence the `|| true`.
REMOTE_UP="mkdir -p /tmp/gnss && sudo systemctl reset-failed gnss-node 2>/dev/null || true; \
      test -r '$CFG' && sudo systemd-run --unit=gnss-node \
        --working-directory=$K \
        --property=StandardOutput=append:$LOG \
        --property=StandardError=append:$LOG \
        '$BIN' --config '$CFG' --bind-address 0.0.0.0:12049"

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
    preflight
    ssh -t "$N" "$REMOTE_UP" \
      && sleep 3 && printf "%-6s " "$N" && ssh -o BatchMode=yes "$N" systemctl is-active gnss-node
    ;;
  restart)
    # STOP AND START IN ONE SSH SESSION -- the entire point. sudo's credential cache is keyed to
    # the TTY, `ssh -t` allocates a new one per call, so `down` then `up` is two passwords per
    # node. Here the stop and the systemd-run share one session and one prompt.
    #
    # The stop is `|| true` on purpose: the unit is TRANSIENT, so on a node that is already down
    # it does not exist and `systemctl stop` exits non-zero. That is the normal case for a node
    # that crashed, not an error, and must not abort the start that follows.
    #
    # Preflight BEFORE stopping. Checking afterwards would take a healthy node down and then
    # refuse to bring it back because of a typo in GNSS_CFG -- the one outcome worse than not
    # restarting at all.
    preflight
    ssh -t "$N" "sudo systemctl stop gnss-node 2>/dev/null || true; $REMOTE_UP" \
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
    ssh -t "$N" "mkdir -p /tmp/gnss && sudo systemctl reset-failed gnss-node 2>/dev/null || true; \
      test -x '$DBG' && sudo systemd-run --unit=gnss-node \
        --working-directory=$K \
        --property=StandardOutput=append:/tmp/gnss_node_dbg.log \
        --property=StandardError=append:/tmp/gnss_node_dbg.log \
        /usr/bin/gdb --batch -ex run -ex 'thread apply all bt' -ex 'info registers' \
          --args '$DBG' --config '$CFG' --bind-address 0.0.0.0:12049" \
      && sleep 5 && printf "%-6s " "$N" && ssh -o BatchMode=yes "$N" systemctl is-active gnss-node \
      && echo "backtrace (if it faults) lands in /tmp/gnss_node_dbg.log on $N"
    ;;
  up-premerge)
    GNSS_BIN=$PREMERGE exec "$0" "$N" up
    ;;
  *) echo "unknown action '$ACT' (up|down|restart|status|debug|up-premerge)"; exit 2 ;;
esac
