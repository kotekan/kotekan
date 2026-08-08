#!/bin/bash
# Restart the CHORD GNSS broker in the background, on the host it is run from.
#
# WHY THIS EXISTS: the obvious one-liner
#     ssh cf06 'pkill -f gps_distributed_broker; nohup ... broker_up.sh ... &'
# KILLS ITS OWN SHELL. pkill -f matches against full command lines, and the remote
# shell's command line CONTAINS the pattern -- so pkill signals the shell that is running
# it, the rest of the line never executes, and no broker comes back. Worse, the usual
# `pgrep -f gps_distributed_broker && echo up` health check matches ITSELF the same way and
# cheerfully reports "broker up" with nothing running. That combination cost 20 minutes on
# 2026-08-04: the broker was dead while both the kill and the check claimed success.
#
# The bracket trick ([g]ps...) is a regex that matches the broker's command line but NOT
# this script's own, because the literal text here contains the brackets.
#
# ⚠️ THIS RESTARTS THE UNIFIED MULTI-CHAIN BROKER (task #27), which is what CHORD runs:
# broker_multi.py over config/gnss_chains_chord.yaml, every constellation in ONE process on
# ONE port. It used to exec broker_up.sh -- the single-chain GPS-only launcher -- and kept
# doing so for a day after the unified broker was deployed, so the obvious "restart the
# broker" command would have quietly downgraded the instrument to one chain. I hit exactly
# that on 2026-08-09 and restarted by hand instead. A stale launcher is a live trap, not
# clutter.
#
# GNSS_CHAINS points it at a different manifest (one chain, a test fleet); the old
# single-chain launcher is still there as broker_up.sh if you truly want it.
#
# usage:  broker_restart.sh [extra broker_multi args...]
set -u
K=/home/kvand/gnss/kotekan
LOG=${GNSS_BROKER_LOG:-/tmp/gnss_broker.log}
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}
CHAINS=${GNSS_CHAINS:-$K/config/gnss_chains_chord.yaml}

if [ ! -r "$CHAINS" ]; then
    echo "FAILED: chain manifest not readable: $CHAINS" >&2
    exit 1
fi

# Both names, because a tree mid-transition can have either running: broker_multi is the
# driver, gps_distributed_broker the single-chain process it replaced.
pkill -f "[b]roker_multi.py" 2>/dev/null || true
pkill -f "[g]ps_distributed_broker" 2>/dev/null || true
sleep 3
if pgrep -f "[b]roker_multi.py|[g]ps_distributed_broker" > /dev/null; then
    pkill -9 -f "[b]roker_multi.py" 2>/dev/null || true
    pkill -9 -f "[g]ps_distributed_broker" 2>/dev/null || true
    sleep 2
fi

# ABSOLUTE PATHS THROUGHOUT. This is normally invoked over ssh, which lands in $HOME rather
# than the repo, and a relative script path there fails with a bare "can't open file" after
# the old broker is already dead.
nohup setsid "$PY" -u "$K/scripts/gnss/broker_multi.py" "$CHAINS" "$@" \
    > "$LOG" 2>&1 < /dev/null &
disown
sleep 10

if pgrep -f "[b]roker_multi.py" > /dev/null; then
    echo "broker up (log $LOG) chains: $CHAINS ${*:+args: $*}"
    grep -a "starting .* chain" "$LOG" | tail -1
else
    echo "FAILED to start -- last lines of $LOG:"
    tail -15 "$LOG"
    exit 1
fi
