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
# usage:  broker_restart.sh [extra broker args...]     e.g. --carrier-gain 0.25
set -u
K=/home/kvand/gnss/kotekan
LOG=${GNSS_BROKER_LOG:-/tmp/gnss_broker.log}

pkill -f "[g]ps_distributed_broker" 2>/dev/null || true
sleep 3
if pgrep -f "[g]ps_distributed_broker" > /dev/null; then
    pkill -9 -f "[g]ps_distributed_broker" 2>/dev/null || true
    sleep 2
fi

nohup setsid bash "$K/scripts/gnss/broker_up.sh" "$@" > "$LOG" 2>&1 < /dev/null &
disown
sleep 10

if pgrep -f "[g]ps_distributed_broker" > /dev/null; then
    echo "broker up (log $LOG) args: $*"
else
    echo "FAILED to start -- last lines of $LOG:"
    tail -15 "$LOG"
    exit 1
fi
