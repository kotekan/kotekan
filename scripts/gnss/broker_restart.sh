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
# ⚠️ THE DEFAULT IS THE FREE-THREADED INTERPRETER. The broker has run on 3.14t at venv-ft since
# 2026-09-0x; under the 3.12 GIL venv one thread pins a core, the telemetry receiver misses the
# gather's 200 ms whole-frame deadline, the gather drops it every ~15 s, every instance reads
# stale, nothing is armed, and every chain but L5 goes dark while the stack looks up (2026-09-14,
# two hours). GNSS_PY still overrides -- the 3.12 venv is the rollback.
PY=${GNSS_PY:-/home/kvand/gnss/venv-ft/bin/python}
CHAINS=${GNSS_CHAINS:-$K/config/gnss_chains_chord.yaml}

if [ ! -r "$CHAINS" ]; then
    echo "FAILED: chain manifest not readable: $CHAINS" >&2
    exit 1
fi

# THIS SCRIPT ACTS ON THE HOST IT RUNS ON -- and the pkill below only reaches LOCAL
# processes. Run casually from a dev node it starts a SECOND broker beside the real one
# and the fleet has two masters silently fighting over every seed (hit 2026-08-09 from
# cx19: six minutes of seed churn before the duplicate was noticed; the local pkill had
# nothing to kill, the health check saw the new local process, everything looked green).
# So: refuse anywhere but the canonical broker host unless explicitly overridden.
HOST=${GNSS_BROKER_HOST:-cf06}
if [ "$(hostname -s)" != "$HOST" ]; then
    echo "REFUSING: this restarts the broker ON $(hostname -s), but the broker host is" >&2
    echo "$HOST -- the pkill cannot reach a broker running there, so this would START A" >&2
    echo "SECOND ONE. Use:  ssh $HOST '$0${*:+ $*}'   (or set GNSS_BROKER_HOST to" >&2
    echo "$(hostname -s) if you truly mean to run a broker here)." >&2
    exit 1
fi

# ⚠️ THE SUPERVISOR FIRST. The launch below wraps the broker in a restart loop, so killing
# the python alone just makes the loop relaunch it 20 s into this script's own startup --
# two brokers, or a race with the new one. Its argv carries the marker below precisely so it
# can be named separately from the broker it babysits.
# ⚠️ PIDFILES, NOT PATTERN MATCHING. Every `pgrep -f`/`pkill -f` over this process table has
# a false positive waiting in it: the pattern matches the shell that runs it (the header's
# 20-minute outage on 2026-08-04), and once the broker is supervised it also matches the
# BABYSITTER, whose argv necessarily contains "broker_multi.py" -- so the health check
# reported "broker up" while the loop merely slept between relaunches. The prototype hit the
# same class and moved to pidfiles; this follows it, including the part that matters: a pid
# is only believed if /proc/<pid>/cmdline still says it is what we think it is, because pids
# are reused.
SUP_PIDFILE=${GNSS_BROKER_SUP_PIDFILE:-/tmp/gnss_broker_supervisor.pid}
BROKER_PIDFILE=${GNSS_BROKER_PIDFILE:-/tmp/gnss_broker.pid}

# pid_is <pidfile> <string that must appear in its /proc cmdline> -> echoes the live pid
pid_is() {
    local pf=$1 want=$2 pid
    [ -f "$pf" ] || return 1
    pid=$(cat "$pf" 2>/dev/null)
    case $pid in ''|*[!0-9]*) return 1 ;; esac
    kill -0 "$pid" 2>/dev/null || return 1
    grep -qa -- "$want" /proc/"$pid"/cmdline 2>/dev/null || return 1
    echo "$pid"
}

broker_running() { pid_is "$BROKER_PIDFILE" "broker_multi.py" > /dev/null; }

sup_pid=$(pid_is "$SUP_PIDFILE" "gnss-broker-supervisor") && {
    kill "$sup_pid" 2>/dev/null || true
    sleep 1
    kill -0 "$sup_pid" 2>/dev/null && kill -9 "$sup_pid" 2>/dev/null
}
brk_pid=$(pid_is "$BROKER_PIDFILE" "broker_multi.py") && kill "$brk_pid" 2>/dev/null
rm -f "$SUP_PIDFILE" "$BROKER_PIDFILE"

# Transitional, and harmless to keep: a broker launched by a version of this script from
# before the pidfiles has none to find. Over-killing here costs nothing -- unlike the health
# check above, where a false positive is the whole problem.
pkill -f "[g]nss-broker-supervisor" 2>/dev/null || true


# Both names, because a tree mid-transition can have either running: broker_multi is the
# driver, gps_distributed_broker the single-chain process it replaced.
pkill -f "[b]roker_multi.py" 2>/dev/null || true
pkill -f "[g]ps_distributed_broker" 2>/dev/null || true
sleep 3
if broker_running || pgrep -f "[g]ps_distributed_broker" > /dev/null; then
    pkill -9 -f "[b]roker_multi.py" 2>/dev/null || true
    pkill -9 -f "[g]ps_distributed_broker" 2>/dev/null || true
    sleep 2
fi

# ABSOLUTE PATHS THROUGHOUT. This is normally invoked over ssh, which lands in $HOME rather
# than the repo, and a relative script path there fails with a bare "can't open file" after
# the old broker is already dead.
# ⚠️ ONE BLAS THREAD. numpy's OpenBLAS pool defaults to one BUSY-SPINNING worker per core;
# the joint filter's small matmuls (P is ~57x57) wake all of them, and measured 2026-08-15
# 17:14 the broker sat at ~53 cores of spin -- which starved the TELEMETRY READER at the OS
# level: the gather dropped its client every 200 ms and 5 of 6 frames were lost (gaps 95k
# vs frames 18k). At these matrix sizes single-threaded BLAS is also simply faster.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# ROTATE, DO NOT TRUNCATE (buglist #65). `> "$LOG"` destroyed the evidence twice: the log a
# restart is diagnosing is the very one it overwrites, and both times the burst that prompted
# the restart was only in the bytes that got dropped. Keep the last three runs -- /tmp here is
# a 2.9 T volume, and even the aggregator's ~1 GB/day log costs nothing against that.
if [ -s "$LOG" ]; then
    mv -f "$LOG" "$LOG.$(date -u +%Y%m%d_%H%M%S)" || true
    # shellcheck disable=SC2012  # ls -t is the point: newest first, drop everything past 3
    ls -1t "$LOG".20*[0-9] 2>/dev/null | tail -n +4 | xargs -r rm -f
fi
# ⚠️ SUPERVISED, NOT BARE. The broker stops itself when the F-engine's frame 0 changes
# (gps_distributed_broker.py, exit 3): its anchor is latched once per process, so every seed
# it computes after a re-base is wrong, and the only way to a coherent anchor is through the
# startup path. On 2026-09-16 that condition was a log line only -- it fired 5960 times while
# seven of eight chains sat blind for twelve hours -- so the exit is now real and something
# has to bring the broker back.
#
# A `while` loop rather than a systemd unit because nothing on this host is a unit today and
# a user unit would not survive the weekly 03:03 reboot either; this at least makes a re-base
# self-healing. Exit 0 is a deliberate stop and ends the loop, so `pkill` still works.
nohup setsid bash -c '
    K="$1"; PY="$2"; CHAINS="$3"; LOG="$4"; SUP_PIDFILE="$5"; BROKER_PIDFILE="$6"; shift 6
    echo $$ > "$SUP_PIDFILE"
    trap "rm -f \"$SUP_PIDFILE\" \"$BROKER_PIDFILE\"" EXIT
    fast=0
    while :; do
        started=$(date +%s)
        "$PY" -u "$K/scripts/gnss/broker_multi.py" "$CHAINS" "$@" >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$BROKER_PIDFILE"
        wait $!
        rc=$?
        [ "$rc" -eq 0 ] && break
        ran=$(( $(date +%s) - started ))
        # ⚠️ A RESTART LOOP MUST NOT BE INFINITE. An exit 3 after hours of running is the
        # F-engine re-base this exists for and a relaunch fixes it. An exit seconds after
        # startup is a bad config or a bad tree, which relaunching cannot fix -- and a broker
        # that thrashes forever is harder to notice than one that is simply down.
        if [ "$ran" -ge 120 ]; then
            fast=0
        else
            fast=$((fast + 1))
        fi
        if [ "$fast" -ge 5 ]; then
            echo "[supervisor $(date -u +%H:%M:%S)] broker exited $rc after only ${ran}s, 5 times" \
                 "in a row -- NOT relaunching. This is a startup failure, not a re-base; fix it" \
                 "and run broker_restart.sh again." >> "$LOG"
            break
        fi
        if [ "$rc" -eq 3 ]; then
            echo "[supervisor $(date -u +%H:%M:%S)] broker exited 3 (F-engine frame0 changed)" \
                 "after ${ran}s; relaunching in 20 s so it re-latches the anchor" >> "$LOG"
        else
            echo "[supervisor $(date -u +%H:%M:%S)] broker exited $rc after ${ran}s;" \
                 "relaunching in 20 s (${fast}/5 rapid failures)" >> "$LOG"
        fi
        sleep 20
    done
' gnss-broker-supervisor "$K" "$PY" "$CHAINS" "$LOG" "$SUP_PIDFILE" "$BROKER_PIDFILE" "$@" \
    > /dev/null 2>&1 < /dev/null &
disown
sleep 10

if broker_running; then
    echo "broker up (log $LOG) chains: $CHAINS ${*:+args: $*}"
    grep -a "starting .* chain" "$LOG" | tail -1
else
    echo "FAILED to start -- last lines of $LOG:"
    tail -15 "$LOG"
    exit 1
fi
