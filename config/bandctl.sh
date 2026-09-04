#!/usr/bin/env bash
# bandctl -- reliable start/stop/status for the GNSS band launchers.
#
#   ./config/bandctl.sh start  {l1|l2c|l5|3band}
#   ./config/bandctl.sh stop   {l1|l2c|l5|3band}
#   ./config/bandctl.sh status [band]
#
# WHY THIS EXISTS (2026-07-17): a day of debugging lost real time to process-lifecycle
# chaos -- pgrep self-matches, launchers whose SIGTERM propagated back into the CALLING
# shell's process group (killing the killer mid-script, leaving the target alive), stray
# kotekans surviving pattern-kills, and monitors latching onto transient PIDs.
#
# The mechanism: `start` runs the launcher under setsid, so the launcher's PID == its
# PROCESS-GROUP id and the whole tree (kotekan + brokers + loggers + viewers) lives in
# that one group, recorded in a pidfile. `stop` sends SIGTERM to the GROUP (-PGID):
# one syscall, the entire tree, no patterns, no self-matches, and the caller's shell is
# in a different group so nothing bounces back. kotekan is never SIGKILLed (CUDA teardown
# must run; a -9 kills sibling GPU contexts) -- stop escalates to -9 only for NON-kotekan
# stragglers after a grace period.
set -uo pipefail
DIR=/tmp/gnss_run
mkdir -p "$DIR"
KOT=/home/lwlab/airspy_gps/kotekan
CMD="${1:-status}"
BAND="${2:-}"

launcher_for() {
    case "$1" in
        l1|l2c|l5) echo "config/run_band.sh $1" ;;
        3band)     echo "config/run_3band.sh" ;;
        *) echo ""; return 1 ;;
    esac
}

start() {
    local band="$1" pf="$DIR/$1.pgid"
    if [ -f "$pf" ] && kill -0 -- -"$(cat "$pf")" 2>/dev/null; then
        echo "band $band already running (pgid $(cat "$pf")) -- stop it first"
        return 1
    fi
    local launch
    launch=$(launcher_for "$band") || { echo "unknown band: $band"; return 1; }
    cd "$KOT"
    setsid nohup bash $launch > "$DIR/$band.log" 2>&1 < /dev/null &
    local pgid=$!
    echo "$pgid" > "$pf"
    echo "started $band: pgid $pgid, log $DIR/$band.log"
    sleep 20
    status "$band"
}

stop() {
    local band="$1" pf="$DIR/$1.pgid"
    [ -f "$pf" ] || { echo "no pidfile for $band ($pf)"; return 1; }
    local pgid
    pgid=$(cat "$pf")
    if ! kill -0 -- -"$pgid" 2>/dev/null; then
        echo "band $band (pgid $pgid) already gone"; rm -f "$pf"; return 0
    fi
    echo "stopping $band (pgid $pgid): SIGTERM to the group..."
    kill -TERM -- -"$pgid" 2>/dev/null
    # wait for the kotekan in this group to exit (clean CUDA teardown)
    for i in $(seq 1 30); do
        pgrep -g "$pgid" -f "build_cuda/kotekan/kotekan" >/dev/null 2>&1 || break
        sleep 2
    done
    if pgrep -g "$pgid" >/dev/null 2>&1; then
        if pgrep -g "$pgid" -f "build_cuda/kotekan/kotekan" >/dev/null 2>&1; then
            echo "WARNING: kotekan still alive after 60 s -- NOT escalating (never -9 a GPU"
            echo "kotekan); investigate: pgrep -g $pgid -a"
            return 1
        fi
        echo "escalating to SIGKILL for non-kotekan stragglers..."
        kill -KILL -- -"$pgid" 2>/dev/null
    fi
    rm -f "$pf"
    echo "band $band stopped"
}

status() {
    local bands="${1:-l1 l2c l5 3band}"
    for band in $bands; do
        local pf="$DIR/$band.pgid"
        if [ ! -f "$pf" ]; then continue; fi
        local pgid
        pgid=$(cat "$pf")
        if ! kill -0 -- -"$pgid" 2>/dev/null; then
            echo "$band: pidfile stale (pgid $pgid gone)"
            continue
        fi
        local kot
        kot=$(pgrep -g "$pgid" -f "build_cuda/kotekan/kotekan" | head -1)
        echo "$band: pgid $pgid | kotekan ${kot:-MISSING} | procs $(pgrep -g "$pgid" | wc -l) | log $DIR/$band.log"
    done
}

case "$CMD" in
    start)  start "$BAND" ;;
    stop)   stop "$BAND" ;;
    status) status ${BAND:+"$BAND"} ;;
    *) echo "usage: $0 {start|stop|status} [band]"; exit 1 ;;
esac
