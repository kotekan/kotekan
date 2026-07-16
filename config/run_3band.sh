#!/usr/bin/env bash
# THE CHORD-LIKE NODE: one kotekan, three bands (L1/L2C/L5), seven constellation chains.
#
#   ./config/run_3band.sh                      (Ctrl-C stops everything)
#
# Architecture (see gen_3band_config.py): the merge is at the INSTANCE level -- nothing crosses
# frequency, so per-band chains run side by side in one process, each on its own airspy with its
# own absolute-time anchor (shared GPSDO -> constant inter-band offsets, absorbed per band by the
# sky clock solve). The control plane reuses run_live.sh's battle-tested broker/logger machinery
# three times in SKIP_KOTEKAN + STAGE_PREFIX mode: each invocation PARSES its ORIGINAL per-band
# config (so signals, hop rates, chip tables, TLE groups, clock profile all stay correct) but
# talks to the ONE merged kotekan with l1_/l2c_/l5_-prefixed stage names.
#
# Fallback: the three separate per-band kotekans (run_band.sh l1|l2c|l5) work unchanged.
set -u
cd "$(dirname "$0")/.."

PORT=${PORT:-12048}
CFG3=config/live_3band.yaml
KOTEKAN=${KOTEKAN:-./build_cuda/kotekan/kotekan}
export LAT=${LAT:-43.968697} LON=${LON:--79.252106} ALT=${ALT:-260}
LOG=/tmp/gps_3band.log

# Bands: original config (parsed by run_live for all derived quantities), stage prefix, TAG
# (log stems + l-a code-bias files -- SAME TAGs as run_band.sh so the converged per-dongle l-a
# estimates carry over), and the viewer HTTP/WS ports (must match the merged config's spawns).
BANDS="l1 l2c l5"
cfg_of()  { case "$1" in l1) echo config/live_l1_dual20.yaml;; l2c) echo config/live_l2c_gpu.yaml;; l5) echo config/live_l5_gpu.yaml;; esac; }
http_of() { case "$1" in l1) echo 8080;; l2c) echo 8081;; l5) echo 8082;; esac; }
ws_of()   { case "$1" in l1) echo 8539;; l2c) echo 8639;; l5) echo 8739;; esac; }

# ---- teardown of anything stale (we own the whole box's GNSS control plane here) ----
# Graceful kotekan stop -- NEVER -9 first (GPU-context death would take siblings down; here
# there are no siblings, but the rule stands).
kill_kotekan() {
    local pids; pids=$(pgrep -f "$1"); [ -z "$pids" ] && return
    kill -TERM $pids 2>/dev/null
    for _ in $(seq 20); do sleep 0.5; pgrep -f "$1" >/dev/null || return; done
    kill -9 $pids 2>/dev/null
}
echo "sweeping stale GNSS processes..."
# stale single-band launchers first (their traps would murder our processes hours later)
for f in /tmp/run_live_launcher_*.pid; do
    [ -f "$f" ] || continue
    oldpid=$(cat "$f" 2>/dev/null)
    [ -n "$oldpid" ] && grep -qa run_live /proc/"$oldpid"/cmdline 2>/dev/null && kill -9 "$oldpid" 2>/dev/null
    rm -f "$f"
done
kill_kotekan "[k]otekan .*-b 0.0.0.0"
pkill -9 -f "[g]ps_distributed_broker.py" 2>/dev/null
pkill -9 -f "[g]ps_status_logger.py" 2>/dev/null
pkill -9 -f "[g]nss_observables.py" 2>/dev/null
pkill -9 -f "[l]ivebeam_server.py" 2>/dev/null
sleep 1

# ---- our own pidfile + cleanup ----
LAUNCHPID=/tmp/run_3band_launcher.pid
if [ -f "$LAUNCHPID" ]; then
    oldpid=$(cat "$LAUNCHPID" 2>/dev/null)
    [ -n "$oldpid" ] && [ "$oldpid" != "$$" ] && \
        grep -qa run_3band /proc/"$oldpid"/cmdline 2>/dev/null && kill -9 "$oldpid" 2>/dev/null
fi
echo $$ > "$LAUNCHPID"
SUBPIDS=""
cleanup() {
    echo; echo "stopping the 3-band node..."
    # TERM the three run_live control planes first: their traps kill their own brokers/loggers
    # by PID (and, in SKIP_KOTEKAN mode, leave OUR kotekan alone).
    kill -TERM $SUBPIDS 2>/dev/null
    sleep 3
    kill_kotekan "[k]otekan .*live_3band"
    pkill -9 -f "[l]ivebeam_server.py" 2>/dev/null
    [ "$(cat "$LAUNCHPID" 2>/dev/null)" = "$$" ] && rm -f "$LAUNCHPID"
    exit 0
}
trap cleanup INT TERM

# ---- regenerate + launch the merged instance ----
python3 config/gen_3band_config.py || { echo "config generation FAILED"; exit 1; }
echo "starting the merged kotekan ($CFG3) -> $LOG"
$KOTEKAN -c $CFG3 -b 0.0.0.0:$PORT > $LOG 2>&1 &
sleep 8
if ! pgrep -f "[k]otekan .*live_3band" >/dev/null; then
    echo "kotekan DIED at startup:"; grep -iE "FATAL|ERROR" $LOG | head -5; exit 1
fi
for b in $BANDS; do
    echo "  $b front end: $(curl -s localhost:$PORT/${b}_airspy_in/adcstat | tr -d '\n ' | head -c 120)"
done

# ---- three control planes against the one instance ----
for b in $BANDS; do
    SKIP_KOTEKAN=1 STAGE_PREFIX=${b}_ PORT=$PORT TAG=gps_${b} \
        CFG=$(cfg_of $b) HTTP_PORT=$(http_of $b) WS_PORT=$(ws_of $b) \
        bash config/run_live.sh > /tmp/gps_${b}_ctl.log 2>&1 &
    SUBPIDS="$SUBPIDS $!"
    echo "  $b control plane up (brokers/loggers; log /tmp/gps_${b}_ctl.log)"
done

echo "=== 3-band node running: kotekan :$PORT, viewers :8080/:8081/:8082 -- Ctrl-C stops all ==="
wait