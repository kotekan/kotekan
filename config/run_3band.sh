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
# rec D (2026-07-19): the receiver IS GPSDO-disciplined -- resolve run_live's derived margins
# (CLK_WIDE_HZ etc.) from the gpsdo profile (0.06 ppm) instead of the 2.0 ppm cold default.
# (The C++ search stage still reads clock_profile from the yamls; that part rides the config
# generator so the generated files and this env stay in one place.)
export CLOCK=${CLOCK:-gpsdo}
CFG3=config/live_3band.yaml
KOTEKAN=${KOTEKAN:-./build_cuda/kotekan/kotekan}
export LAT=${LAT:-43.968697} LON=${LON:--79.252106} ALT=${ALT:-260}
LOG=/tmp/gps_3band.log

# Output dirs for the rawFileWrite record stages + phase dumps. A reboot wipes /tmp and a
# missing dir KILLS kotekan at stage construction ("Cannot open file" on the first record
# file) -- measured 2026-07-18 post-outage: every launch died before the airspys even
# opened, masquerading as a pipeline problem. The launcher owns its output dirs.
mkdir -p /tmp/gpswipe /tmp/gps_l2c_gpu /tmp/gps_l5_gpu /tmp/gnss_run

# Bands: original config (parsed by run_live for all derived quantities), stage prefix, TAG
# (log stems + l-a code-bias files -- SAME TAGs as run_band.sh so the converged per-dongle l-a
# estimates carry over), and the viewer HTTP/WS ports (must match the merged config's spawns).
BANDS="l1 l2c l5"
cfg_of()  { case "$1" in l1) echo config/live_l1_dual20.yaml;; l2c) echo config/live_l2c_gpu.yaml;; l5) echo config/live_l5_gpu.yaml;; esac; }
http_of() { case "$1" in l1) echo 8080;; l2c) echo 8081;; l5) echo 8082;; esac; }
# (viewer WS ports are baked into each config's spawn_pyviewer exec -- no env plumbing)

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

# ---- AUTO-ROTATE the prior run's capture (2026-07-21) ----
# The rawFileWrite record stages write ONE growing *_level_0000000.raw per band; a fresh
# kotekan OVERWRITES it, and the obs/status loggers would otherwise APPEND across runs
# (mixing pre/post-relaunch data in one file). Before launching, move the prior run's data
# aside to a timestamped dir so every relaunch preserves it -- essential for A/B tests
# (e.g. the airspy L1<->L5 swap) where the old capture IS the comparison baseline. Same
# filesystem (/tmp and /home are both nvme0n1p2), so `mv` is instant regardless of size.
# Runs AFTER teardown (nothing is writing) and BEFORE the launch. Skips a fresh boot
# (nothing to rotate). Override the destination root with ROTATE_ROOT; disable with
# ROTATE_ROOT=none.
ROTATE_ROOT=${ROTATE_ROOT:-/home/lwlab/gnss_archive}
if [ "$ROTATE_ROOT" != "none" ]; then
    stamp="run_$(date +%Y-%m-%d_%H%M%S)"
    dst="$ROTATE_ROOT/$stamp"
    rotated=0
    for d in /tmp/gpswipe /tmp/gps_l2c_gpu /tmp/gps_l5_gpu; do
        bn=$(basename "$d")
        # per-pattern (a compound `ls a b c` returns non-zero if ANY glob is empty, which
        # skipped record-LESS dirs like l2c -- fixed 2026-07-21 same session). compgen -G
        # tests one glob at a time.
        for pat in '*.raw' 'obs_*.jsonl' 'status_log*.jsonl'; do
            compgen -G "$d/$pat" >/dev/null || continue
            mkdir -p "$dst/$bn"
            mv $d/$pat "$dst/$bn/" 2>/dev/null && rotated=1
        done
    done
    if ls /tmp/gps_*_broker*.log >/dev/null 2>&1; then
        mkdir -p "$dst/broker_logs"; mv /tmp/gps_*_broker*.log "$dst/broker_logs/" 2>/dev/null; rotated=1
    fi
    if [ "$rotated" = 1 ]; then
        echo "$(date -Iseconds)  rotated by run_3band.sh launch" > "$dst/ROTATED"
        echo "rotated prior run -> $dst"
    else
        rmdir "$dst" 2>/dev/null   # fresh boot: nothing was there
    fi
fi

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
python3 config/gen_band_config.py --out-dir config || { echo "band-config generation FAILED"; exit 1; }
python3 config/gen_3band_config.py || { echo "config generation FAILED"; exit 1; }
echo "starting the merged kotekan ($CFG3) -> $LOG"
# Core dumps for the SILENT-KILL hunt (2026-07-28): kotekan has vanished twice with no crash
# line / no core / not-OOM. A self-crash (SIGSEGV/SIGABRT) leaves a symbolizable core here
# (the binary is not stripped) for a postmortem backtrace; a SIGKILL leaves none, which itself
# says "external/OOM, not a self-crash". core_pattern is a sysctl set out-of-band to
# /tmp/gnss_cores (bypassing apport, which discards cores for non-packaged binaries) -- see
# diag/silent_kill_forensics.md; this line just lifts the per-process limit so a core is
# actually written. Harmless when nothing crashes.
ulimit -c unlimited 2>/dev/null || true
$KOTEKAN -c $CFG3 -b 0.0.0.0:$PORT > $LOG 2>&1 &
sleep 8
if ! pgrep -f "[k]otekan .*live_3band" >/dev/null; then
    echo "kotekan DIED at startup:"; grep -iE "FATAL|ERROR" $LOG | head -5; exit 1
fi
for b in $BANDS; do
    # --max-time: this probe is what HUNG the launcher during THE WEDGE (an unbounded
    # cv.wait in adcstat_callback on a half-open device). Both ends are bounded now -- the
    # handler times out at adcstat_timeout_ms and returns 402 -- but a startup probe must
    # never be the thing that blocks a launch, so bound this end too.
    echo "  $b front end: $(curl -s --max-time 5 localhost:$PORT/${b}_airspy_in/adcstat \
                            | tr -d '\n ' | head -c 120)"
done

# ---- three control planes against the one instance ----
for b in $BANDS; do
    # (the carrier trim pre-shift env (TPC/TRIM_PRECOMP_CARRIER) is gone -- the broker
    # flags were deleted in the 07-19 audit A4 after the bench rejected both signs; the
    # BOOTSTRAP re-pull owns step recovery and --dop-continuous removes the steps.)
    # Model-primary seeding fleet-wide (2026-07-19 eve): --dop-continuous was already the
    # run_band.sh single-band default; the 3band transition silently dropped it and the
    # fleet ran frozen-seed for two days (the release/escape churn era -- see
    # docs/gnss_architecture_audit_2026-07-19.md). A/B replay legs + the 18:15 live L5
    # flip validate it (coh duty 78->87%, RELEASE 9->0 on the GPS leg). NOTE the literal
    # BROKER_EXTRA=$BX prefix: a ${X:+VAR=v} expansion is NOT an assignment post-expansion
    # (bash executes it as a command and the launch line dies -- measured 07-19 am).
    # --nav-bits-brdc 1 is part of THE NODE'S default (2026-07-27), not something an operator
    # has to remember. It was passed by hand via BROKER_EXTRA for days, which meant any launch
    # that forgot it silently ran the GPS peel without constructed nav bits -- a foot-gun with
    # no symptom except worse depth. The broker's OWN default stays 0 on purpose: it was turned
    # off on 2026-07-25 when the 30 s tables broke the seed push, and that is a real caveat for
    # other callers. The transport was since fixed (c839b3ca: parse outside seed_mtx +
    # NAVBITS-BAD counter, broker CPU 98->31%), and the node has run it on all day.
    BX="${BROKER_EXTRA:---dop-continuous --nav-bits-brdc 1}"
    # L2C-CM carries CNAV (FEC+CRC), not LNAV: route its nav_obs to the CNAV decoder or the
    # LNAV frame-sync churns on CNAV symbols forever (S3, 2026-07-28). A band property, appended
    # regardless of BROKER_EXTRA; the cnav path ignores --nav-bits-brdc (LNAV-only), so no clash.
    [ "$b" = l2c ] && BX="$BX --nav-decoder cnav"
    SKIP_KOTEKAN=1 STAGE_PREFIX=${b}_ PORT=$PORT TAG=gps_${b} \
        CFG=$(cfg_of $b) HTTP_PORT=$(http_of $b) \
        BROKER_EXTRA=$BX \
        bash config/run_live.sh > /tmp/gps_${b}_ctl.log 2>&1 &
    SUBPIDS="$SUBPIDS $!"
    echo "  $b control plane up (brokers/loggers; log /tmp/gps_${b}_ctl.log)"
done

# HONEST BANNER (2026-07-27). This used to print unconditionally, the instant the control
# planes were BACKGROUNDED -- so "3-band node running" appeared even when a broker had already
# died, and the launcher looked successful while the node was not. Check what we actually
# claim: kotekan alive, REST answering, and every control plane still up.
sleep 5
_bad=""
pgrep -f "[k]otekan .*live_3band" >/dev/null || _bad="$_bad kotekan"
_metrics=$(curl -s --max-time 5 "localhost:$PORT/metrics") || _bad="$_bad rest"
for pid in $SUBPIDS; do
    kill -0 "$pid" 2>/dev/null || _bad="$_bad ctl-plane($pid)"
done
for b in $BANDS; do
    pgrep -f "[g]ps_distributed_broker.py.*${b}_" >/dev/null || _bad="$_bad ${b}-broker"
done
# ARE THE FRONT ENDS ACTUALLY STREAMING? (2026-07-27, fix (a)/(c) of the wedge post-mortem.)
# Every airspy call can succeed against a HALF-OPEN device -- claimed, configured, start_rx
# returns SUCCESS -- and still never deliver a sample. Before this check the band was simply
# absent, with no error anywhere, behind a banner claiming the node was up; the outage that
# taught us this ran for hours. airspyInput's stream watchdog publishes the verdict, and
# /metrics is the right place to read it because it answers even when /adcstat cannot.
for b in $BANDS; do
    case "$(printf '%s\n' "$_metrics" \
            | grep "^kotekan_airspy_streaming{stage_name=\"/${b}_airspy_in\"}" \
            | awk '{print $2}')" in
        1|1.0*) ;;                                  # streaming
        "")     _bad="$_bad ${b}-airspy(no-watchdog)" ;;  # stage never started
        *)      _bad="$_bad ${b}-airspy(NOT-STREAMING/half-open)" ;;
    esac
done
if [ -n "$_bad" ]; then
    echo "=== 3-band node came up DEGRADED -- not running:$_bad ==="
    echo "    logs: $LOG, /tmp/gps_{l1,l2c,l5}_ctl.log"
else
    echo "=== 3-band node running: kotekan :$PORT, unified viewer :8080 -- Ctrl-C stops all ==="
fi
wait