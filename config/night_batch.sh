#!/usr/bin/env bash
# NIGHT BATCH (2026-07-17 -> weekend): one scheduled ~4-min interruption of the 3-band soak.
#   1. TERM run_3band.sh (its trap cascades: kotekan + brokers + loggers + viewers)
#   2. 60 s airspy_rx L1 grab at 20 MSPS -- tonight ~03:10 EDT catches C31 ~31 / C34 ~76 /
#      C38 ~55 deg (from last night's LOGGED passes, sidereal -4 min; celestrak PRN labels
#      for C38 are WRONG, do not trust TLE here) -> first-ever C38 overlay test + strong C31.
#   3. relaunch run_3band.sh (brokers pick up the v3 hardened carrier loop via run_live.sh).
# Schedule with:  echo "config/night_batch.sh" | at 03:10   (or nohup + sleep-until).
set -uo pipefail
STAMP="$(date +%Y-%m-%d_%H%M)"
DIR=/home/lwlab/airspy_gps/captures
KOT=/home/lwlab/airspy_gps/kotekan
LOG=/tmp/night_batch_${STAMP}.log
exec >> "$LOG" 2>&1
echo "=== night batch $STAMP ==="

# 1. stop the node via bandctl (pidfile group-TERM; NEVER -9 a kotekan sharing the GPU).
# Pre-bandctl this pattern-killed the launcher, which left bandctl's pidfile stale and the
# relaunched node UNMANAGED (2026-07-18 morning: bandctl status lied, recovery needed a
# manual pgid hunt). Stop whichever band is up; fall back to the old pattern-kill only if
# no pidfile matches (e.g. a hand-launched node).
STOPPED_BAND=""
for band in 3band l1 l2c l5; do
    if [ -f /tmp/gnss_run/$band.pgid ] && kill -0 -- -"$(cat /tmp/gnss_run/$band.pgid)" 2>/dev/null; then
        "$KOT/config/bandctl.sh" stop $band && STOPPED_BAND=$band
        break
    fi
done
if [ -z "$STOPPED_BAND" ]; then
    PID=$(pgrep -f "bash.*run_3band[.]sh" | head -1 || true)
    [ -n "${PID:-}" ] && kill -TERM "$PID"
    for i in $(seq 1 30); do pgrep -f "kotekan.*live_3band" >/dev/null || break; sleep 2; done
fi
pgrep -f "kotekan.*live_3band" >/dev/null && { echo "kotekan would not stop; ABORT (no grab)"; exit 1; }
sleep 3

# 2. the grab (L1 dongle by serial; 60 s x 20 MSPS real x int16 = 2.4 GB)
OUT="$DIR/l1_20msps_${STAMP}.bin"
airspy_rx -s 0x03A464C837525967 -t 3 -a 0 -f 1575.42 -l 11 -m 11 -v 11 -b 0 -n 1200000000 -r "$OUT"
cat > "${OUT%.bin}.txt" <<EOF
capture start: $(date -u -d '-70 seconds' +'%Y-%m-%d %H:%M:%S') UTC (approx; file mtime = end)
format: int16 real 20 MSPS, IF=Fs/4=5 MHz, bias EXTERNAL (biast off), gains l11/m11/v11
purpose: B1C overlay tests -- C38 (first sky test) + C31 (soft-scan confirmation)
EOF
ls -lh "$OUT"

# 2b. rotate + gzip the observables/status logs while the writers are down (nightly ~7.8 GB
# of JSONL -> ~1 GB archived; fresh files on relaunch keep every offline tool's tail-seek
# fast). Probe rows are KEPT (the pedestal calibration needs below-horizon noise references);
# the volume win is compression + the 2 s cadence (OBS_INTERVAL in run_live.sh).
ARC=/home/lwlab/airspy_gps/captures/obs_archive/$(date +%Y-%m-%d)
mkdir -p "$ARC"
for f in /tmp/gpswipe/obs_*.jsonl /tmp/gpswipe/status_log*.jsonl \
         /tmp/gps_l2c_gpu/obs_*.jsonl /tmp/gps_l2c_gpu/status_log*.jsonl \
         /tmp/gps_l5_gpu/obs_*.jsonl /tmp/gps_l5_gpu/status_log*.jsonl; do
    [ -f "$f" ] && mv "$f" "$ARC/$(basename "${f%.jsonl}")_$(date +%H%M).jsonl"
done
nohup gzip "$ARC"/*.jsonl > /dev/null 2>&1 &

# 3. relaunch the node under bandctl (records the pgid; the morning session manages it
# with bandctl status/stop instead of hunting the process tree). Relaunch the SAME band
# we stopped (default 3band).
"$KOT/config/bandctl.sh" start "${STOPPED_BAND:-3band}" \
    && echo "node relaunched OK under bandctl (${STOPPED_BAND:-3band})" \
    || echo "RELAUNCH FAILED -- check /tmp/gnss_run/${STOPPED_BAND:-3band}.log"
