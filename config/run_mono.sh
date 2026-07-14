#!/usr/bin/env bash
# Launch the MONOLITHIC ground-truth correlator on the live airspy + watch per-PRN SNR.
# The monolithic (GpsReplicaCorrelator) is the validated full-band FFT search/track; this
# tells us the SNR we SHOULD be getting, independent of the channelized live pipeline.
#
#   LAT=43.968697 LON=-79.252106 ALT=260 ./config/run_mono.sh                 # L1
#   GNSS_SIGNAL=L2C_CM LAT=.. LON=.. ALT=.. ./config/run_mono.sh             # L2C
#
# Run from the repo root. Ctrl-C to stop.
set -u
KOTEKAN=./build_mac/kotekan/kotekan
SIG=${GNSS_SIGNAL:-L1CA}
case "$SIG" in
  L1CA)   CFG=config/live_mono.yaml ;;
  L2C_CM) CFG=config/live_mono_l2c.yaml ;;
  L5_Q)   CFG=config/live_mono_l5.yaml ;;
  *) echo "GNSS_SIGNAL must be L1CA, L2C_CM or L5_Q"; exit 1 ;;
esac
LOG=/tmp/gpsmono.log

cleanup() { echo; echo "stopping..."; pkill -9 -f build_mac/kotekan/kotekan 2>/dev/null
            [ -n "${RUNCFG:-}" ] && [ "${RUNCFG:-}" != "$CFG" ] && rm -f "$RUNCFG"; exit 0; }
trap cleanup INT TERM
pkill -9 -f build_mac/kotekan/kotekan 2>/dev/null

# Refresh prns:/n_prn: to what's overhead now (same as run_live.sh).
RUNCFG="$CFG"
if [ -n "${LAT:-}" ] && [ -n "${LON:-}" ]; then
  RUNCFG="${TMPDIR:-/tmp}/mono_cfg_$$.yaml"
  python3 python/scripts/gnss/gps_visible_prns.py --lat "$LAT" --lon "$LON" \
      --alt "${ALT:-100}" --patch "$CFG" --out "$RUNCFG" \
      || { echo "PRN refresh failed -- using $CFG as-is"; cp "$CFG" "$RUNCFG"; }
fi
NPRN=$(grep -oE '^n_prn:[[:space:]]*[0-9]+' "$RUNCFG" | grep -oE '[0-9]+')
RECDIR=$(grep -oE 'base_dir:[[:space:]]*"[^"]*"' "$RUNCFG" | head -1 | sed -E 's/.*"([^"]*)".*/\1/')
RECDIR=${RECDIR:-/tmp/gpsmono}; mkdir -p "$RECDIR"; rm -f "$RECDIR"/* 2>/dev/null

echo "starting MONOLITHIC kotekan ($SIG, $CFG) -> $LOG"
$KOTEKAN -c "$RUNCFG" > "$LOG" 2>&1 &
sleep 4
echo "front end: $(curl -s --max-time 2 localhost:12048/airspy_in/adcstat | tr -d '\n ')"
echo "=== monolithic SNR (Ctrl-C to stop). This is the ground truth. ==="
python3 python/scripts/gnss/gps_mono_watch.py "$RECDIR" "${NPRN:-11}"
