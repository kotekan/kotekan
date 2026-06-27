#!/usr/bin/env bash
# One-command launcher for the full live airspy GNSS run + diagnostic viewer.
#
#   ./config/run_live.sh
#
# Starts kotekan (config/live_full.yaml: airspy -> PFB -> split -> transmit ->
# gather -> search, + trackers -> combiner -> recorded |A|, + the fine-spectrum
# power viewer), then the broker, then prints detections + combined level every
# few seconds. Watch the power waterfall + ADC stats in a browser at
#   http://localhost:8080
# Recorded signal level lands in /tmp/gpslive/level_*.raw.  Ctrl-C to stop all.
#
# Run from the repo root (the dir containing config/ and build_mac/).
set -u
KOTEKAN=./build_mac/kotekan/kotekan
CFG=${CFG:-config/live_full.yaml}        # CFG=config/live_lowrate.yaml for the 5 MSPS test
# NB: the brace range MUST live in its own var -- nesting it inside ${TRK:-...} makes
# bash match ${ to the first } and append a stray '}' (e.g. TRK=track -> "track}",
# which 404s every set_seeds). Keep the default brace-free of the ${} wrapper.
TRK_DEFAULT='track_{12..28}'             # tracker range (track_{02..18} for low-rate)
TRK=${TRK:-$TRK_DEFAULT}
BROKER=python/scripts/gps_distributed_broker.py
LOG=/tmp/gpslive.log

cleanup() { echo; echo "stopping..."; kill "${BPID:-}" 2>/dev/null
            pkill -9 -f build_mac/kotekan/kotekan 2>/dev/null
            pkill -9 -f livebeam_server 2>/dev/null
            [ -n "${RUNCFG:-}" ] && [ "${RUNCFG:-}" != "$CFG" ] && rm -f "$RUNCFG"
            exit 0; }
trap cleanup INT TERM

pkill -9 -f build_mac/kotekan/kotekan 2>/dev/null; pkill -9 -f livebeam_server 2>/dev/null

# Refresh the search PRN list to what's actually overhead NOW (the constellation rotates
# ~half an orbit in ~8 h, so a hardcoded list goes stale -> zero detections). Needs
# LAT/LON; patches a temp copy so the committed config is untouched. Launch RUNCFG.
RUNCFG="$CFG"
if [ -n "${LAT:-}" ] && [ -n "${LON:-}" ]; then
  # PID-based temp (portable: BSD/macOS mktemp rejects a .yaml suffix after the X's).
  RUNCFG="${TMPDIR:-/tmp}/live_cfg_$$.yaml"
  if ! python3 python/scripts/gps_visible_prns.py --lat "$LAT" --lon "$LON" \
         --alt "${ALT:-100}" --patch "$CFG" --out "$RUNCFG"; then
    echo "PRN refresh failed (network/time?) -- using $CFG as-is"; cp "$CFG" "$RUNCFG"
  fi
fi

# Record dir comes from the config's rawFileWrite base_dir (rawFileWrite open()s with
# O_CREAT -> makes the file but NOT the dir; a missing dir exit()s kotekan at the first
# write -> "waiting for pipeline" forever). Create + clean whatever the config points at.
RECDIR=$(grep -oE 'base_dir:[[:space:]]*"[^"]*"' "$RUNCFG" | head -1 | sed -E 's/.*"([^"]*)".*/\1/')
RECDIR=${RECDIR:-/tmp/gpslive}
mkdir -p "$RECDIR"; rm -f "$RECDIR"/* 2>/dev/null
echo "recording to $RECDIR"
sleep 1

echo "starting kotekan ($CFG) -> $LOG"
$KOTEKAN -c $RUNCFG > $LOG 2>&1 &
sleep 4
echo "front end: $(curl -s localhost:12048/airspy_in/adcstat | tr -d '\n ')"
echo "browser waterfall + ADC: http://localhost:8080"

# Almanac assist (predicted Doppler) when a location is given:
#   LAT=43.66 LON=-79.40 ./config/run_live.sh
# Watch /tmp/gpslive_broker.log for the predicted-vs-measured Doppler + clock bias.
ALM=""
if [ -n "${LAT:-}" ] && [ -n "${LON:-}" ]; then
  ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100}"
  echo "almanac assist ON @ ($LAT, $LON) -- predicted-Doppler seeding"
else
  echo "almanac assist OFF (set LAT= LON= to enable predicted-Doppler seeding)"
fi
echo "starting broker..."
python3 $BROKER --detectors search --trackers "$TRK" --combiner combiner \
        --acquire-snr 6 --interval 0.2 $ALM > /tmp/gpslive_broker.log 2>&1 &
BPID=$!

echo "=== watching (Ctrl-C to stop) ==="
while true; do
  sleep 3
  curl -s localhost:12048/search/get_detections 2>/dev/null | python3 -c "
import sys,json,urllib.request
d=json.load(sys.stdin)
amp=json.load(urllib.request.urlopen('http://localhost:12048/combiner/get_status'))
deep=any(r.get('deep_amplitude',0)>0 for r in amp)
top=sorted(amp,key=lambda r:-(r.get('deep_amplitude',0) or r['amplitude']))[:3]
adc=json.load(urllib.request.urlopen('http://localhost:12048/airspy_in/adcstat'))
hdr='rms=%.0f rail=%.2f'%(adc['rms'],adc['railfrac'])
if d:
    s='  DETECT: '+'; '.join('PRN%d dop%+.0f cp%.0f snr%.1f'%(x['prn'],x['doppler_hz'],x['code_phase_chips'],x['snr']) for x in sorted(d,key=lambda r:-r['snr'])[:4])
else:
    s='  searching...'
if deep:
    lvl='  |A|/deep: '+' '.join('PRN%d=%.2f/%.2f'%(r['prn'],r['amplitude'],r.get('deep_amplitude',0)) for r in top if r['amplitude']>0)
else:
    lvl='  |A|: '+' '.join('PRN%d=%.2f'%(r['prn'],r['amplitude']) for r in top if r['amplitude']>0) or '  |A|: --'
print('[%s]%s%s'%(hdr,s,lvl))
" 2>/dev/null || echo "  (waiting for pipeline...)"
done
