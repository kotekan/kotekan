#!/usr/bin/env bash
# One-command launcher for the lean live airspy GNSS run + GPS sky viewer.
#
#   LAT=43.968697 LON=-79.252106 ALT=260 ./config/run_live.sh
#
# Starts kotekan (config/live.yaml: airspy -> PFB -> { search, valve -> track ->
# combiner -> recorded |A| }, + the GPS-only browser viewer), then the broker,
# then prints detections + combined level every few seconds. Watch the overhead
# sky + locked PRNs in a browser at
#   http://localhost:8080
# Recorded signal level lands in /tmp/gpslive/level_*.raw.  Ctrl-C to stop all.
#
# Run from the repo root (the dir containing config/ and build_mac/).
set -u
KOTEKAN=./build_mac/kotekan/kotekan
CFG=${CFG:-config/live.yaml}             # L1 lean valved distributed config
#   CFG=config/live_l2c.yaml  -> L2C (1227.6 MHz), CFG=config/live_wipe.yaml -> navwipe demo
# Derive the tracker stage names + the carrier straight from the config so any band/signal
# works unchanged: live.yaml -> track_00..11, live_l2c.yaml -> track_02..10 (covering subset),
# live_wipe.yaml -> a single "track". The broker accepts this comma list (it also expands
# {a..b} ranges itself, but a derived list needs no brace-quoting gymnastics).
TRK=${TRK:-$(grep -oE '^track[_0-9]*' "$CFG" | tr '\n' ',' | sed 's/,$//')}
# Loud warning if a requested tracker stage isn't actually in the config -- the classic
# trap is passing TRK=track to the distributed live.yaml (whose trackers are track_00..11):
# the broker POSTs to track/set_seeds, gets a 404, never seeds -> the trackers despread at
# cp=0 -> |A| stays pinned at the noise floor (~0.13) even though the SEARCH still detects.
if [[ "$TRK" != *"{"* ]]; then  # skip brace ranges (the broker expands those itself)
  for _t in ${TRK//,/ }; do
    grep -qE "^${_t}:" "$CFG" || echo "WARNING: tracker '$_t' is not a stage in $CFG -> set_seeds will 404 and |A| stays at noise. Omit TRK to auto-derive: $(grep -oE '^track[_0-9]*' "$CFG" | tr '\n' ' ')"
  done
fi
CARRIER_HZ=$(awk '/^[[:space:]]*freq:/{printf "%.0f", $2*1e6; exit}' "$CFG")  # for the broker almanac Doppler
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
echo "GPS sky viewer: http://localhost:8080"

# Almanac assist (predicted Doppler) when a location is given:
#   LAT=43.66 LON=-79.40 ./config/run_live.sh
# Watch /tmp/gpslive_broker.log for the predicted-vs-measured Doppler + clock bias.
ALM=""
if [ -n "${LAT:-}" ] && [ -n "${LON:-}" ]; then
  # --carrier-hz (from the config's freq) makes the predicted Doppler correct for whatever
  # band is loaded -- L1 ~1.5 kHz, L2 ~0.779x that. Essential for L2C, where the heavier
  # 20 ms acquisition really wants the broker to narrow the Doppler search.
  # --narrow-search (default on): also push the predicted Doppler to the SEARCH so it scans a
  # tight window per PRN instead of the blind grid (cheaper + more sensitive; the search collapses
  # toward a code-phase pin once the clock-freq bias is solved). Override the window via
  # SEARCH_MARGIN_HZ; set NARROW_SEARCH=0 to fall back to the blind grid (A/B baseline -- also the
  # safe mode if the almanac --doppler-sign is still unconfirmed, since a wrong sign narrows onto
  # the wrong half and finds nothing). DOPPLER_SIGN=-1 flips the predicted-Doppler sign.
  ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100} --carrier-hz ${CARRIER_HZ:-1575420000}"
  ALM="$ALM --doppler-sign ${DOPPLER_SIGN:-1}"
  if [ "${NARROW_SEARCH:-1}" != "0" ]; then
    ALM="$ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500}"
    _ns="+ narrowed search"
  else
    _ns="(blind search -- NARROW_SEARCH=0)"
  fi
  echo "almanac assist ON @ ($LAT, $LON), carrier ${CARRIER_HZ} Hz -- predicted-Doppler seeding $_ns"
else
  echo "almanac assist OFF (set LAT= LON= to enable predicted-Doppler seeding)"
fi
echo "starting broker (trackers: $TRK)..."
# --coast-budget: hold a visible sat (seed + forecast Doppler) through a signal dropout this many
# seconds before dropping it, so a radar sweep / brief fade doesn't lose the lock (raise it with a
# disciplined clock). Default 30 s -- inside the free-running-TCXO code-prediction horizon.
python3 $BROKER --detectors search --trackers "$TRK" --combiner combiner \
        --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-30} $ALM \
        > /tmp/gpslive_broker.log 2>&1 &
BPID=$!

echo "=== watching (Ctrl-C to stop) ==="
while true; do
  sleep 3
  curl -s localhost:12048/search/get_detections 2>/dev/null | python3 -c "
import sys,json,urllib.request
d=json.load(sys.stdin)
amp=json.load(urllib.request.urlopen('http://localhost:12048/combiner/get_status'))
# sig = detection significance (sigma above noise): deep nav-wiped SNR or the noise-debiased
# incoherent SNR. >>1 = a real lock; ~1 = noise (the raw |A| sits at the noise floor for weak sats).
sigf=lambda r: max(r.get('deep_snr',0) or 0, r.get('amp_snr',0) or 0)
ampf=lambda r: r.get('deep_amplitude') or r.get('unbiased_amplitude') or 0  # unbiased Â (not the noise-biased |A|)
top=sorted(amp,key=lambda r:-sigf(r))[:3]
adc=json.load(urllib.request.urlopen('http://localhost:12048/airspy_in/adcstat'))
hdr='rms=%.0f rail=%.2f'%(adc['rms'],adc['railfrac'])
if d:
    s='  DETECT: '+'; '.join('PRN%d dop%+.0f cp%.0f snr%.1f'%(x['prn'],x['doppler_hz'],x['code_phase_chips'],x['snr']) for x in sorted(d,key=lambda r:-r['snr'])[:4])
else:
    s='  searching...'
lvl='  sig(Â): '+(' '.join('PRN%d=%.1fσ(%.2f)'%(r['prn'],sigf(r),ampf(r)) for r in top if sigf(r)>0) or '--')
print('[%s]%s%s'%(hdr,s,lvl))
" 2>/dev/null || echo "  (waiting for pipeline...)"
done
