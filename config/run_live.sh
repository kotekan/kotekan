#!/usr/bin/env bash
# One-command launcher for the lean live airspy GNSS run + GPS sky viewer.
#
#   LAT=43.968697 LON=-79.252106 ALT=260 ./config/run_live.sh
#
# Starts kotekan (config/live_l1.yaml: airspy -> PFB -> { search, valve -> track ->
# combiner -> recorded |A| }, + the GPS-only browser viewer), then the broker,
# then prints detections + combined level every few seconds. Watch the overhead
# sky + locked PRNs in a browser at
#   http://localhost:8080
# Recorded signal level lands in /tmp/gpslive/level_*.raw.  Ctrl-C to stop all.
#
# Run from the repo root (the dir containing config/ and build/).
set -u
KOTEKAN=${KOTEKAN:-./build/kotekan/kotekan}
CFG=${CFG:-config/live_l1.yaml}          # L1 lean valved distributed config
#   CFG: live_l2c.yaml -> L2C (1227.6 MHz); live_l5.yaml -> L5 (1176.45 MHz); live_l5_wipe.yaml ->
#        L5 Q5 pilot DEEP overlay-wipe (rolling, NH20 wiped in the combiner -> deep |A| past 1 ms);
#        live_l1c.yaml -> L1C-P (1575.42 MHz, BOC(1,1) Block-III civil pilot, track_00..09);
#        live_l1c_wipe.yaml -> L1C-P DEEP per-PRN L1CO overlay-wipe (rolling, ~36 s window);
#        live_l1_wipe.yaml -> L1 navwipe demo
# Derive the tracker stage names + the carrier straight from the config so any band/signal
# works unchanged: live_l1.yaml -> track_00..11, live_l2c.yaml -> track_02..10 (covering subset),
# live_l1_wipe.yaml -> a single "track". The broker accepts this comma list (it also expands
# {a..b} ranges itself, but a derived list needs no brace-quoting gymnastics).
# GPU-chain configs name the seed target EXPLICITLY (cudaGnssTrack's seed_endpoint:
# "/track/set_seeds" inside a cudaProcess commands: list, where the ^track stage-name grep
# can't see it) -- collect those first; the stage-name grep remains for the CPU configs.
# 2026-07-12: the stage-name-only grep launched the GPS broker of live_l1_dual20.yaml with
# ZERO trackers -- seeds never POSTed, GPS tracking silently dead while search looked normal.
# Skip the gal_/bds_ constellations: their own broker sections below seed them.
TRK=${TRK:-$(
  { grep -oE 'seed_endpoint:[[:space:]]*"/[a-z_0-9]+/set_seeds"' "$CFG" \
      | sed -E 's|.*"/([a-z_0-9]+)/set_seeds"|\1|' | grep -vE '^(gal|bds)_';
    grep -oE '^track[_0-9]*' "$CFG"; } | sort -u | tr '\n' ',' | sed 's/,$//')}
# Also hand any GnssVoltagePeel stage to the broker's --trackers: it POSTs the same consensus seeds
# {cp, Doppler, cp_rate(+l-a)} to /<peel>/set_seeds, so the peel reconstructs + subtracts each sat
# on-peak. Its residual feeds search_resid (peeled sats should drop). Chain L5/L1C peels the same way.
PEEL=$(grep -oE '^[a-z_0-9]+: \{ kotekan_stage: GnssVoltagePeel' "$CFG" | grep -oE '^[a-z_0-9]+' | tr '\n' ',' | sed 's/,$//')
[ -n "$PEEL" ] && TRK="${TRK:+$TRK,}$PEEL" && echo "voltage-peel stage(s) seeded by the broker: $PEEL"
# L2C CL pilot trackers (per-stage signal: GPS_L2C_CL): turn on the broker's time-assist -- it
# lifts each seed's cp by k*10230 with the CL segment k computed from the capture's absolute UTC
# anchor (airspy /adcstat utc0_sample0) + almanac range. Needs the almanac (LAT/LON).
CLA=""
grep -qE 'signal: GPS_L2C_CL' "$CFG" && CLA="--cl-assist --carrier-gain ${CARRIER_GAIN:-0.2}" \
  && echo "L2C CL time-assist ON (k from capture UTC + almanac range) + shared carrier loop"
# SHARED CARRIER LOOP: any config whose tracker sets carrier_shared: true runs pure feed-forward
# (seed + almanac Doppler-rate ramp) + an NCO fed the broker's carrier_trim_hz. The combiner measures
# the residual carrier as a bit-robust phase-SLOPE fit over the deep window and the broker integrates
# it at LOW bandwidth (--carrier-gain) -> residual driven <1 Hz -> the deep coheres a full 1 s+
# (REPLAY-VALIDATED: PRN19 80 sigma@1 s ON vs 7 sigma OFF). Auto-enabled here (CL adds its own
# --carrier-gain above, so skip it there). Tune via CARRIER_GAIN / CARRIER_MAX_HZ / CARRIER_LEAK env.
CARG=""
if grep -qE 'carrier_shared:[[:space:]]*true' "$CFG" && ! grep -qE 'signal: GPS_L2C_CL' "$CFG"; then
  # leak 0.0005 (was 0.005): the leaky integrator's equilibrium leaves a STANDING residual
  # = trim*leak/gain. GPS held seeds sit up to their 100 Hz fence off -> leak 0.005 left
  # ~0.5-1 Hz standing carrier -> 300+ deg of phase wrap across a 1 s deep window, capping
  # the GPS ladder at 0.125-0.25 s while E/C (10/25 Hz fences) held 1 s (2026-07-12; the
  # offline ladder on the raw grab proved the LO coheres to 2 s+, sqrt-T to 494 sigma).
  # 10x lower leak -> bias ~0.1 Hz worst-case; the slip-proof two-stage resid estimator
  # (GnssCoherentCombiner::carrier_resid_hz) keeps the trim from random-walking instead.
  CARG="--carrier-gain ${CARRIER_GAIN:-0.5} --carrier-max-hz ${CARRIER_MAX_HZ:-100} --carrier-leak ${CARRIER_LEAK:-0.0005}"
  echo "shared carrier loop ON (combiner slope-fit resid -> --carrier-gain ${CARRIER_GAIN:-0.5} -> tracker NCO)"
fi
# Loud warning if a requested tracker stage isn't actually in the config -- the classic
# trap is passing TRK=track to the distributed live_l1.yaml (whose trackers are track_00..11):
# the broker POSTs to track/set_seeds, gets a 404, never seeds -> the trackers despread at
# cp=0 -> |A| stays pinned at the noise floor (~0.13) even though the SEARCH still detects.
# A GPU-chain tracker registers /<name>/set_seeds from its seed_endpoint, NOT from a
# top-level stage name (the stage is the enclosing cudaProcess), so accept EITHER spelling
# here -- checking stage names alone cried wolf on every GPU config.
if [[ "$TRK" != *"{"* ]]; then  # skip brace ranges (the broker expands those itself)
  for _t in ${TRK//,/ }; do
    grep -qE "^${_t}:" "$CFG" || grep -qE "seed_endpoint:[[:space:]]*\"/${_t}/set_seeds\"" "$CFG" \
      || echo "WARNING: tracker '$_t' is neither a stage nor a seed_endpoint in $CFG -> set_seeds will 404 and |A| stays at noise. Omit TRK to auto-derive: $(grep -oE '^track[_0-9]*' "$CFG" | tr '\n' ' ')"
  done
fi
CARRIER_HZ=$(awk '/^[[:space:]]*freq:/{printf "%.0f", $2*1e6; exit}' "$CFG")  # for the broker almanac Doppler
# F-engine hop rate = sample_rate / fft_len (fft_len = 2*spectrum_length, real input). The broker
# needs this to convert the fitted cp slope (chips/hop) into chips/s AND for the code-rate clock-bias
# estimate (l-a). Derived from the config so any band works. (L5's 10.23 Mcps code also needs
# --chip-rate-hz 10230000 --code-length 10230 for the code-bias -- add those to BROKER_EXTRA for L5.)
HOPS_PER_SEC=$(awk '/^[[:space:]]*sample_rate:/{sr=$2} /^[[:space:]]*spectrum_length:/{n=$2} END{if(sr>0&&n>0)printf "%.4f", sr/(2*n)}' "$CFG")
# Signal-specific chip rate + primary-code length for the broker's cp-fit and code-rate clock-bias,
# parsed straight from the C++ SignalDescriptor table (the single source of truth) keyed by the
# config's signal:. So any band works with no per-band flags -- L1CA 1.023e6/1023, L1C_P 1.023e6/
# 10230, L2C_CM 511.5e3/10230, L5_Q 10.23e6/10230. Unknown/absent -> broker defaults (L1 C/A).
SIGNAL=$(awk '/^[[:space:]]*signal:/{print $2; exit}' "$CFG")
SIGHDR="lib/stages/gnss/gnssSignal.hpp"
CHIP_HZ=""; CODELEN=""
if [ -n "${SIGNAL:-}" ] && [ -f "$SIGHDR" ]; then
  CHIP_HZ=$(awk -F, -v s="$SIGNAL" '$1 ~ ("^[[:space:]]*\"" s "\"$"){c=$3;gsub(/[^0-9.eE+-]/,"",c);print c;exit}' "$SIGHDR")
  CODELEN=$(awk -F, -v s="$SIGNAL" '$1 ~ ("^[[:space:]]*\"" s "\"$"){l=$4;gsub(/[^0-9]/,"",l);print l;exit}' "$SIGHDR")
fi
# --- Receiver clock profile (clockProfile.hpp is the canonical table) ---------------------------
# One knob for clock quality across airspy TCXO ... GPSDO ... maser. The search STAGE reads
# clock_profile from the config directly (sizes the Doppler grid from accuracy_ppm); here we resolve
# the SAME profile to (a) cap integration_length from coherence_s and (b) hand the broker a matching
# cold per-PRN margin. Name = CLOCK= env OR the config's clock_profile.name (default auto); explicit
# accuracy_ppm / coherence_s in the block override the preset.
CLKHDR="lib/stages/gnss/clockProfile.hpp"
_clkline=$(grep -oE '^clock_profile:[^#]*' "$CFG" | head -1)
CLK_NAME=${CLOCK:-$(echo "$_clkline" | grep -oE 'name:[[:space:]]*[A-Za-z]+' | awk '{print $NF}')}
CLK_NAME=${CLK_NAME:-auto}
CLK_ACC_CFG=$(echo "$_clkline" | grep -oE 'accuracy_ppm:[[:space:]]*[0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$')
CLK_COH_CFG=$(echo "$_clkline" | grep -oE 'coherence_s:[[:space:]]*[0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$')
if [ -f "$CLKHDR" ]; then
  read -r CLK_ACC CLK_COH < <(awk -v p="$CLK_NAME" \
    '$0 ~ ("name == \"" p "\"") && match($0, /\{[0-9eE.+-]+,[[:space:]]*[0-9eE.+-]+\}/){
       s=substr($0,RSTART+1,RLENGTH-2); gsub(/[[:space:]]/,"",s); split(s,a,","); print a[1],a[2]; exit}' "$CLKHDR")
fi
CLK_ACC=${CLK_ACC_CFG:-${CLK_ACC:-2.0}}   # auto/unknown -> conservative 2.0 ppm / 0.2 s
CLK_COH=${CLK_COH_CFG:-${CLK_COH:-0.2}}
# Broker COLD per-PRN margin from the accuracy bound (clock offset ~ accuracy*f_c, + 500 Hz almanac).
CLK_WIDE_HZ=$(awk -v a="$CLK_ACC" -v f="${CARRIER_HZ:-1575420000}" 'BEGIN{printf "%.0f", a*1e-6*f + 500}')
# Coherent-integration cap (records; 1 record = 1 primary code period = CODELEN/CHIP_HZ). Applied only
# for a COMMITTED profile (named non-auto clock OR explicit coherence_s) and only if it TIGHTENS the
# config -- pure auto leaves integration_length alone (v2 will measure + grow it live).
CLK_MAXINT=""
if { [ "$CLK_NAME" != auto ] || [ -n "$CLK_COH_CFG" ]; } && [ -n "${CHIP_HZ:-}" ] && [ -n "${CODELEN:-}" ]; then
  CLK_MAXINT=$(awk -v c="$CLK_COH" -v ch="$CHIP_HZ" -v cl="$CODELEN" 'BEGIN{rp=cl/ch; if(rp>0)printf "%d",int(c/rp)}')
fi
echo "clock profile '$CLK_NAME': accuracy ${CLK_ACC} ppm, coherence ${CLK_COH} s -> broker cold margin +-${CLK_WIDE_HZ} Hz${CLK_MAXINT:+, integ cap ${CLK_MAXINT} rec}"
BROKER=python/scripts/gnss/gps_distributed_broker.py
LOG=/tmp/gpslive.log

cleanup() { echo; echo "stopping..."; kill "${BPID:-}" "${LOGPID:-}" "${GALPID:-}" "${GALLOGPID:-}" "${BDSPID:-}" "${BDSLOGPID:-}" 2>/dev/null
            pkill -9 -f kotekan/kotekan 2>/dev/null
            pkill -9 -f livebeam_server 2>/dev/null
            pkill -9 -f gps_status_logger 2>/dev/null
            [ -n "${RUNCFG:-}" ] && [ "${RUNCFG:-}" != "$CFG" ] && rm -f "$RUNCFG"
            exit 0; }
trap cleanup INT TERM

pkill -9 -f kotekan/kotekan 2>/dev/null; pkill -9 -f livebeam_server 2>/dev/null

# Refresh the search PRN list to what's actually overhead NOW (the constellation rotates
# ~half an orbit in ~8 h, so a hardcoded list goes stale -> zero detections). Needs
# LAT/LON; patches a temp copy so the committed config is untouched. Launch RUNCFG.
# SKIP for require_hint configs: those carry the whole constellation and let the search self-select
# the visible set from the broker's hints (which follow the sky), so pinning a visible-now list would
# only shrink the candidate set and stop new risers from ever being acquired.
RUNCFG="$CFG"
if grep -qE 'require_hint:[[:space:]]*true' "$CFG"; then
  echo "require_hint config: search self-selects visible PRNs from broker hints (no visible-now patch)"
elif [ -n "${LAT:-}" ] && [ -n "${LON:-}" ]; then
  # PID-based temp (portable: BSD/macOS mktemp rejects a .yaml suffix after the X's).
  RUNCFG="${TMPDIR:-/tmp}/live_cfg_$$.yaml"
  if ! python3 python/scripts/gnss/gps_visible_prns.py --lat "$LAT" --lon "$LON" \
         --alt "${ALT:-100}" ${SIGNAL:+--signal "$SIGNAL"} --patch "$CFG" --out "$RUNCFG"; then
    echo "PRN refresh failed (network/time?) -- using $CFG as-is"; cp "$CFG" "$RUNCFG"
  fi
fi

# Apply the clock profile to the RUNCFG (temp copy only -- never the committed config): a CLOCK= env
# name override, and the coherence integration cap for a committed profile. The search stage reads
# clock_profile itself for the Doppler grid; this only touches name + integration_length.
if [ -n "${CLOCK:-}" ] || [ -n "${CLK_MAXINT:-}" ]; then
  if [ "$RUNCFG" = "$CFG" ]; then RUNCFG="${TMPDIR:-/tmp}/live_cfg_$$.yaml"; cp "$CFG" "$RUNCFG"; fi
  if [ -n "${CLOCK:-}" ]; then
    if grep -q '^clock_profile:' "$RUNCFG"; then
      sed -i -E "s/^(clock_profile:[[:space:]]*\{[^}]*name:[[:space:]]*)[A-Za-z]+/\1$CLOCK/" "$RUNCFG"
    else
      printf 'clock_profile: { name: %s }\n' "$CLOCK" >> "$RUNCFG"
    fi
  fi
  if [ -n "${CLK_MAXINT:-}" ]; then   # cap integration_length DOWN to the coherence limit, never up
    awk -v m="$CLK_MAXINT" '{ if (match($0, /integration_length:[[:space:]]*[0-9]+/)) {
        v=$0; sub(/.*integration_length:[[:space:]]*/,"",v); sub(/[^0-9].*/,"",v);
        if (v+0 > m+0) sub(/integration_length:[[:space:]]*[0-9]+/, "integration_length: " m) }
      print }' "$RUNCFG" > "$RUNCFG.tmp" && mv "$RUNCFG.tmp" "$RUNCFG"
  fi
fi

# Record dir comes from the config's rawFileWrite base_dir (rawFileWrite open()s with
# O_CREAT -> makes the file but NOT the dir; a missing dir exit()s kotekan at the first
# write -> "waiting for pipeline" forever). Create + clean whatever the config points at.
RECDIR=$(grep -oE 'base_dir:[[:space:]]*"[^"]*"' "$RUNCFG" | head -1 | sed -E 's/.*"([^"]*)".*/\1/')
RECDIR=${RECDIR:-/tmp/gpslive}
mkdir -p "$RECDIR"
# AUTO-ARCHIVE, never delete (2026-07-13: a bare rm -f here silently destroyed the first
# clean overnight beam-map run at a morning relaunch). Status logs move to a timestamped
# soak dir; only the level_*.raw scratch recordings are cleared.
if ls "$RECDIR"/status_log*.jsonl >/dev/null 2>&1; then
  ARCH="$(dirname "$0")/../../captures/soaks/auto_$(date +%Y-%m-%d_%H%M%S)"
  mkdir -p "$ARCH" && mv "$RECDIR"/status_log*.jsonl "$ARCH"/     && echo "previous status logs archived -> $ARCH"
fi
rm -f "$RECDIR"/level_*.raw 2>/dev/null
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
  # 2 deepest-below-horizon PRNs stay seeded as NOISE PROBES: genuine signal-free emits
  # calibrate the beam map's pedestal (the almanac gate otherwise never tracks one and
  # the GPS pedestal fell back to a signal percentile, blinding the map's low end).
  ALM="$ALM --noise-probes ${NOISE_PROBES:-4}"
  # Beam-map coasting (default ON; COAST_TO_HORIZON=0 to disable): visible sats coast on
  # the pure model through fades/nulls until they SET, so the unbiased power observables
  # sample the whole beam instead of only the locked stretches.
  if [ "${COAST_TO_HORIZON:-1}" != "0" ]; then ALM="$ALM --coast-to-horizon"; fi
  # Dead-reckoned cp seeding (default ON; DEAD_RECKON=0 to disable): BRDC ephemeris + the
  # receiver clock solved from detected sats seed CODE PHASE for every visible sat the
  # search hasn't found -- sub-threshold sats despread on-peak with no detection needed
  # (the search demotes to bootstrap/fallback/integrity; watch 'dead-reckon' broker lines).
  if [ "${DEAD_RECKON:-1}" != "0" ]; then ALM="$ALM --dead-reckon --dr-constellation G"; fi
  if [ "${NARROW_SEARCH:-1}" != "0" ]; then
    # --search-margin-wide-hz = the COLD per-PRN window (pre-clock-bias), sized from the clock
    # profile's accuracy bound; the warm margin (--search-margin-hz) applies once the bias solves.
    ALM="$ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500} --search-margin-wide-hz ${CLK_WIDE_HZ:-3000}"
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
# Persist the receiver code-rate clock offset (l-a) across runs: a strong band (L1 C/A) converges it
# and writes it here; a weak band (L1C) reads it at startup and seeds its pilots on-peak from cycle 1
# instead of self-calibrating. Override with BROKER_EXTRA="--code-bias-init <ppm>" to pin a value; rm
# the file to reset (e.g. after a cold start, once warm). The OCXO will make l-a small + stable.
CODE_BIAS_FILE=${CODE_BIAS_FILE:-/tmp/gps_code_bias.ppm}
echo "signal $SIGNAL: hops/s ${HOPS_PER_SEC:-default}, chip ${CHIP_HZ:-default} Hz, code ${CODELEN:-default} chips, l-a file $CODE_BIAS_FILE"
python3 $BROKER --detectors gps_search --trackers "$TRK" --combiner gps_combiner \
        --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-30} \
        ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file "$CODE_BIAS_FILE" \
        ${CHIP_HZ:+--chip-rate-hz $CHIP_HZ} ${CODELEN:+--code-length $CODELEN} \
        ${BROKER_EXTRA:-} $ALM $CLA $CARG \
        > /tmp/gpslive_broker.log 2>&1 &
BPID=$!

# ---- Second constellation: a config with a gal_track stage gets its OWN broker + logger
# (Galileo E1 shares the L1 tune; the stages are parallel consumers of the same channelized
# stream). Separate TLE group, seed endpoints, code-bias file; same receiver clock, so the
# two l-a estimates should agree (a nice cross-check). E1 chips 1.023 Mcps, 4092-chip code.
GALPID=""
GALLOGPID=""
if grep -qE '^gal_track:' "$RUNCFG"; then
  GAL_TLE="https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"
  GAL_ALM=""
  if [ -n "$LAT" ] && [ -n "$LON" ]; then
    GAL_ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100} --carrier-hz ${CARRIER_HZ:-1575420000}"
    GAL_ALM="$GAL_ALM --doppler-sign ${DOPPLER_SIGN:-1} --tle $GAL_TLE"
    GAL_ALM="$GAL_ALM --noise-probes ${NOISE_PROBES:-4}"
    if [ "${DEAD_RECKON:-1}" != "0" ]; then GAL_ALM="$GAL_ALM --dead-reckon --dr-constellation E"; fi
    GAL_ALM="$GAL_ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500} --search-margin-wide-hz ${CLK_WIDE_HZ:-3000}"
  else
    echo "WARNING: gal_track present but LAT/LON unset -- Galileo require_hint search will scan NOTHING"
  fi
  echo "starting GALILEO broker (gal_search/gal_track/gal_combiner, TLE group=galileo)..."
  python3 $BROKER --detectors gal_search --trackers gal_track --combiner gal_combiner           --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-30}           ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file /tmp/gps_code_bias_gal.ppm           --chip-rate-hz 1.023e6 --code-length 4092           $GAL_ALM $CARG           > /tmp/gpslive_broker_gal.log 2>&1 &
  GALPID=$!
  python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:12048           --combiner gal_combiner --search gal_search --airspy "$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')"           --out "$RECDIR/status_log_gal.jsonl" > /tmp/gpslive_logger_gal.log 2>&1 &
  GALLOGPID=$!
  echo "Galileo C/N0 status log -> $RECDIR/status_log_gal.jsonl"
fi

# Third constellation: BeiDou B1C (bds_* stages), same pattern as Galileo.
BDSPID=""
BDSLOGPID=""
if grep -qE '^bds_track:' "$RUNCFG"; then
  BDS_TLE="https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"
  BDS_ALM=""
  if [ -n "$LAT" ] && [ -n "$LON" ]; then
    BDS_ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100} --carrier-hz ${CARRIER_HZ:-1575420000}"
    BDS_ALM="$BDS_ALM --doppler-sign ${DOPPLER_SIGN:-1} --tle $BDS_TLE"
    # B1C is BDS-3 only: BDS-2 birds in the group TLE don't transmit it. Their predictions
    # poisoned the clock-freq bias (2026-07-12: lone cross-corr 'C14' lock swallowed -1550 Hz
    # as clock bias and deadlocked the narrowed search for the whole constellation).
    BDS_ALM="$BDS_ALM --tle-name-filter BEIDOU-3"
    BDS_ALM="$BDS_ALM --noise-probes ${NOISE_PROBES:-4}"
    if [ "${DEAD_RECKON:-1}" != "0" ]; then BDS_ALM="$BDS_ALM --dead-reckon --dr-constellation C"; fi
    BDS_ALM="$BDS_ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500} --search-margin-wide-hz ${CLK_WIDE_HZ:-3000}"
  else
    echo "WARNING: bds_track present but LAT/LON unset -- BeiDou require_hint search will scan NOTHING"
  fi
  echo "starting BEIDOU broker (bds_search/bds_track/bds_combiner, TLE group=beidou)..."
  python3 $BROKER --detectors bds_search --trackers bds_track --combiner bds_combiner \
          --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-30} \
          ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file /tmp/gps_code_bias_bds.ppm \
          --chip-rate-hz 1.023e6 --code-length 10230 \
          $BDS_ALM $CARG \
          > /tmp/gpslive_broker_bds.log 2>&1 &
  BDSPID=$!
  python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:12048 \
          --combiner bds_combiner --search bds_search --airspy "$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
          --out "$RECDIR/status_log_bds.jsonl" > /tmp/gpslive_logger_bds.log 2>&1 &
  BDSLOGPID=$!
  echo "BeiDou C/N0 status log -> $RECDIR/status_log_bds.jsonl"
fi

# Persist per-PRN C/N0 + detection stats (deep_snr/coherence_s live only in REST/the browser -- the
# recorded level_*.raw has Â but not those). One JSONL line per active PRN per poll, wall-clock
# stamped, alongside the raw records. Offline: captures/gps_beam_survey.py --status <this file>.
python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:12048 \
        --combiner gps_combiner --search gps_search --airspy "$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
        --out "$RECDIR/status_log.jsonl" > /tmp/gpslive_logger.log 2>&1 &
LOGPID=$!
echo "C/N0 status log -> $RECDIR/status_log.jsonl"

# OBSERVABLES RECORD (the instrument's primary data product; OBSERVABLES=0 to skip). One JSONL
# row per satellite per band per EMIT: code phase + accumulated carrier phase (with its arc id)
# + C/N0 + the BRDC geometry at that epoch -- raw, unlevelled, nothing subtracted. This is what
# TEC, scintillation and precise positioning are all computed FROM, offline. One logger per
# chain, because per-band is exactly how the geometry-free combination will consume them.
OBSPIDS=""
if [ "${OBSERVABLES:-1}" != "0" ]; then
  for _o in "gps_combiner gps_search G GPS_L1CA 1023 obs_gps_l1" \
            "gal_combiner gal_search E GAL_E1C 4092 obs_gal_e1" \
            "bds_combiner bds_search C BDS_B1C_P 10230 obs_bds_b1c"; do
    set -- $_o
    grep -qE "^($1|${1#gps_}):" "$RUNCFG" || continue
    python3 python/scripts/gnss/gnss_observables.py --url http://localhost:12048 \
            --combiner "$1" --search "$2" --sys "$3" --band "$4" --code-length "$5" \
            --carrier-hz "${CARRIER_HZ:-1575420000}" --chip-rate-hz 1.023e6 \
            ${LAT:+--lat $LAT} ${LON:+--lon $LON} ${ALT:+--alt $ALT} \
            --out "$RECDIR/$6.jsonl" > "/tmp/gpslive_$6.log" 2>&1 &
    OBSPIDS="$OBSPIDS $!"
    echo "observables ($4) -> $RECDIR/$6.jsonl"
  done
fi

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
