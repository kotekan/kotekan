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
# Multi-band single-instance mode (run_3band.sh): SP prefixes every stage name this launcher
# passes to brokers/loggers (the merged config namespaces stages as l1_/l2c_/l5_), and
# SKIP_KOTEKAN=1 makes this invocation a BROKER/LOGGER-ONLY control plane against a kotekan
# the wrapper owns -- skipping the kotekan launch/kill, the launcher pidfile, and the
# port-scoped orphan sweeps (all three bands share one port there; a port-scoped pkill from
# band 2 would murder band 1's brokers).
SP=${STAGE_PREFIX:-}
SKIP_KOTEKAN=${SKIP_KOTEKAN:-0}
CFG=${CFG:-config/live_l1.yaml}          # L1 lean valved distributed config

# ---- PER-BAND INSTANCE KNOBS (2026-07-14) -------------------------------------------------
# Everything below used to be hardcoded, which meant a second kotekan silently talked to the
# FIRST one's REST server: the L2C broker would seed the L1 trackers and nothing would look
# wrong. Each band now owns its own ports, record dir, log names and l-a file, so L1/L2C/L5
# run side by side. Defaults are exactly the old values, so an unset environment behaves as
# before. `config/run_band.sh l2c` sets these for you.
#   NOTE l-a (the LO-vs-ADC code-rate offset) is a property of the DONGLE, not the band, so
#   each airspy MUST have its own CODE_BIAS_FILE -- sharing one cross-contaminates the seeds.
PORT=${PORT:-12048}                 # kotekan REST
HTTP_PORT=${HTTP_PORT:-8080}        # viewer HTTP
TAG=${TAG:-gpslive}                 # log-file stem: /tmp/$TAG*.log
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
# Skip the gal_/bds_/l1c_ constellations: their own broker sections below seed them (l1c_ is
# GPS-constellation but a SEPARATE chain/signal -- if left in, the C/A broker POSTs 1023-chip
# seeds to the 10230-chip L1C tracker; see the L1C broker block below). Skip cl_ too: the
# L2C CL pilot tracker is seeded by the CM broker's --cl-tracker DERIVATION (phase lifted by
# k*10230) -- raw CM seeds POSTed to it would despread the wrong 1/75th of the CL code.
TRK=${TRK:-$(
  { grep -oE 'seed_endpoint:[[:space:]]*"/[a-z_0-9]+/set_seeds"' "$CFG" \
      | sed -E 's|.*"/([a-z_0-9]+)/set_seeds"|\1|' | grep -vE '^(gal|bds|l1c|cl)_';
    grep -oE '^track[_0-9]*' "$CFG"; } | sort -u | tr '\n' ',' | sed 's/,$//')}
# Also hand any GnssVoltagePeel stage to the broker's --trackers: it POSTs the same consensus seeds
# {cp, Doppler, cp_rate(+l-a)} to /<peel>/set_seeds, so the peel reconstructs + subtracts each sat
# on-peak. Its residual feeds search_resid (peeled sats should drop). Chain L5/L1C peels the same way.
PEEL=$(grep -oE '^[a-z_0-9]+: \{ kotekan_stage: GnssVoltagePeel' "$CFG" | grep -oE '^[a-z_0-9]+' | tr '\n' ',' | sed 's/,$//')
# In prefixed (merged-instance) mode the derived names are from the ORIGINAL per-band config;
# the running stages carry the band prefix.
[ -n "$SP" ] && TRK=$(echo "$TRK" | sed "s/[^,][^,]*/${SP}&/g")
[ -n "$PEEL" ] && TRK="${TRK:+$TRK,}$PEEL" && echo "voltage-peel stage(s) seeded by the broker: $PEEL"
# L2C CL pilot. TWO shapes, detected in order:
#  (1) SIBLING CHAIN (the shared-knowledge plan's Mechanism A; gen_band_config emits it from
#      the cl chain row): a cl_track seed_endpoint beside the CM tracker. The CM broker gets
#      --cl-tracker/--cl-combiner and DERIVES one lifted CL row per CM row (cp + k*10230, the
#      segment k pinned from capture UTC + model range + SV clock, snapped to the measured
#      cp). CM keeps its own seeds and stays up as the in-run control; no CL search, no CL
#      broker, and the normal shared carrier loop (CARG below) serves both -- CL's trim IS
#      CM's, copied with the row.
#  (2) LEGACY single-chain (--cl-assist, superseded but kept): the MAIN trackers despread
#      GPS_L2C_CL and their seeds are lifted IN PLACE. Only fires when no cl_track endpoint
#      exists, so old configs behave as before. Needs the almanac (LAT/LON) either way.
# S4 CROSS-BAND CNAV. A chain whose combiner carries the CNAV DATA component but which is not
# this broker's own --combiner: at L5 the broker owns the Q pilot, while the CNAV symbols come
# off the derived L5-I sibling (i_combiner). Handed over as --cnav-combiner so its nav_obs reach
# the CNAV decoder and get cross-checked against BRDC -- a SECOND independent decode of the same
# message set L2C already decodes, so an ephemeris can be verified three ways.
# Detected the same way as the CL sibling: by the chain existing in the band's config.
CNAVA=""
if grep -qE '^i_combiner:' "$CFG"; then
  CNAVA="--cnav-combiner ${SP}i_combiner"
  echo "L5-I cross-band CNAV: broker decodes ${SP}i_combiner's symbols + BRDC cross-check"
fi
CLA=""
if grep -qE 'seed_endpoint:[[:space:]]*"/cl_track/set_seeds"' "$CFG"; then
  CLA="--cl-tracker ${SP}cl_track"
  grep -qE '^cl_combiner:' "$CFG" && CLA="$CLA --cl-combiner ${SP}cl_combiner"
  echo "L2C CL sibling chain: broker derives CL seeds (k from capture UTC + model range + SV clock) -> ${SP}cl_track"
elif grep -qE 'signal: GPS_L2C_CL' "$CFG"; then
  CLA="--cl-assist --carrier-gain ${CARRIER_GAIN:-0.2} --carrier-min-sig ${CARRIER_MIN_SIG:-15} --carrier-max-step ${CARRIER_MAX_STEP:-1.0} --carrier-fleet-seed"
  echo "L2C CL time-assist ON, LEGACY in-place mode (k from capture UTC + almanac range) + shared carrier loop"
fi
# SHARED CARRIER LOOP: any config whose tracker sets carrier_shared: true runs pure feed-forward
# (seed + almanac Doppler-rate ramp) + an NCO fed the broker's carrier_trim_hz. The combiner measures
# the residual carrier as a bit-robust phase-SLOPE fit over the deep window and the broker integrates
# it at LOW bandwidth (--carrier-gain) -> residual driven <1 Hz -> the deep coheres a full 1 s+
# (REPLAY-VALIDATED: PRN19 80 sigma@1 s ON vs 7 sigma OFF). Auto-enabled here EXCEPT in the
# legacy in-place CL mode, whose CLA already carries its own carrier args. The SIBLING-chain CL
# config also contains `signal: GPS_L2C_CL` (its second tracker command) but needs the normal
# loop for the CM chain -- so key the suppression on the MODE (--cl-assist in CLA), not on the
# signal string, which stopped implying single-chain the day the sibling chain landed.
# Tune via CARRIER_GAIN / CARRIER_MAX_HZ / CARRIER_LEAK env.
CARG=""
if grep -qE 'carrier_shared:[[:space:]]*true' "$CFG" && [[ "$CLA" != *--cl-assist* ]]; then
  # leak 0.0005 (was 0.005): the leaky integrator's equilibrium leaves a STANDING residual
  # = trim*leak/gain. GPS held seeds sit up to their 100 Hz fence off -> leak 0.005 left
  # ~0.5-1 Hz standing carrier -> 300+ deg of phase wrap across a 1 s deep window, capping
  # the GPS ladder at 0.125-0.25 s while E/C (10/25 Hz fences) held 1 s (2026-07-12; the
  # offline ladder on the raw grab proved the LO coheres to 2 s+, sqrt-T to 494 sigma).
  # 10x lower leak -> bias ~0.1 Hz worst-case; the slip-proof two-stage resid estimator
  # (GnssCoherentCombiner::carrier_resid_hz) keeps the trim from random-walking instead.
  # --carrier-min-sig: HOLD the trim through fades (coast on the feed-forward). Without it a
  # faded sat's noise residual random-walks the trim at full gain -> the model phase runs off
  # -> deeper fade: a ~4 s limit cycle that broke ADR by ~8 cycles per dip on the 1176 MHz
  # chains (median certified stretch 4 s; the gf-TEC floor, 2026-07-17).
  CARG="--carrier-gain ${CARRIER_GAIN:-0.5} --carrier-max-hz ${CARRIER_MAX_HZ:-100} --carrier-leak ${CARRIER_LEAK:-0.0005} --carrier-min-sig ${CARRIER_MIN_SIG:-15} --carrier-max-step ${CARRIER_MAX_STEP:-1.0} --carrier-innov-hz ${CARRIER_INNOV_HZ:-3.0} --carrier-step-accept ${CARRIER_STEP_ACCEPT:-3} --carrier-fleet-seed"
  echo "shared carrier loop ON (combiner slope-fit resid -> --carrier-gain ${CARRIER_GAIN:-0.5} -> tracker NCO, min-sig ${CARRIER_MIN_SIG:-15}, max-step ${CARRIER_MAX_STEP:-1.0}, innov ${CARRIER_INNOV_HZ:-3.0}, fleet-seed)"
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
# Parse chip-rate / primary-code length for a signal name straight from the C++ SignalDescriptor
# table (single source of truth). Reused for the gal_/bds_ parallel chains below so they are no
# longer hardwired to E1C/B1C -- an L5-band config (E5a-Q, B2a-P) derives 10.23e6/10230 the same way.
sig_chip()    { [ -f "$SIGHDR" ] && awk -F, -v s="$1" '$1 ~ ("^[[:space:]]*\"" s "\"$"){c=$3;gsub(/[^0-9.eE+-]/,"",c);print c;exit}' "$SIGHDR"; }
sig_codelen() { [ -f "$SIGHDR" ] && awk -F, -v s="$1" '$1 ~ ("^[[:space:]]*\"" s "\"$"){l=$4;gsub(/[^0-9]/,"",l);print l;exit}' "$SIGHDR"; }
# The signal: inside a named search block (multi-line YAML), bounded to that block so a later
# stage's signal can't leak in. Default = the L1-band signal (backward compat for dual20).
blk_signal()  { awk -v b="^$1:" '$0 ~ b {f=1;next} /^[^[:space:]]/{f=0} f&&/^[[:space:]]*signal:/{print $2;exit}' "$CFG"; }
CHIP_HZ=$(sig_chip "$SIGNAL"); CODELEN=$(sig_codelen "$SIGNAL")
# Per-constellation signal params for the parallel gal_/bds_ chains (see the brokers below).
# GAL_E1C -> 1.023e6/4092/obs_gal_e1 ; GAL_E5A_Q -> 10.23e6/10230/obs_gal_e5a. Likewise BDS.
GAL_SIGNAL=$(blk_signal gal_search); GAL_SIGNAL=${GAL_SIGNAL:-GAL_E1C}
BDS_SIGNAL=$(blk_signal bds_search); BDS_SIGNAL=${BDS_SIGNAL:-BDS_B1C_P}
GAL_CHIP=$(sig_chip "$GAL_SIGNAL");    GAL_CHIP=${GAL_CHIP:-1.023e6}
GAL_CODELEN=$(sig_codelen "$GAL_SIGNAL"); GAL_CODELEN=${GAL_CODELEN:-4092}
BDS_CHIP=$(sig_chip "$BDS_SIGNAL");    BDS_CHIP=${BDS_CHIP:-1.023e6}
BDS_CODELEN=$(sig_codelen "$BDS_SIGNAL"); BDS_CODELEN=${BDS_CODELEN:-10230}
case "$GAL_SIGNAL" in GAL_E5A*) GAL_OBS=obs_gal_e5a ;; *) GAL_OBS=obs_gal_e1 ;; esac
case "$BDS_SIGNAL" in BDS_B2A*) BDS_OBS=obs_bds_b2a ;; *) BDS_OBS=obs_bds_b1c ;; esac
# Escape-referee threshold, CHIP-SCALED (2026-07-19 census): the broker default 0.4 chips is
# calibrated in L1-scale chips. Detection-timing noise is constant in TIME, so in CHIPS it
# scales with chip rate -- at 10.23 Mcps (L5/E5a/B2a) the same fit noise reads 10x more chips
# and the referee yanked healthy sats every ~20 s (l5_gps: 149 escapes/12 min, all strong
# sats). Coefficient 1.2 (not 0.4): the measured search-fit scatter is ~+-1 us on BOTH L1
# (0.4-1 chip, 35 yanks/h at the old 0.4) and L5 (+-9 chips, SIGN-FLIPPING within seconds =
# reference noise, not track drift), so the threshold must clear ~1 us everywhere. True
# mislocks (cross-correlation peaks) sit tens of chips off and are still caught. FOLLOW-UP:
# reference the referee against the BRDC integrity residual (+-0.2 chips) instead of the
# search fit -- 'check the model is RIGHT', per the 07-19 audit.
cperr_of() { awk -v c="$1" 'BEGIN{printf "%.2f", 1.2*c/1.023e6}'; }
CPERR=$(cperr_of "$CHIP_HZ"); GAL_CPERR=$(cperr_of "$GAL_CHIP"); BDS_CPERR=$(cperr_of "$BDS_CHIP")
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
LOG=/tmp/$TAG.log

# GRACEFUL kotekan stop. NEVER SIGKILL (-9) a kotekan that owns a GPU context: on the shared GB10
# (gpu_id 0, both L1 and L2C), a -9 skips CUDA teardown and the abrupt context death takes DOWN
# the OTHER band's kotekan too -- the whole run of "mysterious L1 deaths" was a -9 of the sibling
# GPU process. SIGTERM lets kotekan release the GPU cleanly; escalate to -9 only if it hangs.
kill_kotekan() {  # $1 = pgrep pattern (port-scoped to ONE band)
    local pids; pids=$(pgrep -f "$1"); [ -z "$pids" ] && return
    kill -TERM $pids 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do sleep 0.5; pgrep -f "$1" >/dev/null || return; done
    kill -9 $pids 2>/dev/null   # last resort, only if it ignored TERM for 5 s
}

cleanup() { echo; echo "stopping..."; kill "${BPID:-}" "${LOGPID:-}" "${GALPID:-}" "${GALLOGPID:-}" "${BDSPID:-}" "${BDSLOGPID:-}" "${L1CPID:-}" "${L1CLOGPID:-}" ${OBSPIDS:-} 2>/dev/null
            if [ "$SKIP_KOTEKAN" != "1" ]; then
              kill_kotekan "kotekan .*-b 0.0.0.0:$PORT"   # only THIS band's kotekan, GRACEFULLY
              pkill -9 -f "livebeam_server.*--http-port $HTTP_PORT" 2>/dev/null
              # port-scoped: NEVER in merged mode (all bands share the port; PIDs above suffice)
              pkill -9 -f "gps_status_logger.*localhost:$PORT" 2>/dev/null
              pkill -9 -f "gnss_observables.*localhost:$PORT" 2>/dev/null
            fi
            [ -n "${RUNCFG:-}" ] && [ "${RUNCFG:-}" != "$CFG" ] && rm -f "$RUNCFG"
            [ -f "${LAUNCHPID:-}" ] && [ "$(cat "$LAUNCHPID" 2>/dev/null)" = "$$" ] && rm -f "$LAUNCHPID"
            exit 0; }
trap cleanup INT TERM

# Kill only THIS band's instance -- and GRACEFULLY (see kill_kotekan): an unscoped or -9 kill would
# tear down the other bands' runs (unscoped by pattern, -9 by GPU-context collapse).
if [ "$SKIP_KOTEKAN" != "1" ]; then
kill_kotekan "kotekan .*-b 0.0.0.0:$PORT"
pkill -9 -f "livebeam_server.*--http-port $HTTP_PORT" 2>/dev/null
fi
# A previous launcher SHELL for this band must die BEFORE its trap can ever fire: a stale
# launcher's cleanup() would pkill this run's port-scoped processes hours later. -9 on
# purpose (skip its trap); pidfile-scoped so other bands' launchers are untouched.
LAUNCHPID=/tmp/run_live_launcher_${PORT}${SP:+_$SP}.pid
if [ -f "$LAUNCHPID" ]; then
    oldpid=$(cat "$LAUNCHPID" 2>/dev/null)
    [ -n "$oldpid" ] && [ "$oldpid" != "$$" ] && \
        grep -qa run_live /proc/"$oldpid"/cmdline 2>/dev/null && kill -9 "$oldpid" 2>/dev/null
fi
echo $$ > "$LAUNCHPID"
# Sweep this band's ORPHANED control-plane processes. A launcher that died without its trap
# (or whose kotekan alone was killed) leaves brokers/loggers running; a surviving broker
# FIGHTS the new one over the same trackers -- two independent clock solves alternately
# re-seeding every 0.2 s = intermittent per-sat carrier decoherence (the 2026-07-15 'E1C
# bistable') -- and surviving loggers double-write the jsonl files. Port-scoped.
if [ "$SKIP_KOTEKAN" != "1" ]; then
pkill -9 -f "gps_distributed_broker.py.*localhost:$PORT" 2>/dev/null
pkill -9 -f "gps_status_logger.py.*localhost:$PORT" 2>/dev/null
pkill -9 -f "gnss_observables.py.*localhost:$PORT" 2>/dev/null
else
# merged mode: sweep only THIS band's control plane, by its prefixed stage names
pkill -9 -f "gps_distributed_broker.py.*--detectors ${SP}gps_search" 2>/dev/null
pkill -9 -f "gps_status_logger.py.*--combiner ${SP}" 2>/dev/null
pkill -9 -f "gnss_observables.py.*--combiner ${SP}" 2>/dev/null
fi

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
# ...and clear the level_*.raw scratch ONLY when this script owns the kotekan. In
# SKIP_KOTEKAN mode (run_3band's per-band control-plane invocations) the merged kotekan is
# ALREADY RUNNING with these files open: the rm unlinks a LIVE output and every GPS-chain
# record from then on silently vanishes into a deleted inode. (Found 2026-07-24 -- this had
# eaten the L1 GPS level archive since the 07-22 launch; gal_/bds_/l1c_level survived only
# because the glob anchors at 'level_'. 318 MB salvaged via /proc/<pid>/fd.)
[ "$SKIP_KOTEKAN" != "1" ] && rm -f "$RECDIR"/level_*.raw 2>/dev/null
echo "recording to $RECDIR"
sleep 1

if [ "$SKIP_KOTEKAN" != "1" ]; then
echo "starting kotekan ($CFG) -> $LOG"
$KOTEKAN -c $RUNCFG -b 0.0.0.0:$PORT > $LOG 2>&1 &
sleep 4
fi
echo "front end: $(curl -s localhost:$PORT/${SP}airspy_in/adcstat | tr -d '\n ')"
echo "GPS sky viewer: http://localhost:$HTTP_PORT   (kotekan REST :$PORT)"

# Almanac assist (predicted Doppler) when a location is given:
#   LAT=43.66 LON=-79.40 ./config/run_live.sh
# Watch /tmp/${TAG}_broker.log for the predicted-vs-measured Doppler + clock bias.
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
  # --constellation: BRDC-almanac constellation gate (the default almanac source since
  # 2026-07-17; PRN-indexed = immune to the celestrak label rot. --tle stays as fallback).
  ALM="$ALM --constellation G --doppler-sign ${DOPPLER_SIGN:-1}"
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
# PERSISTENT, NOT /tmp (2026-07-27). These are the warm-start state that lets a chain come up
# "solved from cycle 1"; in /tmp a reboot wipes them and EVERY chain restarts cold. Measured cost
# on the 07-27 power-outage restart: L5 GPS took 2h45m to acquire ANYTHING -- it could measure only
# one satellite (PRN11), --bias-min-sats 2 kept the bias UNSOLVED for 603 cycles, and it waited for
# PRN21 to physically rise to 40 deg. Its own siblings held the right answer (+32 Hz) the whole time.
# ~/.cache/kotekan_gps already survives reboots (BRDC, TLE, Earthdata token live there).
BIAS_DIR=${BIAS_DIR:-$HOME/.cache/kotekan_gps}
mkdir -p "$BIAS_DIR" 2>/dev/null
CODE_BIAS_FILE=${CODE_BIAS_FILE:-$BIAS_DIR/gps_code_bias_${TAG}.ppm}
# rec D (2026-07-19): carrier clock bias is a per-chain CONSTANT on the GPSDO -- persist and
# warm-start it like l-a. Solved-from-cycle-1 = narrow margins + immediate seeding (Tier-2 cut).
CLOCK_BIAS_FILE=${CLOCK_BIAS_FILE:-$BIAS_DIR/gps_clock_bias_${TAG}.hz}
# BAND-SHARED clock bias (2026-07-22): all chains of a band despread the SAME carrier
# frequency through ONE LO, so their clock-freq biases are one physical number. Each
# broker fuses its siblings' persisted estimates (sat-count weighted, <60 s fresh) --
# a 2-sat chain then rides the band's ~10-sat consensus instead of its own swinging
# median (the L5 NCO-kick root, measured -16..+44 Hz in 70 s vs L1's steady -152).
# Nonexistent sibling files (band without that chain) are skipped harmlessly.
# SIGNAL-CAPABILITY gate (2026-07-22): seed/hint only the GPS blocks that transmit this
# band's modernized signal (L2C -> IIR-M+, L5 -> IIF+, L1C -> III). The BeiDou-3 phantom fix
# was a numeric PRN cutoff (--dr-min-prn) that can't express GPS's interspersed III sats.
# L1 C/A is on every sat -> no gate (and no risk of excluding a sat missing from the TLE).
case "$SIGNAL" in *L1CA*|*L1_CA*) SIG_CAP="";; *) SIG_CAP="--signal-capability $SIGNAL";; esac
CBP=$BIAS_DIR/gps_clock_bias_${TAG}
SIB_GPS="--clock-bias-siblings ${CBP}_gal.hz ${CBP}_bds.hz ${CBP}_l1c.hz"
SIB_GAL="--clock-bias-siblings ${CBP}.hz ${CBP}_bds.hz ${CBP}_l1c.hz"
SIB_BDS="--clock-bias-siblings ${CBP}.hz ${CBP}_gal.hz ${CBP}_l1c.hz"
SIB_L1C="--clock-bias-siblings ${CBP}.hz ${CBP}_gal.hz ${CBP}_bds.hz"
echo "signal $SIGNAL: hops/s ${HOPS_PER_SEC:-default}, chip ${CHIP_HZ:-default} Hz, code ${CODELEN:-default} chips, l-a file $CODE_BIAS_FILE"
python3 $BROKER --rest-url "http://localhost:$PORT" --detectors ${SP}gps_search --trackers "$TRK" --combiner ${SP}gps_combiner \
        --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-300} --adc-stage "${SP}airspy_in" \
        ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file "$CODE_BIAS_FILE" --clock-bias-file "$CLOCK_BIAS_FILE" $SIB_GPS \
        ${CHIP_HZ:+--chip-rate-hz $CHIP_HZ} ${CODELEN:+--code-length $CODELEN} ${CPERR:+--hold-max-cp-err $CPERR} \
        --watchdog-s ${WATCHDOG_S:-45} --watchdog-det-snr ${WATCHDOG_DET_SNR:-100} \
        --carrier-det-gate-s ${CARRIER_DET_GATE_S:-10} \
        ${BROKER_EXTRA:-} $ALM $CLA $CNAVA $CARG $SIG_CAP \
        > /tmp/${TAG}_broker.log 2>&1 &
BPID=$!
# WATCHDOG NOW FLEET-WIDE (was BeiDou-only until 2026-07-19 eve): the L2C duty timeline
# exposed the cost of the gap -- zombie-anchor cohorts (strong detection, chips-off-peak
# despread, zero coherence) form in the birth/wide-margin window, are then PROTECTED
# indefinitely by hold + the escape amp-veto, and only cleared by accidental relaunches
# (every 07-19 'config improvement' was really the relaunch clearing a cohort: duty
# 18->41->62->74% at each restart, then 38% again after the next birth). The watchdog's
# lifecycle rescue is the designed answer; nothing about it is BeiDou-specific.

# ---- Second constellation: a config with a gal_track stage gets its OWN broker + logger
# (Galileo E1 shares the L1 tune; the stages are parallel consumers of the same channelized
# stream). Separate TLE group, seed endpoints, code-bias file; same receiver clock, so the
# two l-a estimates should agree (a nice cross-check). E1 chips 1.023 Mcps, 4092-chip code.
GALPID=""
GALLOGPID=""
if grep -qE '^gal_track:|seed_endpoint:[[:space:]]*"/gal_track/set_seeds"' "$RUNCFG"; then
  GAL_TLE="https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"
  GAL_ALM=""
  if [ -n "$LAT" ] && [ -n "$LON" ]; then
    GAL_ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100} --carrier-hz ${CARRIER_HZ:-1575420000}"
    GAL_ALM="$GAL_ALM --constellation E --doppler-sign ${DOPPLER_SIGN:-1} --tle $GAL_TLE"
    GAL_ALM="$GAL_ALM --noise-probes ${NOISE_PROBES:-4}"
    if [ "${DEAD_RECKON:-1}" != "0" ]; then GAL_ALM="$GAL_ALM --dead-reckon --dr-constellation E"; fi
    GAL_ALM="$GAL_ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500} --search-margin-wide-hz ${CLK_WIDE_HZ:-3000}"
  else
    echo "WARNING: gal_track present but LAT/LON unset -- Galileo require_hint search will scan NOTHING"
  fi
  echo "starting GALILEO broker ($GAL_SIGNAL: gal_search/gal_track/gal_combiner, TLE group=galileo)..."
  python3 $BROKER --rest-url "http://localhost:$PORT" --detectors ${SP}gal_search --trackers ${SP}gal_track --combiner ${SP}gal_combiner           --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-300} --adc-stage "${SP}airspy_in"           ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file $BIAS_DIR/gps_code_bias_${TAG}_gal.ppm --clock-bias-file $BIAS_DIR/gps_clock_bias_${TAG}_gal.hz $SIB_GAL           --chip-rate-hz $GAL_CHIP --code-length $GAL_CODELEN --hold-max-cp-err $GAL_CPERR           --watchdog-s ${WATCHDOG_S:-45} --watchdog-det-snr ${WATCHDOG_DET_SNR:-100} --carrier-det-gate-s ${CARRIER_DET_GATE_S:-10}           ${BROKER_EXTRA:-} $GAL_ALM $CARG           > /tmp/${TAG}_broker_gal.log 2>&1 &
  GALPID=$!
  python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:$PORT           --combiner ${SP}gal_combiner --search ${SP}gal_search --airspy "${SP}$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')"           --out "$RECDIR/status_log_gal.jsonl" > /tmp/${TAG}_logger_gal.log 2>&1 &
  GALLOGPID=$!
  echo "Galileo C/N0 status log -> $RECDIR/status_log_gal.jsonl"
fi

# Third constellation: BeiDou B1C (bds_* stages), same pattern as Galileo.
BDSPID=""
BDSLOGPID=""
if grep -qE '^bds_track:|seed_endpoint:[[:space:]]*"/bds_track/set_seeds"' "$RUNCFG"; then
  BDS_TLE="https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"
  BDS_ALM=""
  if [ -n "$LAT" ] && [ -n "$LON" ]; then
    BDS_ALM="--almanac --lat $LAT --lon $LON --alt ${ALT:-100} --carrier-hz ${CARRIER_HZ:-1575420000}"
    BDS_ALM="$BDS_ALM --constellation C --doppler-sign ${DOPPLER_SIGN:-1} --tle $BDS_TLE"
    # B1C is BDS-3 only: BDS-2 birds don't transmit it. Their predictions poisoned the
    # clock-freq bias (2026-07-12: lone cross-corr 'C14' lock swallowed -1550 Hz as clock
    # bias and deadlocked the narrowed search for the whole constellation). The BRDC
    # almanac gates PRN >= 19 itself; the name filter covers the TLE fallback path.
    BDS_ALM="$BDS_ALM --tle-name-filter BEIDOU-3"
    BDS_ALM="$BDS_ALM --noise-probes ${NOISE_PROBES:-4}"
    if [ "${DEAD_RECKON:-1}" != "0" ]; then BDS_ALM="$BDS_ALM --dead-reckon --dr-constellation C"; fi
    BDS_ALM="$BDS_ALM --narrow-search --search-margin-hz ${SEARCH_MARGIN_HZ:-500} --search-margin-wide-hz ${CLK_WIDE_HZ:-3000}"
    # nh TIME-ASSIST: if the B1C combiner opts in (nh_assist: true), the broker POSTs predicted
    # overlay indices so the weak sats get the right alignment instead of losing the 1800-way
    # search. B1C overlay = 1800 chips (E5a/B2a CS100 would use 100).
    # RUNCFG is the per-band SOURCE config whose stage names are UNPREFIXED (gen_3band_config
    # adds ${SP} only in the merged file) -- so match with-or-without the prefix. Requiring
    # the prefix silently dropped --nh-assist for every 3-band launch (found 2026-07-18: the
    # C20/C23 off-grid overlay mislocks the assist had fixed on 07-17 were back all soak).
    if grep -qE "^(${SP})?bds_combiner:.*nh_assist:[[:space:]]*true" "$RUNCFG"; then
      BDS_ALM="$BDS_ALM --nh-assist --nh-overlay-len ${NH_OVERLAY_LEN:-1800}"
      echo "  BeiDou nh time-assist ON (predicted overlay -> combiner /set_nh_hint)"
    fi
  else
    echo "WARNING: bds_track present but LAT/LON unset -- BeiDou require_hint search will scan NOTHING"
  fi
  echo "starting BEIDOU broker ($BDS_SIGNAL: bds_search/bds_track/bds_combiner, TLE group=beidou)..."
  python3 $BROKER --rest-url "http://localhost:$PORT" --detectors ${SP}bds_search --trackers ${SP}bds_track --combiner ${SP}bds_combiner \
          --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-300} --adc-stage "${SP}airspy_in" \
          ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file $BIAS_DIR/gps_code_bias_${TAG}_bds.ppm --clock-bias-file $BIAS_DIR/gps_clock_bias_${TAG}_bds.hz $SIB_BDS \
          --chip-rate-hz $BDS_CHIP --code-length $BDS_CODELEN --hold-max-cp-err $BDS_CPERR \
          --watchdog-s ${WATCHDOG_S:-45} --watchdog-det-snr ${WATCHDOG_DET_SNR:-100} \
          --carrier-det-gate-s ${CARRIER_DET_GATE_S:-10} \
          ${BROKER_EXTRA:-} $BDS_ALM $CARG \
          > /tmp/${TAG}_broker_bds.log 2>&1 &
  BDSPID=$!
  python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:$PORT \
          --combiner ${SP}bds_combiner --search ${SP}bds_search --airspy "${SP}$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
          --out "$RECDIR/status_log_bds.jsonl" > /tmp/${TAG}_logger_bds.log 2>&1 &
  BDSLOGPID=$!
  echo "BeiDou C/N0 status log -> $RECDIR/status_log_bds.jsonl"
fi

# Fourth chain: GPS L1C-P (l1c_* stages). GPS constellation (shares C/A's ephemeris/almanac,
# constellation G, PRNs 1-32) but a SEPARATE signal -- a 10230-chip BOC(1,1) pilot under the
# per-PRN 1800-symbol L1CO overlay. Its OWN broker + bias files + obs so it never cross-seeds
# the C/A chain (its 10230-chip code needs different cp params than C/A's 1023). Reuses the GPS
# $ALM (constellation G, same birds). Added 2026-07-21 as the clean-L1-band overlay-pilot
# ADR/TEC comparison for the overlay-wipe de-rotation fix (isolates the wipe math from the L5 leg).
L1CPID=""
L1CLOGPID=""
if grep -qE '^l1c_track:|seed_endpoint:[[:space:]]*"/l1c_track/set_seeds"' "$RUNCFG"; then
  L1C_SIGNAL=$(blk_signal l1c_search); L1C_SIGNAL=${L1C_SIGNAL:-GPS_L1C_P}
  L1C_CHIP=$(sig_chip "$L1C_SIGNAL");     L1C_CHIP=${L1C_CHIP:-1.023e6}
  L1C_CODELEN=$(sig_codelen "$L1C_SIGNAL"); L1C_CODELEN=${L1C_CODELEN:-10230}
  L1C_CPERR=$(cperr_of "$L1C_CHIP")
  # nh TIME-ASSIST (like B1C): the 1800-symbol L1CO overlay is ambiguous over the 1 s window, so
  # the blind alignment search flickers -- the broker POSTs the predicted overlay index (period =
  # code_length/chip_rate = 10 ms, len 1800) and the combiner pins its wipe there. Same convention
  # path as B1C; the combiner self-calibrates the constant from confidently-locked sats.
  L1C_NH=""
  if grep -qE "^(${SP})?l1c_combiner:.*nh_assist:[[:space:]]*true" "$RUNCFG"; then
    L1C_NH="--nh-assist --nh-overlay-len ${NH_OVERLAY_LEN:-1800}"
    echo "  GPS-L1C nh time-assist ON (predicted L1CO overlay -> combiner /set_nh_hint)"
  fi
  { [ -z "$LAT" ] || [ -z "$LON" ]; } && echo "WARNING: l1c_track present but LAT/LON unset -- L1C require_hint search will scan NOTHING"
  echo "starting GPS-L1C broker ($L1C_SIGNAL: l1c_search/l1c_track/l1c_combiner, constellation G, chip $L1C_CHIP code $L1C_CODELEN)..."
  python3 $BROKER --rest-url "http://localhost:$PORT" --detectors ${SP}l1c_search --trackers ${SP}l1c_track --combiner ${SP}l1c_combiner \
          --acquire-snr 6 --interval 0.2 --coast-budget ${COAST_BUDGET:-300} --adc-stage "${SP}airspy_in" \
          ${HOPS_PER_SEC:+--hops-per-sec $HOPS_PER_SEC} --code-bias-file $BIAS_DIR/gps_code_bias_${TAG}_l1c.ppm --clock-bias-file $BIAS_DIR/gps_clock_bias_${TAG}_l1c.hz $SIB_L1C \
          --chip-rate-hz $L1C_CHIP --code-length $L1C_CODELEN --hold-max-cp-err $L1C_CPERR \
          --watchdog-s ${WATCHDOG_S:-45} --watchdog-det-snr ${WATCHDOG_DET_SNR:-100} \
          --carrier-det-gate-s ${CARRIER_DET_GATE_S:-10} \
          ${BROKER_EXTRA:-} $ALM $CARG $L1C_NH --signal-capability ${L1C_SIGNAL:-GPS_L1C_P} \
          > /tmp/${TAG}_broker_l1c.log 2>&1 &
  L1CPID=$!
  python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:$PORT \
          --combiner ${SP}l1c_combiner --search ${SP}l1c_search --airspy "${SP}$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
          --out "$RECDIR/status_log_l1c.jsonl" > /tmp/${TAG}_logger_l1c.log 2>&1 &
  L1CLOGPID=$!
  echo "GPS-L1C C/N0 status log -> $RECDIR/status_log_l1c.jsonl"
fi

# Persist per-PRN C/N0 + detection stats (deep_snr/coherence_s live only in REST/the browser -- the
# recorded level_*.raw has Â but not those). One JSONL line per active PRN per poll, wall-clock
# stamped, alongside the raw records. Offline: captures/gps_beam_survey.py --status <this file>.
python3 python/scripts/gnss/gps_status_logger.py --url http://localhost:$PORT \
        --combiner ${SP}gps_combiner --search ${SP}gps_search --airspy "${SP}$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
        --out "$RECDIR/status_log.jsonl" > /tmp/${TAG}_logger.log 2>&1 &
LOGPID=$!
echo "C/N0 status log -> $RECDIR/status_log.jsonl"

# OBSERVABLES RECORD (the instrument's primary data product; OBSERVABLES=0 to skip). One JSONL
# row per satellite per band per EMIT: code phase + accumulated carrier phase (with its arc id)
# + C/N0 + the BRDC geometry at that epoch -- raw, unlevelled, nothing subtracted. This is what
# TEC, scintillation and precise positioning are all computed FROM, offline. One logger per
# chain, because per-band is exactly how the geometry-free combination will consume them.
OBSPIDS=""
if [ "${OBSERVABLES:-1}" != "0" ]; then
  # The PRIMARY chain's band comes from the CONFIG (SIGNAL/CODELEN/CHIP_HZ, derived above), not
  # from a hardcoded L1 triple. It used to be hardcoded, so an L2C run logged its observables as
  # "GPS_L1CA, 1023 chips, 1.023 Mcps" -- right carrier, wrong everything else, and the file was
  # even named obs_gps_l1.jsonl. Silently wrong data is worse than no data.
  case "${SIGNAL:-GPS_L1CA}" in
    GPS_L2C*) _obs=obs_gps_l2c ;;
    GPS_L5*)  _obs=obs_gps_l5  ;;
    GPS_L1C_*) _obs=obs_gps_l1c ;;
    *)        _obs=obs_gps_l1  ;;
  esac
  # gal_/bds_ band params are DERIVED (GAL_SIGNAL/BDS_SIGNAL etc., above) so the same loop covers
  # the L1 tune (E1C/B1C) and the L5 tune (E5a-Q/B2a-P); the grep skips whichever chain is absent.
  for _o in "gps_combiner gps_search G ${SIGNAL:-GPS_L1CA} ${CODELEN:-1023} $_obs ${CHIP_HZ:-1.023e6}" \
            "gal_combiner gal_search E $GAL_SIGNAL $GAL_CODELEN $GAL_OBS $GAL_CHIP" \
            "bds_combiner bds_search C $BDS_SIGNAL $BDS_CODELEN $BDS_OBS $BDS_CHIP" \
            "l1c_combiner l1c_search G ${L1C_SIGNAL:-GPS_L1C_P} ${L1C_CODELEN:-10230} obs_gps_l1c ${L1C_CHIP:-1.023e6}"; do
    set -- $_o
    grep -qE "^($1|${1#gps_}):" "$RUNCFG" || continue
    python3 python/scripts/gnss/gnss_observables.py --url http://localhost:$PORT \
            --combiner "${SP}$1" --search "${SP}$2" --sys "$3" --band "$4" --code-length "$5" \
            --airspy "${SP}$(grep -oE '^airspy[_a-z0-9]*:' "$RUNCFG" | head -1 | tr -d ':')" \
            --carrier-hz "${CARRIER_HZ:-1575420000}" --chip-rate-hz "$7" \
            ${LAT:+--lat $LAT} ${LON:+--lon $LON} ${ALT:+--alt $ALT} \
            --interval "${OBS_INTERVAL:-2.0}" \
            --out "$RECDIR/$6.jsonl" > "/tmp/${TAG}_$6.log" 2>&1 &
    OBSPIDS="$OBSPIDS $!"
    echo "observables ($4) -> $RECDIR/$6.jsonl"
  done
fi

echo "=== watching (Ctrl-C to stop) ==="
while true; do
  sleep 3
  curl -s localhost:$PORT/search/get_detections 2>/dev/null | python3 -c "
import sys,json,urllib.request
d=json.load(sys.stdin)
amp=json.load(urllib.request.urlopen('http://localhost:$PORT/combiner/get_status'))
# sig = detection significance (sigma above noise): deep nav-wiped SNR or the noise-debiased
# incoherent SNR. >>1 = a real lock; ~1 = noise (the raw |A| sits at the noise floor for weak sats).
sigf=lambda r: max(r.get('deep_snr',0) or 0, r.get('amp_snr',0) or 0)
ampf=lambda r: r.get('deep_amplitude') or r.get('unbiased_amplitude') or 0  # unbiased Â (not the noise-biased |A|)
top=sorted(amp,key=lambda r:-sigf(r))[:3]
adc=json.load(urllib.request.urlopen('http://localhost:$PORT/airspy_in/adcstat'))
hdr='rms=%.0f rail=%.2f'%(adc['rms'],adc['railfrac'])
if d:
    s='  DETECT: '+'; '.join('PRN%d dop%+.0f cp%.0f snr%.1f'%(x['prn'],x['doppler_hz'],x['code_phase_chips'],x['snr']) for x in sorted(d,key=lambda r:-r['snr'])[:4])
else:
    s='  searching...'
lvl='  sig(Â): '+(' '.join('PRN%d=%.1fσ(%.2f)'%(r['prn'],sigf(r),ampf(r)) for r in top if sigf(r)>0) or '--')
print('[%s]%s%s'%(hdr,s,lvl))
" 2>/dev/null || echo "  (waiting for pipeline...)"
done
