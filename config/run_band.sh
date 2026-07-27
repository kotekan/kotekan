#!/usr/bin/env bash
# Launch ONE band's live GNSS pipeline. Three airspys, three kotekans, three brokers -- all
# running side by side, each pinned to its own board and its own ports.
#
#   ./config/run_band.sh l1     # GPS L1 C/A + Galileo E1C + BeiDou B1C (the GPU tri-chain)
#   ./config/run_band.sh l2c    # GPS L2C CM
#   ./config/run_band.sh l5     # GPS L5 Q5
#
# Each band is INDEPENDENT on purpose: separate kotekan, separate broker, separate record dir.
# Bands will be merged eventually, but there is a lot of up-and-down dev per band and a shared
# harness would mean every L5 restart bounced the L1 soak.
#
# WHY THE PORT MAP EXISTS (this is the whole point of the script). run_live.sh used to hardcode
# kotekan's REST port at 12048. Start a second band and its broker would POST seeds to the FIRST
# band's kotekan -- seeding L1's trackers with L5 Doppler. Nothing would crash; L5 would just
# never lock, and L1 would quietly degrade. Every shared resource is now per-band:
#
#   band  airspy serial          kotekan  viewer  ws    bufferSend   recdir        l-a file
#   L1    0x03A464C837525967     12048    8080    8539  -- (GPU)     /tmp/gpswipe  ..._l1.ppm
#   L2C   0x744C60C821692F4F     12148    8081    8639  15101-15107  /tmp/gps_l2c  ..._l2c.ppm
#   L5    0x03A464C837264F67     12248    8082    8739  15201-15210  /tmp/gps_l5   ..._l5.ppm
#
# The l-a (LO-vs-ADC code-rate offset) file MUST be per-band: l-a is a property of the DONGLE,
# not the band, and the three boards have three different crystals. Sharing one file feeds the
# wrong clock offset into another board's code-phase seeds.
#
# Serial pinning is what keeps a band on its own board. Without it airspyInput opens whichever
# board enumerates first -- the bands would race, and the loser tunes correctly and simply never
# acquires (indistinguishable from a dead antenna). Verify a new pin by BREAKING it: a wrong
# serial must fail with AIRSPY_ERROR_NOT_FOUND.
set -u
BAND=${1:-l1}
cd "$(dirname "$0")/.."

case "$BAND" in
  l1)  CFG=config/live_l1_dual20.yaml; PORT=12048; HTTP_PORT=8080
       # (b) dop-continuous: the broker seeds a continuously-true Doppler instead of a fenced
       # one. Safe -- and better -- only since the tracker's f_ref re-pin became phase-continuous.
       EXTRA=${BROKER_EXTRA:---dop-continuous} ;;
  l2c) CFG=config/live_l2c_gpu.yaml;   PORT=12148; HTTP_PORT=8081
       # GPU chain is now the L2C default: it exports the coherent-C/N0 / ADR / S4 / CMC
       # observables the old CPU chain lacked. `l2c_cpu` still launches the CPU chain.
       EXTRA=${BROKER_EXTRA:---dop-continuous} ;;
  l2c_cpu) CFG=config/live_l2c.yaml;   PORT=12148; HTTP_PORT=8081
       EXTRA=${BROKER_EXTRA:-} ;;
  l5)  CFG=config/live_l5_gpu.yaml;    PORT=12248; HTTP_PORT=8082
       # GPU chain is the L5 default: coherent-C/N0/ADR/S4/boundary_f observables AND the
       # segmented despread -- L5's NH20 (one chip per 1 ms record, ~50% transitions) is
       # fully exposed to the period-boundary null on the CPU chain (head==P compat).
       EXTRA=${BROKER_EXTRA:---dop-continuous} ;;
  l5_cpu) CFG=config/live_l5.yaml;     PORT=12248; HTTP_PORT=8082
       EXTRA=${BROKER_EXTRA:-} ;;
  *)   echo "usage: $0 {l1|l2c|l5}" >&2; exit 1 ;;
esac

# Site (enables the almanac/BRDC assist -- without LAT/LON there is no predicted-Doppler seeding
# and no dead reckoning).
export LAT=${LAT:-43.968697} LON=${LON:--79.252106} ALT=${ALT:-260}
export KOTEKAN=${KOTEKAN:-./build_cuda/kotekan/kotekan}
export CFG PORT HTTP_PORT
export TAG="gps_$BAND"                                   # -> /tmp/gps_l5*.log
# PERSISTENT, not /tmp -- see the BIAS_DIR note in run_live.sh (a reboot wiping this cost
# L5 GPS 2h45m of cold-start acquisition on 2026-07-27).
export BIAS_DIR=${BIAS_DIR:-$HOME/.cache/kotekan_gps}
mkdir -p "$BIAS_DIR" 2>/dev/null
export CODE_BIAS_FILE=${CODE_BIAS_FILE:-$BIAS_DIR/gps_code_bias_$BAND.ppm}
export BROKER_EXTRA="$EXTRA"

echo "=== band $BAND ==============================================="
echo "  config   $CFG"
echo "  kotekan  REST :$PORT     viewer http://localhost:$HTTP_PORT"
echo "  logs     /tmp/$TAG*.log  l-a $CODE_BIAS_FILE"
echo "  serial   $(grep -oE '^\s*serial: 0x[0-9A-Fa-f]+' "$CFG" | grep -oE '0x[0-9A-Fa-f]+')"
echo "============================================================="
exec bash config/run_live.sh
