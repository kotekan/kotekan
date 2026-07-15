#!/usr/bin/env bash
# Grab an L2C raw-voltage pass (1227.6 MHz) with the L2C airspy, as a REFERENCE dataset for
# repeatable dev + a CPU-vs-GPU A/B (replay the SAME voltage through both chains). kotekan's
# rawFileWrite cannot dump the airspyInput buffer (NDArray descriptor), so capture with
# airspy_rx directly -- flags match the live L2C front end (grab_l1.sh is the L1 twin).
#
# Usage: ./grab_l2c.sh [seconds]   (default 60)
set -euo pipefail
SECS="${1:-60}"
DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y-%m-%d)"
OUT="$DIR/l2c_5msps_${STAMP}.bin"
N=$(( SECS * 5000000 ))                 # samples = seconds * 5 MSPS real
SERIAL=0x744C60C821692F4F               # L2C unit
echo "grabbing ${SECS}s (${N} samples) at L2C 1227.6 MHz from $SERIAL -> $OUT"
# -b 0: dongle bias OFF (antenna is externally bias-teed now). Gains 11/11/11 = the live front end.
# Note: airspy_rx cannot disable the R820T PLL dither, so this single-unit reference is captured
# WITH dither -- irrelevant for a single-band CPU/GPU A/B (dither only matters for INTER-unit
# coherence). The live multi-unit runs disable it via airspyInput.
airspy_rx -t 3 -a 1 -f 1227.6 -l 11 -m 11 -v 11 -b 0 -s "$SERIAL" -n "$N" -r "$OUT"
cat > "${OUT%.bin}.txt" <<EOF
L2C-band GNSS capture (5 MSPS real, matches the live L2C pipeline front end)
file: $(basename "$OUT")  (${SECS}s), date: ${STAMP}
device: airspy $SERIAL, ACTIVE antenna via EXTERNAL bias tee (dongle bias off)
format: int16 real (airspy_rx -t 3), little-endian, NO header (PRE-flip)
sample_rate: 5 MSPS real (-a 1); center 1227.6 MHz; IF = Fs/4 = 1.25 MHz after the (-1)^n flip
gains: lna 11, mix 11, if 11 ; dither: ON (airspy_rx can't disable it)
capture cmd: airspy_rx -t 3 -a 1 -f 1227.6 -l 11 -m 11 -v 11 -b 0 -s $SERIAL -n ${N} -r <file>
EOF
echo "captured $(du -h "$OUT" | cut -f1); descriptor ${OUT%.bin}.txt"
echo "NEXT: python3 '$DIR/prep_replay.py' '$OUT' /tmp/gpsin_l2c   # -> replay bench"
