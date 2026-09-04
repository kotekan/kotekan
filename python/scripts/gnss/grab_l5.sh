#!/usr/bin/env bash
# Grab an L5-band raw-voltage pass (1176.45 MHz: GPS L5 + Galileo E5a + BeiDou B2a) with the
# L5 airspy -- the REFERENCE dataset for the E5a/B2a code-generator bring-up (python decode
# first, then the C++ port must reproduce the sky-validated tables). grab_l2c.sh is the twin.
#
# Usage: ./grab_l5.sh [seconds]   (default 60)
# NB: stop the live L5 kotekan first (it owns the dongle); relaunch run_band.sh l5 after.
set -euo pipefail
SECS="${1:-60}"
DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y-%m-%d)"
OUT="$DIR/l5_20msps_${STAMP}.bin"
N=$(( SECS * 20000000 ))                # samples = seconds * 20 MSPS real (10 MSPS mode, -a 0)
SERIAL=0x03A464C837264F67               # L5 unit
echo "grabbing ${SECS}s (${N} samples) at 1176.45 MHz from $SERIAL -> $OUT"
# -b 0: dongle bias OFF (antenna is externally bias-teed). Gains 8/8/8 = the live L5 front end
# (d528e8cd). Dither: ON under airspy_rx (irrelevant for single-band offline decode).
airspy_rx -t 3 -a 0 -f 1176.45 -l 8 -m 8 -v 8 -b 0 -s "$SERIAL" -n "$N" -r "$OUT"
cat > "${OUT%.bin}.txt" <<DESC
L5-band GNSS capture (20 MSPS real, matches the live L5 GPU pipeline front end)
file: $(basename "$OUT")  (${SECS}s), date: ${STAMP}
device: airspy $SERIAL, ACTIVE antenna via EXTERNAL bias tee (dongle bias off)
format: int16 real (airspy_rx -t 3), little-endian, NO header (PRE-flip)
sample_rate: 20 MSPS real (-a 0); center 1176.45 MHz; IF = Fs/4 = 5 MHz after the (-1)^n flip
gains: lna 8, mix 8, if 8 ; dither: ON (airspy_rx can't disable it)
band tenants: GPS L5 (I5/Q5), Galileo E5a (I/Q, CS20/CS100), BeiDou B2a (data/pilot, CS5/CS100)
capture cmd: airspy_rx -t 3 -a 0 -f 1176.45 -l 8 -m 8 -v 8 -b 0 -s $SERIAL -n ${N} -r <file>
DESC
echo "captured $(du -h "$OUT" | cut -f1); descriptor ${OUT%.bin}.txt"
