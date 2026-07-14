#!/usr/bin/env bash
# Grab an L1 raw-voltage pass with the airspy. kotekan's rawFileWrite CANNOT dump the airspyInput
# voltage buffer (NDArray descriptor incompatibility), so capture with airspy_rx directly -- proven
# flags matching the live 5 MSPS front end (active antenna -> bias-tee ON). A longer grab gives a
# better clock-offset baseline and a better chance of >=2 strong sats for the LO-vs-ADC confirmation.
#
# Usage: ./grab_l1.sh [seconds]        (default 60)
set -euo pipefail
SECS="${1:-60}"
DIR="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y-%m-%d)"
OUT="$DIR/l1_5msps_${STAMP}.bin"
N=$(( SECS * 5000000 ))                 # samples = seconds * 5 MSPS
echo "grabbing ${SECS}s (${N} samples) from an OPEN sky (want multiple strong sats) -> $OUT"
airspy_rx -t 3 -a 1 -f 1575.42 -l 14 -m 15 -v 15 -b 1 -n "$N" -r "$OUT"
cat > "${OUT%.bin}.txt" <<EOF
L1-band GNSS capture (5 MSPS, matches the live pipeline front end)
file: $(basename "$OUT")  (${SECS}s), date: ${STAMP}
device: airspy R2, ACTIVE antenna, bias-tee ON
format: int16 real (airspy_rx -t 3), little-endian, NO header (PRE-flip)
sample_rate: 5 MSPS real (-a 1); center 1575.42 MHz; IF = Fs/4 = 1.25 MHz after the (-1)^n flip
gains: lna 14, mix 15, if 15
capture cmd: airspy_rx -t 3 -a 1 -f 1575.42 -l 14 -m 15 -v 15 -b 1 -n ${N} -r <file>
EOF
echo "captured $(du -h "$OUT" | cut -f1); wrote descriptor ${OUT%.bin}.txt"
echo ""
echo "NEXT:"
echo "  1) confirm the LO-vs-ADC offset (l-a) is COMMON across sats (the decisive check):"
echo "       python3 '$DIR/l1_code_drift.py' '$OUT'"
echo "  2) deep-integration curve on the new capture (edit FILE or import):"
echo "       python3 '$DIR/l1_deep_integ.py'"
echo "  3) (optional) replay through the kotekan pipeline + broker to watch the empirical cp_rate"
echo "     fit absorb the large code-rate offset:"
echo "       python3 '$DIR/prep_replay.py' '$OUT'   # -> /tmp/gpsin, then run offline_l1ca_replay.yaml"
