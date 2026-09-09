#!/bin/bash
# Start the GNSS OBSERVABLES loggers -- one per chain, into the NFS record under fixtures/.
#
# WHY THIS EXISTS: these five processes were started by hand, and nothing recorded HOW. On
# 2026-08-11, bringing the system back up after an outage, the invocation had to be
# reverse-engineered from the schema of the rows they had already written -- checking which
# endpoints the broker answered, guessing --combiner/--search/--airspy against a script whose
# defaults are all AIRSPY names (--url 12048, gps_combiner, airspy_in, 1575.42 MHz, 1023
# chips, and a Toronto lat/lon). Every one of those defaults is wrong for CHORD and NONE of
# them fails loudly: you get rows, with a plausible schema, computed against the wrong
# geometry. A data product whose generator flags live only in a dead shell's history is a
# souvenir, not a record.
#
# THE PARAMETERS ARE NOT FREE PARAMETERS. carrier/chip-rate/code-length must match the
# SignalDef the broker is actually running; the broker prints it at startup, e.g.
#     signal gps_l5: SignalDef(gps_l5: GPS_L5_Q/GPS_L5_Q_NH, 1176.450 MHz, 10230 chips @ ...)
# If a chain's signal changes, take the numbers from that line rather than from here. The
# table below is those lines, one per chain. ⚠️ L2C's chips are CM chips at 0.5115 Mcps with
# comb_mult 2 (the CM/CL multiplex): the chip is 586 m, not 29 m, and the code residual is
# taken from the broker in SECONDS for exactly that reason.
#
# The lat/lon/alt MUST BE THE BROKER'S (gnss_chains_chord.yaml). The code residual the rows
# carry is measured-minus-model with the model evaluated at the broker's site, so the PVT
# self-survey that consumes it reports an offset from THAT point; az/el/range_m here are
# evaluated at the same point so the two never disagree about where "here" is.
#
# usage:  obs_up.sh              # every chain, into fixtures/obs/<chain>_<UTCdate>.jsonl
#         OBS_OUT_DIR=/tmp/x obs_up.sh    # somewhere else (a test run)
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}
OUT=${OBS_OUT_DIR:-/home/kvand/gnss/fixtures/obs}
BROKER=${GNSS_BROKER_URL:-http://localhost:12060}
# ⚠️ THE DATE IS NOT A LAUNCH-TIME CONSTANT. Baked in at start, a logger keeps writing one
# day's filename for as long as it lives, while every consumer opens today's and finds
# nothing. gnss_observables strftime-expands --out per row and rolls at UTC midnight, so the
# pattern below goes through intact.
FRAME0_URL=${GNSS_FRAME0_URL:-http://cx43:12048}

# RUNS WHERE THE BROKER RUNS: --url defaults to localhost, and the loggers poll it once a
# second. Started from elsewhere they would silently log nothing.
HOST=${GNSS_BROKER_HOST:-cf06}
if [ "$(hostname -s)" != "$HOST" ] && [ "$BROKER" = "http://localhost:12060" ]; then
    echo "REFUSING: the broker is on $HOST but this is $(hostname -s), so localhost:12060" >&2
    echo "would poll nothing. Use:  ssh $HOST '$0'  (or set GNSS_BROKER_URL)." >&2
    exit 1
fi

# The bracket trick -- a bare pkill -f pattern matches this script's own command line.
pkill -f "[g]nss_observables.py" 2>/dev/null || true
sleep 2
mkdir -p "$OUT"

# chain  sys  carrier_hz     -- sys is the RINEX constellation letter (G/E/C)
# chain  sys  carrier_Hz   chip_rate_Hz  code_len  comb_mult
CHAINS="
gps_l5  G 1176450000 10230000 10230 1
gal_e5a E 1176450000 10230000 10230 1
bds_b2a C 1176450000 10230000 10230 1
gal_e5b E 1207140000 10230000 10230 1
bds_b2b C 1207140000 10230000 10230 1
bds_b3i C 1268520000 10230000 10230 1
gal_e6  E 1278750000  5115000  5115 1
gps_l2c G 1227600000   511500 10230 2
"

n=0
while read -r chain sys carrier chiprate codelen combmult; do
    [ -z "$chain" ] && continue
    nohup setsid "$PY" -u "$K/python/scripts/gnss/gnss_observables.py" \
        --url "$BROKER" --combiner "$chain" --search "$chain" --airspy "$chain" \
        --sys "$sys" --band "$chain" \
        --carrier-hz "$carrier" --chip-rate-hz "$chiprate" --code-length "$codelen" \
        --comb-mult "$combmult" \
        --lat 49.32001414 --lon -119.62262691 --alt 545 \
        --frame0-url "$FRAME0_URL" \
        --out "$OUT/${chain}_%Y%m%d.jsonl" \
        > "/tmp/obs_${chain}.log" 2>&1 < /dev/null &
    disown
    n=$((n + 1))
done <<< "$CHAINS"

sleep 12
up=$(pgrep -fc "[g]nss_observables.py" 2>/dev/null || echo 0)
echo "observables loggers: $up/$n up -> $OUT/<chain>_<UTC date>.jsonl (anchor $FRAME0_URL)"
if [ "$up" -lt "$n" ]; then
    echo "NOT ALL STARTED -- check /tmp/obs_<chain>.log" >&2
    exit 1
fi
