#!/bin/sh
# One observables writer, in the foreground, for systemd to supervise. $GNSS_CHAIN names it.
#
# WHY IT EXECS INSTEAD OF BACKGROUNDING: obs_up.sh launches all eight with `nohup setsid ... &`
# because nothing was supervising them. Under systemd the opposite is wanted -- one process per
# unit, in the foreground, so the exit status reaches the supervisor.
#
# THE RF CONSTANTS ARE READ OUT OF obs_up.sh, NOT COPIED. Carrier, chip rate, code length and
# comb multiplier live in one table there and nowhere else; a second copy here would be right on
# the day it was written and silently wrong after the first edit. If the table moves, this fails
# loudly rather than guessing.
set -eu

K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}
BROKER=${GNSS_BROKER_URL:-http://127.0.0.1:12060}
FRAME0_URL=${GNSS_FRAME0_URL:-http://chive:54321/get-frame0-time}
OUT=${GNSS_OBS_OUT:-/home/kvand/gnss/fixtures/obs}
UP=$K/scripts/gnss/obs_up.sh

: "${GNSS_CHAIN:?set GNSS_CHAIN to the chain name (the systemd template's %i)}"

row=$(sed -n '/^CHAINS="/,/^"$/p' "$UP" | awk -v c="$GNSS_CHAIN" '$1 == c {print; exit}')
[ -n "$row" ] || {
    echo "FAILED: no row for chain '$GNSS_CHAIN' in the CHAINS table of $UP." >&2
    echo "  Either the chain name is wrong, or the table has moved and this extraction" >&2
    echo "  needs updating -- do not guess the RF constants." >&2
    exit 1
}
set -- $row
chain=$1 sys=$2 carrier=$3 chiprate=$4 codelen=$5 combmult=$6

mkdir -p "$OUT"
exec "$PY" -u "$K/python/scripts/gnss/gnss_observables.py" \
    --url "$BROKER" --combiner "$chain" --search "$chain" --airspy "$chain" \
    --sys "$sys" --band "$chain" \
    --carrier-hz "$carrier" --chip-rate-hz "$chiprate" --code-length "$codelen" \
    --comb-mult "$combmult" \
    --lat 49.32001414 --lon -119.62262691 --alt 545 \
    --frame0-url "$FRAME0_URL" \
    --out "$OUT/${chain}_%Y%m%d.jsonl"
