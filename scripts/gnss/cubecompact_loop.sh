#!/bin/bash
# The compactor loop body -- started by cubecompact_up.sh, never by hand (see that file).
#   $1 python  $2 tool  $3 raw dir  $4 root  $5 utc0 (v2 epoch)  $6 period seconds
set -u
PY=$1; TOOL=$2; RAW=$3; ROOT=$4; UTC0=$5; PERIOD=$6
last_rung_day=""
while true; do
    echo "== $(date -u +%FT%TZ) compact"
    "$PY" -u "$TOOL" compact --raw "$RAW" --out "$ROOT" --utc0 "$UTC0" 2>&1
    # Yesterday's rungs, once the day has been closed for 20 min (the last raw file of a day
    # closes ~10 s after midnight; 20 min is slack for a compactor pass to have folded it).
    today=$(date -u +%Y%m%d)
    if [ "$(date -u +%H%M)" -ge 20 ] && [ "$last_rung_day" != "$today" ]; then
        yday=$(date -u -d yesterday +%Y%m%d)
        files=$(ls "$ROOT"/l0/*/*/"$yday".h5 2>/dev/null)
        if [ -n "$files" ]; then
            echo "== $(date -u +%FT%TZ) rungs for $yday"
            for n in 12 60; do
                # shellcheck disable=SC2086
                "$PY" -u "$TOOL" rung --n $n --out "$ROOT" --l0-root "$ROOT" $files 2>&1
            done
            echo "== $(date -u +%FT%TZ) ls $yday"
            # shellcheck disable=SC2086
            "$PY" -u "$TOOL" ls --rel "$ROOT/l0" $files 2>&1 | tail -n 4
        fi
        last_rung_day=$today
    fi
    sleep "$PERIOD"
done
