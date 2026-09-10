#!/bin/bash
# The compactor loop body -- started by cubecompact_up.sh, never by hand (see that file).
#   $1 python  $2 tool  $3 raw dir  $4 root  $5 utc0 (v2 epoch)  $6 period seconds
set -u
PY=$1; TOOL=$2; RAW=$3; ROOT=$4; UTC0=$5; PERIOD=$6
SELFDIR=$(cd "$(dirname "$0")" && pwd)   # beamcube_daily.sh is this script's sibling
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
            # `last_rung_day` lives in memory, so a restart used to re-cut a day whose rungs
            # were already complete -- ~15 GB rewritten and several minutes of no compaction,
            # every time. Skip only on a STRICT match: one rung12 file per L0 file. A count
            # that differs (an interrupted cut) re-cuts, because a partial rung silently
            # becomes a partial map.
            n_l0=$(echo "$files" | wc -l)
            n_r12=$(ls "$ROOT"/rung12/*/*/"$yday".h5 2>/dev/null | wc -l)
            if [ "$n_r12" -eq "$n_l0" ]; then
                echo "== $(date -u +%FT%TZ) rungs for $yday already complete ($n_r12/$n_l0)"
            else
                echo "== $(date -u +%FT%TZ) rungs for $yday ($n_r12/$n_l0 present)"
                for n in 12 60; do
                    # shellcheck disable=SC2086
                    "$PY" -u "$TOOL" rung --n $n --out "$ROOT" --l0-root "$ROOT" $files 2>&1
                done
                echo "== $(date -u +%FT%TZ) ls $yday"
                # shellcheck disable=SC2086
                "$PY" -u "$TOOL" ls --rel "$ROOT/l0" $files 2>&1 | tail -n 4
            fi
            # ⚡ PUBLISH THE DAY. The rungs are what the map builder reads, so this is the
            # first moment yesterday can be drawn -- and until 2026-09-10 the two commands
            # that do it were hand-run, with a failure mode nobody can see (the viewer keeps
            # serving its last export, which looks exactly like a healthy viewer).
            # Inline and synchronous on purpose: a ~6 min pause in compaction once a day
            # costs nothing (raw is never deleted, the next pass just folds more) and keeps
            # the ordering legible. A failure here must NEVER take the compactor down -- the
            # archive is the primary duty and the map is a by-product of it.
            echo "== $(date -u +%FT%TZ) beam-cube publish $yday"
            GNSS_PY="$PY" "$SELFDIR/beamcube_daily.sh" "$yday" 2>&1 \
                || echo "== $(date -u +%FT%TZ) beamcube_daily FAILED rc=$? (compactor continues)"
        fi
        last_rung_day=$today
    fi
    sleep "$PERIOD"
done
