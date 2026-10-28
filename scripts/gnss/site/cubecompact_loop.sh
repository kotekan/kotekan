#!/bin/bash
# The compactor loop body: started by cubecompact_up.sh or gnss-cubecompact.service
# (run_component.sh cubecompact), never by hand (see cubecompact_up.sh).
#   $1 python  $2 tool  $3 raw dir  $4 root  $5 period seconds
set -u
PY=$1; TOOL=$2; RAW=$3; ROOT=$4; PERIOD=$5
SELFDIR=$(cd "$(dirname "$0")" && pwd)   # beamcube_daily.sh is this script's sibling
last_rung_day=""
# A stop is a SIGTERM to this loop alone (gnss-cubecompact.service, KillMode=mixed). bash runs
# the trap only once the foreground command completes, so an in-flight gnss_cube_compact.py or
# publish finishes and nothing is left half-written: two compactors on one L0 tree, or half a
# pass, corrupt it. The sleep is backgrounded so a stop during it is immediate.
stopping=0
trap 'stopping=1' TERM
while [ "$stopping" = 0 ]; do
    echo "== $(date -u +%FT%TZ) compact"
    # Newest first, at most 600 raw files a pass (~1 h): the newest days fold first and the
    # rung step below runs between passes, while a backlog is worked through backwards. Raw
    # files are read ahead on 4 threads: reading the raw store is the bound, and it saturates
    # near 4 readers.
    "$PY" -u "$TOOL" compact --raw "$RAW" --out "$ROOT" --newest-first --max-files 600 \
        --readers 4 2>&1
    # Yesterday's rungs, once the day has been closed for 20 min (the last raw file of a day
    # closes ~10 s after midnight; 20 min is slack for a compactor pass to have folded it).
    today=$(date -u +%Y%m%d)
    if [ "$(date -u +%H%M)" -ge 20 ] && [ "$last_rung_day" != "$today" ]; then
        yday=$(date -u -d yesterday +%Y%m%d)
        files=$(ls "$ROOT"/l0/*/*/"$yday".h5 2>/dev/null)
        if [ -n "$files" ]; then
            # last_rung_day lives in memory, so after a restart the files decide whether a
            # day's rungs are done (a re-cut rewrites the whole day). Skip only on one rung12
            # and one rung60 file per L0 file; a lower count (a cut interrupted between files,
            # or before rung60) re-cuts, because a partial rung becomes a partial map. The tool
            # renames each rung file into place once written, so a file that exists is whole.
            n_l0=$(echo "$files" | wc -l)
            n_r12=$(ls "$ROOT"/rung12/*/*/"$yday".h5 2>/dev/null | wc -l)
            n_r60=$(ls "$ROOT"/rung60/*/*/"$yday".h5 2>/dev/null | wc -l)
            if [ "$n_r12" -eq "$n_l0" ] && [ "$n_r60" -eq "$n_l0" ]; then
                echo "== $(date -u +%FT%TZ) rungs for $yday already complete ($n_r12/$n_l0)"
            else
                echo "== $(date -u +%FT%TZ) rungs for $yday ($n_r12/$n_l0 present, rung60 $n_r60)"
                for n in 12 60; do
                    # shellcheck disable=SC2086
                    "$PY" -u "$TOOL" rung --n $n --out "$ROOT" --l0-root "$ROOT" $files 2>&1
                done
                echo "== $(date -u +%FT%TZ) ls $yday"
                # shellcheck disable=SC2086
                "$PY" -u "$TOOL" ls --rel "$ROOT/l0" $files 2>&1 | tail -n 4
            fi
            # Publish the day. The map builder reads the rungs, so this is the first moment
            # yesterday can be drawn. Inline and synchronous: a pause of minutes in compaction
            # once a day costs nothing (raw is never deleted; the next pass folds more) and
            # keeps the ordering plain. A failure here must never stop the compactor: the
            # archive is the primary duty and the map a by-product of it.
            echo "== $(date -u +%FT%TZ) beam-cube publish $yday"
            GNSS_PY="$PY" "$SELFDIR/beamcube_daily.sh" "$yday" 2>&1 \
                || echo "== $(date -u +%FT%TZ) beamcube_daily FAILED rc=$? (compactor continues)"
        fi
        last_rung_day=$today
    fi
    [ "$stopping" = 0 ] || break
    sleep "$PERIOD" & wait $! || true
done
echo "== $(date -u +%FT%TZ) compactor loop stopped on SIGTERM after a complete pass"
