#!/bin/bash
# Rungs (gnss_cube_compact.py rung --n 12 60, one read of each L0 file) for every closed UTC day
# of the last DAYS, newest first: yesterday's map comes first and a backlog is worked backwards.
# A day is closed once none of its L0 files has changed for 20 min (the fold reaches a day's
# last raw file a few minutes after midnight). The tool skips an L0 file whose rungs are newer
# than it and renames each rung into place once whole, so a stopped run resumes where it left
# off. Started in the background by cube_staging_loop.sh; also safe by hand.
#   cube_rungs.sh PY TOOL ROOT [MAX_READ_MBPS] [DAYS]
set -u
PY=$1; TOOL=$2; ROOT=$3; MBPS=${4:-0}; DAYS=${5:-14}
child=
trap '[ -z "$child" ] || kill -TERM "$child" 2>/dev/null; exit 143' TERM
log() { echo "== $(date -u +%FT%TZ) rungs: $*"; }

count() { ls "$ROOT"/"$1"/*/*/"$2".h5 2>/dev/null | wc -l; }

now=$(date -u +%s)
for k in $(seq 1 "$DAYS"); do
    day=$(date -u -d "@$((now - k * 86400))" +%Y%m%d)
    files=$(ls "$ROOT"/l0/*/*/"$day".h5 2>/dev/null)
    [ -n "$files" ] || continue
    # shellcheck disable=SC2086
    if [ -n "$(find $files -mmin -20)" ]; then
        log "$day still being written; left for the next run"
        continue
    fi
    n_l0=$(echo "$files" | wc -l)
    log "$day: $n_l0 L0 file(s), $MBPS MB/s"
    # shellcheck disable=SC2086
    "$PY" -u "$TOOL" rung --n 12 60 --skip-done --max-read-mbps "$MBPS" --out "$ROOT" \
        --l0-root "$ROOT" $files 2>&1 &
    child=$!
    wait "$child"
    rc=$?
    child=
    log "$day rc=$rc: rung12 $(count rung12 "$day")/$n_l0, rung60 $(count rung60 "$day")/$n_l0"
done
