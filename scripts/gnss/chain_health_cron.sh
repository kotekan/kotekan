#!/bin/bash
# Live chain-health check -- every 15 minutes from cron on cf06.
#
# Reads the TAIL of today's (and, in the first hour, yesterday's) observables files, so a pass
# costs ~40 MB per chain over NFS rather than the day's several GB, and flags any chain whose
# fleet_present fraction has been under 30% for 15+ minutes ending within the last hour.
#
#   flag   : $OBS/health/ALERT exists (contents = the episode lines); removed when clear
#   log    : $OBS/health/health.log
#   status : $OBS/health/current.json (the same mask format the beam-cube build reads)
#
# Why a file and not a page: nothing on site pages. The viewer, a shell prompt or a human can
# test for the file; the daily beam-cube build reads the per-day mask independently.
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}
OBS=${GNSS_OBS_OUT:-/home/kvand/gnss/fixtures/obs}
H=$OBS/health
mkdir -p "$H" || exit 1
exec 9>"$H/.cron.lock" || exit 1
flock -n 9 || exit 0
TODAY=$(date -u +%Y%m%d); YDAY=$(date -u -d yesterday +%Y%m%d)
shopt -s nullglob
FILES=("$OBS"/*_"$TODAY".jsonl)
[ "$(date -u +%H)" = "00" ] && FILES+=("$OBS"/*_"$YDAY".jsonl)
shopt -u nullglob
[ "${#FILES[@]}" -eq 0 ] && { echo "$(date -u +%FT%TZ) no observables for $TODAY" >> "$H/health.log"; exit 0; }
{
    echo "== $(date -u +%FT%TZ)"
    nice -n 19 "$PY" "$K/python/scripts/gnss/gnss_chain_health.py" "${FILES[@]}" \
        --tail-bytes 40000000 --recent 3600 --min-run 900 \
        --out "$H/current.json" --alert-file "$H/ALERT"
    echo "exit $?"
} >> "$H/health.log" 2>&1
# keep the log bounded (~a week of 15-min passes)
tail -n 20000 "$H/health.log" > "$H/.health.log.tmp" && mv -f "$H/.health.log.tmp" "$H/health.log"
