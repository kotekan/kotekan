#!/bin/bash
# #154 safety net for the shared element model's phase pin -- every 5 minutes from cron on gnss.
#
# Compares every assembler's shared model with the committed fleet reference (the manifest's
# elem-shared-ref, the file the node configs were generated from):
#
#   flag   : $OBS/elem_ref/ALERT exists while something needs a person; removed when clear
#   log    : $OBS/elem_ref/watch.log
#   status : $OBS/elem_ref/current.json (per band R, per instance offset and similarity)
#
# ELEM_REF_ACT=1 lets it act (elem_shared_ref.py watch --act): re-post the reference to an
# instance that lost it or is pinned off it, and re-take a band's reference from the fleet
# consensus when the reference no longer describes the models (new F-engine gains), into
# $OBS/elem_ref/auto_ref.json until that file is committed. Unset, it only reports.
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv/bin/python}
OBS=${GNSS_OBS_OUT:-/home/kvand/gnss/fixtures/obs}
REF=${ELEM_REF_FILE:-$K/$(grep -oP '^\s+elem-shared-ref:\s*\K\S+' "$K/config/gnss_fleet_chord.yaml")}
MODE=$(grep -oP '^\s+elem-shared-ref-mode:\s*\K\S+' "$K/config/gnss_fleet_chord.yaml")
D=$OBS/elem_ref
mkdir -p "$D" || exit 1
exec 9>"$D/.cron.lock" || exit 1
flock -n 9 || exit 0
[ -f "$REF" ] || { echo "$(date -u +%FT%TZ) no reference file $REF" >> "$D/watch.log"; exit 0; }
ACT=()
[ "${ELEM_REF_ACT:-0}" = "1" ] && ACT=(--act)
nice -n 19 "$PY" "$K/python/scripts/gnss/elem_shared_ref.py" watch "$REF" "$D" --mode "${MODE:-live}" "${ACT[@]}" \
    >> "$D/watch.log" 2>&1
# keep the log bounded (~a week of 5-min passes)
tail -n 20000 "$D/watch.log" > "$D/.watch.log.tmp" && mv -f "$D/.watch.log.tmp" "$D/watch.log"
