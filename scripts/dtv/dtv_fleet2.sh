#!/bin/bash
# rev2 (11-04): DTV subtraction on every node, from cf06 (REST only; dtv_live.py per node, in parallel), for a stage
# with any number of sources (lags -16..+47, 3 sources from the 11-04 cycle). Own states/work/log dirs, so nothing
# is shared with rev1's states (49 lags, 2 sources).
#   dtv_fleet2.sh setup [nodes...]                    # eigen starts + fits of every source in turn, judge window
#   dtv_fleet2.sh refit-loop EVERY_S UNTIL [nodes...]  # UNTIL: UTC, anything date -d reads (2026-11-04T13:00Z)
#       source 0 every round; sources 1.. every EXTRA_EVERY-th round (default 2), never 09:45-11:45Z (Tau A transit
#       ~10:38Z on 11-04, 4 min earlier per day: a weak source's timestream would take Tau A)
# Refuses to run while a rev1 dtv_fleet.sh refit-loop is alive (it arms the same dump gates).
set -u
ACT=${1:?setup|refit-loop}; shift
export DTV_STATES=/home/kvand/gnss/builds/dtv-live/states-rev2 DTV_WORK=/var/tmp/kvand-dtvlive2
PY=~/gnss/venv/bin/python3; L=~/gnss/bb1030/dtv_live.py; LOG=$DTV_WORK/log; mkdir -p $LOG $DTV_STATES
OLD=$(ps -eo pid=,args= | awk '$3 ~ /\/dtv_fleet\.sh$/ && $4 == "refit-loop" {print $1}')
if [ -n "$OLD" ]; then echo "rev1 refit loop still running (PID $OLD): stop it first (kill $OLD)"; exit 3; fi
NSRC=$(curl -s -m 5 http://cx51:12048/run_send_voltage/gpu_0/commands/0/status | $PY -c 'import json,sys; s=json.load(sys.stdin); print(len(s["channels"][0]["act"]), s["lag_min"], s["n_lag"])')
echo "stage on cx51: sources/lag_min/n_lag = $NSRC"
if [ "$ACT" = setup ]; then
  NODES=${*:-cx19 cx27 cx42 cx43 cx44 cx47 cx51 cx52}
  for n in $NODES; do ($PY $L --node $n setup >> $LOG/$n.log 2>&1; echo "$n setup exit $?" >> $LOG/$n.log) & done; wait
  for n in $NODES; do echo "== $n"; grep -E "beam dump|subtracting|exit|rror" $LOG/$n.log | tail -10 | sed -e 's/; mean |s|.*//' -e 's/act per source.*subtracting in/subtracting in/'; done
elif [ "$ACT" = refit-loop ]; then
  EVERY=${1:?every_s}; UNTIL=${2:?until}; shift 2
  END=$(date -u -d "$UNTIL" +%s) || exit 2
  NODES=${*:-cx19 cx27 cx42 cx43 cx44 cx47 cx51 cx52}
  N=$(echo $NSRC | cut -d' ' -f1); EXTRA=$(seq -s' ' 1 $((N - 1)))
  i=0
  while [ "$(date -u +%s)" -lt "$END" ]; do
    i=$((i + 1)); B=""; T=$(date -u +%H%M)
    if [ -n "$EXTRA" ] && [ $((i % ${EXTRA_EVERY:-2})) -eq 0 ] && { [ "$T" \< "0945" ] || [ "$T" \> "1145" ]; }; then B="--extra $EXTRA"; fi
    for n in $NODES; do ($PY $L --node $n refit --seconds 15 $B >> $LOG/$n.log 2>&1) & done; wait
    for n in $NODES; do grep "before the refit" $LOG/$n.log | tail -1; done >> $LOG/refit_summary.log
    sleep $EVERY
  done
fi
