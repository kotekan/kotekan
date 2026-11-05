#!/bin/bash
# DTV subtraction on every node, from cf06 (REST only; dtv_live.py per node, in parallel).
#   dtv_fleet.sh setup [nodes...]                    # eigen starts + fits of sources 0, 1, judge window
#   dtv_fleet.sh refit-loop EVERY_S UNTIL [nodes...]   # UNTIL: UTC, anything date -d reads (2026-11-04T13:00Z)
#                                                      # source 0 every round, source 1 every 4th
#                                                           (not 09:45-11:45Z: the Tau A transit)
# Logs: /var/tmp/kvand-dtvlive/log/<node>.log
set -u
ACT=${1:?setup|refit-loop}; shift
PY=~/gnss/venv/bin/python3; L=~/gnss/bb1030/dtv_live.py; LOG=/var/tmp/kvand-dtvlive/log; mkdir -p $LOG
if [ "$ACT" = setup ]; then
  NODES=${*:-cx19 cx27 cx42 cx43 cx44 cx47 cx51 cx52}
  for n in $NODES; do ($PY $L --node $n setup >> $LOG/$n.log 2>&1; echo "$n setup exit $?" >> $LOG/$n.log) & done; wait
  for n in $NODES; do echo "== $n"; grep -E "beam dump|subtracting|exit|rror" $LOG/$n.log | tail -8 | sed -e 's/; mean |s|.*//' -e 's/act per source.*subtracting in/subtracting in/'; done
elif [ "$ACT" = refit-loop ]; then
  EVERY=${1:?every_s}; UNTIL=${2:?until}; shift 2
  END=$(date -u -d "$UNTIL" +%s) || exit 2
  NODES=${*:-cx19 cx27 cx42 cx43 cx44 cx47 cx51 cx52}
  i=0
  while [ "$(date -u +%s)" -lt "$END" ]; do
    i=$((i + 1)); B=""; T=$(date -u +%H%M)
    # source 1 every 4th round, never near the Tau A transit (its weak timestream would take Tau A)
    if [ $((i % 4)) -eq 0 ] && { [ "$T" \< "0945" ] || [ "$T" \> "1145" ]; }; then B=--both; fi
    for n in $NODES; do ($PY $L --node $n refit --seconds 15 $B >> $LOG/$n.log 2>&1) & done; wait
    for n in $NODES; do grep "before the refit" $LOG/$n.log | tail -1; done >> $LOG/refit_summary.log
    sleep $EVERY
  done
fi
