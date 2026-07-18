#!/usr/bin/env bash
# Replay bench v2 (2026-07-17 late eve): the honest single-pass anchor replay.
# Fixes vs the dead 07-17am legs: (1) SINGLE PASS -- no 30 s loop (the loop sawtoothed
# per-sat code phases, poisoning cp-fit slopes by ~2 ppm and breaking every hold);
# (2) v3 carrier flags (the old legs ran an unclamped v1 loop that chased ±50 Hz garbage
# residuals with ±70 Hz trims); (3) BRDC almanac at --almanac-epoch (day-matched; ±2 Hz
# seeds instead of the ±125 Hz blind grid); (4) the carrier squared-branch binary.
#   usage: replay_bench2.sh <legname> <config.yaml>
set -uo pipefail
LEG="$1"; CFG="$2"
SP=/tmp/claude-1000/-home-lwlab-airspy-gps/19ffc4a5-abfb-4bf5-970c-216ad32aea52/scratchpad
cd /home/lwlab/airspy_gps/kotekan
PORT=12148
EPOCH=1784224053.47   # capture sample-0 UTC (mtime - 30 s), 2026-07-16T17:47:33Z
LAT=43.968697; LON=-79.252106; ALT=260
taskset -c 12-19 nice -n 5 ./build_cuda/kotekan/kotekan -c "$CFG" -b 0.0.0.0:$PORT \
    > "$SP/replay_${LEG}_kotekan.log" 2>&1 &
KPID=$!
sleep 6
CARG="--carrier-gain 0.5 --carrier-max-hz 100 --carrier-leak 0.0005 --carrier-min-sig 15 --carrier-max-step 1.0 --carrier-innov-hz 3.0 --carrier-fleet-seed"
ALM="--almanac --lat $LAT --lon $LON --alt $ALT --almanac-epoch $EPOCH --doppler-sign 1 --narrow-search --search-margin-hz 500 --search-margin-wide-hz 3000 --carrier-hz 1575420000"
BARGS="--rest-url http://localhost:$PORT --acquire-snr 6 --interval 0.2 --coast-budget 30
       --hops-per-sec 1000000.0000 --chip-rate-hz 1.023e6"
python3 python/scripts/gnss/gps_distributed_broker.py $BARGS $CARG $ALM \
    --constellation G \
    --detectors gps_search --trackers gps_track --combiner gps_combiner \
    --code-length 1023 --code-bias-file /tmp/replay_cb_gps.ppm \
    > "$SP/replay_${LEG}_broker_gps.log" 2>&1 &
BG=$!
python3 python/scripts/gnss/gps_distributed_broker.py $BARGS $CARG $ALM \
    --constellation C --nh-assist --nh-overlay-len 1800 \
    --detectors bds_search --trackers bds_track --combiner bds_combiner \
    --code-length 10230 --code-bias-file /tmp/replay_cb_bds.ppm \
    > "$SP/replay_${LEG}_broker_bds.log" 2>&1 &
BB=$!
: > "$SP/replay_${LEG}_status.jsonl"
for i in $(seq 1 60); do
    kill -0 $KPID 2>/dev/null || break
    T=$(date +%s.%N)
    for C in gps_combiner bds_combiner; do
        J=$(curl -s -m 2 "http://localhost:$PORT/$C/get_status" || true)
        [ -n "$J" ] && echo "{\"t\":$T,\"combiner\":\"$C\",\"status\":$J}" >> "$SP/replay_${LEG}_status.jsonl"
    done
    sleep 1
done
kill -TERM $BG $BB 2>/dev/null
kill -TERM $KPID 2>/dev/null
sleep 3
echo "leg $LEG done: $(wc -l < "$SP/replay_${LEG}_status.jsonl") status snapshots"
