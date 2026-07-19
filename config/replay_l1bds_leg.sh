#!/usr/bin/env bash
# L1-BDS replay leg: the 07-18 17:37 capture through replay_l1_1737.yaml + the broker.
# v3 bench: baseline / no-regression / poisoned-trim watchdog rescue.
# Usage: replay_l1bds_leg.sh <legname> [extra broker args...]
set -uo pipefail
LEG="$1"; shift
EXTRA="$@"
SP=${REPLAY_OUT:-/tmp/replay_legs}; mkdir -p "$SP"
cd /home/lwlab/airspy_gps/kotekan
PORT=12158
EPOCH=1784410658.0    # capture sample-0 UTC (mtime - 60 s), 2026-07-18 21:37:38 UTC
LAT=43.968697; LON=-79.252106; ALT=260
mkdir -p /tmp/gpswipe
taskset -c 8-13 nice -n 10 ./build_cuda/kotekan/kotekan -c config/replay_l1_1737.yaml -b 0.0.0.0:$PORT \
    > "$SP/replay_${LEG}_kotekan.log" 2>&1 &
KPID=$!
sleep 6
CARG="--carrier-gain 0.5 --carrier-max-hz 100 --carrier-leak 0.0005 --carrier-min-sig 15 --carrier-max-step 1.0 --carrier-innov-hz 3.0 --carrier-fleet-seed"
ALM="--almanac --lat $LAT --lon $LON --alt $ALT --almanac-epoch $EPOCH --doppler-sign 1 --tle https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle --tle-name-filter BEIDOU-3 --narrow-search --search-margin-hz 500 --search-margin-wide-hz 3651"
GNSS_SEED_DEBUG=20,23 taskset -c 8-13 nice -n 10 python3 python/scripts/gnss/gps_distributed_broker.py \
    --rest-url http://localhost:$PORT --acquire-snr 6 --interval 0.2 --coast-budget 30 \
    --hops-per-sec 1000000.0000 --chip-rate-hz 1.023e6 --code-length 10230 \
    --constellation C --noise-probes 4 --dead-reckon --dr-constellation C \
    --detectors bds_search --trackers bds_track --combiner bds_combiner \
    --carrier-hz 1575420000 --nh-assist --nh-overlay-len 1800 \
    --code-bias-file /tmp/replay_cb_l1bds.ppm $CARG $ALM $EXTRA \
    > "$SP/replay_${LEG}_broker.log" 2>&1 &
BG=$!
: > "$SP/replay_${LEG}_status.jsonl"
: > "$SP/replay_${LEG}_det.jsonl"
for i in $(seq 1 100); do
    kill -0 $KPID 2>/dev/null || break
    T=$(date +%s.%N)
    J=$(curl -s -m 2 "http://localhost:$PORT/bds_combiner/get_status" || true)
    [ -n "$J" ] && echo "{\"t\":$T,\"status\":$J}" >> "$SP/replay_${LEG}_status.jsonl"
    D=$(curl -s -m 2 "http://localhost:$PORT/bds_search/get_detections" || true)
    [ -n "$D" ] && echo "{\"t\":$T,\"det\":$D}" >> "$SP/replay_${LEG}_det.jsonl"
    sleep 1
done
kill -TERM $BG 2>/dev/null
kill -TERM $KPID 2>/dev/null
sleep 3
kill -0 $KPID 2>/dev/null && kill -9 $KPID 2>/dev/null
echo "leg $LEG done: $(wc -l < "$SP/replay_${LEG}_status.jsonl") status snapshots"
