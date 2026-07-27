#!/usr/bin/env bash
# L5 PEEL A/B HARNESS -- run the SAME captured voltage through the L5 chain twice, peel off
# vs peel on, and snapshot the combiner state each time.
#
#   ./peel_ab_l5.sh <label>        # e.g. ./peel_ab_l5.sh A_peeloff
#
# Reads config/replay_l5_peel.yaml (whatever peel state it currently has) over the capture in
# /tmp/gpsin_l5, brings up the real L5 control plane against it, lets it settle, then dumps
# every combiner's per-PRN status to /tmp/peel_ab/<label>_*.json.
#
# WHY A SCRIPT: an A/B is only an A/B if both legs run the IDENTICAL procedure. Doing this by
# hand twice invites a difference in timing or teardown that then masquerades as a result.
#
# Run from the kotekan repo root. Bench cores only (14-19) -- the soak owns 0-13.
set -u
LABEL=${1:?usage: peel_ab_l5.sh <label>}
CFG=config/replay_l5_peel.yaml
PORT=12448   # DEDICATED bench port: never collide with the live soak on 12048
SETTLE=${SETTLE:-75}          # seconds of replay before the snapshot
OUT=/tmp/peel_ab
mkdir -p "$OUT"

cleanup() {
    echo "  tearing down..."
    [ -n "${CPPID:-}" ] && kill -TERM "$CPPID" 2>/dev/null
    sleep 3
    pkill -f "gps_distributed_broker.py.*localhost:$PORT" 2>/dev/null
    pkill -f "gps_status_logger.py.*localhost:$PORT" 2>/dev/null
    pkill -f "gnss_observables.py.*localhost:$PORT" 2>/dev/null
    [ -n "${KPID:-}" ] && kill -TERM "$KPID" 2>/dev/null
    sleep 3
    [ -n "${KPID:-}" ] && kill -9 "$KPID" 2>/dev/null
}
trap cleanup EXIT

if curl -s --max-time 2 -o /dev/null http://localhost:$PORT/endpoints; then
    echo "ABORT: something is already serving :$PORT -- a stale bench would answer instead" >&2
    exit 1
fi

echo "=== leg $LABEL: peel state in $CFG ==="
grep -c "peel: true" "$CFG" | sed 's/^/  peel:true count = /'

rm -rf /tmp/gps_l5_replay && mkdir -p /tmp/gps_l5_replay
nice -n 5 taskset -c 14-19 ./build_cuda/kotekan/kotekan -c "$CFG" -b 0.0.0.0:$PORT \
    > "/tmp/replay_l5_${LABEL}.log" 2>&1 &
KPID=$!
echo "  kotekan pid=$KPID, waiting for REST..."
for i in $(seq 1 40); do
    sleep 1
    [ "$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' http://localhost:$PORT/endpoints)" = "200" ] && break
done
if ! kill -0 "$KPID" 2>/dev/null; then
    echo "ABORT: kotekan died during startup:" >&2
    grep -E "^ERROR|FATAL" "/tmp/replay_l5_${LABEL}.log" | head -3 >&2
    exit 1
fi
echo "  REST up after ${i}s (our kotekan pid=$KPID alive)"

# Real control plane, kotekan owned by US (SKIP_KOTEKAN=1). No almanac assist: the replay has
# no /adcstat endpoint, so dead-reckon/CL time-assist have nothing to anchor to.
# --bias-min-sats 99: pin the broker PRE-SOLVE for the whole bench. Measured three times
# (2026-07-27): with search-measured seeds + the shared carrier loop, PRN20 reaches asnr 81,
# 0.5 s coherence, resid 0.6 Hz -- then the instant the clock-bias solves and seeding switches
# to pred+bias, the seed steps by 20-40 Hz (replay epoch/pacing skew the common bias cannot
# absorb per-sat) and the tracker is yanked off. The bench tests the PEEL, not the bias
# machinery: keep seeds search-measured, let the CAR loop own the fine carrier.
# ALMANAC PINNED TO THE CAPTURE EPOCH (--almanac-epoch). Search alone seeds Doppler only to
# its 100 Hz grid, and 50 Hz of error destroys the 1 s coherent deep integration -- measured
# 2026-07-27: DLL locked hard (disc +-0.03 chips) while deep_snr sat at 2.5, i.e. code fine,
# carrier incoherent. Live, the almanac supplies ~1 Hz; here it must be evaluated at the time
# the DATA was taken, not now, or it is worse than useless (this capture is hours old).
# NO --dop-continuous: dead-reckon wants an /adcstat anchor the replay cannot serve, and the
# broker just spins on "adcstat anchor unavailable (404); retrying".
: "${CAP_EPOCH:?set CAP_EPOCH to the sample-0 UTC of the capture}"
# FROZEN WARM-STARTED CLOCK BIAS. --bias-min-sats 99 alone DEADLOCKS: the bias never solves,
# and the broker's bias-unsolved first-seed guard (a "~2 s hold-off" that assumes the bias
# solves next cycle) then blocks every real satellite's FIRST seed forever -- measured
# 2026-07-27: PRN20/21 detected for 4 min straight, active=[probes] only. Warm-starting via
# --clock-bias-file makes the bias non-None from cycle 1 (guard passes); min-sats 99 then
# FREEZES it (the EMA update sits behind the same gate), which is the point: the replay's
# epoch/pacing skew poisons any re-solve. BENCH_BIAS: the LO offset the live L5 band
# measured around the capture (+24..+42 Hz on 2026-07-27; persistent file read +34.9).
BENCH_BIAS=${BENCH_BIAS:-35.0}
BIAS_DIR=${BIAS_DIR:-$HOME/.cache/kotekan_gps}
printf '%s 5 %s\n' "$BENCH_BIAS" "$(date +%s)" > "$BIAS_DIR/gps_clock_bias_l5ab_${LABEL}.hz"
echo "  bench clock bias warm-start: $BENCH_BIAS Hz -> gps_clock_bias_l5ab_${LABEL}.hz"
SKIP_KOTEKAN=1 STAGE_PREFIX="" CFG="$CFG" PORT=$PORT HTTP_PORT=8092 TAG="l5ab_$LABEL" \
    LAT=43.968697 LON=-79.252106 ALT=260 \
    BROKER_EXTRA="--almanac-epoch $CAP_EPOCH --bias-min-sats 99 --bias-stale-s 0" \
    nice -n 5 taskset -c 14-19 ./config/run_live.sh > "/tmp/ctl_l5_${LABEL}.log" 2>&1 &
CPPID=$!
echo "  control plane pid=$CPPID; settling ${SETTLE}s..."
sleep "$SETTLE"

echo "  snapshotting..."
for st in gps_combiner gal_combiner bds_combiner; do
    curl -s --max-time 8 "http://localhost:$PORT/$st/get_status" \
        > "$OUT/${LABEL}_${st}.json" 2>/dev/null
    n=$(python3 -c "
import json;r=json.load(open('$OUT/${LABEL}_${st}.json'))
print('%d rows, %d locked' % (len(r), sum(1 for x in r if (x.get('deep_snr') or 0)>10)))" 2>/dev/null || echo "unreadable")
    echo "    $st: $n"
done
# The peel health lines are the other half of the story (ratio/floor, HURTING).
grep "peel\[" "/tmp/replay_l5_${LABEL}.log" | tail -20 > "$OUT/${LABEL}_peelhealth.txt" 2>/dev/null
echo "  peel health lines captured: $(wc -l < "$OUT/${LABEL}_peelhealth.txt")"
echo "=== leg $LABEL done ==="
