#!/usr/bin/env bash
# A/B the trim pre-compensation on the replay bench. Usage: trim_bench.sh {on|off}
set -u
MODE=$1
cd /home/lwlab/airspy_gps/kotekan
PRECOMP=""
[ "$MODE" = "on" ] && PRECOMP="--trim-precomp"
./build_cuda/kotekan/kotekan -c config/replay_trim_bench.yaml -b 0.0.0.0:12070 \
    > /tmp/trimbench_kotekan_$MODE.log 2>&1 &
KPID=$!
sleep 6
# Flap regime: --hold-snr 1e9 disables hold -> the seed re-fits (grid-scattered doppler)
# every broker cycle, sub-fence steps at ~5 Hz. Shared carrier loop args = live values.
python3 python/scripts/gps_distributed_broker.py \
    --rest-url http://localhost:12070 \
    --detectors search --trackers track --combiner combiner \
    --acquire-snr 6 --interval 0.2 --coast-budget 30 \
    --hops-per-sec 1000000.0 --chip-rate-hz 1.023e6 --code-length 1023 \
    --hold-snr 1e9 \
    --carrier-gain 0.5 --carrier-max-hz 100 --carrier-leak 0.0005 \
    $PRECOMP > /tmp/trimbench_broker_$MODE.log 2>&1 &
BPID=$!
# poll the combiner once a second until kotekan exits (EOF)
while kill -0 $KPID 2>/dev/null; do
    curl -s http://localhost:12070/combiner/get_status 2>/dev/null \
        | python3 -c "
import json,sys,time
try: d=json.load(sys.stdin)
except: sys.exit()
for r in d:
    if (r.get('amp_snr') or 0) > 20:
        print('%.1f PRN%d amp %.0f deep %.0f coh %.2f resid %+.2f'
              % (time.time(), r['prn'], r['amp_snr'], r.get('deep_snr') or 0,
                 r.get('coherence_s') or 0, r.get('carrier_hz_resid') or 0))
" >> /tmp/trimbench_trace_$MODE.log
    sleep 1
done
kill $BPID 2>/dev/null
echo "$MODE done"
