#!/usr/bin/env bash
set -uo pipefail
N=${1:-4}; WATCH_S=${2:-480}
KOT=/home/lwlab/airspy_gps/kotekan
SP=/tmp/claude-1000/-home-lwlab-airspy-gps/19ffc4a5-abfb-4bf5-970c-216ad32aea52/scratchpad
OUT=$SP/binomial_results.txt
: > $OUT
for i in $(seq 1 $N); do
    $KOT/config/bandctl.sh stop 3band >/dev/null 2>&1
    $KOT/config/bandctl.sh start 3band >/dev/null 2>&1
    T0=$(date +%s)
    echo "trial $i: started $(date +%H:%M:%S)" >> $OUT
    until curl -s -m 2 http://localhost:12048/l1_gps_combiner/get_status >/dev/null 2>&1; do
        sleep 5
        [ $(( $(date +%s) - T0 )) -gt 180 ] && break
    done
    while [ $(( $(date +%s) - T0 )) -lt $WATCH_S ]; do sleep 15; done
    VERDICT=$(python3 - <<'EOF'
import json, urllib.request
try:
    st=json.load(urllib.request.urlopen("http://localhost:12048/l1_gps_combiner/get_status", timeout=4))
    strong=[r for r in st if r.get("amp_snr",0)>30]
    rails=[r for r in st if abs(r.get("carrier_hz_resid",0))>50 and r.get("amp_snr",0)>30]
    certs=sum(1 for r in st if r.get("coherence_s",0)>0.5)
    healthy_deep=sum(1 for r in strong if r.get("deep_snr",0)>80)
    lab="DISEASED" if len(rails)>=2 else ("HEALTHY" if healthy_deep>=3 else "MARGINAL")
    print("%s  strong=%d rails=%d certs=%d deep80=%d" % (lab, len(strong), len(rails), certs, healthy_deep))
except Exception as e:
    print("POLL-FAIL %s" % e)
EOF
)
    FS=$(grep -c 'FIRST SEED' /tmp/gps_l1_broker.log 2>/dev/null || echo 0)
    FB=$(grep -m1 'FIRST SEED' /tmp/gps_l1_broker.log 2>/dev/null | grep -o 'bias [+-][0-9.]*' || echo n/a)
    echo "trial $i: $VERDICT  (first-seeds=$FS, first bias: $FB)" >> $OUT
done
echo "=== TRIAL COMPLETE ===" >> $OUT
cat $OUT
