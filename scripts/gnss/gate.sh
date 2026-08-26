#!/bin/bash
# THE REFACTOR GATE: all seven broker_equiv fixtures, IN PARALLEL, one verdict.
#
# Sequential this is ~5 minutes, which is too slow to run after every extraction -- and a gate
# you skip because it is slow is not a gate. The replays are independent processes over
# independent transcripts, so they parallelise perfectly; wall time becomes the slowest single
# fixture (~90 s, the 172-cycle b2a).
#
#   scripts/gnss/gate.sh            # check every fixture against its golden digest
#   scripts/gnss/gate.sh -v         # ... and show each fixture's line
#
# Exit 0 only if EVERY fixture reports EQUIVALENT. Anything else -- a moved digest, a crash, a
# missing blob -- is a failure, because "the gate did not run" and "the gate passed" must never
# look alike (the fleetdll lesson: eight days green while every frame was BAD_HEADER).
set -u
K=/home/kvand/gnss/kotekan
PY=${GNSS_PY:-/home/kvand/gnss/venv-ft/bin/python}
F=/home/kvand/gnss/fixtures
cd "$K/scripts/gnss" || exit 2

FIXTURES=(
    fixtures/broker_fake_l5.jsonl
    $F/broker_onsky_l5.jsonl.gz
    $F/broker_onsky_l5_holds.jsonl.gz
    $F/broker_onsky_e5a.jsonl.gz
    $F/broker_onsky_e5a_20260826.jsonl.gz
    $F/broker_onsky_l5_20260826.jsonl.gz
    $F/broker_onsky_b2a_20260826.jsonl.gz
)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

for f in "${FIXTURES[@]}"; do
    ( "$PY" broker_equiv.py check "$f" > "$OUT/$(basename "$f").log" 2>&1
      echo $? > "$OUT/$(basename "$f").rc" ) &
done
wait

bad=0
for f in "${FIXTURES[@]}"; do
    b=$(basename "$f")
    rc=$(cat "$OUT/$b.rc" 2>/dev/null || echo 99)
    line=$(grep -E "EQUIVALENT|DIFFERENT|MISSING|Traceback|Error" "$OUT/$b.log" | head -1)
    if [ "$rc" = 0 ] && grep -q EQUIVALENT "$OUT/$b.log"; then
        [ "${1:-}" = "-v" ] && printf "  ok    %-38s %s\n" "$b" "$line"
    else
        bad=$((bad+1))
        printf "  RED   %-38s rc=%s\n" "$b" "$rc"
        tail -12 "$OUT/$b.log" | sed 's/^/          /'
    fi
done

n=${#FIXTURES[@]}
if [ $bad = 0 ]; then
    echo "GATE GREEN  $n/$n EQUIVALENT"
else
    echo "GATE RED    $((n-bad))/$n equivalent, $bad failed"
fi
exit $bad
