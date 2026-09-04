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

# ---- THE UNIT TESTS ---------------------------------------------------------------------
# The digest gate is blind by construction to anything the transcript cannot drive: the #90
# admission gate strikes on the WALL clock, and #92's handover posts to a gather that never
# answers in a replay. Those live in gnss_broker/test_*.py, and they belong to the same
# verdict -- a refactor is safe only if BOTH halves are green.
units=0
for t in "$K"/python/scripts/gnss/gnss_broker/test_*.py; do
    [ -e "$t" ] || continue
    m="gnss_broker.$(basename "$t" .py)"
    if ! ( cd "$K/python/scripts/gnss" && "$PY" -m "$m" > "$OUT/$m.log" 2>&1 ); then
        units=$((units+1))
        printf "  RED   %-38s (unit)\n" "$m"
        grep -E "FAIL|Error|Traceback" "$OUT/$m.log" | head -8 | sed 's/^/          /'
    elif [ "${1:-}" = "-v" ]; then
        printf "  ok    %-38s %s\n" "$m" "$(grep -c PASS "$OUT/$m.log") checks"
    fi
done

# ⚠️ selftest.py DOES NOT MATCH THE GLOB ABOVE, AND THAT COST FOUR DAYS OF RED (2026-08-27).
# The mean->median gauge change (08-23) invalidated two of its assertions and NOBODY SAW,
# because nothing ran it -- the same defect class as the fleetdll symlink that was red for
# eight days while reporting green. It is the broker's oldest check suite (the filter's
# whole robustness ledger lives there); it belongs to the same verdict as everything else.
if ! ( cd "$K/python/scripts/gnss" && "$PY" gnss_broker/selftest.py > "$OUT/selftest.log" 2>&1 ); then
    units=$((units+1))
    printf "  RED   %-38s (unit)\n" "gnss_broker.selftest"
    grep -E "FAIL|Error|Traceback" "$OUT/selftest.log" | head -8 | sed 's/^/          /'
elif [ "${1:-}" = "-v" ]; then
    printf "  ok    %-38s %s\n" "gnss_broker.selftest" "$(grep -c ' ok ' "$OUT/selftest.log") checks"
fi

# ---- THE STATIC PASS --------------------------------------------------------------------
# 2026-08-26: trimarm.py used C_LIGHT with no import for ~25 min of production. The digest
# gate CANNOT see it (the shadow needs a live gather, so the line never runs in replay) and
# the except clause turned the NameError into a "shadow only" log line. Undefined names in
# unexecuted paths are exactly what a static checker exists for. Proven able to fail: the
# pre-fix trimarm.py yields "undefined name 'C_LIGHT'" here.
static=0
if ! "$PY" -m pyflakes "$K"/python/scripts/gnss/gnss_broker/*.py \
        "$K"/python/scripts/gnss/gps_distributed_broker.py 2>&1 \
        | grep -E "undefined name|referenced before assignment" > "$OUT/static.log"; then
    :  # no undefined names -- green
else
    static=1
    printf "  RED   %-38s (static)\n" "pyflakes undefined-name pass"
    head -8 "$OUT/static.log" | sed 's/^/          /'
fi

n=${#FIXTURES[@]}
if [ $bad = 0 ] && [ $units = 0 ] && [ $static = 0 ]; then
    echo "GATE GREEN  $n/$n EQUIVALENT + unit tests"
else
    echo "GATE RED    $((n-bad))/$n equivalent, $bad fixtures / $units unit modules / $static static failed"
fi
exit $((bad + units + static))
