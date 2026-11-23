#!/usr/bin/env bash

# Starts kotekan with no config and drives it over REST, as choco does. A
# config that fails the static checks is refused with 400 and leaves kotekan
# idle. A good config is accepted at once and built on its own thread; /status
# goes to running, and /stop takes it back to idle. A config that passes the
# checks but fails in a stage constructor makes kotekan exit with an error.
#
# Usage: test_daemon_start.sh [kotekan_binary]
# Without an argument the binary is taken from <repo>/${KOTEKAN_BUILD_DIRNAME:-build}.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KOTEKAN_EXECUTABLE="${1:-${SCRIPT_DIR}/../../../${KOTEKAN_BUILD_DIRNAME:-build}/kotekan/kotekan}"
if [ ! -x "${KOTEKAN_EXECUTABLE}" ]; then
    echo "kotekan executable not found: ${KOTEKAN_EXECUTABLE}"
    exit 1
fi
KOTEKAN_EXECUTABLE="$(realpath "${KOTEKAN_EXECUTABLE}")"

WORK_DIR="${TMPDIR:-/tmp}/kotekan-${USER}/daemonStart"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}" || exit 1

URL="127.0.0.1:12948"
ERROR=0

fail() {
    echo "FAIL: $*"
    ERROR=1
}

# config <stage type> <value>: testDataGen -> ExampleConsumer, with the
# producer's stage type and its constant `value` (JSON) given.
config() {
    cat <<EOF
{
    "log_level": "warn",
    "cpu_affinity": [1],
    "telescope": {"num_polarizations": 2, "num_dishes": 2},
    "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 4},
    "data_buffer": {"kotekan_buffer": "standard", "num_frames": 2, "frame_size": 2048,
                    "metadata_pool": "main_pool"},
    "gen_data": {"kotekan_stage": "$1", "type": "const", "value": $2, "out_buf": "data_buffer",
                 "array_shape": [2048], "dim_name": ["S"]},
    "consume": {"kotekan_stage": "ExampleConsumer", "in_buf": "data_buffer"}
}
EOF
}

# post_start <json>: POST a config to /start; prints the HTTP status.
post_start() {
    curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d "$1" \
        "${URL}/start"
}

# wait_status <running> <starting>: poll /status until it reports both, as
# Python literals (True/False). Fails after 30 s.
wait_status() {
    for _ in $(seq 60); do
        if curl -sf "${URL}/status" | python3 -c "
import json, sys
s = json.load(sys.stdin)
sys.exit(0 if (s['running'], s.get('starting')) == ($1, $2) else 1)" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

"${KOTEKAN_EXECUTABLE}" -b "${URL}" > kotekan.log 2>&1 &
KOTEKAN_PID=$!
wait_status False False || fail "kotekan did not come up idle"

code=$(post_start "$(config noSuchStage 1)")
[ "$code" = 400 ] || fail "/start with an unknown stage type returned $code, not 400"
wait_status False False || fail "kotekan is not idle after a refused /start"

code=$(post_start "$(config testDataGen 1)")
[ "$code" = 200 ] || fail "/start returned $code, not 200"
wait_status True False || fail "the pipeline did not reach running"
code=$(curl -s -o /dev/null -w '%{http_code}' "${URL}/config")
[ "$code" = 200 ] || fail "/config returned $code after the start, not 200"

code=$(curl -s -o /dev/null -w '%{http_code}' "${URL}/stop")
[ "$code" = 200 ] || fail "/stop returned $code, not 200"
wait_status False False || fail "kotekan is not idle after /stop"

# A string `value` passes the static checks and fails in testDataGen's constructor.
code=$(post_start "$(config testDataGen '"one"')")
[ "$code" = 200 ] || fail "/start of a config that fails to build returned $code, not 200"
for _ in $(seq 60); do
    kill -0 ${KOTEKAN_PID} 2>/dev/null || break
    sleep 0.5
done
if kill -0 ${KOTEKAN_PID} 2>/dev/null; then
    fail "kotekan did not exit after a failed start"
    kill -9 ${KOTEKAN_PID}
fi
wait ${KOTEKAN_PID}
status=$?
[ $status -ne 0 ] || fail "kotekan exited with status 0 after a failed start"

if [ $ERROR -ne 0 ]; then
    echo "daemon start test failed! Log follows:"
    cat kotekan.log
    exit 1
fi

echo "daemon start test passed (exit status $status after the failed start)."
exit 0
