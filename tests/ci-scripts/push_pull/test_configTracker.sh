#!/usr/bin/env bash

# Runs three kotekan instances in an A -> B -> C chain (see the yaml comments)
# and checks that the configTrackerWriter on C writes A's, B's and its own
# config to disk, matching what each instance serves on its /config endpoint.
#
# Usage: test_configTracker.sh [kotekan_binary]
# Without an argument the binary is taken from <repo>/${KOTEKAN_BUILD_DIRNAME:-build}.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KOTEKAN_EXECUTABLE="${1:-${SCRIPT_DIR}/../../../${KOTEKAN_BUILD_DIRNAME:-build}/kotekan/kotekan}"
if [ ! -x "${KOTEKAN_EXECUTABLE}" ]; then
    echo "kotekan executable not found: ${KOTEKAN_EXECUTABLE}"
    exit 1
fi
KOTEKAN_EXECUTABLE="$(realpath "${KOTEKAN_EXECUTABLE}")"

# Per-user scratch directory: several users may run this on one node. The
# writer's base_dir is relative, so run the instances from this directory.
WORK_DIR="${TMPDIR:-/tmp}/kotekan-${USER}/configTracker"
CONFIG_OUT_DIR="${WORK_DIR}/config_writes"
rm -rf "${WORK_DIR}"
mkdir -p "${CONFIG_OUT_DIR}"
cd "${WORK_DIR}" || exit 1

REST_PORT_1=12048
REST_PORT_2=12748
REST_PORT_3=12848

"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_1.yaml" > kotekan_1.log 2>&1 &
KOTEKAN_PID_1=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_2.yaml" -b 127.0.0.1:${REST_PORT_2} > kotekan_2.log 2>&1 &
KOTEKAN_PID_2=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_3.yaml" -b 127.0.0.1:${REST_PORT_3} > kotekan_3.log 2>&1 &
KOTEKAN_PID_3=$!

# Wait for C to hold both upstream configs (B's via step 1, A's via step 2).
for _ in $(seq 60); do
    sleep 1
    n=$(curl -sf "127.0.0.1:${REST_PORT_3}/config_tracker_upstream_hashes" 2>/dev/null | grep -c '"host"')
    [ "$n" -ge 2 ] && break
done
if [ "$n" -lt 2 ]; then
    echo "Timed out waiting for instance 3 to fetch both upstream configs (got $n)"
fi
sleep 1 # let the writer flush

# Snapshot each instance's full config before shutting down.
curl -sf "127.0.0.1:${REST_PORT_1}/config" > config_1.json
curl -sf "127.0.0.1:${REST_PORT_2}/config" > config_2.json
curl -sf "127.0.0.1:${REST_PORT_3}/config" > config_3.json

ERROR=0
for i in 1 2 3; do
    eval pid=\$KOTEKAN_PID_$i
    kill $pid
    wait $pid
    status=$?
    echo "kotekan instance $i exit status: $status"
    if [ $status -ne 0 ]; then
        echo "kotekan instance $i did not exit cleanly! Log follows:"
        cat kotekan_$i.log
        ERROR=1
    fi
done

# Each written file must hold the matching instance's config, minus the blocks
# with a kotekan_update_endpoint, which the tracker strips.
compare_config() {
    python3 - "$1" "$2" <<'PY'
import json, sys
written, served = sys.argv[1:3]
try:
    got = json.load(open(written))["config"]
except (OSError, ValueError, KeyError) as e:
    print(f"Cannot read config from {written}: {e}")
    sys.exit(1)
want = json.load(open(served))
want = {k: v for k, v in want.items()
        if not (isinstance(v, dict) and "kotekan_update_endpoint" in v)}
if got != want:
    print(f"{written} does not match {served}")
    print(json.dumps(got, indent=1, sort_keys=True))
    sys.exit(1)
PY
}
compare_config "${CONFIG_OUT_DIR}/127.0.0.1_${REST_PORT_1}.json" config_1.json || ERROR=1
compare_config "${CONFIG_OUT_DIR}/127.0.0.1_${REST_PORT_2}.json" config_2.json || ERROR=1
compare_config "${CONFIG_OUT_DIR}/local.json" config_3.json || ERROR=1

if [ "$(ls -1 "${CONFIG_OUT_DIR}" | wc -l)" -ne 3 ]; then
    echo "Expected exactly 3 files in ${CONFIG_OUT_DIR}:"
    ls -1 "${CONFIG_OUT_DIR}"
    ERROR=1
fi

if [ $ERROR -ne 0 ]; then
    echo "configTrackerWriter test failed!"
    exit 1
fi

echo "configTrackerWriter test passed."
exit 0
