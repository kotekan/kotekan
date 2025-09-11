#!/usr/bin/env bash

# Variable containing an absolute path to the directory this script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if kotekan executable exists in an expected location
if [ -f "${SCRIPT_DIR}/../../${KOTEKAN_BUILD_DIRNAME}/kotekan/kotekan" ]; then
    KOTEKAN_EXECUTABLE="${SCRIPT_DIR}/../../${KOTEKAN_BUILD_DIRNAME}/kotekan/kotekan"
elif [ -x "${SCRIPT_DIR}/../../build/kotekan/kotekan" ]; then
    KOTEKAN_EXECUTABLE="${SCRIPT_DIR}/../../build/kotekan/kotekan"
else
    echo "kotekan executable not found in expected location: ${SCRIPT_DIR}/../../${KOTEKAN_BUILD_DIRNAME}/kotekan/kotekan"
    echo "or in alternate location: ${SCRIPT_DIR}/../../build/kotekan/kotekan"
    exit 1
fi

# Prepare output directory for config writes
CONFIG_OUT_DIR="${SCRIPT_DIR}/config_writes"
rm -rf "${CONFIG_OUT_DIR}"
mkdir -p "${CONFIG_OUT_DIR}"

# Run two instances of kotekan, keep track of the process IDs to kill them later.
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_1.yaml" &
KOTEKAN_PID_1=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_2.yaml" -b localhost:12748 &
KOTEKAN_PID_2=$!

# Allow some time for kotekan to start and exchange tracker info, then kill the processes.
sleep 5
kill $KOTEKAN_PID_1 $KOTEKAN_PID_2

# Verify that the config writer produced at least one JSON file
num_json=$(find "${CONFIG_OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)
if [ "$num_json" -lt 1 ]; then
    echo "No config JSON files written to ${CONFIG_OUT_DIR}!"
    exit 1
fi

echo "configTrackerWriter test passed: ${num_json} file(s) in ${CONFIG_OUT_DIR}"
exit 0
