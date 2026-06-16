#!/usr/bin/env bash

# Variable containing an absolute path to the directory this script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Directory name where kotekan was built, relative to the root of the repository
# Use KOTEKAN_BUILD_DIRNAME environment variable if set, otherwise default to "build"
KOTEKAN_BUILD_DIR="${SCRIPT_DIR}/../../../${KOTEKAN_BUILD_DIRNAME:-build}"
# Resolve to absolute path
KOTEKAN_BUILD_DIR="$(realpath "${KOTEKAN_BUILD_DIR}")"
# Create build directory if it does not exist
mkdir -p "${KOTEKAN_BUILD_DIR}"

# Check if kotekan executable exists in an expected location
if [ -f "${KOTEKAN_BUILD_DIR}/kotekan/kotekan" ]; then
    KOTEKAN_EXECUTABLE="${KOTEKAN_BUILD_DIR}/kotekan/kotekan"
else
    echo "kotekan executable not found in expected location: ${KOTEKAN_BUILD_DIR}/kotekan/kotekan. Attempting to build kotekan."
    # Attempt to build kotekan if the executable is not found
    cd "${KOTEKAN_BUILD_DIR}"
    cmake -Wdev -Werror=dev -Wdeprecated -Werror=deprecated -DWERROR=ON -DCMAKE_LINK_WHAT_YOU_USE=ON -DCMAKE_BUILD_TYPE=Test -DNO_MEMLOCK=ON -DUSE_OMP=ON -DCCACHE=ON ..
    # Check for errors
    if [ $? -ne 0 ]; then
        echo "CMake configuration failed. Please check the output for errors."
        exit 1
    fi
    make -j $(nproc)
    # Check for errors
    if [ $? -ne 0 ]; then
        echo "Build failed. Please check the output for errors."
        exit 1
    fi
    # Check again for the kotekan executable
    if [ -f "${KOTEKAN_BUILD_DIR}/kotekan/kotekan" ]; then
        KOTEKAN_EXECUTABLE="${KOTEKAN_BUILD_DIR}/kotekan/kotekan"
    else
        echo "kotekan executable still not found after build attempt. Exiting."
        exit 1
    fi
fi


# Prepare output directory for config writes
CONFIG_OUT_DIR="${KOTEKAN_BUILD_DIR}/config_writes"
rm -rf "${CONFIG_OUT_DIR}"
mkdir -p "${CONFIG_OUT_DIR}"

# Run three instances of kotekan in an A -> B -> C chain. This exercises both
# steps of getUpstreamConfigs on instance 3:
#   - step 1: fetches B's local config via /config_tracker_local
#   - step 2: discovers and fetches A's config via B's upstream-hashes index
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_1.yaml" &
KOTEKAN_PID_1=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_2.yaml" -b 127.0.0.1:12748 &
KOTEKAN_PID_2=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_3.yaml" -b 127.0.0.1:12848 &
KOTEKAN_PID_3=$!

# Allow some time for kotekan to start and the chain to converge (frames must
# flow A -> B and trigger B -> A config fetch, then frames B -> C and trigger
# C -> B config fetch, including the upstream-by-hash step).
sleep 3

# Kill kotekan processes, make sure they exit cleanly.
kill $KOTEKAN_PID_1
wait $KOTEKAN_PID_1
EXIT_STATUS_1=$?
kill $KOTEKAN_PID_2
wait $KOTEKAN_PID_2
EXIT_STATUS_2=$?
kill $KOTEKAN_PID_3
wait $KOTEKAN_PID_3
EXIT_STATUS_3=$?

sleep 1 # Wait a moment to ensure output is flushed

ERROR=0

# Print exit statuses
echo "kotekan instance 1 exit status: $EXIT_STATUS_1"
echo "kotekan instance 2 exit status: $EXIT_STATUS_2"
echo "kotekan instance 3 exit status: $EXIT_STATUS_3"
# Exit with error if any instance did not exit cleanly
if [ $EXIT_STATUS_1 -ne 0 ] || [ $EXIT_STATUS_2 -ne 0 ] || [ $EXIT_STATUS_3 -ne 0 ]; then
    echo "One or more kotekan instances did not exit cleanly!"
    ERROR=1
fi

# Verify that the config writer (on instance 3) produced exactly three JSON
# files:
#   local.json              -- instance 3's own (local) config; identity-by-content
#   127.0.0.1_12748.json    -- B's local config, fetched by C's bufferRecv via
#                              /config_tracker_local and re-keyed under
#                              (client_ip, upstream_rest_port) = (127.0.0.1, 12748)
#                              (step 1)
#   127.0.0.1_12048.json    -- A's config, discovered via B's upstream-hashes
#                              index and fetched as
#                              /config_tracker_upstream_configs?hash=...
#                              (step 2)
if [ "$(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)" -ne 3 ]; then
    echo "Expected 3 JSON files in ${CONFIG_OUT_DIR}, found $(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)"
    ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null
    ERROR=1
fi

# Prune node/code-dependent lines from json output before comparing
for file in "${CONFIG_OUT_DIR}"/*.json; do
    # Remove lines with "kotekan_build_branch", "kotekan_git_commit_hash", "kotekan_version", "kotekan_cmake_options"
    sed -i '/"kotekan_build_branch":/d' "$file"
    sed -i '/"kotekan_git_commit_hash":/d' "$file"
    sed -i '/"kotekan_version":/d' "$file"
    sed -i '/"kotekan_cmake_options":/d' "$file"
done

# Compare files against expected md5 sums.
#
# NOTE: these literals must be regenerated whenever the canonical
# ConfigTracker JSON format or hashing changes. After the local/upstream
# split, the on-disk JSONs include a `json_hash` field whose value depends on
# how the entry is keyed (local: hash over JSON only; upstream: hash over
# host:port|JSON), so the file md5 changes when that protocol changes. On
# mismatch the script prints the actual hash and the (pruned) file contents
# so the literal can be updated in one shot.
check_file_hash() {
    local file="$1"
    local expected="$2"
    if [ ! -f "$file" ]; then
        echo "Expected file not found: $file"
        return 1
    fi
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "MD5 checksum for $file does not match expected value!"
        echo "  Expected $expected, got ${actual} instead."
        echo "File contents:"
        echo "--------------"
        cat "$file"
        echo ""
        echo "--------------"
        return 1
    fi
    return 0
}

# Instance 3's own (local) config
EXPECTED_LOCAL_HASH="92d32f5195eaeb8f9354a3a8eb860115"
if ! check_file_hash "${CONFIG_OUT_DIR}/local.json" "$EXPECTED_LOCAL_HASH"; then
    ERROR=1
fi

# B's local config as seen by C (step 1, re-keyed under 127.0.0.1:12748)
EXPECTED_B_HASH="3cf1d34c37eb9179cf078dedd9ba0a68"
if ! check_file_hash "${CONFIG_OUT_DIR}/127.0.0.1_12748.json" "$EXPECTED_B_HASH"; then
    ERROR=1
fi

# A's config, discovered via B's upstream-hashes (step 2; B already had A
# stored under 127.0.0.1:12048, and C trusts that key transitively).
EXPECTED_A_HASH="7a4f644e999a4c6789d187b2c385406e"
if ! check_file_hash "${CONFIG_OUT_DIR}/127.0.0.1_12048.json" "$EXPECTED_A_HASH"; then
    ERROR=1
fi

if [ $ERROR -ne 0 ]; then
    echo "configTrackerWriter test failed!"
    exit 1
fi

echo "configTrackerWriter test passed: three matching config files found in ${CONFIG_OUT_DIR}."
exit 0
