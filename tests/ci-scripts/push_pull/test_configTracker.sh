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
    cmake -Wdev -Werror=dev -Wdeprecated -Werror=deprecated -DWERROR=ON -DCMAKE_LINK_WHAT_YOU_USE=ON -DCMAKE_BUILD_TYPE=Test -DNO_MEMLOCK=ON -DUSE_OMP=ON -DUSE_CUDA=ON -DCCACHE=ON ..
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

# Run two instances of kotekan, keep track of the process IDs to kill them later.
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_1.yaml" &
KOTEKAN_PID_1=$!
"${KOTEKAN_EXECUTABLE}" -c "${SCRIPT_DIR}/test_configTracker_2.yaml" -b 127.0.0.1:12748 &
KOTEKAN_PID_2=$!

# Allow some time for kotekan to start and exchange tracker info, then kill the processes.
sleep 2

# Kill kotekan processes, make sure they exit cleanly.
kill $KOTEKAN_PID_1
wait $KOTEKAN_PID_1
EXIT_STATUS_1=$?
kill $KOTEKAN_PID_2
wait $KOTEKAN_PID_2
EXIT_STATUS_2=$?

sleep 1 # Wait a moment to ensure output is flushed

ERROR=0

# Print exit statuses
echo "kotekan instance 1 exit status: $EXIT_STATUS_1"
echo "kotekan instance 2 exit status: $EXIT_STATUS_2"
# Exit with error if either instance did not exit cleanly
if [ $EXIT_STATUS_1 -ne 0 ] || [ $EXIT_STATUS_2 -ne 0 ]; then
    echo "One or both kotekan instances did not exit cleanly!"
    ERROR=1
fi

# Verify that the config writer produced exactly two JSON files
if [ "$(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)" -ne 2 ]; then
    echo "Expected 2 JSON files in ${CONFIG_OUT_DIR}, found $(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)"
    ERROR=1
fi

# Prune node/code-dependent lines from json output before comparing
for file in "${CONFIG_OUT_DIR}"/*.json; do
    # Remove lines with "kotekan_build_branch", "kotekan_git_commit_hash", "kotekan_version"
    sed -i '/"kotekan_build_branch":/d' "$file"
    sed -i '/"kotekan_git_commit_hash":/d' "$file"
    sed -i '/"kotekan_version":/d' "$file"
done

# We expect the modified 127.0.0.1_12048.json to exist and have a md5sum c46b468ea28873a80ebc76f9f1648076
md5sum -c --status <(echo "c46b468ea28873a80ebc76f9f1648076  ${CONFIG_OUT_DIR}/127.0.0.1_12048.json")
if [ $? -ne 0 ]; then
    echo "MD5 checksum for ${CONFIG_OUT_DIR}/127.0.0.1_12048.json does not match expected value"
    echo "File contents:"
    cat "${CONFIG_OUT_DIR}/127.0.0.1_12048.json"
    ERROR=1
fi

# We expect the modified 127.0.0.1_12748.json to exist and have a md5sum 01c90bf3d9c22a2222b9b17252b1d464
md5sum -c --status <(echo "01c90bf3d9c22a2222b9b17252b1d464  ${CONFIG_OUT_DIR}/127.0.0.1_12748.json")
if [ $? -ne 0 ]; then
    echo "MD5 checksum for ${CONFIG_OUT_DIR}/127.0.0.1_12748.json does not match expected value."
    echo "File contents:"
    cat "${CONFIG_OUT_DIR}/127.0.0.1_12748.json"
    ERROR=1
fi

if [ $ERROR -ne 0 ]; then
    echo "configTrackerWriter test failed!"
    exit 1
fi

echo "configTrackerWriter test passed: ${num_json} file(s) in ${CONFIG_OUT_DIR}"
exit 0
