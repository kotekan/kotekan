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
    cmake -Wdev -Werror=dev -Wdeprecated -Werror=deprecated -DWERROR=ON -DCMAKE_LINK_WHAT_YOU_USE=ON -DCMAKE_BUILD_TYPE=Test -DUSE_ASDF=ON -DUSE_GDAL=ON -DUSE_HDF5=ON -DUSE_LAPACK_BLAZE=ON -DNO_MEMLOCK=ON -DUSE_OMP=ON -DUSE_CUDA=ON -DUSE_FFTW=ON -DWITH_TESTS=ON -DWITH_BOOST_TESTS=ON -DCCACHE=ON ..
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

# Print exit statuses
echo "kotekan instance 1 exit status: $EXIT_STATUS_1"
echo "kotekan instance 2 exit status: $EXIT_STATUS_2"
# Exit with error if either instance did not exit cleanly
if [ $EXIT_STATUS_1 -ne 0 ] || [ $EXIT_STATUS_2 -ne 0 ]; then
    echo "One or both kotekan instances did not exit cleanly!"
    exit 1
fi

# Verify that the config writer produced exactly two JSON files
if [ "$(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)" -ne 2 ]; then
    echo "Expected 2 JSON files in ${CONFIG_OUT_DIR}, found $(ls -1 "${CONFIG_OUT_DIR}"/*.json 2>/dev/null | wc -l)"
    exit 1
fi

# We expect 127.0.0.1_12048.json to exist and have a md5sum 60972de19e3a2c6e780e77744b557050
md5sum -c --status <(echo "60972de19e3a2c6e780e77744b557050  ${CONFIG_OUT_DIR}/127.0.0.1_12048.json")
if [ $? -ne 0 ]; then
    echo "MD5 checksum for ${CONFIG_OUT_DIR}/127.0.0.1_12048.json does not match expected value"
    exit 1
fi

# We expect 127.0.0.1_12748.json to exist and have a md5sum 311dd13a157ff40d14ebe5c95b8cfeb9
md5sum -c --status <(echo "311dd13a157ff40d14ebe5c95b8cfeb9  ${CONFIG_OUT_DIR}/127.0.0.1_12748.json")
if [ $? -ne 0 ]; then
    echo "MD5 checksum for ${CONFIG_OUT_DIR}/127.0.0.1_12748.json does not match expected value"
    exit 1
fi

echo "configTrackerWriter test passed: ${num_json} file(s) in ${CONFIG_OUT_DIR}"
exit 0
