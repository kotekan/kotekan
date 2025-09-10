#!/bin/bash

# Default timeout
TIMEOUT=60
VERBOSE=0

# Parse command-line arguments
while getopts "t:v" opt; do
    case $opt in
        t) TIMEOUT="$OPTARG" ;;
        v) VERBOSE=1 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Shift past options to get positional arguments
shift $((OPTIND-1))

# Check if directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [-t timeout] [-v] <directory>" >&2
    exit 1
fi

# Validate directory
TEST_DIR="$1"
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory '$TEST_DIR' does not exist" >&2
    exit 1
fi

# Validate timeout is a positive integer
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -le 0 ]; then
    echo "Error: Timeout must be a positive integer" >&2
    exit 1
fi

# Find all executable files in the directory that start with "test_"
mapfile -t EXECUTABLES < <(find "$TEST_DIR" -maxdepth 1 -type f -executable -name "test_*")

# Check if any executables were found
if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo "No executables found in '$TEST_DIR'" >&2
    exit 1
fi

# Run each executable with the specified timeout and track results
passed=0
failed=0
declare -a failed_tests

for exe in "${EXECUTABLES[@]}"; do
    echo "Running $exe..."
    # Use temporary files to capture output
    tmp_output=$(mktemp)
    tmp_stderr=$(mktemp)

    if [ $VERBOSE -eq 1 ]; then
        # Verbose mode
        output=$(timeout "$TIMEOUT" "$exe" 2>&1)
        exit_code=$?
        echo "$output"
    else
        # Non-verbose mode: Minimize output
        timeout "$TIMEOUT" "$exe" >"$tmp_output" 2>"$tmp_stderr"
        exit_code=$?
        [ $exit_code -ne 0 ] && [ -s "$tmp_stderr" ] && cat "$tmp_stderr" >&2
    fi
    if [ $exit_code -eq 0 ]; then
        echo "$exe: PASSED"
        ((passed++))
    elif [ $exit_code -eq 124 ]; then
        echo "$exe: FAILED (timed out after ${TIMEOUT}s)" >&2
        failed_tests+=("$exe (timed out)")
        ((failed++))
    else
        echo "$exe: FAILED (exit code $exit_code)" >&2
        failed_tests+=("$exe (exit code $exit_code)")
        ((failed++))
    fi
    echo "----------------"
done

# Summary
echo "Summary: $passed passed, $failed failed"

# List failed tests
if [ ${#failed_tests[@]} -gt 0 ]; then
    echo "Failed tests:" >&2
    for test in "${failed_tests[@]}"; do
        echo "  $test" >&2
    done
fi

[ $failed -eq 0 ] && exit 0 || exit -1
