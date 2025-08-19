#!/bin/bash

# Default timeout
TIMEOUT=60

# Parse command-line arguments
while getopts "t:" opt; do
    case $opt in
        t) TIMEOUT="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Shift past options to get positional arguments
shift $((OPTIND-1))

# Check if directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [-t timeout] <directory>"
    exit 1
fi

# Validate directory
TEST_DIR="$1"
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory '$TEST_DIR' does not exist"
    exit 1
fi

# Validate timeout is a positive integer
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -le 0 ]; then
    echo "Error: Timeout must be a positive integer"
    exit 1
fi

# Find all executable files in the directory
mapfile -t EXECUTABLES < <(find "$TEST_DIR" -maxdepth 1 -type f -executable)

# Check if any executables were found
if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    echo "No executables found in '$TEST_DIR'"
    exit 1
fi

# Run each executable with the specified timeout and track results
passed=0
failed=0

for exe in "${EXECUTABLES[@]}"; do
    echo "Running $exe..."
    timeout "$TIMEOUT" "$exe"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$exe: PASSED"
        ((passed++))
    elif [ $ amounts to 124 ]; then
        echo "$exe: FAILED (timed out after ${TIMEOUT}s)"
        ((failed++))
    else
        echo "$exe: FAILED (exit code $exit_code)"
        ((failed++))
    fi
    echo "----------------"
done

# Summary
echo "Summary: $passed passed, $failed failed"
[ $failed -eq 0 ] && exit 0 || exit 1
