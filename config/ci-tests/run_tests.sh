#!/usr/bin/env bash
# Runs Kotekan tests from directories of configs and/or individual configs, with
# timeout protection. Directories are searched for .yaml, .yml and .j2 configs.
# Tracks pass/fail/timeout status and provides a summary at the end.
# Usage: ./run_tests.sh <kotekan_binary> <timeout_duration> <test_config_dir|test_config>...

# Check if required arguments are provided
if [ $# -lt 3 ]; then
  echo "Usage: $0 <kotekan_binary> <timeout_duration> <test_config_dir|test_config>..."
  echo "Example: $0 ./build-2404/kotekan/kotekan 2m config/ci-tests"
  echo "Example: $0 ./build-2404/kotekan/kotekan 2m config/ci-tests/cpu_batch/foo.yaml"
  exit 1
fi

KOTEKAN_BINARY="$1"
TIMEOUT_DURATION="$2"
shift 2

format_duration() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))

  if [ $hours -gt 0 ]; then
    printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
  elif [ $minutes -gt 0 ]; then
    printf "%02dm:%02ds" "$minutes" "$seconds"
  else
    printf "%02ds" "$seconds"
  fi
}

# Verify kotekan binary exists
if [ ! -f "$KOTEKAN_BINARY" ]; then
  echo "Error: Kotekan binary not found at $KOTEKAN_BINARY"
  exit 1
fi

# Collect the configs to run: every config in a directory argument, or the file itself
CONFIG_FILES=()
for target in "$@"; do
  if [ -d "$target" ]; then
    for config_file in "$target"/*.yaml "$target"/*.yml "$target"/*.j2; do
      [ -f "$config_file" ] && CONFIG_FILES+=("$config_file")
    done
  elif [ -f "$target" ]; then
    CONFIG_FILES+=("$target")
  else
    echo "Error: Test config not found at $target"
    exit 1
  fi
done

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
  echo "Error: No .yaml, .yml or .j2 configs found in $*"
  exit 1
fi

# Initialize test tracking
PASSED_TESTS=()
PASSED_TEST_TIMES=()
FAILED_TESTS=()
FAILED_TEST_EXIT_CODES=()
FAILED_TEST_TIMES=()
TIMED_OUT_TESTS=()
TIMED_OUT_TEST_TIMES=()
TIMED_OUT_TEST_EXIT_CODES=()

# Run tests
for config_file in "${CONFIG_FILES[@]}"; do
  echo "Running test with config: $config_file"
  
  # Run the test with timeout
  start_time=$(date +%s)
  timeout "$TIMEOUT_DURATION" "$KOTEKAN_BINARY" --config "$config_file"
  EXIT_CODE=$?
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  
  if [ $EXIT_CODE -eq 124 ]; then
    echo "Test timed out for config: $config_file"
    TIMED_OUT_TESTS+=("$config_file")
    TIMED_OUT_TEST_EXIT_CODES+=("$EXIT_CODE")
    TIMED_OUT_TEST_TIMES+=("$elapsed")
  elif [ $EXIT_CODE -ne 0 ]; then
    echo "Test failed for config: $config_file with exit code $EXIT_CODE"
    FAILED_TESTS+=("$config_file")
    FAILED_TEST_EXIT_CODES+=("$EXIT_CODE")
    FAILED_TEST_TIMES+=("$elapsed")
  else
    echo "Test passed for config: $config_file"
    PASSED_TESTS+=("$config_file")
    PASSED_TEST_TIMES+=("$elapsed")
  fi
  echo ""
done

# Print summary
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Total tests: $((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]} + ${#TIMED_OUT_TESTS[@]}))"
echo "Passed: ${#PASSED_TESTS[@]}"
echo "Failed: ${#FAILED_TESTS[@]}"
echo "Timed out: ${#TIMED_OUT_TESTS[@]}"
echo ""

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
  echo "Passed tests:"
  for i in "${!PASSED_TESTS[@]}"; do
    duration=$(format_duration "${PASSED_TEST_TIMES[$i]}")
    echo "  - ${PASSED_TESTS[$i]} (duration: $duration)"
  done
  echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  echo "Failed tests:"
  for i in "${!FAILED_TESTS[@]}"; do
    duration=$(format_duration "${FAILED_TEST_TIMES[$i]}")
    echo "  - ${FAILED_TESTS[$i]} (exit code: ${FAILED_TEST_EXIT_CODES[$i]}, duration: $duration)"
  done
  echo ""
fi

if [ ${#TIMED_OUT_TESTS[@]} -gt 0 ]; then
  echo "Timed out tests:"
  for i in "${!TIMED_OUT_TESTS[@]}"; do
    duration=$(format_duration "${TIMED_OUT_TEST_TIMES[$i]}")
    echo "  - ${TIMED_OUT_TESTS[$i]} (exit code: ${TIMED_OUT_TEST_EXIT_CODES[$i]}, duration: $duration)"
  done
  echo ""
fi

# Exit with failure if any tests failed or timed out
if [ ${#FAILED_TESTS[@]} -gt 0 ] || [ ${#TIMED_OUT_TESTS[@]} -gt 0 ]; then
  exit 1
fi

echo "All tests passed!"
