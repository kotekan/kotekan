#!/usr/bin/env bash
# Runs Kotekan tests from a directory of YAML configs with timeout protection.
# Tracks pass/fail/timeout status and provides a summary at the end.
# Usage: ./run_tests.sh <kotekan_binary> <timeout_duration> <test_config_dir>


# Check if required arguments are provided
if [ $# -lt 3 ]; then
  echo "Usage: $0 <kotekan_binary> <timeout_duration> <test_config_dir>"
  echo "Example: $0 ./build-2404/kotekan/kotekan 2m config/ci-tests"
  exit 1
fi

KOTEKAN_BINARY="$1"
TIMEOUT_DURATION="$2"
TEST_CONFIG_DIR="$3"

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

# Verify test config directory exists
if [ ! -d "$TEST_CONFIG_DIR" ]; then
  echo "Error: Test config directory not found at $TEST_CONFIG_DIR"
  exit 1
fi

# Initialize test tracking
PASSED_TESTS=()
PASSED_TEST_TIMES=()
FAILED_TESTS=()
FAILED_TEST_EXIT_CODES=()
FAILED_TEST_TIMES=()
TIMED_OUT_TESTS=()

# Run tests
for config_file in "$TEST_CONFIG_DIR"/*.yaml; do
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
  for test in "${TIMED_OUT_TESTS[@]}"; do
    echo "  - $test"
  done
  echo ""
fi

# Exit with failure if any tests failed or timed out
if [ ${#FAILED_TESTS[@]} -gt 0 ] || [ ${#TIMED_OUT_TESTS[@]} -gt 0 ]; then
  exit 1
fi

echo "All tests passed!"
