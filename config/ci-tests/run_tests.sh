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
FAILED_TESTS=()
TIMED_OUT_TESTS=()

# Run tests
for config_file in "$TEST_CONFIG_DIR"/*.yaml; do
  echo "Running test with config: $config_file"
  
  # Run the test with timeout
  timeout "$TIMEOUT_DURATION" "$KOTEKAN_BINARY" --config "$config_file"
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 124 ]; then
    echo "Test timed out for config: $config_file"
    TIMED_OUT_TESTS+=("$config_file")
  elif [ $EXIT_CODE -ne 0 ]; then
    echo "Test failed for config: $config_file with exit code $EXIT_CODE"
    FAILED_TESTS+=("$config_file (exit code: $EXIT_CODE)")
  else
    echo "Test passed for config: $config_file"
    PASSED_TESTS+=("$config_file")
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

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  echo "Failed tests:"
  for test in "${FAILED_TESTS[@]}"; do
    echo "  - $test"
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
