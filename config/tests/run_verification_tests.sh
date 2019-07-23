#!/usr/bin/env bash
#
# Script to run each of the CHIME verification tests
#

# Return codes
TEST_FAILED=1
TEST_PASSED=2

print_message () {
    echo " "
    echo "#########################################################"
    echo "# $1"
    echo "#########################################################"
    echo " "
}

print_message "Running $0 script"

# Run each test in the directory
for config_file in ./*.yaml
do
  print_message "Running $config_file test"

  # Run test and exit if the test fails
  ../../build/kotekan/kotekan -c $config_file

  RETURN_CODE=$?

  if [[ $RETURN_CODE == $TEST_FAILED ]]; then
    print_message "$config_file TEST FAILED"
    exit 1
  elif [[ $RETURN_CODE == $TEST_PASSED ]]; then
    print_message "$config_file TEST PASSED"
  else
    print_message "$config_file returned unknown code"
    exit 1
  fi

done

# All tests have passed
exit 0
