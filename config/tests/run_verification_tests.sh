#!/usr/bin/env bash
#
# Script to run each of the CHIME verification tests
#

# Exit upon error, avoiding cascading errors
set -o errexit

# Run each test in the directory
for config_file in ./*
do
  echo " "
  echo "#################################################"
  echo "# Running $config_file test."
  echo "#################################################"
  echo " "

  # Run test and exit if the test fails
  {
    ../../build/kotekan/kotekan -c $config_file
  } || {
    echo " "
    echo "#################################################"
    echo "# $config_file TEST FAILED"
    echo "#################################################"
    echo " "
    exit
  }
  
  echo " "
  echo "#################################################"
  echo "# $config_file TEST PASSED"
  echo "#################################################"
  echo " "

done
