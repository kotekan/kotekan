#
# Script to run each of the CHIME verification tests
#

import os
import glob
import pytest

BIN_DIR = "../build/kotekan"
TEST_DIR = "../../config/tests"
TEST_FAILED = 1
TEST_PASSED = 2

def print_msg(msg):
    print("\n#########################################################")
    print("{}".format(msg))
    print("#########################################################\n")

@pytest.mark.skip(reason="Need to update CI server first")
def test_gpu_kernels():

    # Change to kotekan bin directory
    os.chdir(BIN_DIR)

    # Get list of config files in tests/ directory
    config_files = glob.glob(TEST_DIR + "/*.yaml")
    print config_files

    # Run each test in TEST_DIR
    for config in config_files:

        print_msg("Running " + os.path.basename(config) + " test")

        # Run test
        status = os.system("./kotekan -c " + config)
        
        # Get return code from kotekan in the highest 8-bits
        return_code = status >> 8

        # Check that the test passed
        assert return_code == TEST_PASSED
        
        print_msg(os.path.basename(config) + " TEST PASSED")
