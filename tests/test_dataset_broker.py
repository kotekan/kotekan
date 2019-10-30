# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import os.path
import re
import time
import tempfile
import signal

from subprocess import call, Popen

producer_path = "../build/tests/boost/dataset_broker_producer"
producer2_path = "../build/tests/boost/dataset_broker_producer2"
consumer_path = "../build/tests/boost/dataset_broker_consumer"
broker_path = "../build/ext/src/ch_acq/dataset_broker.py"


def test_produce_consume():
    if (
        not os.path.isfile(producer_path)
        or not os.path.isfile(consumer_path)
        or not os.path.isfile(producer2_path)
    ):
        print("Deactivated! Build with -DBOOST_TESTS=ON to activate this test")
        return
    if not os.path.isfile(broker_path):
        print("Deactivated! Make sure the dataset_broker is at {}".format(broker_path))
        return

    with tempfile.NamedTemporaryFile(mode="w") as f_out:
        # Start comet with a random port
        broker = Popen(
            [broker_path, "-p", "0", "--recover", "False"], stdout=f_out, stderr=f_out
        )
        time.sleep(3)

        # Find port in the log
        regex = re.compile("Selected random port: ([0-9]+)$")
        log = open(f_out.name, "r").read().split("\n")
        port = None
        for line in log:
            print(line)
            match = regex.search(line)
            if match:
                port = match.group(1)
                print("Test found comet port in log: %s" % port)
                break
        if not match:
            print("Could not find comet port in logs.")
            exit(1)

        try:
            assert call([producer_path, "--", port]) == 0
            assert call([producer2_path, "--", port]) == 0
            assert call([consumer_path, "--", port]) == 0
        finally:
            pid = broker.pid
            os.kill(pid, signal.SIGINT)
            broker.terminate()
            log = open(f_out.name, "r").read().split("\n")
            for line in log:
                print(line)
