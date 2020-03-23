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
import shutil
import signal

from subprocess import call, Popen

producer_path = "../build/tests/boost/dataset_broker_producer"
producer2_path = "../build/tests/boost/dataset_broker_producer2"
consumer_path = "../build/tests/boost/dataset_broker_consumer"


def has_redis(host="localhost", port=6379):
    """Check if redis is available."""

    try:
        import redis
        r = redis.Redis(host, port)
        return redis.ping()

    except Exception as e:
        print(e)
        return False


def test_produce_consume():
    broker_path = shutil.which("comet")
    if (
        not os.path.isfile(producer_path)
        or not os.path.isfile(consumer_path)
        or not os.path.isfile(producer2_path)
    ):
        pytest.skip("Build with -DBOOST_TESTS=ON to activate this test")
    if not broker_path:
        pytest.skip(
            "Make sure PYTHONPATH is set to where the comet dataset broker is installed."
        )
    if not has_redis():
        pytest.skip("Redis is not available and so comet will fail.")

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
