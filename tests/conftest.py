#
# Configures tests based upon command line arguments to pytest
#

import pytest
import subprocess

# Optional debugging aid for hangs (e.g. in CI). When FAULTHANDLER_DUMP_INTERVAL
# is set (seconds), every process that imports this conftest -- the pytest-xdist
# controller and each worker -- periodically dumps the stack of all of its
# threads. A stalled run then shows *where* it is stuck (collection, a fixture,
# the xdist controller waiting on a worker, a test, teardown, ...), even when the
# hang is not inside a test. Inert (no overhead) when the variable is unset.
import os
import sys
import faulthandler
import re
import signal
import shutil
import tempfile

from time import sleep

faulthandler.enable()
_fh_interval = os.environ.get("FAULTHANDLER_DUMP_INTERVAL")
if _fh_interval:
    faulthandler.dump_traceback_later(
        float(_fh_interval), repeat=True, file=sys.__stderr__
    )


def pytest_addoption(parser):
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="only run tests matching the environment NAME.",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "env(name): mark test to run only on named environment"
    )
    config.addinivalue_line(
        "markers", "serial: mark test to not run in parallel with other tests"
    )


def pytest_runtest_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
    if envnames:
        if item.config.getoption("-E") not in envnames:
            pytest.skip("test requires env in {!r}".format(envnames))


#
# Shared fixtures for gain updates
#
from pytest_localserver.http import WSGIServer
from flask import Flask, jsonify, request as flask_req
import base64


def encode_gains(gain, weight):
    # encode base64
    res = {
        "gain": {
            "dtype": "complex64",
            "shape": gain.shape,
            "data": base64.b64encode(gain.tobytes()).decode(),
        },
        "weight": {
            "dtype": "bool",
            "shape": weight.shape,
            "data": base64.b64encode(weight.tobytes()).decode(),
        },
    }
    return res


@pytest.fixture(scope="module")
def cal_broker(request, old_gains, new_gains):
    # get updates IDs from module
    new_update_id = getattr(request.module, "new_update_id", None)
    old_update_id = getattr(request.module, "old_update_id", None)

    # Create a basic flask server
    app = Flask("cal_broker")

    @app.route("/gain", methods=["POST"])
    def gain_app():
        content = flask_req.get_json()
        update_id = content["update_id"]
        if update_id == new_update_id:
            gains = encode_gains(*new_gains)
        elif update_id == old_update_id:
            gains = encode_gains(*old_gains)
        else:
            raise Exception("Did not recognize update_id {}.".format(update_id))
        print(f"Served gains with {update_id}")

        return jsonify(gains)

    # hand to localserver fixture
    server = WSGIServer(application=app)
    server.start()

    yield server

    server.stop()


def has_redis(host="localhost", port=6379):
    """Check if redis is available."""

    try:
        import redis

        r = redis.Redis(host, port)
        return r.ping()

    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="module")
def comet_broker_port(*broker_args):
    """Yield the port on which the comet broker is running."""
    broker_path = shutil.which("comet")
    if not broker_path:
        pytest.skip(
            "Make sure PYTHONPATH is set to where the comet dataset broker is installed."
        )

    if not has_redis():
        pytest.skip("Redis is not available and so comet will fail.")

    if not broker_args:
        # By default, open on a random port
        broker_args = ["-p", "0"]

    with tempfile.NamedTemporaryFile(mode="w") as f_out:
        # Start comet
        broker = subprocess.Popen(
            [broker_path, "broker", *broker_args], stdout=f_out, stderr=f_out
        )
        sleep(3)

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
            yield port
        finally:
            pid = broker.pid
            os.kill(pid, signal.SIGTERM)
            broker.terminate()
            log = open(f_out.name, "r").read().split("\n")
            for line in log:
                print(line)
