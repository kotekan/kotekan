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
import shutil
import socket

from time import monotonic, sleep

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


@pytest.fixture(scope="module")
def comet_broker_port(tmp_path_factory):
    """Run a Comet broker with an isolated, temporary Redis database."""
    broker_path = shutil.which("comet")
    redis_path = shutil.which("redis-server")
    if not broker_path or not redis_path:
        pytest.skip("Comet tests require comet and redis-server on PATH")
    redis = pytest.importorskip("redis")
    import requests

    work = tmp_path_factory.mktemp("comet")
    redis_log = work / "redis.log"
    broker_log = work / "broker.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        redis_port = sock.getsockname()[1]

    processes = []
    with redis_log.open("w") as redis_out, broker_log.open("w") as broker_out:
        try:
            server = subprocess.Popen(
                [
                    redis_path,
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(redis_port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                cwd=work,
                stdout=redis_out,
                stderr=subprocess.STDOUT,
            )
            processes.append(server)
            client = redis.Redis("127.0.0.1", redis_port, socket_timeout=1)
            deadline = monotonic() + 10
            while monotonic() < deadline and server.poll() is None:
                try:
                    if client.ping():
                        break
                except redis.ConnectionError:
                    pass
                sleep(0.1)
            else:
                pytest.fail("Redis did not start:\n" + redis_log.read_text())
            client.close()

            broker = subprocess.Popen(
                [broker_path, "broker", "-p", "0"],
                stdout=broker_out,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "REDIS_HOST": "127.0.0.1",
                    "REDIS_PORT": str(redis_port),
                    "PYTHONUNBUFFERED": "1",
                },
            )
            processes.append(broker)
            deadline = monotonic() + 20
            while monotonic() < deadline and broker.poll() is None:
                match = re.search(
                    r"Selected random port: (\d+)", broker_log.read_text()
                )
                if match:
                    port = int(match.group(1))
                    try:
                        response = requests.get(
                            f"http://127.0.0.1:{port}/status", timeout=1
                        )
                        if response.ok and response.json().get("running"):
                            break
                    except (requests.ConnectionError, requests.Timeout):
                        pass
                sleep(0.1)
            else:
                pytest.fail("Comet did not start:\n" + broker_log.read_text())
            yield port
        finally:
            for process in reversed(processes):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print(broker_log.read_text())
            print(redis_log.read_text())
