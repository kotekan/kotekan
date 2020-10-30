# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

#
# Configures tests based upon command line arguments to pytest
#

import pytest
import subprocess


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
