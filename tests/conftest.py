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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    df = subprocess.Popen(["df", "-h"], stdout=subprocess.PIPE)
    output = df.communicate()[0]
    print(output.decode())
    yield
    df = subprocess.Popen(["df", "-h"], stdout=subprocess.PIPE)
    output = df.communicate()[0]
    print(output.decode())


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
