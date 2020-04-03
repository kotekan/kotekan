# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import copy
import logging
import numpy as np
import os
import pytest
import re
import signal
from subprocess import Popen
import shutil
import tempfile
import threading
from time import sleep

from kotekan import runner, testing

# use tempfile creation to get exclusive random strings
useless_file = tempfile.NamedTemporaryFile()
fname_buf = "calBuffer_" + os.path.split(useless_file.name)[-1]

logging.basicConfig(level=logging.INFO)


@pytest.fixture()
def comet_broker():
    broker_path = shutil.which("comet")
    if not broker_path:
        pytest.skip(
            "Make sure PYTHONPATH is set to where the comet dataset broker is installed."
        )

    with tempfile.NamedTemporaryFile(mode="w") as f_out:
        # Start comet with a random port
        broker = Popen(
            [broker_path, "-p", "0", "--recover", "False"], stdout=f_out, stderr=f_out
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
            os.kill(pid, signal.SIGINT)
            broker.terminate()
            log = open(f_out.name, "r").read().split("\n")
            for line in log:
                print(line)


# number of frames to ignore in validation
ignore_frames = 6
params = {
    "num_elements": 21,
    "num_ev": 1,
    "total_frames": 21,
    "cadence": 1,
    "dataset_manager": {"use_dataset_broker": True},
    "mode": "default",
}

params_fakevis = {
    "freq_ids": [7, 13, 14, 33, 44],
    "num_frames": params["total_frames"],
    "mode": params["mode"],
    "wait": True,
}

params_writer_stage = {"nsamples": 3, "fname": fname_buf}


@pytest.fixture()
def vis_data(tmpdir_factory, comet_broker):

    # keeping all the data this test produced here (probably do not need it)
    # using FakeVisBuffer to produce fake data
    fakevis_buffer = runner.FakeVisBuffer(**params_fakevis)

    # pass comet port to kotekan
    params["dataset_manager"]["ds_broker_port"] = comet_broker

    # KotekanStageTester is used to run kotekan with my config
    test = runner.KotekanStageTester(
        stage_type="VisSharedMemWriter",
        stage_config=params_writer_stage,
        buffers_in=fakevis_buffer,
        buffers_out=None,
        global_config=params,
    )
    yield test


# This test still needs to run alone, because multiple comet instances would have conflicts
# accessing redis.
@pytest.mark.serial
def test_shared_mem_buffer(vis_data, comet_broker):
    # start kotekan writer in a thread, to read before it's done (it will delete the shm on exit)
    threading.Thread(target=vis_data.run).start()
    sleep(2)
    view_size = [1, 2, 3, 3]
    update_interval = [0.1, 1, 0.5, 5]
    len_test = params["total_frames"] - ignore_frames

    # allow the last reader to miss a lot of frames, but still exit before kotekan
    len_test = [len_test] * (len(view_size) - 1) + [9]

    config = copy.copy(params)
    config["fakevis"] = params_fakevis
    config["writer"] = params_writer_stage
    validation = testing.SharedMemValidationTest(
        len_test,
        config,
        len(view_size),
        fname_buf,
        view_size,
        params["mode"],
        update_interval,
        threshold_frame_age_error=1,
        threshold_cadence_error=0,
    )
    validation.run()
