import copy
import logging
import os
import pytest
import tempfile
import threading
from time import sleep

from kotekan import runner, testing

# use tempfile creation to get exclusive random strings
useless_file = tempfile.NamedTemporaryFile()
fname_buf = "calBuffer_" + os.path.split(useless_file.name)[-1]

logging.basicConfig(level=logging.INFO)


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

params_writer_stage = {"num_samples": 3, "name": fname_buf}


@pytest.fixture()
def vis_data(tmpdir_factory, comet_broker_port):

    # keeping all the data this test produced here (probably do not need it)
    # using FakeVisBuffer to produce fake data
    fakevis_buffer = runner.FakeVisBuffer(**params_fakevis)

    # pass comet port to kotekan
    params["dataset_manager"]["ds_broker_port"] = comet_broker_port

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
def test_shared_mem_buffer(vis_data):
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
