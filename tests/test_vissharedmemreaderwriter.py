# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import logging
import posix_ipc
import pytest
import threading
from time import sleep

from kotekan import runner, shared_memory_buffer

sem_name = "kotekan"
fname_buf = "calBuffer"

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="module")
def semaphore():
    sem = posix_ipc.Semaphore(sem_name)
    yield sem
    sem.release()
    sem.unlink()


params = {
    "num_elements": 7,
    "num_ev": 0,
    "total_frames": 10,
    "cadence": 1.0,
    "mode": "default",
    "dataset_manager": {"use_dataset_broker": False},
}

params_fakevis = {
    "freq_ids": [0, 1, 2],
    "num_frames": params["total_frames"],
    "mode": params["mode"],
    "wait": True,
}

params_writer_stage = {"nsamples": 5}


@pytest.fixture()
def vis_data_slow(tmpdir_factory):

    # keeping all the data this test produced here (probably do not need it)
    # using FakeVisBuffer to produce fake data
    fakevis_buffer = runner.FakeVisBuffer(**params_fakevis)

    # KotekanStageTester is used to run kotekan with my config
    test = runner.KotekanStageTester(
        stage_type="visSharedMemWriter",
        stage_config=params_writer_stage,
        buffers_in=fakevis_buffer,
        buffers_out=None,
        global_config=params,
    )
    yield test


def test_shared_mem_buffer(vis_data_slow):
    threading.Thread(target=vis_data_slow.run).start()
    sleep(3)
    buffer = shared_memory_buffer.SharedMemoryReader(sem_name, fname_buf, 4)

    assert buffer.num_time == params_writer_stage["nsamples"]
    assert buffer.num_freq == len(params_fakevis["freq_ids"])

    # access_record = []
    # for t in
    # assert buffer._access_record() == access_record
    print(buffer._access_record())

    i = 0
    with pytest.raises(shared_memory_buffer.SharedMemoryError):
        while True:
            print("round {}".format(i))
            sleep(1)
            print(buffer._access_record())
            print(buffer.read_last(3))
            i += 1
    assert i >= 2
