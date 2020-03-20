# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import logging
import numpy as np
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
    "total_frames": 11,
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
    num_freq = len(params_fakevis["freq_ids"])
    num_ev = params["num_ev"]
    num_elements = params["num_elements"]

    threading.Thread(target=vis_data_slow.run).start()
    sleep(2)
    buffer = shared_memory_buffer.SharedMemoryReader(sem_name, fname_buf, 4)

    assert buffer.num_time == params_writer_stage["nsamples"]
    assert buffer.num_freq == num_freq

    # access_record = []
    # for t in
    # assert buffer._access_record() == access_record

    n_times_to_read = 3

    i = 0
    with pytest.raises(shared_memory_buffer.SharedMemoryError):
        while True:
            sleep(1)
            print(buffer._access_record())
            visraw = buffer.read_last(n_times_to_read)
            assert visraw.num_freq == len(params_fakevis["freq_ids"])
            assert visraw.num_time == n_times_to_read

            ds = np.array(visraw.metadata["dataset_id"]).copy().view("u8,u8")
            unique_ds = np.unique(ds)

            evals = visraw.data["eval"]
            evecs = visraw.data["evec"]
            erms = visraw.data["erms"]

            # Check datasets are present
            assert evals.shape == (n_times_to_read, num_freq, num_ev)
            assert evecs.shape == (n_times_to_read, num_freq, num_ev * num_elements)
            assert erms.shape == (n_times_to_read, num_freq)

            evecs = evecs.reshape(n_times_to_read, num_freq, num_ev, num_elements)

            # Check that the datasets have the correct values
            assert (evals == np.arange(num_ev)[np.newaxis, np.newaxis, :]).all()
            assert (
                evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
            ).all()
            assert (
                evecs.imag
                == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
            ).all()
            # assert (erms == 1.0).all()

            i += 1
    assert i >= 2
