# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import pytest

from kotekan import runner, shared_memory_buffer

sem_name = "kotekan"
fname_access_record = "calBufferAccessRecord"
fname_buf = "calBuffer"

params = {
    "num_elements": 7,
    "num_ev": 0,
    "total_frames": 128,
    "cadence": 10.0,
    "mode": "default",
    "dataset_manager": {"use_dataset_broker": False},
}


@pytest.fixture(scope="module")
def vis_data(tmpdir_factory):

    # keeping all the data this test produced here (probably do not need it)
    # using FakeVisBuffer to produce fake data
    fakevis_buffer = runner.FakeVisBuffer(
        num_frames=params["total_frames"], mode=params["mode"]
    )

    # KotekanStageTester is used to run kotekan with my config
    test = runner.KotekanStageTester(
        stage_type="visSharedMemWriter",
        stage_config={},
        buffers_in=fakevis_buffer,
        buffers_out=None,
        global_config=params,
    )

    test.run()


@pytest.fixture()
def buffer():
    yield shared_memory_buffer.SharedMemoryReader(sem_name, fname_buf)


def test_shared_mem_buffer(vis_data, buffer):
    assert buffer
