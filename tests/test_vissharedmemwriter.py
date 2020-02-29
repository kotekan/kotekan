# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import pytest
import posix_ipc
import mmap
import os

from kotekan import runner

sem_name = "kotekan"
fname_met = "calBufferMetadata"
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


@pytest.fixture(scope="module")
def semaphore():
    sem = posix_ipc.Semaphore(sem_name)
    yield sem
    sem.release()
    sem.unlink()


@pytest.fixture(scope="module")
def memory_map_data():
    memory = posix_ipc.SharedMemory(fname_buf)
    mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
    os.close(memory.fd)
    yield mapfile
    mapfile.close()
    posix_ipc.unlink_shared_memory(fname_buf)

@pytest.fixture(scope="module")
def memory_map_metadata():
    memory = posix_ipc.SharedMemory(fname_met)
    mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
    os.close(memory.fd)
    yield mapfile
    mapfile.close()
    posix_ipc.unlink_shared_memory(fname_met)


def test_sharedmem(vis_data, semaphore, memory_map_metadata, memory_map_data):
    for memory_map in [memory_map_metadata, memory_map_data]:
        memory_map.seek(0)
    import struct

    for i in range(0, 100):
        semaphore.acquire()
        print(struct.unpack("<c", memory_map_data.read(1))[0])
        semaphore.release()
        print(struct.unpack("<Q", memory_map_metadata.read(8))[0])
