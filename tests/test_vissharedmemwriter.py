# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import * #noqa pylint: disable=W0401, W0614
from future.builtins.disabled import * #noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import pytest
import posix_ipc
import mmap
import os

from kotekan import runner

SEM_NAME = "kotekan"
MEM_NAME = "ringBufferMetadata"

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
            num_frames=params["total_frames"],
            mode=params["mode"],
    )

    # KotekanStageTester is used to run kotekan with my config
    test = runner.KotekanStageTester(
            stage_type="visSharedMemWriter",
            stage_config={},
            buffers_in=fakevis_buffer,
            buffers_out=None,
            global_config=params
    )

    test.run()

@pytest.fixture(scope="module")
def semaphore():
    sem = posix_ipc.Semaphore(SEM_NAME)
    print("Test is waiting to acquire the semaphore")
    sem.acquire()
    print("Acquired!")
    yield sem
    sem.release()
    sem.unlink()

@pytest.fixture(scope="module")
def memory_map():
    memory = posix_ipc.SharedMemory(MEM_NAME)
    mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
    os.close(memory.fd)
    yield mapfile
    mapfile.close()
    posix_ipc.unlink_shared_memory(MEM_NAME)

def test_sharedmem(vis_data, semaphore, memory_map):
    memory_map.seek(0)
    import struct
    for i in range(0, 800, 8):
        print(struct.unpack("<Q", memory_map.read(8))[0])

