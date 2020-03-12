# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import pytest
import posix_ipc
import mmap
import os

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


@pytest.fixture(scope="module")
def semaphore():
    sem = posix_ipc.Semaphore(sem_name)
    yield sem
    sem.release()
    sem.unlink()


# @pytest.fixture(scope="module")
# def memory_map_buf():
#     memory = posix_ipc.SharedMemory(fname_buf)
#     mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
#     os.close(memory.fd)
#     yield mapfile
#     mapfile.close()
#     posix_ipc.unlink_shared_memory(fname_buf)
#
# @pytest.fixture(scope="module")
# def mem_map_access_record():
#     memory = posix_ipc.SharedMemory(fname_access_record)
#     mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
#     os.close(memory.fd)
#     yield mapfile
#     mapfile.close()
#     posix_ipc.unlink_shared_memory(fname_access_record)


@pytest.fixture()
def buffer():
    yield shared_memory_buffer.SharedMemoryReader(sem_name, fname_buf)


def test_shared_mem_buffer(vis_data, buffer):
    assert buffer

# def test_sharedmem(vis_data, semaphore, mem_map_access_record, memory_map_buf):
#     for memory_map in [mem_map_access_record, memory_map_buf]:
#         memory_map.seek(0)
#     import struct
#
#     for i in range(0, 100):
#         semaphore.acquire()
#         print(struct.unpack("<c", memory_map_buf.read(1))[0])
#         print(struct.unpack("<c", memory_map_buf.read(1))[0])
#         print(struct.unpack("<c", memory_map_buf.read(1))[0])
#         semaphore.release()
#         print(struct.unpack("<Q", mem_map_access_record.read(8))[0])
