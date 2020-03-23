# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import mmap
import os
import posix_ipc
import pytest
import struct

from kotekan import runner

sem_name = "kotekan"
fname_access_record = "calBufferAccessRecord"
fname_buf = "calBuffer"
page_size = 4096

params = {
    "num_elements": 7,
    "num_ev": 0,
    "total_frames": 7,
    "cadence": 10.0,
    "mode": "default",
    "dataset_manager": {"use_dataset_broker": False},
}

params_fakevis = {
    "freq_ids": [0, 1, 2],
    "num_frames": params["total_frames"],
    "mode": params["mode"],
}

params_fakevis_small = {
    "freq_ids": [0, 1, 2],
    "num_frames": params["total_frames"] - 4,
    "mode": params["mode"],
}

params_fakevis_large = {
    "freq_ids": [0, 1, 2],
    "num_frames": params["total_frames"] + 1,
    "mode": params["mode"],
}

global num_frames

params_writer_stage = {"nsamples": 7}


@pytest.fixture(
    scope="module", params=[params_fakevis, params_fakevis_small, params_fakevis_large]
)
def vis_data(tmpdir_factory, request):
    global num_frames

    # keeping all the data this test produced here (probably do not need it)
    # using FakeVisBuffer to produce fake data
    fakevis_buffer = runner.FakeVisBuffer(**request.param)
    num_frames = request.param["num_frames"]

    # KotekanStageTester is used to run kotekan with my config
    test = runner.KotekanStageTester(
        stage_type="visSharedMemWriter",
        stage_config=params_writer_stage,
        buffers_in=fakevis_buffer,
        buffers_out=None,
        global_config=params,
    )

    test.run()


@pytest.fixture(scope="module")
def semaphore(vis_data):
    sem = posix_ipc.Semaphore(sem_name)
    yield sem
    sem.release()
    sem.unlink()


@pytest.fixture(scope="module")
def memory_map_buf(vis_data):
    memory = posix_ipc.SharedMemory(fname_buf)
    mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
    os.close(memory.fd)
    yield mapfile
    mapfile.close()
    posix_ipc.unlink_shared_memory(fname_buf)


def test_access_record(semaphore, memory_map_buf):
    global num_frames
    size_of_uint64 = 8
    num_structural_params = 6
    pos_access_record = size_of_uint64 * num_structural_params

    semaphore.acquire()

    num_writes = struct.unpack("<Q", memory_map_buf.read(8))[0]
    num_time = struct.unpack("<Q", memory_map_buf.read(8))[0]
    num_freq = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame_meta = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame_data = struct.unpack("<Q", memory_map_buf.read(8))[0]

    assert num_writes >= 0
    assert num_time == params_writer_stage["nsamples"]
    assert num_freq == len(params_fakevis["freq_ids"])
    assert size_frame == page_size
    print("TODO: test if frame metadata size should be {}".format(size_frame_meta))
    print("TODO: test if frame data size should be {}".format(size_frame_data))
    print("NUMBER OF FRAMES" + str(num_frames))

    memory_map_buf.seek(pos_access_record)
    fpga_seq = 0

    if num_time == num_frames:
        # if ring buffer is the same size as the number of frames
        for t in range(num_time):
            for f in range(num_freq):
                access_record = struct.unpack("q", memory_map_buf.read(size_of_uint64))[
                    0
                ]
                assert access_record == fpga_seq
            fpga_seq += 800e6 / 2048 * params["cadence"]

    elif num_time > num_frames:
        # if ring buffer is larger than the number of frames
        for t in range(num_time):
            for f in range(num_freq):
                access_record = struct.unpack("q", memory_map_buf.read(size_of_uint64))[
                    0
                ]
                assert access_record == fpga_seq
            if t + 1 < num_frames:
                fpga_seq += 800e6 / 2048 * params["cadence"]
            else:
                fpga_seq = -1

    elif num_time < num_frames:
        # if ring buffer is smaller than number of frames
        fpga_seqs = []
        fpga_seqs.append(fpga_seq)

        for t in range(1, num_frames):
            fpga_seq += 800e6 / 2048 * params["cadence"]
            fpga_seqs.append(fpga_seq)

        for t in range(num_time):
            for f in range(num_freq):
                access_record = struct.unpack("q", memory_map_buf.read(size_of_uint64))[
                    0
                ]
                if t == 0:
                    assert access_record == fpga_seqs[-1]
                else:
                    assert access_record == fpga_seqs[t]

        semaphore.release()
