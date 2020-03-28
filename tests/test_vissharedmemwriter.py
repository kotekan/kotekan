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
import tempfile

from kotekan import runner, shared_memory_buffer

# use tempfile creation to get exclusive random strings
useless_file = tempfile.NamedTemporaryFile()
sem_name = "kotekan_" + os.path.split(useless_file.name)[-1]
fname_buf = "calBuffer_" + os.path.split(useless_file.name)[-1]

page_size = 4096

global_params = {
    "num_elements": 4,
    "num_ev": 2,
    "total_frames": 10,
    "cadence": 5.0,
    "freq": [3, 777, 554],
    "mode": "default",
    "dataset_manager": {"use_dataset_broker": False},
}

params_fakevis = {
    "freq_ids": global_params["freq"],
    "num_frames": global_params["total_frames"],
    "mode": global_params["mode"],
    "cadence": global_params["cadence"],
}

params_fakevis_small = {
    "freq_ids": global_params["freq"],
    "num_frames": global_params["total_frames"] - 4,
    "mode": global_params["mode"],
    "cadence": global_params["cadence"],
}

params_fakevis_large = {
    "freq_ids": global_params["freq"],
    "num_frames": global_params["total_frames"] + 1,
    "mode": global_params["mode"],
    "cadence": global_params["cadence"],
}

global num_frames

params_writer_stage = {"nsamples": global_params["total_frames"], "sem_name": sem_name, "fname_buf": fname_buf}

size_of_uint64 = 8
num_structural_params = 6
pos_access_record = size_of_uint64 * num_structural_params
pos_ring_buffer = pos_access_record + size_of_uint64 * params_writer_stage["nsamples"] * len(global_params["freq"])

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
        stage_type="VisSharedMemWriter",
        stage_config=params_writer_stage,
        buffers_in=fakevis_buffer,
        buffers_out=None,
        global_config=global_params,
    )

    test.run()


@pytest.fixture(scope="module")
def semaphore(vis_data):
    sem = posix_ipc.Semaphore(sem_name)
    yield sem


@pytest.fixture(scope="module")
def memory_map_buf(vis_data):
    memory = posix_ipc.SharedMemory(fname_buf)
    mapfile = mmap.mmap(memory.fd, memory.size, prot=mmap.PROT_READ)
    os.close(memory.fd)
    yield mapfile

def test_structured_data(semaphore, memory_map_buf):
    semaphore.acquire()

    ## Test Structured Data
    num_writes = struct.unpack("<Q", memory_map_buf.read(8))[0]
    num_time = struct.unpack("<Q", memory_map_buf.read(8))[0]
    num_freq = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame_meta = struct.unpack("<Q", memory_map_buf.read(8))[0]
    size_frame_data = struct.unpack("<Q", memory_map_buf.read(8))[0]

    assert num_writes == 0
    assert num_time == params_writer_stage["nsamples"]
    assert num_freq == len(params_fakevis["freq_ids"])
    assert size_frame == page_size
    print("TODO: test if frame metadata size should be {}".format(size_frame_meta))
    print("TODO: test if frame data size should be {}".format(size_frame_data))

    semaphore.release()


def test_access_record(semaphore, memory_map_buf):
    global num_frames

    semaphore.acquire()

    num_time = params_writer_stage["nsamples"]
    num_freq = len(global_params["freq"])

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
            fpga_seq += 800e6 / 2048 * global_params["cadence"]

    elif num_time > num_frames:
        # if ring buffer is larger than the number of frames
        for t in range(num_time):
            for f in range(num_freq):
                access_record = struct.unpack("q", memory_map_buf.read(size_of_uint64))[
                    0
                ]
                assert access_record == fpga_seq
            if t + 1 < num_frames:
                fpga_seq += 800e6 / 2048 * global_params["cadence"]
            else:
                fpga_seq = -1

    elif num_time < num_frames:
        # if ring buffer is smaller than number of frames
        fpga_seqs = []
        fpga_seqs.append(fpga_seq)

        for t in range(1, num_frames):
            fpga_seq += 800e6 / 2048 * global_params["cadence"]
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

def test_ring_buffer(vis_data):
    global num_frames

    num_time = params_writer_stage["nsamples"]
    num_freq = len(global_params["freq"])
    num_ev = global_params["num_ev"]
    num_elements = global_params["num_elements"]

    ring_buffer = shared_memory_buffer.SharedMemoryReader(sem_name, fname_buf, params_writer_stage["nsamples"])
