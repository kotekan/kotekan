# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa pylint: disable=W0401, W0614

# == End Python 2/3 compatibility

import logging
import numpy as np
import os
import posix_ipc
import pytest
import re
import signal
from subprocess import Popen
import shutil
import tempfile
import threading
from time import sleep

from comet import Manager

from kotekan import runner, shared_memory_buffer

# use tempfile creation to get exclusive random strings
useless_file = tempfile.NamedTemporaryFile()
fname_buf = "calBuffer_" + os.path.split(useless_file.name)[-1]

logging.basicConfig(level=logging.DEBUG)


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


@pytest.fixture(scope="module")
def semaphore():
    sem = posix_ipc.Semaphore(fname_buf)
    yield sem
    sem.release()
    sem.unlink()


params = {
    "num_elements": 7,
    "num_ev": 0,
    "total_frames": 8,
    "cadence": 1.0,
    "mode": "change_state",
    "dataset_manager": {"use_dataset_broker": True},
}

start_time = 1_500_000_000
flagging_update_time = start_time + 4
gain_update_time = start_time + 5
params_fakevis = {
    "freq_ids": [1, 2, 3, 4, 7, 10],
    "num_frames": params["total_frames"],
    "mode": params["mode"],
    "wait": True,
    "start_time": start_time,
    "state_changes": [
        {"timestamp": flagging_update_time, "type": "flags"},
        {"timestamp": gain_update_time, "type": "gains"},
    ],
}

params_writer_stage = {"num_samples": 5, "name": fname_buf}


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
    ds_manager = Manager("localhost", comet_broker)
    sleep(2)
    reader = []
    view_size = [2, 8, 17]

    for i in range(len(view_size)):
        reader.append(
            shared_memory_buffer.SharedMemoryReader(fname_buf, view_size[i], ds_manager)
        )

    for i in range(len(reader)):
        assert reader[i].num_time == params_writer_stage["num_samples"]
        assert reader[i].num_freq == len(params_fakevis["freq_ids"])

    i = 0
    visraw = [None] * len(view_size)
    with pytest.raises(shared_memory_buffer.SharedMemoryError):
        # Give this some extra time, because maybe it's just reading nothing a few times, but make
        # sure it eventually raises because the shared memory got removed by the writer.
        while i <= params["total_frames"] * 2:
            sleep(0.5)
            for j in range(len(reader)):
                visraw[j] = reader[j].update()
                assert visraw[j].num_time == reader[j].view_size
                check_visraw(visraw[j])
            i += 1
    assert i >= params["total_frames"] / 2


def check_visraw(visraw):
    """Test content of valid frames."""
    num_freq = len(params_fakevis["freq_ids"])
    num_ev = params["num_ev"]
    num_elements = params["num_elements"]
    valid = visraw.valid_frames.astype(np.bool)

    num_time = visraw.num_time
    assert visraw.num_freq == len(params_fakevis["freq_ids"])

    num_prod = int(num_elements * (num_elements + 1) / 2)

    # check valid frames only
    assert (visraw.num_prod[valid] == num_prod).all()
    assert (visraw.metadata["num_elements"][valid] == num_elements).all()
    assert (visraw.metadata["num_ev"][valid] == num_ev).all()

    # check gain/flag update IDs VisRaw got from comet
    assert (
        visraw.update_id["flags"][valid & (visraw.time["ctime"] < flagging_update_time)]
        == "None"
    ).all()
    assert (
        visraw.update_id["gains"][valid & (visraw.time["ctime"] < gain_update_time)]
        == "None"
    ).all()
    assert (
        visraw.update_id["flags"][
            valid & (visraw.time["ctime"] >= flagging_update_time)
        ]
        == "flag_update_0"
    ).all()
    assert (
        visraw.update_id["gains"][valid & (visraw.time["ctime"] >= gain_update_time)]
        == "gain_update_0"
    ).all()

    evals = visraw.data["eval"]
    evecs = visraw.data["evec"]
    erms = visraw.data["erms"]

    # Check datasets are present
    assert evals.shape == (num_time, num_freq, num_ev)
    assert evecs.shape == (num_time, num_freq, num_ev * num_elements)
    assert erms.shape == (num_time, num_freq)

    evecs = evecs.reshape(num_time, num_freq, num_ev, num_elements)

    # Check that the datasets have the correct values
    assert (evals == np.arange(num_ev)[np.newaxis, np.newaxis, :]).all()
    assert (
        evecs.real == np.arange(num_ev)[np.newaxis, np.newaxis, :, np.newaxis]
    ).all()
    assert (
        evecs.imag == np.arange(num_elements)[np.newaxis, np.newaxis, np.newaxis, :]
    ).all()
    assert (erms[valid] == 1).all()

    ftime = visraw.time["fpga_count"]
    ctime = visraw.time["ctime"]
    freq = visraw.metadata["freq_id"]

    # check "default" test pattern in vis
    vis = visraw.data["vis"].view(np.complex64)
    assert vis.shape == (num_time, num_freq, num_prod)
    i = 0
    for time_slot_vis, time_slot_valid in zip(vis, visraw.valid_frames):
        f = 0
        for frame, valid in zip(time_slot_vis, time_slot_valid):
            if valid:
                assert (frame.real[0] == ftime[i].astype(np.float32)).all()
                assert (frame.real[1] == ctime[i].astype(np.float32)).all()
                assert frame.real[2] == freq[i][f].astype(np.float32)
                assert (frame.real[3:] == 0).all()
                k = num_elements
                l = 0
                m = 0
                for j in range(num_prod):
                    if j == l:
                        assert frame.imag[j] == m
                        l += k
                        k -= 1
                        m += 1
                    else:
                        assert frame.imag[j] == 0
            f += 1
        i += 1

    if num_time > 0:
        # find last valid timestamp
        valid_times = visraw.time[valid]
        return num_time, valid_times[-1]["fpga_count"]
    return 0, None
