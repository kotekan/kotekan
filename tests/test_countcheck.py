# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
import time

from kotekan import runner
from kotekan import visbuffer
from kotekan import n2buffer

root_params = {
    "log_level": "DEBUG",
    "num_elements": 16,
    "num_prod": 136,
    "num_ev": 2,
    "total_frames": 128,
    "cadence": 5.0,
    "mode": "fill_ij",
    "freq_ids": [250],
    "buffer_depth": 5,
}

count_params = {}


@pytest.fixture(scope="module")
def kotekan_output(tmpdir_factory):
    def make_kotekan_output(restartcount=True):
        counts_per_second = 390625
        start_time = time.time()

        tmpdir = tmpdir_factory.mktemp("countcheck")

        # Generate a list of 'VisBuffer' objects.
        frame_list = []
        for ii in range(root_params["buffer_depth"]):
            frame_list.append(
                visbuffer.VisBuffer.new_from_params(
                    root_params["num_elements"],
                    root_params["num_prod"],
                    root_params["num_ev"],
                )
            )
            frame_list[ii].metadata.fpga_seq = (
                counts_per_second * int(root_params["cadence"]) * ii
            )
            frame_list[ii].metadata.ctime.tv = int(
                start_time + int(root_params["cadence"]) * ii
            )

        if restartcount:
            # Have the last frame restart FPGA counts:
            frame_list[-1].metadata.fpga_seq = 0

        # ReadVisBuffer receives a list of frames and writes them down to disk.
        read_buffer = runner.ReadVisBuffer(str(tmpdir), frame_list)
        read_buffer.write()

        test = runner.KotekanStageTester(
            "countCheck",
            count_params,
            read_buffer,
            None,  # buffers_out is None
            root_params,
            expect_failure=True,
        )

        test.run()

        return test.output

    return make_kotekan_output


def test_countcheck_restart(kotekan_output):
    output = kotekan_output(restartcount=True)
    # Kotekan returns 0 whether 'countcheck' raises SIGTERM or not.
    # So I can't check the return code to test 'countcheck'.
    # For now, I parse the output for specific messages from 'countcheck'
    # and for a SIGTERM message. Not very elegant...
    msg = "Found wrong start time"
    assert (msg in output) and ("SIGTERM" in output)


def test_countcheck_norestart(kotekan_output):
    # Test that countCheck is not raising SIGTERM without reason.
    output = kotekan_output(restartcount=False)
    msg = "Found wrong start time"
    assert msg not in output


# =============================
# N2 buffer tests
# =============================


@pytest.fixture(scope="module")
def kotekan_output_n2(tmpdir_factory):
    def make_kotekan_output(restartcount=True):
        counts_per_second = 390625
        tick_len_ns = int(1_000_000_000 / counts_per_second)
        cadence = int(root_params["cadence"])
        delta_ticks = counts_per_second * cadence
        delta_ns = tick_len_ns * delta_ticks
        start_time_ns = int(time.time()) * 1_000_000_000

        tmpdir = tmpdir_factory.mktemp("countcheck_n2")

        frame_list = []
        for ii in range(root_params["buffer_depth"]):
            buf = n2buffer.N2Buffer.new_from_params(
                root_params["num_elements"],
                root_params["num_prod"],
                root_params["num_ev"],
            )

            meta = buf.metadata
            meta.num_elements = root_params["num_elements"]
            meta.num_prod = root_params["num_prod"]
            meta.num_ev = root_params["num_ev"]
            meta.nfreq = 1
            meta.freq_id = root_params["freq_ids"][0]
            meta.vis_layout = 0  # FullUpperTri
            meta.fpga_start_tick = delta_ticks * ii
            meta.frame_start_time_ns = start_time_ns + delta_ns * ii
            meta.frame_length_fpga_ticks = delta_ticks
            meta.n_valid_fpga_ticks = delta_ticks
            meta.n_rfi_fpga_ticks = 0

            frame_list.append(buf)

        if restartcount:
            frame_list[-1].metadata.fpga_start_tick = 0
            frame_list[-1].metadata.frame_start_time_ns = start_time_ns + delta_ns * (
                root_params["buffer_depth"] - 1
            )

        read_buffer = runner.ReadN2Buffer(str(tmpdir), frame_list)
        read_buffer.write()

        global_config = dict(root_params)
        global_config["vis_layout"] = "FullUpperTri"

        test = runner.KotekanStageTester(
            "countCheck",
            count_params,
            read_buffer,
            None,  # buffers_out is None
            global_config,
            expect_failure=True,
        )

        test.run()

        return test.output

    return make_kotekan_output


def test_countcheck_n2_restart(kotekan_output_n2):
    output = kotekan_output_n2(restartcount=True)
    msg = "Found wrong start time"
    assert (msg in output) and ("SIGTERM" in output)


def test_countcheck_n2_norestart(kotekan_output_n2):
    output = kotekan_output_n2(restartcount=False)
    msg = "Found wrong start time"
    assert msg not in output
