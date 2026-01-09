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


# --- N2 countCheck tests using FakeN2 with simulate_fpga_restart_at_frame ---

t_start_s = 1_760_000_000
GIGA = 1_000_000_000

n2_root_params = {
    "log_level": "DEBUG",
    "fft_length": 16384,
    "sampling_rate_MHz": 3.2e3,
    "num_elements": 16,
    "num_dishes": 8,
    "num_ev": 0,
    "samples_per_data_set": 16384,
    "sub_integration_ntime": 4096,
    "buffer_depth": 5,
    "telescope": {
        "name": "CHORDTelescope",
        "origin_itrs_lat_deg": 50.0,
        "origin_itrs_lon_deg": -120.0,
        "grid_x_axis": [1.0, 0.0, 0.0],
        "grid_y_axis": [0.0, 1.0, 0.0],
        "dish_elev_axis": [1.0, 0.0, 0.0],
        "dish_vert_axis": [0.0, 0.0, 1.0],
        "dish_coelev_deg": 90.0,
        "require_gps": False,
        "updatable_config": "/earth_rotation_data",
        "num_dishes_x": 4,
        "num_dishes_y": 4,
        "dish_separation_x_m": 6.3,
        "dish_separation_y_m": 8.5,
        "dish_inputs": [
            {
                "dish_idx": i,
                "grid_x_idx": i % 4,
                "grid_y_idx": i // 4,
                "feed_pos_disp_m": [0.0, 0.0, 0.0],
                "coelev_disp_deg": 0.0,
                "type": "ArrayDish",
                "label": f"D{i:02d}",
            }
            for i in range(8)
        ],
    },
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "earth_orientation_parameter_table": [
            {
                "time_inst_ns": (t_start_s - 1000) * GIGA,
                "delta_UT1_inst": 0.0,
                "x_pm": 0.0,
                "y_pm": 0.0,
            },
            {
                "time_inst_ns": (t_start_s + 100000) * GIGA,
                "delta_UT1_inst": 0.0,
                "x_pm": 1.0,
                "y_pm": 1.0,
            },
        ],
    },
    "gps_time": {"frame0_nano": t_start_s * GIGA},
}


@pytest.fixture(scope="module")
def n2_kotekan_output(tmpdir_factory):
    def make_n2_output(restartcount=True):
        tmpdir = tmpdir_factory.mktemp("countcheck_n2")

        num_frames = 8
        # Frame at which to simulate restart (if requested)
        restart_at = num_frames - 2 if restartcount else -1

        input_buffer = runner.FakeN2Buffer(
            num_elements=n2_root_params["num_elements"],
            num_ev=n2_root_params["num_ev"],
            cadence=5.0,
            num_frames=num_frames,
            wait=False,
            start_time=t_start_s,
            simulate_fpga_restart_at_frame=restart_at,
            end_interrupt=True,
        )

        test = runner.KotekanStageTester(
            "countCheck",
            count_params,
            input_buffer,
            None,  # buffers_out is None
            n2_root_params,
            expect_failure=restartcount,
        )

        test.run()

        return test.output

    return make_n2_output


def test_countcheck_n2_restart(n2_kotekan_output):
    output = n2_kotekan_output(restartcount=True)
    msg = "Found wrong start time"
    assert (msg in output) and ("SIGTERM" in output)


def test_countcheck_n2_norestart(n2_kotekan_output):
    output = n2_kotekan_output(restartcount=False)
    msg = "Found wrong start time"
    assert msg not in output
