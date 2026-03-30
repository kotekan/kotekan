# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import time

from kotekan import runner
from kotekan import visbuffer


# -----------------------------------------------------------------------------
# Shared configuration
# -----------------------------------------------------------------------------

T_START_S = 1_760_000_000
GIGA = 1_000_000_000


def make_chord_telescope_config(num_dishes=8):
    """Create a standard CHORDTelescope configuration for testing."""
    return {
        "name": "CHORDTelescope",
        "origin_itrs_lat_deg": 50.0,
        "origin_itrs_lon_deg": -120.0,
        "grid_x_axis": [1.0, 0.0, 0.0],
        "grid_y_axis": [0.0, 1.0, 0.0],
        "dish_elev_axis": [1.0, 0.0, 0.0],
        "dish_vert_axis": [0.0, 0.0, 1.0],
        "dish_coelev_deg": 90.0,
        "require_gps": False,
        "eop_updatable_config": "/earth_rotation_data",
        "num_dishes": num_dishes,
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
            for i in range(num_dishes)
        ],
    }


EARTH_ROTATION_DATA = {
    "kotekan_update_endpoint": "json",
    "earth_orientation_parameter_table": [
        {
            "t_inst_ns": (T_START_S - 1000) * GIGA,
            "delta_UT1_inst": 0.0,
            "xp_as": 0.0,
            "yp_as": 0.0,
        },
        {
            "t_inst_ns": (T_START_S + 100000) * GIGA,
            "delta_UT1_inst": 0.0,
            "xp_as": 1.0,
            "yp_as": 1.0,
        },
    ],
}

GPS_TIME = {"frame0_nano": T_START_S * GIGA}


# -----------------------------------------------------------------------------
# Buffer wrapper classes
# -----------------------------------------------------------------------------


class FakeHFBBuffer(runner.InputBuffer):
    """Input buffer using FakeHFB stage with HFBMetadata."""

    _buf_ind = 0

    def __init__(self, **kwargs):
        self.name = f"fakehfb_buf{self._buf_ind}"
        stage_name = f"fakehfb{self._buf_ind}"
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "hfb",
                "metadata_pool": "HFB_pool",
                "num_frames": "buffer_depth",
            },
            "HFB_pool": {
                "kotekan_metadata_pool": "HFBMetadata",
                "num_metadata_objects": "30 * buffer_depth",
            },
        }

        stage_config = {
            "kotekan_stage": "FakeHFB",
            "out_buf": self.name,
            "freq_ids": [0],
            "wait": False,
        }
        stage_config.update(kwargs)
        self.stage_block = {stage_name: stage_config}


class ChordMetadataBuffer(runner.InputBuffer):
    """Input buffer using testDataGen stage with chordMetadata."""

    _buf_ind = 0

    def __init__(self, frame_size=1024, **kwargs):
        self.name = f"testdatagen_buf{self._buf_ind}"
        stage_name = f"testdatagen{self._buf_ind}"
        self.__class__._buf_ind += 1

        self.buffer_block = {
            self.name: {
                "kotekan_buffer": "standard",
                "frame_size": frame_size,
                "metadata_pool": "main_pool",
                "num_frames": "buffer_depth",
            },
        }

        stage_config = {
            "kotekan_stage": "testDataGen",
            "out_buf": self.name,
            "type": "const",
            "value": 0,
            "wait": False,
        }
        stage_config.update(kwargs)
        self.stage_block = {stage_name: stage_config}


# -----------------------------------------------------------------------------
# Test runner helper
# -----------------------------------------------------------------------------


def run_countcheck_test(input_buffer, root_params, expect_failure):
    """Run countCheck stage and return output."""
    test = runner.KotekanStageTester(
        "countCheck",
        {},  # count_params
        input_buffer,
        None,
        root_params,
        expect_failure=expect_failure,
    )
    test.run()
    return test.output


def assert_restart_detected(output):
    """Assert that countCheck detected an FPGA restart."""
    assert "Found wrong start time" in output and "SIGTERM" in output


def assert_no_restart_detected(output):
    """Assert that countCheck did not detect an FPGA restart."""
    assert "Found wrong start time" not in output


# -----------------------------------------------------------------------------
# VisMetadata tests (uses ReadVisBuffer with pre-generated frames)
# -----------------------------------------------------------------------------

VIS_ROOT_PARAMS = {
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


@pytest.fixture(scope="module")
def vis_output(tmpdir_factory):
    """Factory fixture for VisMetadata countCheck tests."""

    def _run(simulate_restart):
        counts_per_second = 390625
        start_time = time.time()
        tmpdir = tmpdir_factory.mktemp("countcheck_vis")

        # Generate VisBuffer frames
        frame_list = []
        for ii in range(VIS_ROOT_PARAMS["buffer_depth"]):
            frame = visbuffer.VisBuffer.new_from_params(
                VIS_ROOT_PARAMS["num_elements"],
                VIS_ROOT_PARAMS["num_prod"],
                VIS_ROOT_PARAMS["num_ev"],
            )
            frame.metadata.fpga_seq = (
                counts_per_second * int(VIS_ROOT_PARAMS["cadence"]) * ii
            )
            frame.metadata.ctime.tv = int(
                start_time + int(VIS_ROOT_PARAMS["cadence"]) * ii
            )
            frame_list.append(frame)

        if simulate_restart:
            frame_list[-1].metadata.fpga_seq = 0

        read_buffer = runner.ReadVisBuffer(str(tmpdir), frame_list)
        read_buffer.write()

        return run_countcheck_test(read_buffer, VIS_ROOT_PARAMS, expect_failure=True)

    return _run


@pytest.mark.parametrize(
    "simulate_restart", [True, False], ids=["restart", "norestart"]
)
def test_countcheck_vis(vis_output, simulate_restart):
    output = vis_output(simulate_restart)
    if simulate_restart:
        assert_restart_detected(output)
    else:
        assert_no_restart_detected(output)


# -----------------------------------------------------------------------------
# N2Metadata tests (uses FakeN2Buffer)
# -----------------------------------------------------------------------------

N2_ROOT_PARAMS = {
    "log_level": "DEBUG",
    "fft_length": 16384,
    "sampling_rate_MHz": 3.2e3,
    "num_elements": 16,
    "num_dishes": 8,
    "num_ev": 0,
    "samples_per_data_set": 16384,
    "sub_integration_ntime": 4096,
    "buffer_depth": 5,
    "telescope": make_chord_telescope_config(),
    "earth_rotation_data": EARTH_ROTATION_DATA,
    "gps_time": GPS_TIME,
}


@pytest.fixture(scope="module")
def n2_output(tmpdir_factory):
    """Factory fixture for N2Metadata countCheck tests."""

    def _run(simulate_restart):
        num_frames = 8
        restart_at = num_frames - 2 if simulate_restart else -1

        input_buffer = runner.FakeN2Buffer(
            num_elements=N2_ROOT_PARAMS["num_elements"],
            num_ev=N2_ROOT_PARAMS["num_ev"],
            cadence=5.0,
            num_frames=num_frames,
            wait=False,
            start_time=T_START_S,
            simulate_fpga_restart_at_frame=restart_at,
            end_interrupt=True,
        )

        return run_countcheck_test(
            input_buffer, N2_ROOT_PARAMS, expect_failure=simulate_restart
        )

    return _run


@pytest.mark.parametrize(
    "simulate_restart", [True, False], ids=["restart", "norestart"]
)
def test_countcheck_n2(n2_output, simulate_restart):
    output = n2_output(simulate_restart)
    if simulate_restart:
        assert_restart_detected(output)
    else:
        assert_no_restart_detected(output)


# -----------------------------------------------------------------------------
# HFBMetadata tests (uses FakeHFBBuffer)
# -----------------------------------------------------------------------------

HFB_ROOT_PARAMS = {
    "log_level": "DEBUG",
    "num_frb_total_beams": 4,
    "factor_upchan": 16,
    "buffer_depth": 5,
    "dataset_manager": {"use_dataset_broker": False},
    # FakeHFB uses tick rate of 800e6/2048 = 390625 Hz
    # CHORDTelescope tick length = 1e9 * fft_length / sampling_rate
    # To match: 1e9/390625 = 2560 ns => use fft_length=2048, sampling_rate=800 MHz
    "fft_length": 2048,
    "sampling_rate_MHz": 800,
    "telescope": make_chord_telescope_config(),
    "earth_rotation_data": EARTH_ROTATION_DATA,
    "gps_time": GPS_TIME,
}


@pytest.fixture(scope="module")
def hfb_output(tmpdir_factory):
    """Factory fixture for HFBMetadata countCheck tests."""

    def _run(simulate_restart):
        num_frames = 8
        restart_at = num_frames - 2 if simulate_restart else -1

        input_buffer = FakeHFBBuffer(
            num_frb_total_beams=HFB_ROOT_PARAMS["num_frb_total_beams"],
            factor_upchan=HFB_ROOT_PARAMS["factor_upchan"],
            cadence=5.0,
            num_frames=num_frames,
            wait=False,
            start_time=T_START_S,
            simulate_fpga_restart_at_frame=restart_at,
            end_interrupt=True,
        )

        return run_countcheck_test(
            input_buffer, HFB_ROOT_PARAMS, expect_failure=simulate_restart
        )

    return _run


@pytest.mark.parametrize(
    "simulate_restart", [True, False], ids=["restart", "norestart"]
)
def test_countcheck_hfb(hfb_output, simulate_restart):
    output = hfb_output(simulate_restart)
    if simulate_restart:
        assert_restart_detected(output)
    else:
        assert_no_restart_detected(output)


# -----------------------------------------------------------------------------
# chordMetadata tests (uses ChordMetadataBuffer/testDataGen)
# -----------------------------------------------------------------------------

CHORD_ROOT_PARAMS = {
    "log_level": "DEBUG",
    "buffer_depth": 5,
    "samples_per_data_set": 32768,
    "fft_length": 16384,
    "sampling_rate_MHz": 3200,
    "telescope": make_chord_telescope_config(),
    "earth_rotation_data": EARTH_ROTATION_DATA,
    "gps_time": GPS_TIME,
}


@pytest.fixture(scope="module")
def chord_output(tmpdir_factory):
    """Factory fixture for chordMetadata countCheck tests."""

    def _run(simulate_restart):
        num_frames = 8
        restart_at = num_frames - 2 if simulate_restart else -1

        input_buffer = ChordMetadataBuffer(
            frame_size=1024,
            samples_per_data_set=CHORD_ROOT_PARAMS["samples_per_data_set"],
            num_frames=num_frames,
            wait=False,
            # Start with a large fpga_seq so restart detection works
            # (first_packet_recv_time is wall clock, so we need fpga_seq
            # to contribute significantly to the computed start_time)
            first_frame_index=100000,
            simulate_fpga_restart_at_frame=restart_at,
            end_interrupt=True,
        )

        return run_countcheck_test(
            input_buffer, CHORD_ROOT_PARAMS, expect_failure=simulate_restart
        )

    return _run


@pytest.mark.parametrize(
    "simulate_restart", [True, False], ids=["restart", "norestart"]
)
def test_countcheck_chord(chord_output, simulate_restart):
    output = chord_output(simulate_restart)
    if simulate_restart:
        assert_restart_detected(output)
    else:
        assert_no_restart_detected(output)
