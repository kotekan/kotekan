# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

downsamp_params = {
    "num_samples": 2,
    "total_frames": 11,
    "fakevis_mode": "fill_ij_missing",
    "cadence": 2.0,
}

global_params = {
    "num_elements": 4,
    "num_ev": 4,
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "earth_orientation_parameter_table": [
            {
                "time_inst_ns": 1578010000_000_000_000,
                "delta_UT1_inst": 0.5,
                "x_pm": 0.1,
                "y_pm": 0.1,
            },
            {
                "time_inst_ns": 1578010001_000_000_000,
                "delta_UT1_inst": 0.4,
                "x_pm": 0.12,
                "y_pm": 0.12,
            },
            {
                "time_inst_ns": 1578010002_000_000_000,
                "delta_UT1_inst": 0.3,
                "x_pm": 0.123,
                "y_pm": 0.123,
            },
        ],
    },
    "telescope": {
        "name": "CHORDTelescope",
        "require_gps": False,
        "inst_long_deg": -119.62081125,
        "inst_lat_deg": 49.32075144444,
        "inst_grid_axis_x": [1, 0, 0],
        "inst_grid_axis_y": [0, 1, 0],
        "inst_alt_axis_polar_angle_deg": 90.0,
        "inst_alt_axis_azimuth_angle_deg": 0.0,
        "inst_alt_deg": 90.0,
        "dish_positions": [[0.0, 0.0, 0.0]],
        "updatable_config": "/earth_rotation_data",
    },
    "gps_time": {"frame0_nano": 0},
}


@pytest.fixture(scope="module")
def n2_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("n2_data")

    dump_buffer = runner.DumpN2Buffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "N2TimeDownsample",
        downsamp_params,
        runner.FakeVisBuffer(
            num_frames=downsamp_params["total_frames"],
            mode=downsamp_params["fakevis_mode"],
            cadence=downsamp_params["cadence"],
        ),
        dump_buffer,
        global_params,
    )

    test.run()

    yield dump_buffer.load()


def test_structure(n2_data):

    n = global_params["num_elements"]

    # Check that each samples is the expected shape
    for frame in n2_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == global_params["num_ev"]

    # Check that we have the expected number of samples
    nsamp = downsamp_params["total_frames"] // downsamp_params["num_samples"]
    assert len(n2_data) == nsamp


def test_metadata(n2_data):

    input_frame_length = int(800e6 / 2048 * downsamp_params["cadence"])
    frame_length = input_frame_length * downsamp_params["num_samples"]
    frame_total = (input_frame_length - 2) * downsamp_params["num_samples"]
    rfi_total = downsamp_params["num_samples"]

    for frame in n2_data:
        assert frame.metadata.freq_id == 0
        assert frame.metadata.frame_length_fpga_ticks == frame_length
        assert frame.metadata.n_valid_fpga_ticks_in_frame == frame_total
        assert frame.metadata.n_rfi_fpga_ticks == rfi_total


def test_time(n2_data):
    def timespec_to_float(ts):
        return ts.tv + ts.tv_nsec * 1e-9

    time_ns = np.array([v.metadata.frame_start_time_ns for v in n2_data])
    time_s = time_ns * 1.0e-9

    # Check downsampled cadence
    assert np.all(
        np.diff(time_s) == downsamp_params["cadence"] * downsamp_params["num_samples"]
    )


def test_contents(n2_data):

    n = global_params["num_elements"]
    n_ev = global_params["num_ev"]

    # Reproduce expected fakeVis output
    model_vis = np.zeros(n * (n + 1) // 2, dtype=np.complex64)
    ind = 0
    for i in range(n):
        for j in range(i, n):
            model_vis[ind] = i + j * 1j
            ind += 1
    model_evec = np.zeros(n_ev * n, dtype=np.complex64)
    model_eval = np.zeros(n_ev, dtype=np.float32)
    for i in range(n_ev):
        model_eval[i] = i
        for j in range(n):
            model_evec[i * n + j] = i + 1j * j

    # Averaging shouldn't change vis, eigenstuff
    for frame in n2_data:
        assert np.all(frame.vis == model_vis)
        assert np.all(frame.evec == model_evec)
        assert np.all(frame.eval == model_eval)
        assert frame.erms == 1.0

    # weights get an extra factor of nsamp
    for frame in n2_data:
        assert np.all(frame.weight == downsamp_params["num_samples"])
