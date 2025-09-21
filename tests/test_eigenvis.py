# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

# Skip if LAPACK support not built into kotekan
if not runner.has_lapack():
    pytest.skip("LAPACK support not available.", allow_module_level=True)


default_params = {
    "num_elements": 200,
    "num_ev": 4,
    "total_frames": 16,
    "cadence": 10.0,
    "mode": "phase_ij",
    # 'mode': 'fill_ij',
    "freq": [0],
    "buffer_depth": 5,
    "num_diagonals_filled": 0,
    "dataset_manager": {"use_dataset_broker": False},
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "earth_orientation_parameter_table": [
            {"time_inst_ns": 0, "delta_UT1_inst": 0.0, "x_pm": 0.0, "y_pm": 0.0,},
            {
                "time_inst_ns": 3_000_000_000 * 1_000_000_000,
                "delta_UT1_inst": 0.0,
                "x_pm": 0.0,
                "y_pm": 0.0,
            },
        ],
    },
    "telescope": {
        "name": "CHORDTelescope",
        "require_gps": False,
        "inst_long_deg": -119.62081125,
        "inst_lat_deg": 49.32075144444,
        "inst_grid_x_axis": [1, 0, 0],
        "inst_grid_y_axis": [0, 1, 0],
        "inst_dish_alt_axis": [1, 0, 0],
        "inst_dish_vert_axis": [0, 0, 1],
        "inst_alt_deg": 90.0,
        "dish_positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "updatable_config": "/earth_rotation_data",
    },
    "gps_time": {"frame0_nano": 0},
}


def run_eigenvis(tdir_factory, params=None):

    if not params:
        params = default_params

    tmpdir = tdir_factory.mktemp("eigenvis")

    fakevis_buffer = runner.FakeN2VisBuffer(
        freq_ids=params["freq"], num_frames=params["total_frames"], mode=params["mode"]
    )

    dump_buffer = runner.DumpN2Buffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "eigenVis", {}, fakevis_buffer, dump_buffer, params
    )

    test.run()
    return dump_buffer.load()


def test_basic(tmpdir_factory):

    params = default_params
    num_elements = params["num_elements"]
    expected_evec_phase_fact = np.exp(1j * np.arange(num_elements))

    eigen_data = run_eigenvis(tmpdir_factory)
    for frame in eigen_data:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - num_elements) / num_elements < 1e-6
        assert np.allclose(largest_evec / largest_evec[0], expected_evec_phase_fact)
        assert np.allclose(abs(largest_evec), 1 / np.sqrt(num_elements))
        zero_eval = frame.eval[1]
        assert zero_eval / num_elements < 1e-6
        # The erms error (below) may be larger due to accumulated error.
        # If this test is failing, it might be ok; some additional discussion is at
        # https://github.com/kotekan/kotekan/issues/1034.
        assert frame.erms < 1e-5


def test_filled(tmpdir_factory):

    params = dict(default_params)
    params["num_diagonals_filled"] = 10
    num_elements = params["num_elements"]
    expected_evec_phase = np.exp(1j * np.arange(num_elements))

    eigen_data = run_eigenvis(tmpdir_factory, params)
    for frame in eigen_data[8:]:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - num_elements) / num_elements < 1e-4
        assert np.allclose(
            largest_evec / largest_evec[0], expected_evec_phase, rtol=1e-3
        )
        assert np.allclose(abs(largest_evec), 1 / np.sqrt(num_elements), rtol=1e-3)
        assert frame.erms < 1e-3


def test_input_excluded(tmpdir_factory):

    params = dict(default_params)
    params["exclude_inputs"] = [5, 10, 6]
    nexclude = len(params["exclude_inputs"])
    num_elements = params["num_elements"]
    expected_eval = num_elements - nexclude
    expected_evec_phase_fact = np.exp(1j * np.arange(num_elements))
    expected_evec_phase_fact[params["exclude_inputs"]] = 0

    eigen_data = run_eigenvis(tmpdir_factory, params)
    for frame in eigen_data:
        largest_eval = frame.eval[0]
        largest_evec = frame.evec[:num_elements]
        assert abs(largest_eval - expected_eval) / num_elements < 1e-6
        assert np.allclose(largest_evec / largest_evec[0], expected_evec_phase_fact)
