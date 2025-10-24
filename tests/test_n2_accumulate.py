# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

t_start_s = 1_760_000_000
GIGA = 1_000_000_000


global_params = {
    "num_elements": 64,
    "num_ev": 0,
    "samples_per_data_set": 16384,
    "sub_integration_ntime": 8192,
    "rfi_downsampling_factor": 256,
    "num_local_freq": 1,
    "total_frames": 128,
    "telescope": {
        "name": "CHORDTelescope",
        "inst_lat_deg": 50.0,
        "inst_long_deg": -120.0,
        "inst_alt_deg": 90.0,
        "inst_grid_x_axis": [1.0, 0.0, 0.0],
        "inst_grid_y_axis": [0.0, 1.0, 0.0],
        "inst_dish_alt_axis": [1.0, 0.0, 0.0],
        "inst_dish_vert_axis": [0.0, 0.0, 1.0],
        "dish_positions": [
            [0.0, 0.0, 0.0],
            [6.3, 0.0, 0.0],
            [0.0, 8.4, 0.0],
            [6.3, 8.4, 0.0],
            [151.0, 0.0, 0.0],
            [0.0, 204.0, 0.0],
            [151.0, 204.0, 0.0]],
        "require_gps": False,
        "updatable_config": "/earth_rotation_data"
    },
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "earth_orientation_parameter_table": [
            {
                "time_inst_ns": (t_start_s - 1000)* GIGA,
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

accumulate_default_params = {
    "num_freq_per_n2k_frame": "num_local_freq",
    "num_frames_to_accumulate": 32,
    "packet_loss_is_scalar": True,
}


@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    input_buffers = runner.FakeN2KBuffers(global_params['samples_per_data_set'],
                                          global_params['num_local_freq'], {}, {})
    dump_buffer = runner.DumpN2Buffer(str(tmpdir), "total_frames * num_local_freq")

    accumulate_params = accumulate_default_params.copy()

    accumulate_params["in_counts_buf"] = input_buffers.counts_name
    accumulate_params["in_rfimask_buf"] = input_buffers.rfi_name

    test = runner.KotekanStageTester(
        "N2Accumulate",
        accumulate_params,
        input_buffers,
        dump_buffer,
        global_params,
    )

    test.run()

    yield dump_buffer.load()



def test_structure(accumulate_data):

    n = global_default_params["num_elements"]

    # Check that each samples is the expected shape
    for frame in accumulate_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == 0

    # Check that we have the expected number of samples
    nsamp = accumulate_params["total_frames"] // accumulate_params["int_frames"]
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for frame in accumulate_data:
        assert frame.metadata.freq_id == accumulate_params["freq"]


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(accumulate_params["num_elements"])

    pat = (row + 1.0j * col).astype(np.complex64)

    for frame in accumulate_data:

        assert (frame.vis == pat).all()
        assert (frame.weight == 8.0).all()
        assert (frame.flags == 1.0).all()
        assert (frame.gain == 1.0).all()
