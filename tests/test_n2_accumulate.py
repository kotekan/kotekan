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

nfreq = 3
freq_ids = [1, 1024, 8192]

count_vals = [0, 8192, 8191, 20, 5000]
rfi_vals = [255, 0, 255, 255]

global_params = {
    "fft_length": 16384,
    "sampling_rate_Hz": 3.2e9,
    "num_elements": 64,
    "num_dishes": 32,
    "dish_separation_ew_m": 6.3,
    "dish_separation_ns_m": 8.5,
    "num_ev": 0,
    "samples_per_data_set": 16384,
    "sub_integration_ntime": 4096,
    "rfi_downsampling_factor": 256,
    "num_local_freq": nfreq,
    "total_frames": 32,
    "telescope": {
        "name": "CHORDTelescope",
        "inst_lat_deg": 50.0,
        "inst_long_deg": -120.0,
        "inst_coelev_deg": 90.0,
        "inst_grid_x_axis": [1.0, 0.0, 0.0],
        "inst_grid_y_axis": [0.0, 1.0, 0.0],
        "inst_dish_elev_axis": [1.0, 0.0, 0.0],
        "inst_dish_vert_axis": [0.0, 0.0, 1.0],
        "require_gps": False,
        "updatable_config": "/earth_rotation_data",
    },
    "dish_inputs" : [
        {'dish_idx': 0, 'ew_idx': 0, 'ns_idx': 0, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D00"},
        {'dish_idx': 1, 'ew_idx': 1, 'ns_idx': 0, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D01"},
        {'dish_idx': 2, 'ew_idx': 0, 'ns_idx': 1, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D10"},
        {'dish_idx': 3, 'ew_idx': 1, 'ns_idx': 1, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D11"},
        {'dish_idx': 4, 'ew_idx': 24, 'ns_idx': 0, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D02"},
        {'dish_idx': 5, 'ew_idx': 0, 'ns_idx': 24, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D20"},
        {'dish_idx': 6, 'ew_idx': 24, 'ns_idx': 24, 'pos_disp_m': [0.0, 0.0, 0.0],
         'coelev_disp_deg': 0.0, 'type': 0, 'label': "D22"}],
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
    "DumpN2HDF5": {
        "kotekan_stage": "hdf5FileWrite",
        "base_dir": "fake_n2_data",
        "file_name": "n2_vis",
        "in_buf": "",
    },
    "DumpN2kCountsHDF5": {
        "kotekan_stage": "hdf5FileWrite",
        "base_dir": "fake_n2_data",
        "file_name": "n2k_counts",
        "in_buf": "",
    },
}

accumulate_params = {
    "num_freq_per_n2k_frame": "num_local_freq",
    "num_n2k_samples_to_accumulate": 8,
    "packet_loss_is_scalar": True,
}


@pytest.fixture(scope="module")
def accumulate_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("accumulate")

    input_buffers = runner.FakeN2KBuffers(
        global_params["samples_per_data_set"],
        global_params["num_local_freq"],
        {
            "correlation_type": "f_times_ee_plus_t",
            "counts_type": "const_scalar",
            "counts_values": count_vals,
            "freq_ids": freq_ids,
            "num_frames": global_params["total_frames"],
        },
        {"type": "const", "values": rfi_vals},
    )

    # Number of accumulated frames expected at the output (per freq).
    expected_frames = global_params["num_local_freq"] * (
        global_params["total_frames"]
        // accumulate_params["num_n2k_samples_to_accumulate"]
    )
    dump_buffer = runner.DumpN2Buffer(str(tmpdir), exit_after_n_files=expected_frames)

    accumulate_run_params = accumulate_params.copy()
    accumulate_run_params["in_counts_buf"] = input_buffers.counts_name
    accumulate_run_params["in_rfimask_buf"] = input_buffers.rfi_name

    global_params["DumpN2HDF5"]["in_buf"] = dump_buffer.name
    global_params["DumpN2kCountsHDF5"]["in_buf"] = input_buffers.counts_name

    test = runner.KotekanStageTester(
        "N2Accumulate",
        accumulate_run_params,
        input_buffers,
        dump_buffer,
        global_params,
    )

    test.run()

    yield dump_buffer.load()


def gen_vis_data(t_idx, f_idx):

    n = global_params["num_elements"]
    n_prod = (n * (n + 1)) // 2

    row, col = np.triu_indices(n)
    pat0 = (col - 1.0j * row).astype(np.complex64)

    seq0 = (
        t_idx
        * global_params["sub_integration_ntime"]
        * accumulate_params["num_n2k_samples_to_accumulate"]
    )

    t_n2k_0 = t_idx * accumulate_params["num_n2k_samples_to_accumulate"]

    vis = np.zeros(n_prod, dtype=np.complex64)
    weights = np.zeros(n_prod, dtype=np.float64)
    total_counts = 0

    diff_vis_sq = np.zeros(n_prod, dtype=np.float64)
    diff_N_sq = np.zeros(n_prod, dtype=np.float64)

    even_vis = np.zeros(n_prod, dtype=np.complex64)
    even_counts = 0

    for t in range(
        t_n2k_0, t_n2k_0 + accumulate_params["num_n2k_samples_to_accumulate"]
    ):

        count_idx = (t * global_params["num_local_freq"] + f_idx) % len(count_vals)

        vis_pat = pat0 * f_idx + (1.0 - 1.0j) * t
        counts = count_vals[count_idx]

        vis += vis_pat * counts
        total_counts += counts

        if (t - t_n2k_0) % 2 == 0:
            even_vis[:] = vis_pat * counts
            even_counts = counts
        else:
            diff_vis = vis_pat * counts - even_vis
            diff_vis_sq += np.absolute(diff_vis) ** 2
            diff_N_sq += (counts - even_counts) ** 2

    if total_counts > 0:
        vis /= total_counts
        print(diff_vis_sq, diff_N_sq * np.absolute(vis) ** 2)
        weights[:] = total_counts ** 2 / (
            diff_vis_sq - diff_N_sq * np.absolute(vis) ** 2
        )
    else:
        vis[:] = 0.0
        weights[:] = 0.0

    return vis, weights, total_counts


def test_structure(accumulate_data):

    n = global_params["num_elements"]
    n_prod = (n * (n + 1)) // 2
    n_ev = 0

    # Check that each samples is the expected shape
    for frame in accumulate_data:
        assert frame.vis.shape == (n_prod,)
        assert frame.weight.shape == (n_prod,)
        assert frame.flags.shape == (n,)
        assert frame.eval.shape == (n_ev,)
        assert frame.evec.shape == (n_ev * n,)
        assert frame.gain.shape == (n,)
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == n_prod
        assert frame.metadata.num_ev == n_ev

    # Check that we have the expected number of samples
    nsamp = global_params["num_local_freq"] * (
        global_params["total_frames"]
        // accumulate_params["num_n2k_samples_to_accumulate"]
    )
    assert len(accumulate_data) == nsamp


def test_metadata(accumulate_data):

    for idx, frame in enumerate(accumulate_data):

        f_idx = idx % len(freq_ids)

        assert (
            frame.metadata.frame_length_fpga_ticks
            == accumulate_params["num_n2k_samples_to_accumulate"]
            * global_params["sub_integration_ntime"]
        )
        assert frame.metadata.freq_id == freq_ids[f_idx]
        assert (
            frame.metadata.freq_Hz
            == freq_ids[f_idx]
            * global_params["sampling_rate_Hz"]
            / global_params["fft_length"]
        )


def test_time(accumulate_data):

    dt_ns = GIGA * global_params["fft_length"] / global_params["sampling_rate_Hz"]

    for idx, frame in enumerate(accumulate_data):

        t_idx = idx // len(freq_ids)

        assert (
            frame.metadata.fpga_start_tick
            == t_idx
            * accumulate_params["num_n2k_samples_to_accumulate"]
            * global_params["sub_integration_ntime"]
        )
        assert (
            frame.metadata.frame_start_time_ns
            == frame.metadata.fpga_start_tick * dt_ns
            + global_params["gps_time"]["frame0_nano"]
        )


def test_EOP(accumulate_data):

    dt_ns = GIGA * global_params["fft_length"] / global_params["sampling_rate_Hz"]
    t0_ns = global_params["gps_time"]["frame0_nano"]

    eopA = global_params["earth_rotation_data"]["earth_orientation_parameter_table"][0]
    eopB = global_params["earth_rotation_data"]["earth_orientation_parameter_table"][1]

    for idx, frame in enumerate(accumulate_data):

        t_idx = idx // len(freq_ids)
        seq_len = (
            accumulate_params["num_n2k_samples_to_accumulate"]
            * global_params["sub_integration_ntime"]
        )
        seq0 = t_idx * seq_len
        seq = seq0 + seq_len // 2

        t_inst_ns = seq * dt_ns + t0_ns

        wA = (eopB["time_inst_ns"] - t_inst_ns) / (
            eopB["time_inst_ns"] - eopA["time_inst_ns"]
        )
        wB = 1.0 - wA

        assert frame.metadata.eop.t_inst == t_inst_ns
        assert np.isclose(
            frame.metadata.eop.delta_UT1_inst,
            wA * eopA["delta_UT1_inst"] + wB * eopB["delta_UT1_inst"],
            atol=0.0,
            rtol=1.0e-12,
        )
        assert np.isclose(
            frame.metadata.eop.xp_as,
            wA * eopA["x_pm"] + wB * eopB["x_pm"],
            atol=0.0,
            rtol=1.0e-12,
        )
        assert np.isclose(
            frame.metadata.eop.yp_as,
            wA * eopA["y_pm"] + wB * eopB["y_pm"],
            atol=0.0,
            rtol=1.0e-12,
        )


def test_accumulate(accumulate_data):

    row, col = np.triu_indices(global_params["num_elements"])

    pat0 = (col - 1.0j * row).astype(np.complex64)

    for idx, frame in enumerate(accumulate_data):

        f = idx % global_params["num_local_freq"]
        t = idx // global_params["num_local_freq"]

        vis_pat, weights, counts = gen_vis_data(t, f)

        assert np.isclose(frame.vis, vis_pat, rtol=1.0e-6).all()
        assert frame.metadata.n_valid_fpga_ticks == counts
        # TODO: The tolerance on the weights here has to be set really high, this is either
        # because the weight calculation in kotekan has a TON of round-off error, or there is
        # something slightly wrong with the test and/or kotekan calculations
        assert np.isclose(frame.weight, weights, rtol=1.0e-2).all()
        assert (frame.flags == 0.0).all()
        assert (frame.gain == -1.0).all()


def test_rfi(accumulate_data):

    num_full_frames_per_accumulation = accumulate_params[
        "num_n2k_samples_to_accumulate"
    ] // (
        global_params["samples_per_data_set"] // global_params["sub_integration_ntime"]
    )

    for idx, frame in enumerate(accumulate_data):

        start_tick = frame.metadata.fpga_start_tick

        dset_idx = start_tick // global_params["samples_per_data_set"]

        n_rfi = 0
        for _ in range(num_full_frames_per_accumulation):
            mask_val = 0 if rfi_vals[dset_idx % len(rfi_vals)] == 0 else 1
            n_rfi += (1 - mask_val) * global_params["samples_per_data_set"]
            dset_idx += 1

        assert frame.metadata.n_rfi_fpga_ticks == n_rfi
