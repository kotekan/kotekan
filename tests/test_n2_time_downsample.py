# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as units

from kotekan import runner

T_rot_sec = 86400 / 1.00273781191135448
n_bins_per_rot = 360  # 21600  # approx 4 seconds per bin
n_samps_per_bin = 4
n_samps_tot = 101

cadence = T_rot_sec / (n_samps_per_bin * n_bins_per_rot)

dut1 = 0.0

# Set a time a few seconds before ERA=0.0
t_start = Time("2024-07-01 05:00:00", scale='utc')
# Set the DUT1 to our user value
t_start.delta_ut1_utc = dut1
t_end = t_start + (cadence*(n_samps_tot-1)) * units.s
t_end.delta_ut1_utc = dut1

GIGA = 1_000_000_000

t_start_inst_ns = int(t_start.unix) * GIGA
t_end_inst_ns = int(t_end.unix) * GIGA

fake_params = {
    "num_frames": n_samps_tot,
    "mode": "fill_ij_missing",
    "cadence": cadence,
    "start_time": t_start_inst_ns//GIGA,
}

downsamp_params = {
    "num_bins_per_rotation": n_bins_per_rot,
    "max_age": 2 * T_rot_sec / n_bins_per_rot,
}

global_params = {
    "num_elements": 4,
    "num_ev": 4,
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "earth_orientation_parameter_table": [
            {
                "time_inst_ns": t_start_inst_ns - 2000*GIGA,
                "delta_UT1_inst": dut1,
                "x_pm": 0.1,
                "y_pm": 0.1,
            },
            {
                "time_inst_ns": t_end_inst_ns + 2000*GIGA,
                "delta_UT1_inst": dut1,
                "x_pm": 0.12,
                "y_pm": 0.12,
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
        "dish_positions": [[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]],
        "updatable_config": "/earth_rotation_data",
    },
    "gps_time": {"frame0_nano": t_start_inst_ns},
}


def jd_to_s_ns(jd):

    s = int(jd * 86400)
    ns = int((jd*86400 - s) * GIGA)

    return s, ns


def calc_times(t):
    t_inst_s = int(t.unix)
    t_inst_ns = int((t.unix - t_inst_s) * GIGA)
    t_ut1_jd1_s, t_ut1_jd1_ns = jd_to_s_ns(t.ut1.jd1)
    t_ut1_jd2_s, t_ut1_jd2_ns = jd_to_s_ns(t.ut1.jd2)

    t_ut1_s = t_ut1_jd1_s + t_ut1_jd2_s
    t_ut1_ns = t_ut1_jd1_ns + t_ut1_jd2_ns
    while t_ut1_ns > GIGA:
        t_ut1_s += 1
        t_ut1_ns -= GIGA
    while t_ut1_ns < 0:
        t_ut1_s -= 1
        t_ut1_ns += GIGA

    era = t.earth_rotation_angle('tio').to_value('degree')

    return (t_inst_s, t_inst_ns), (t_ut1_s, t_ut1_ns), era


def calc_era_bins():

    t_start_center = (t_start + 0.5 * cadence * units.s)
    t_start_center.delta_ut1_utc = dut1
    t_end_center = (t_end + 0.5 * cadence * units.s)
    t_end_center.delta_ut1_utc = dut1

    era_start = t_start_center.earth_rotation_angle('tio').to_value('deg')
    era_end = t_end_center.earth_rotation_angle('tio').to_value('deg')

    n_rot = int((t_end_center - t_start_center).tai.to_value('s') / T_rot_sec)

    if era_end < era_start:
        era_end += 360.0 * (n_rot+1)
    else:
        era_end += 360.0 * n_rot

    delta_era = 360.0 / n_bins_per_rot

    # integration begins on a full bin.

    bin_idx_start = int(era_start / delta_era) + 1
    bin_idx_end = int(era_end / delta_era)

    bin_indices = np.arange(bin_idx_start, bin_idx_end+1)
    bin_edges_raw = delta_era * bin_indices

    bin_edges = bin_edges_raw.copy()

    return bin_edges


@pytest.fixture(scope="module")
def n2_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("n2_data")

    dump_buffer = runner.DumpN2Buffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "N2TimeDownsample",
        downsamp_params,
        runner.FakeVisBuffer(**fake_params),
        dump_buffer,
        global_params,
    )

    test.run()

    yield dump_buffer.load()


def test_structure(n2_data):

    n = global_params["num_elements"]

    # eras = []

    # Check that each samples is the expected shape
    for frame in n2_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == global_params["num_ev"]
        # eras.append(frame.metadata.eop.ERA_deg)

    """
    with open("test_structure.out", "w") as f:
        eras = np.array(eras)
        line = "N2 ERA:   " + " ".join(["{0:f}".format(x) for x in eras])
        f.write(line + "\n")

        e_e = calc_era_bins()
        e = 0.5*(e_e[1:] + e_e[:-1])

        while (e >= 360.0).any():
            e[e >= 360.0] -= 360.0
        while (e < 0.0).any():
            e[e < 0.0] += 360.0

        line = "BIN CENT: " + " ".join(["{0:f}".format(x) for x in e])
        f.write(line + "\n")
        line = "BIN EDGE: " + " ".join(["{0:f}".format(x) for x in e_e])
        f.write(line + "\n")

        t_start_center = (t_start + 0.5 * cadence * units.s)
        t_start_center.delta_ut1_utc = dut1

        for i in range(10):
            t_center = t_start_center + i * cadence * units.s
            t_center.delta_ut1_utc = dut1

            t_inst, t_ut1, t_era = calc_times(t_center)
            f.write("{0:d}\n".format(i) + str(t_center) + "\n")
            f.write("t:   {0:d}s + {1:d}ns\n".format(*t_inst))
            f.write("ut1: {0:d}s + {1:d}ns\n".format(*t_ut1))
            f.write("era: {0:f}deg\n".format(t_era))
    """

    # Check that we have the expected number of samples
    nsamp = len(calc_era_bins()) - 1
    assert len(n2_data) == nsamp


def test_metadata(n2_data):

    input_frame_length = int(800e6 / 2048 * fake_params["cadence"])
    frame_length = input_frame_length * n_samps_per_bin
    frame_total = (input_frame_length - 2) * n_samps_per_bin
    rfi_total = n_samps_per_bin

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
        np.diff(time_s) == fake_params["cadence"] * n_samps_per_bin
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
        assert np.all(frame.weight == n_samps_per_bin)
