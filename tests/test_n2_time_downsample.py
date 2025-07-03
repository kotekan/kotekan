# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import pytest
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as units

from kotekan import runner

T_rot_sec = 86400 / 1.00273781191135448
n_bins_per_rot = 60  # 21600  # approx 4 seconds per bin
n_samps_per_bin = 10
n_samps_tot = 101

dut1 = 0.0
x_pm = 0.1
y_pm = 0.1

cadence = np.float32(T_rot_sec / (n_samps_per_bin * n_bins_per_rot))

# Set a time a few seconds before ERA=0.0
t_start = Time("2024-07-01 05:00:00", scale="utc")
# Set the DUT1 to our user value
t_start.delta_ut1_utc = dut1
t_end = t_start + (cadence * (n_samps_tot - 1)) * units.s
t_end.delta_ut1_utc = dut1

GIGA = 1_000_000_000

t_start_inst_ns = int(t_start.unix) * GIGA
t_end_inst_ns = int(t_end.unix) * GIGA

era_deg_tol = (360.0 / (1.0e9 * 86400))
eop_ns_tol = 500

fake_params = {
    "num_frames": n_samps_tot,
    "mode": "fill_ij_missing",
    "cadence": float(cadence),
    "start_time": t_start_inst_ns // GIGA,
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
                "time_inst_ns": t_start_inst_ns - 2000 * GIGA,
                "delta_UT1_inst": dut1,
                "x_pm": x_pm,
                "y_pm": y_pm,
            },
            {
                "time_inst_ns": t_end_inst_ns + 2000 * GIGA,
                "delta_UT1_inst": dut1,
                "x_pm": x_pm,
                "y_pm": y_pm,
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
    "gps_time": {"frame0_nano": t_start_inst_ns},
}


def calc_t_at_era(t0, era_deg_target, tol):

    dta = TimeDelta(-12.0*units.hour, scale="ut1")
    dtb = TimeDelta(+12.0*units.hour, scale="ut1")

    t0_ut1 = t0.ut1

    with open('t_search.out', 'w'):
        pass

    while dtb - dta >= tol:

        with open("t_search.out", "a") as f:
            f.write(str(dta) + "   " + str(dtb) + "\n")

        dt = 0.5*(dta + dtb)
        t = t0_ut1 + dt
        dera = t.earth_rotation_angle('tio').to_value('deg') - era_deg_target
        if dera > 180.0:
            dera -= 360.0
        elif dera <= -180.0:
            dera += 360.0

        if dera < 0.0:
            dta = dt
        else:
            dtb = dt

    t = t0_ut1 + 0.5*(dta + dtb)
    t.delta_ut1_utc = dut1

    return t


def jd_to_s_ns(jd):

    s = int(jd * 86400)
    ns = int((jd * 86400 - s) * GIGA)

    return s, ns


def calc_times(t):
    t_inst_s = int(t.utc.unix)
    t_inst_ns = int((t.utc.unix - t_inst_s) * GIGA)
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

    era = t.earth_rotation_angle("tio").to_value("degree")

    return (t_inst_s, t_inst_ns), (t_ut1_s, t_ut1_ns), era


def calc_era_bins():

    t_start_center = t_start + 0.5 * cadence * units.s

    era_start = t_start_center.earth_rotation_angle("tio").to_value("deg")
    delta_era = 360.0 / n_bins_per_rot
    era_idx_start = 1 + int(era_start / delta_era)

    t_first_bin_edge = calc_t_at_era(t_start_center,
                                     era_idx_start * delta_era,
                                     0.1*units.ns)
    t_second_bin_edge = calc_t_at_era(t_start_center,
                                      (era_idx_start + 1) * delta_era,
                                      0.1*units.ns)
    dt = t_second_bin_edge - t_first_bin_edge

    n_bins = int((t_end.ut1 - t_first_bin_edge.ut1) / (dt.ut1)) + 1

    bins_raw = t_first_bin_edge + np.arange(n_bins) * dt

    bins = []
    for t_raw in bins_raw:
        era_ahead = (t_raw.earth_rotation_angle('tio').to_value('deg')
                     + 0.5 * delta_era)
        era_idx = int(era_ahead / delta_era)
        era = era_idx * delta_era
        t = calc_t_at_era(t_raw, era, 0.1*units.ns)
        bins.append(t)

    bins = np.array(bins)

    with open('bins.out', 'w') as f:
        for i, t in enumerate(bins):
            t_inst, t_ut1, era = calc_times(t)
            f.write(
                "{0:d} INST {1:d}s + {2:09d}ns UT1 {3:d}s + {4:09d}ns ERA {5:.17f}\n"
                .format(i, *t_inst, *t_ut1, era))

    return bins


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

    # Check that each samples is the expected shape
    for frame in n2_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == global_params["num_ev"]


    # Check that we have the expected number of samples
    nsamp = len(calc_era_bins()) - 1
    assert len(n2_data) == nsamp


def test_metadata(n2_data):

    # ticks_per_second = int(3.2e9 / 16384)

    input_frame_length = int(3.2e9 / 16384 * fake_params["cadence"])
    frame_length = input_frame_length * n_samps_per_bin
    frame_total = (input_frame_length - 2) * n_samps_per_bin
    rfi_total = n_samps_per_bin

    for frame in n2_data:
        assert frame.metadata.freq_id == 0
        assert frame.metadata.frame_length_fpga_ticks == frame_length
        assert frame.metadata.n_valid_fpga_ticks_in_frame == frame_total
        assert frame.metadata.n_rfi_fpga_ticks == rfi_total


def test_time(n2_data):

    frame_time_ns = np.array([v.metadata.frame_start_time_ns for v in n2_data])

    bins = calc_era_bins()

    t_bin_center = [bins[i] + 0.5 * (bins[i+1] - bins[i])
                    for i in range(len(bins)-1)]

    for i in range(len(t_bin_center)):
        t_bin_center[i].delta_ut1_utc = dut1

    t_bin_cent_tup = [calc_times(t) for t in t_bin_center]
    t_bin_cent_ns = np.array([t[0][0]*GIGA + t[0][1] for t in t_bin_cent_tup])


    # Check downsampled cadence
    assert np.all(frame_time_ns == t_bin_cent_ns)


def test_eop(n2_data):

    eop_t_inst = np.array([(v.metadata.eop.t_inst.tv,
                            v.metadata.eop.t_inst.tv_nsec)
                           for v in n2_data])
    eop_t_ut1 = np.array([(v.metadata.eop.t_ut1.tv,
                           v.metadata.eop.t_ut1.tv_nsec)
                          for v in n2_data])
    eop_dut1 = np.array([v.metadata.eop.delta_UT1_inst for v in n2_data])
    eop_x_pm = np.array([v.metadata.eop.xp_as for v in n2_data])
    eop_y_pm = np.array([v.metadata.eop.yp_as for v in n2_data])
    eop_era = np.array([v.metadata.eop.ERA_deg for v in n2_data])

    bins = calc_era_bins()

    t_bin_center = [bins[i] + 0.5 * (bins[i+1] - bins[i])
                    for i in range(len(bins)-1)]

    for i in range(len(t_bin_center)):
        t_bin_center[i].delta_ut1_utc = dut1

    t_cent = []
    for i in range(len(bins)-1):
        _, _, era_a = calc_times(bins[i])
        _, _, era_b = calc_times(bins[i+1])

        if era_b < era_a:
            era_a -= 360.0
        era = 0.5*(era_a + era_b)
        if era < 0.0:
            era += 360
        elif era >= 360.0:
            era -= 360.0

        t = calc_t_at_era(t_bin_center[i], era, 0.1*units.ns)
        t_cent.append(t)

    t_cent_tup = [calc_times(t) for t in t_cent]

    t_inst_bin = np.array([t[0] for t in t_cent_tup])
    ut1_bin = np.array([t[1] for t in t_cent_tup])
    era_bin = np.array([t[2] for t in t_cent_tup])

    # check EOP
    assert np.all(eop_dut1 == dut1)
    assert np.all(eop_x_pm == x_pm)
    assert np.all(eop_y_pm == y_pm)

    # check seconds
    assert np.all(eop_t_inst[:, 0] == t_inst_bin[:, 0])
    assert np.all(eop_t_ut1[:, 0] == ut1_bin[:, 0])

    with open("era.out", "w") as f:
        for i in range(len(eop_era)):
            f.write("{:04d} ERA_FRAME: {:.17f}\n".format(i, eop_era[i]))
            f.write(  "     ERA_TEST:  {:.17f}\n".format(era_bin[i]))

    # check ERA
    assert np.all(np.fabs(eop_era - era_bin) < era_deg_tol)

    # check nanoseconds
    assert np.all(np.fabs(eop_t_inst[:, 1] - t_inst_bin[:, 1]) < eop_ns_tol)
    assert np.all(np.fabs(eop_t_ut1[:, 1] - ut1_bin[:, 1]) < eop_ns_tol)


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

    with open("test_contents.out", "w"):
        pass

    # Averaging shouldn't change vis, eigenstuff
    for i, frame in enumerate(n2_data):
        with open("test_contents.out", "a") as f:
            f.write("{0:d} {1:d} {2:d}\n".format(i,
                    frame.metadata.fpga_start_tick,
                    frame.metadata.frame_length_fpga_ticks))
        assert np.all(frame.vis == model_vis)
        assert np.all(frame.evec == model_evec)
        assert np.all(frame.eval == model_eval)
        assert frame.erms == 1.0

    # weights get an extra factor of nsamp
    for frame in n2_data:
        assert np.all(frame.weight == n_samps_per_bin)
