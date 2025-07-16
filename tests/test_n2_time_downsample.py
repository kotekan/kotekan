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

T_rot_sec = 86400 # / 1.00273781191135448
n_bins_per_rot = 600  # 21600  # approx 4 seconds per bin
n_samps_per_bin = 10
n_samps_tot = 1001

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
eop_t_ns_tol = 1
eop_ut1_ns_tol = 1

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


def seq_to_t_inst_ns(seq):

    dt_ns = 5120   # 16384/(3.2 GHz)

    return t_start_inst_ns + dt_ns * seq


def calc_raw_frame_times_and_seqs():

    delta_seq = int(3.2e9/16384 * cadence)
    delta_ns = int(cadence * GIGA)

    seq = delta_seq * np.arange(fake_params["num_frames"],
                                dtype=int)

    dseq = np.empty_like(seq)
    dseq[:] = delta_seq

    t_inst_ns = seq_to_t_inst_ns(seq)

    return seq, dseq, t_inst_ns

def calc_t_start_tel(frame0_ns):

    t_start_0h_s = (frame0_ns // (86400 * GIGA)) * 86400
    t_start_diff_ns = frame0_ns - GIGA * t_start_0h_s

    diff_h = t_start_diff_ns // (GIGA * 3600)
    t_start_diff_ns -= GIGA * 3600 * diff_h
    diff_min = t_start_diff_ns // (GIGA * 60)
    t_start_diff_ns -= GIGA * 60 * diff_min
    diff_s = t_start_diff_ns / GIGA
    
    t_start_0h = Time(t_start_0h_s, scale='utc', format='unix')
    t_start_tel = Time({'year': t_start_0h.ymdhms.year,
                        'month': t_start_0h.ymdhms.month,
                        'day': t_start_0h.ymdhms.day,
                        'hour': diff_h,
                        'minute': diff_min,
                        'second': diff_s}, scale='utc', format='ymdhms')

    return t_start_tel


def calc_t_from_t_inst_ns(t_inst_ns):

    frame0_ns = global_params['gps_time']['frame0_nano']

    t_start_tel = calc_t_start_tel(frame0_ns)

    dt_ns = t_inst_ns - frame0_ns

    dt = TimeDelta(dt_ns * units.ns, scale='tai')

    t = t_start_tel + dt
    t.delta_ut1_utc = dut1

    return t


def calc_t_inst_ns_from_t(t):

    frame0_ns = global_params['gps_time']['frame0_nano']
    t_start_tel = calc_t_start_tel(frame0_ns)

    dt = t.tai - t_start_tel.tai

    dt_s, dt_ns = t_jd_to_s_ns(dt)

    return frame0_ns + dt_s*GIGA + dt_ns


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


def t_jd_to_s_ns(t):


    t1_s, t1_ns = jd_to_s_ns(t.jd1)
    t2_s, t2_ns = jd_to_s_ns(t.jd2)

    t_s = t1_s + t2_s
    t_ns = t1_ns + t2_ns

    sec_over = t_ns // GIGA
    t_ns -= GIGA * sec_over
    t_s += sec_over

    return t_s, t_ns


def calc_times(t):
    t_inst_ns = calc_t_inst_ns_from_t(t)
    t_inst_s = t_inst_ns // GIGA
    t_inst_ns -= GIGA * t_inst_s

    t_ut1_s, t_ut1_ns = t_jd_to_s_ns(t.ut1)

    era = t.earth_rotation_angle("tio").to_value("degree")

    return (t_inst_s, t_inst_ns), (t_ut1_s, t_ut1_ns), era


def calc_downsamp_frame_meta():

    seq, dseq, t_ns = calc_raw_frame_times_and_seqs()
    t = calc_t_from_t_inst_ns(t_ns)

    seq_c = seq + dseq // 2
    t_c_ns = seq_to_t_inst_ns(seq_c)
    t_c = calc_t_from_t_inst_ns(t_c_ns)

    era_edges = np.linspace(0.0, 360.0, n_bins_per_rot+1)
    out_frames = []

    def init_frame(in_idx, era_idx):
        return dict(seq_start=seq[in_idx],
                    t_start_ns=t_ns[in_idx],
                    t_start=t[in_idx],
                    t_eop=calc_t_at_era(t[in_idx],
                        0.5*(era_edges[era_idx]+era_edges[era_idx+1]),
                        0.1*units.ns),
                    era_lo=era_edges[era_idx],
                    era_hi=era_edges[era_idx+1],
                    seq_len=0,
                    seq_valid=0,
                    seq_rfi=0,
                    n_frames=0,
                    finalized=False)

    def accum_frame(out_idx, in_idx):
        out_frames[out_idx]["seq_len"] += dseq[in_idx]
        out_frames[out_idx]["n_frames"] += 1
        if fake_params["mode"] == "fill_ij_missing":
            out_frames[out_idx]["seq_valid"] += dseq[in_idx] - 2
            out_frames[out_idx]["seq_rfi"] += 1
        else:
            out_frames[out_idx]["seq_valid"] += dseq[in_idx]
            out_frames[out_idx]["seq_rfi"] += 0

    def get_era_idx(era):
        idx = np.searchsorted(era_edges, era) - 1
        if idx < 0:
            idx += n_bins_per_rot
        if idx >= n_bins_per_rot:
            idx -= n_bins_per_rot
        return idx

    era_idx = None
    waiting = True

    for i in range(len(seq)):
        era = t_c[i].earth_rotation_angle('tio').to_value('deg')

        if era_idx is None:
            era_idx = get_era_idx(era)
            continue

        if era < era_edges[era_idx] or era >= era_edges[era_idx+1]:
            if len(out_frames) > 0:
                out_frames[-1]['finalized'] = True
            era_idx = era_idx + 1 if era_idx < n_bins_per_rot-1 else 0
            out_frames.append(init_frame(i, era_idx))
            accum_frame(-1, i)
            waiting = False
        elif not waiting:
            accum_frame(-1, i)

    if not out_frames[-1]['finalized']:
        out_frames.pop()


    with open("frame_meta.out", "w") as f:
        for frame in out_frames:
            f.write(str(frame) + "\n")

    return out_frames


def calc_era_bin_times(out_frame_metas):

    t = []

    for frame_meta in out_frame_metas:
        t.append(calc_t_at_era(frame_meta['t_start'],
                               frame_meta['era_lo'],
                               0.1*units.ns))

    t.append(calc_t_at_era(out_frame_metas[-1]['t_start'],
                           out_frame_metas[-1]['era_hi'],
                           0.1*units.ns))

    return t


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

    out_frame_metas = calc_downsamp_frame_meta()

    # Check that there are the correct number of samples
    assert len(n2_data) == len(out_frame_metas)

    # Check that each samples is the expected shape
    n = global_params["num_elements"]

    for frame in n2_data:
        assert frame.metadata.num_elements == n
        assert frame.metadata.num_prod == (n * (n + 1) // 2)
        assert frame.metadata.num_ev == global_params["num_ev"]


    # Check that we have the expected number of samples



def test_metadata(n2_data):

    # ticks_per_second = int(3.2e9 / 16384)

    # input_frame_length = int(3.2e9 / 16384 * fake_params["cadence"])
    # frame_length = input_frame_length * n_samps_per_bin
    # frame_total = (input_frame_length - 2) * n_samps_per_bin
    # rfi_total = n_samps_per_bin

    frame_meta = calc_downsamp_frame_meta()

    for i, frame in enumerate(n2_data):
        assert frame.metadata.freq_id == 0
        assert frame.metadata.frame_length_fpga_ticks == frame_meta[i]["seq_len"]
        assert frame.metadata.n_valid_fpga_ticks_in_frame == frame_meta[i]["seq_valid"]
        assert frame.metadata.n_rfi_fpga_ticks == frame_meta[i]["seq_rfi"]


def test_time(n2_data):

    frame_time_ns = np.array([v.metadata.frame_start_time_ns for v in n2_data])
    
    frame_meta = calc_downsamp_frame_meta()

    calculated_time_ns = np.array([meta["t_start_ns"] for meta in frame_meta])

    # Check downsampled cadence
    assert np.all(frame_time_ns == calculated_time_ns)


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

    out_frame_metas = calc_downsamp_frame_meta()

    t_cent_tup = [calc_times(frame['t_eop']) for frame in out_frame_metas]

    t_inst_bin = np.array([t[0] for t in t_cent_tup])
    ut1_bin = np.array([t[1] for t in t_cent_tup])
    era_bin = np.array([t[2] for t in t_cent_tup])

    with open("era.out", "w") as f:
        for i in range(len(eop_era)):
            f.write("{:04d} ERA_FRAME:   {:.17f}\n".format(i, eop_era[i]))
            f.write(  "     ERA_TEST:    {:.17f}\n".format(era_bin[i]))
            f.write(  "     DUT1_FRAME:  {:.17f}\n".format(eop_dut1[i]))
            f.write(  "     DUT1_TEST:   {:.17f}\n".format(dut1))
            f.write(  "     XPM_FRAME:   {:.17f}\n".format(eop_x_pm[i]))
            f.write(  "     XPM_TEST:    {:.17f}\n".format(x_pm))
            f.write(  "     YPM_FRAME:   {:.17f}\n".format(eop_y_pm[i]))
            f.write(  "     YPM_TEST:    {:.17f}\n".format(y_pm))
            f.write(  "     T_DIFF_NS:   {:d}\n".format(eop_t_inst[i, 1]
                                                       - t_inst_bin[i, 1]))
            f.write(  "     UT1_DIFF_NS: {:d}\n".format(eop_t_ut1[i, 1]
                                                       - ut1_bin[i, 1]))

    # check EOP
    assert np.all(np.isclose(eop_dut1, dut1, 1.0e-15, 0.0))
    assert np.all(np.isclose(eop_x_pm, x_pm, 1.0e-15, 0.0))
    assert np.all(np.isclose(eop_y_pm, y_pm, 1.0e-15, 0.0))

    # check seconds
    assert np.all(eop_t_inst[:, 0] == t_inst_bin[:, 0])
    assert np.all(eop_t_ut1[:, 0] == ut1_bin[:, 0])

    # check ERA
    assert np.all(np.fabs(eop_era - era_bin) <= era_deg_tol)

    # check nanoseconds
    assert np.all(np.fabs(eop_t_inst[:, 1] - t_inst_bin[:, 1]) <= eop_t_ns_tol)
    assert np.all(np.fabs(eop_t_ut1[:, 1] - ut1_bin[:, 1]) <= eop_ut1_ns_tol)


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
