# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
import astropy.constants as const
import astropy.units as units
import astropy.time
import erfa

# === End Python 2/3 compatibility

import pytest
import numpy as np

from kotekan import runner

obs_start_time = astropy.time.Time("2024-12-11 00:00:00", scale="utc")

t_gps_0 = astropy.time.Time("1980-01-06 00:00:00", scale="utc")


def calc_dtai(t):

    tdate = t.ymdhms
    return erfa.dat(tdate.year, tdate.month, tdate.day, 0.0)


dut1 = float(obs_start_time.delta_ut1_utc)
dtai = float(calc_dtai(obs_start_time))

t0_nanosec = int((obs_start_time - t_gps_0).to_value("ns"))
t0_sec = int((obs_start_time - t_gps_0).to_value("s"))


downsamp_params = {
    "num_samples": 2,
    "max_age": 1000000.0,
}

fakevis_params = {
    "freq_ids": [6000],
    "wait": False,
    "num_frames": 41,
    "mode": "point_source",
    "cadence": 4.21,
    "start_time": t0_sec,
    "sleep_after": 1.0,
    "ra": -118,
    "dec": 48,
    "stokes_I": 10.0,
    "stokes_Q": 0.0,
    "stokes_U": 0.0,
    "stokes_V": 0.0,
    "noise_var": 0.1,
    "n_rfi_ticks": 0,
    "n_lost_ticks": 0,
}

global_params = {
    "num_elements": 4,
    "num_ev": 4,
    "earth_rotation_data": {
        "kotekan_update_endpoint": "json",
        "DUT1": dut1,
        "DTAI": dtai,
    },
    "telescope": {
        "name": "CHORDTelescope",
        "require_gps": False,
        "inst_long_deg": -119.62081125,
        "inst_lat_deg": 49.32075144444,
        "inst_alt_deg": 90.0,
        "inst_orientation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "dish_positions": [
            [0.0, 0.0, 0.0],
            # [1.0, 0.0, 0.0],
            # [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [10.0, 10.0, 0.0],
            [100.0, 100.0, 0.0],
        ],
        "updatable_config": "/earth_rotation_data",
    },
    "gps_time": {"frame0_nano": t0_nanosec},
}


@pytest.fixture(scope="module")
def n2_data(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("n2_data")

    dump_buffer = runner.DumpN2Buffer(str(tmpdir))

    test = runner.KotekanStageTester(
        "N2TimeDownsample",
        downsamp_params,
        runner.FakeVisBuffer(**fakevis_params),
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
    nsamp = (fakevis_params["num_frames"] - 1) // downsamp_params["num_samples"]
    assert len(n2_data) == nsamp


def test_metadata(n2_data):

    input_frame_length = int(3.2e9 / 16384 * fakevis_params["cadence"])
    frame_length = input_frame_length * downsamp_params["num_samples"]
    frame_total = (
        input_frame_length
        - fakevis_params["n_rfi_ticks"]
        - fakevis_params["n_lost_ticks"]
    ) * downsamp_params["num_samples"]
    rfi_total = fakevis_params["n_rfi_ticks"] * downsamp_params["num_samples"]

    freq_id = fakevis_params["freq_ids"][0]
    freq_Hz = freq_id * 1.6e9 / 8192

    for frame in n2_data:
        assert frame.metadata.freq_id == fakevis_params["freq_ids"][0]
        assert frame.metadata.frame_length_fpga_ticks == frame_length
        assert frame.metadata.n_valid_fpga_ticks_in_frame == frame_total
        assert frame.metadata.n_rfi_fpga_ticks == rfi_total
        assert frame.metadata.freq_Hz == freq_Hz


def test_time(n2_data):

    time_ns = []
    era_deg = []

    for v in n2_data:
        time_ns.append(v.metadata.frame_start_time_ns)
        era_deg.append(v.metadata.era_deg)

    time_ns = np.array(time_ns)
    era_deg = np.array(era_deg)

    time_s = time_ns // 1000000000
    # print("time check", time_ns, time_s)

    dt_downsamp = fakevis_params["cadence"] * downsamp_params["num_samples"]
    n_downsamp = (fakevis_params["num_frames"] - 1) // downsamp_params["num_samples"]

    t_start = obs_start_time + np.arange(n_downsamp) * dt_downsamp * units.second
    t_mid = obs_start_time + (np.arange(n_downsamp) + 0.5) * dt_downsamp * units.second

    era_true = t_mid.earth_rotation_angle("tio").degree

    # check_era = isclose_sym(era_deg, era_true, atol=1.0e-6)
    # print("BAD era:", era_deg[~check_era], era_true[~check_era])

    # assert np.all(time_ns == (t_start - t_gps_0).to_value('ns').astype(int))
    assert (
        np.fabs(time_ns - (t_start - t_gps_0).to_value("ns").astype(int)) < 5121
    ).all()
    assert np.all(isclose_sym(era_deg, era_true, atol=1.0e-6))


def test_vis(n2_data):

    n = global_params["num_elements"]

    # Reproduce expected fakeVis output

    """
    model_vis = np.zeros(n * (n + 1) // 2, dtype=np.complex64)
    ind = 0
    for i in range(n):
        for j in range(i, n):
            model_vis[ind] = i + j * 1j
            ind += 1
    """

    # Averaging shouldn't change vis

    with open("vis_out.txt", "w") as f:
        pass

    for k, frame in enumerate(n2_data):

        model_vis = calc_vis_pointsource(
            frame.metadata.frame_start_time_ns
            + int(0.5 * fakevis_params["cadence"] * 1e9)
            * downsamp_params["num_samples"],
            frame.metadata.freq_id * 1.6e9 / 8192,
        )

        with open("vis_out.txt", "a") as f:
            f.write("{} frame vis:\n".format(k))
            idx = 0
            for i in range(n):
                line_vals = ["-------"] * i + [
                    "{}".format(x) for x in frame.vis[idx : idx + n - i]
                ]
                idx += n - i
                f.write(" ".join(line_vals) + "\n")
            f.write("{} model vis:\n".format(k))
            idx = 0
            for i in range(n):
                line_vals = ["-------"] * i + [
                    "{}".format(x) for x in model_vis[idx : idx + n - i]
                ]
                idx += n - i
                f.write(" ".join(line_vals) + "\n")

        assert np.all(isclose_sym(frame.vis, model_vis, atol=0.0, rtol=1.0e-5))
        assert frame.erms == 1.0

        # weights get an extra factor of nsamp
        assert np.all(frame.weight == downsamp_params["num_samples"])


def test_evec(n2_data):

    n = global_params["num_elements"]
    n_ev = global_params["num_ev"]

    # Averaging shouldn't change eigenstuff

    frame_idx = 0

    with open("ev_out.txt", "w") as f:
        pass

    for frame in n2_data:
        model_vis = calc_vis_pointsource(
            frame.metadata.frame_start_time_ns
            + int(0.5 * fakevis_params["cadence"] * 1e9)
            * downsamp_params["num_samples"],
            frame.metadata.freq_id * 1.6e9 / 8192,
        )

        model_eval, model_evec = calc_eig_from_vis(model_vis)

        """
        vis_square = np.zeros((num_elements, num_elements), dtype=np.complex64)
        idx = 0
        for i in range(num_elements):
            for j in range(i, num_elements):
                vis_square[i, j] = frame.vis[idx]
                idx += 1
        for i in range(num_elements):
            for j in range(i):
                vis_square[i, j] = vis_square[j, i].conj()

        b = np.empty((num_ev, num_elements), dtype=np.complex64)

        res = np.empty(num_ev)

        for k in range(num_ev):
            b[k][:] = 0.0
            for i in range(num_elements):
                for j in range(num_elements):
                    b[k, i] += vis_square[i, j] * frame.evec[k*num_elements+j]

            evec_k = frame.evec[k*num_elements:(k+1)*num_elements]
            res = np.abs(b[k] - evec_k * frame.eval[k]).sum()

            with open("ev_out.txt", "a") as f:
                f.write("{} {} {} {} {}\n".format(frame_idx, k, res,
                                            np.sqrt((np.abs(evec_k)**2).sum()),
                                            frame.eval[k]))
        """
        with open("ev_out.txt", "a") as f:
            for k in range(n_ev):
                f.write(
                    "{} {} model-ev {} frame-ev {}\n".format(
                        frame_idx, k, model_eval[k], frame.eval[k]
                    )
                )

        assert np.all(isclose_sym(frame.eval, model_eval, atol=2.0e-5, rtol=1.0e-6))
        # assert np.all(np.isclose(frame.evec, model_evec.flat,
        #                          atol=1.0e-4, rtol=1.0e-4))
        frame_idx += 1


def calc_eig_from_vis(vis):

    num_el = global_params["num_elements"]
    vis_square = np.empty((num_el, num_el), dtype=vis.dtype)

    idx = 0
    for i in range(num_el):
        for j in range(i, num_el):
            vis_square[i, j] = vis[idx]
            vis_square[j, i] = vis[idx].conj()
            idx += 1

    eigval, eigvec = np.linalg.eigh(vis_square)

    num_ev = global_params["num_ev"]

    return eigval[-num_ev:][::-1], eigvec[:, -num_ev:][:, ::-1].T


def calc_vis_pointsource(time_ns, freq_Hz):

    idx = 0

    num_el = global_params["num_elements"]

    tel_params = global_params["telescope"]

    num_dishes = len(tel_params["dish_positions"])

    # print("Modeling vis at:", time_ns, "ns")

    n_source = calc_n_source(time_ns)

    # print("n_source:", n_source)

    lambda_m = const.c.to_value("m/s") / freq_Hz

    vis = np.empty((num_el * (num_el + 1)) // 2, dtype=np.complex64)

    for el_i in range(num_el):
        for el_j in range(el_i, num_el):
            dish_i = el_i % num_dishes
            dish_j = el_j % num_dishes
            pol_i = el_i // num_dishes
            pol_j = el_j // num_dishes

            x_i = np.array(tel_params["dish_positions"][dish_i])
            x_j = np.array(tel_params["dish_positions"][dish_j])
            u_ij = (x_i - x_j) / lambda_m

            phase = 2 * np.pi * (u_ij * n_source).sum()

            power = 0.0
            if pol_i == 0 and pol_j == 0:
                power = fakevis_params["stokes_I"] + fakevis_params["stokes_Q"]
            elif pol_i == 1 and pol_j == 1:
                power = fakevis_params["stokes_I"] - fakevis_params["stokes_Q"]
            elif pol_i == 0 and pol_j == 1:
                power = fakevis_params["stokes_U"] + 1j * fakevis_params["stokes_V"]
            elif pol_i == 1 and pol_j == 0:
                power = fakevis_params["stokes_U"] - 1j * fakevis_params["stokes_V"]

            vis[idx] = np.exp(1j * phase) * power
            idx += 1

    return vis


def calc_n_source(time_ns):

    t = time_ns * units.ns + t_gps_0

    tel_params = global_params["telescope"]

    era = t.earth_rotation_angle("tio").degree

    # print("ERA:", era, "deg")

    deg2rad = np.pi / 180

    phi = deg2rad * (fakevis_params["ra"] - era)
    theta = deg2rad * (90 - fakevis_params["dec"])

    n_geocen = np.array(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    )

    # print("n_geocen", n_geocen)

    clon = np.cos(deg2rad * tel_params["inst_long_deg"])
    slon = np.sin(deg2rad * tel_params["inst_long_deg"])
    clat = np.cos(deg2rad * (90 - tel_params["inst_lat_deg"]))
    slat = np.sin(deg2rad * (90 - tel_params["inst_lat_deg"]))

    R_geocen_to_topocen = np.array(
        [
            [-slon, clon, 0.0],
            [-clon * clat, -slon * clat, slat],
            [clon * slat, slon * slat, clat],
        ]
    )

    # print("R:", R_geocen_to_topocen[0])
    # print("  ", R_geocen_to_topocen[1])
    # print("  ", R_geocen_to_topocen[2])

    n_topocen = (R_geocen_to_topocen[:, :] * n_geocen[None, :]).sum(axis=1)

    return n_topocen


def isclose_sym(a, b, atol=0.0, rtol=1.0e-6):
    return np.isclose(a, b, atol=atol, rtol=rtol) | np.isclose(
        b, a, atol=atol, rtol=rtol
    )
