from astropy.time import Time
from mpmath import mp
import math
import sys


# 40 digits of decimal precision
mp.dps = 40

# ERA = (A + B * t_jd)
era_A = mp.mpf(  779_057_273_264_000_000) / mp.mpf(1e18)
era_B = mp.mpf(1_002_737_811_911_354_480) / mp.mpf(1e18)


def get_unix_ns(t):

    ymdhms = t.utc.ymdhms
    t0h = Time(
        {
            "year": ymdhms.year,
            "month": ymdhms.month,
            "day": ymdhms.day,
            "hour": 0,
            "minute": 0,
            "second": 0,
        },
        scale="utc",
    )
    ns_from_0 = round((t - t0h).tai.to_value("ns"))

    return int(t0h.unix) * 1_000_000_000 + ns_from_0


def get_s_ns_from_jd(t):

    t_s1 = t.jd1 * 86400.0
    t_s2 = t.jd2 * 86400.0

    s = int(t_s1 + t_s2)
    ns = round(((t_s1 - s) + t_s2) * 1_000_000_000)

    sec_over = ns // 1_000_000_000
    s += sec_over
    ns -= 1_000_000_000 * sec_over

    return s, ns


def get_era_nrot(t):

    era_deg = t.earth_rotation_angle("tio").to_value("deg")

    dt_jd = (t.ut1.jd1 - 2451545.0) + t.ut1.jd2

    nrot = math.floor(0.7790572732640 + dt_jd * 1.00273781191135448)

    return era_deg, nrot


def get_era_nrot_basic(t):

    jd = t.ut1.jd

    mjd = jd - 2451545.0

    nrot = math.floor(0.7790572732640 + mjd * 1.00273781191135448)

    era_deg = 360.0 * ((0.7790572732640 + 1.00273781191135448 * mjd) % 1.0)

    if era_deg < 0.0:
        era_deg += 360.0
        nrot -= 1

    return era_deg, nrot


def get_era_nrot_fancy(t):

    jd1 = t.ut1.jd1
    jd2 = t.ut1.jd2

    if jd1 > jd2:
        jd1 -= 2451545.0
    else:
        jd2 -= 2451545.0

    jd1f, jd1n = math.modf(jd1)
    jd2f, jd2n = math.modf(jd2)

    jd1n_sdf, jd1n_sdn = math.modf(jd1n * 2.73781191135448e-3)
    jd2n_sdf, jd2n_sdn = math.modf(jd2n * 2.73781191135448e-3)

    arg_rot = (
        0.7790572732640
        + jd1f
        + jd2f
        + (jd1f + jd2f) * 2.73781191135448e-3
        + jd1n_sdf
        + jd2n_sdf
    )

    f_rot, fn_rot = math.modf(arg_rot)

    n_rot = (
        math.floor(jd1n)
        + math.floor(jd2n)
        + math.floor(jd1n_sdn)
        + math.floor(jd2n_sdn)
        + math.floor(fn_rot)
    )

    if f_rot < 0.0:
        f_rot += 1.0
        n_rot -= 1

    era_deg = 360.0 * f_rot

    return era_deg, n_rot


def get_era_nrot_precise(t):

    ut1_jd = (mp.mpf(t.ut1.jd1) - mp.mpf(2451545)) + mp.mpf(t.ut1.jd2)

    rot = era_A + era_B * ut1_jd

    nrot = int(mp.floor(rot))
    era_deg = float(360 * mp.frac(rot))

    return era_deg, nrot


def print_isot_times(t_str):

    t = Time(t_str, scale="utc", format="isot")
    print(str(t))
    unix_tot_ns = get_unix_ns(t)
    unix_s = unix_tot_ns // 1_000_000_000
    unix_ns = unix_tot_ns - 1_000_000_000 * unix_s
    print("UNIX {0:d} s {1:d} ns".format(unix_s, unix_ns))

    tai_s, tai_ns = get_s_ns_from_jd(t.tai)
    print("TAI {0:d} s {1:d} ns".format(tai_s, tai_ns))

    print("dUT1 {0:.17e} s".format(t.delta_ut1_utc))

    ut1_s, ut1_ns = get_s_ns_from_jd(t.ut1)
    ut1_s -= 2451545 * 86400
    ut1_ns += ut1_s * 1_000_000_000
    print(
        "UT1 {0:d} ns {1:.17e} jd2000".format(
            ut1_ns, (t.ut1.jd1 - 2451545) + t.ut1.jd2
        )
    )

    era_deg, nrot = get_era_nrot(t)
    mjd = ((era_deg / 360.0 + nrot) - 0.7790572732640) / 1.00273781191135448
    print("ERA {0:.17e} deg {1:d} rot {2:.17e} jd2000".format(era_deg, nrot, mjd))
    era_deg, nrot = get_era_nrot_basic(t)
    mjd = ((era_deg / 360.0 + nrot) - 0.7790572732640) / 1.00273781191135448
    print("ERA2 {0:.17e} deg {1:d} rot {2:.17e} jd2000".format(era_deg, nrot, mjd))
    era_deg, nrot = get_era_nrot_fancy(t)
    mjd = ((era_deg / 360.0 + nrot) - 0.7790572732640) / 1.00273781191135448
    print("ERA3 {0:.17e} deg {1:d} rot {2:.17e} jd2000".format(era_deg, nrot, mjd))
    era_deg, nrot = get_era_nrot_precise(t)
    mjd = ((era_deg / 360.0 + nrot) - 0.7790572732640) / 1.00273781191135448
    print("ERA4 {0:.17e} deg {1:d} rot {2:.17e} jd2000".format(era_deg, nrot, mjd))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit()

    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode == "isot":
        for t_str in args:
            print_isot_times(t_str)
