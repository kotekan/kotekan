import ctypes
import time
import numpy as np
import astropy.units as units
import astropy.utils.iers
from astropy.time import Time, TimeDelta
from mpmath import mp

"""
Telescope and time utilities for Kotekan. This module provides similar implementations of Telescope and timeUtil appropriate for testing.
"""

# Set precision for arbitrary-precision floats.
mp.dps = 40

# Eq 5.14 of IERS Conventions (2010) Ch. 5. defines these values
# for converting from UT1 to ERA
ERA_A = mp.mpf(779_057_273_264_000_000) / mp.mpf(1e18)
ERA_B = mp.mpf(1_002_737_811_911_354_480) / mp.mpf(1e18)


class EOP(ctypes.Structure):

    _fields_ = [
        ("t_inst_ns", ctypes.c_int64),
        ("t_ut1_ns", ctypes.c_int64),
        ("delta_UT1_inst", ctypes.c_double),
        ("ERA_deg", ctypes.c_double),
        ("xp_as", ctypes.c_double),
        ("yp_as", ctypes.c_double),
    ]

    def __repr__(self):
        return "{{t_inst_ns: {:d}, t_ut1_ns: {:d}, delta_UT1_inst: {:f}, ERA_deg: {:f}, xp_as: {:f}, yp_as: {:f}}}".format(
            self.t_inst_ns,
            self.t_ut1_ns,
            self.delta_UT1_inst,
            self.ERA_deg,
            self.xp_as,
            self.yp_as,
        )


def get_sampling_params(tel_config):
    sampling_rate_MHz = (
        tel_config["sampling_rate_MHz"]
        if "sampling_rate_MHz" in tel_config
        else tel_config["sampling_rate"]
    )
    fft_length = tel_config["fft_length"]
    ny_zone = tel_config["nyquist_zone"]

    return sampling_rate_MHz, fft_length, ny_zone


def get_freq_MHz(freq_id, tel_config):

    sampling_rate_MHz, fft_length, ny_zone = get_sampling_params(tel_config)

    freq0_MHz = (ny_zone // 2) * sampling_rate_MHz
    df_MHz = sampling_rate_MHz / fft_length
    if ny_zone % 2 == 0:
        df_MHz *= -1.0

    return freq0_MHz + freq_id * df_MHz


def get_unix_time_ns(time_str, scale="utc"):

    if time_str == "now":
        t = Time.now()
    else:
        t = Time(time_str, scale=scale)

    return int(t.unix * 1e9)


def get_t_inst_ns(seq, tel_config):

    sampling_rate_MHz, fft_length, ny_zone = get_sampling_params(tel_config)
    dseq_ns = int((fft_length / sampling_rate_MHz) * 1e3)

    return tel_config["frame0_nano"] + seq * dseq_ns


def get_EOP_table(t0_ns, ndays):

    t0 = calc_astropy_time_from_unix_ns(t0_ns)
    t = Time(int(t0.mjd) + np.arange(-ndays + 1, ndays + 1), format="mjd", scale="utc")

    iers = astropy.utils.iers.IERS_Auto.open()
    eop_list = build_EOP_table(t, t0_ns, iers)
    iers.close()

    return eop_list


def get_EOP_at_t_inst_ns(t_ns, tel_config, use_eop):

    t_ns = np.atleast_1d(t_ns)

    t = calc_astropy_time_from_inst_ns(t_ns, tel_config["frame0_nano"])
    if not use_eop:
        t.delta_ut1_utc = 0.0

    n = len(t_ns)

    eops = []

    iers = astropy.utils.iers.IERS_Auto.open()
    x, y, status = iers.pm_xy(t, None, True)
    iers.close()

    for i in range(n):
        eop = EOP()
        eop.t_inst_ns = t_ns[i]
        eop.t_ut1_ns = calc_ut1_ns_from_t(t[i])
        eop.delta_UT1_inst = t[i].delta_ut1_utc
        eop.ERA_deg = t[i].earth_rotation_angle("tio").to_value("deg")
        if use_eop:
            eop.xp_as = x[i].to_value("arcsecond")
            eop.yp_as = y[i].to_value("arcsecond")
        else:
            eop.xp_as = 0.0
            eop.yp_as = 0.0
        eops.append(eop)

    if n == 1:
        return eops[0]
    return eops


def get_ut1_ns_at_t_inst_ns(ut1_ns, tel_config, use_eop):

    t = calc_astropy_time_from_inst_ns(ut1_ns, tel_config["frame0_nano"])
    if not use_eop:
        t.delta_ut1_utc = 0.0

    return calc_ut1_ns_from_t(t)


def get_ERA_at_t_inst_ns(t_ns, tel_config, use_eop):

    t = calc_astropy_time_from_inst_ns(t_ns, tel_config["frame0_nano"])
    if not use_eop:
        t.delta_ut1_utc = 0.0

    return get_ERA_at_t(t)


def get_ERA_at_ut1_ns(ut1_ns):

    t = calc_astropy_time_from_ut1_ns(ut1_ns)

    return get_ERA_at_t(t)


def get_ERA_at_t(t):

    return t.earth_rotation_angle("tio").to_value("deg")


def get_nrot_at_t_inst_ns(t_ns, tel_config, use_eop):

    t = calc_astropy_time_from_inst_ns(t_ns, tel_config["frame0_nano"])
    if not use_eop:
        t.delta_ut1_utc = 0.0

    return get_nrot_at_t(t)


def get_nrot_at_ut1_ns(ut1_ns):

    t = calc_astropy_time_from_ut1_ns(ut1_ns)

    return get_nrot_at_t(t)


def get_nrot_at_t(t):

    # ERA_A and ERA_B are greater precision than can fit in a float64,
    # so we do this calculation with large precision floats, switching to a
    # normal integer at the end.

    # Kotekan measures UT1 from 2 451 545 JD UT1, the same point
    # as in the definition of ERA
    dt_jd = (t.ut1.jd1 - mp.mpf(2451545.0)) + t.ut1.jd2

    if np.ndim(t) == 0:
        return int(mp.floor(ERA_A + dt_jd * ERA_B))

    nrot = np.empty(t.shape, dtype=int)
    for i in range(len(nrot.flat)):
        nrot.flat[i] = int(mp.floor(ERA_A + dt_jd[i] * ERA_B))

    return nrot


def get_ut1_ns_at_ERA_nrot(era_deg_in, nrot):

    # ERA_A and ERA_B are greater precision than can fit in a float64,
    # so we do this calculation with large precision floats,

    era_deg = np.atleast_1d(era_deg_in)

    frac_rot = era_deg / mp.mpf(360.0)

    # Invert the definition of ERA to get the elapsed UT1 JD since 2 451 545.
    dt_jd_mp = (nrot + frac_rot - ERA_A) / ERA_B

    # We need to turn this into an astropy time, using both parts to maintain precision.
    # the larger part (jd1) gets the integer number of JD, and the smaller part (jd2) gets
    # the remainder.
    dt1_jd = np.empty_like(era_deg)
    dt2_jd = np.empty_like(era_deg)
    for i in range(len(dt_jd_mp.flat)):
        dt1_jd.flat[i] = float(mp.floor(dt_jd_mp.flat[i]))
        dt2_jd.flat[i] = float(dt_jd_mp[i] - dt1_jd.flat[i])

    if era_deg_in.ndim == 0:
        dt1_jd = float(dt1_jd[0])
        dt2_jd = float(dt2_jd[0])

    # Now we can make the astropy time delta, and get the nanosecond count.
    dt = TimeDelta(dt1_jd, dt2_jd, format="jd", scale="ut1")

    return calc_ut1_ns_from_dt(dt)


def get_t_inst_ns_at_ut1_ns(ut1_ns, tel_config, use_eop):

    t0 = calc_astropy_time_from_unix_ns(tel_config["frame0_nano"])

    t = calc_astropy_time_from_ut1_ns(ut1_ns)
    if not use_eop:
        t.delta_ut1_utc = 0.0

    dt = t.tai - t0.tai

    dt_ns = calc_tai_ns_from_dt(dt)

    return tel_config["frame0_nano"] + dt_ns


def get_t_inst_ns_at_ERA_nrot(era_deg, nrot, tel_config, use_eop):

    ut1_ns = get_ut1_ns_at_ERA_nrot(era_deg, nrot)

    return get_t_inst_ns_at_ut1_ns(ut1_ns, tel_config, use_eop)


def get_local_ERA_at_t_inst_ns(t_ns, tel_config, use_eop):

    t_ns = np.atleast_1d(t_ns)

    t = calc_astropy_time_from_inst_ns(t_ns, tel_config["frame0_nano"])
    if not use_eop:
        t.delta_ut1_utc = 0.0

    lon_deg = (
        tel_config["origin_itrs_lon_deg"]
        if "origin_itrs_lon_deg" in tel_config
        else tel_config["inst_long"]
    )

    return t.earth_rotation_angle(lon_deg * units.deg).to_value("deg")


def calc_delta_tai_utc(t):
    r"""
    Calculate the difference TAI - UTC in seconds at time t. This is the number
    of leap seconds at time t.

    Since astropy's internal representation follows the SOFA standard, its
    representation for UTC during a day which contains a leap second is
    non-uniform. So we first compute t in UTC, then break it into a part
    containing whole days (which will have a uniform representation) and the
    remainder, for which we compute the number of seconds manually. Given
    these we can compute the difference in timestamp between TAI and UTC at t.

    Parameters
    ----------
    t : astropy Time object
        The time at which to calculate TAI - UTC

    Returns
    -------
    delta_tai_utc : float
        Value of TAI-UTC in seconds, rounded to nearest 0.1 ns.
    """

    # Get a representation of t in UTC with years, months, days, etc.
    t_utc = t.utc.ymdhms

    # Form a time object for 0h UTC on the beginning of the given day.
    # This will have a numerical representation (in JD) that can be differenced
    # with the TAI represenatation.
    t_utc_d = Time(
        {
            "year": t_utc.year,
            "month": t_utc.month,
            "day": t_utc.day,
            "hour": 0,
            "minute": 0,
            "second": 0,
        },
        scale="utc",
        precision=9,
    )

    # Compute the remaining time from 0h to the given t, in seconds.
    t_utc_s = 3600 * t_utc.hour + 60 * t_utc.minute + t_utc.second

    # Compute the difference (in seconds) for each part of the time
    # representation.  jd1 is typically the larger value, and has whole days.
    dt1 = 86400 * (t.tai.jd1 - t_utc_d.jd1)
    dt2 = 86400 * (t.tai.jd2 - t_utc_d.jd2) - t_utc_s

    # Due to floating point precision we may have accumulated a few picoseconds
    # of error. In the modern era this dt will always be whole number of
    # seconds, so round the total dt to nearest 0.1 ns.
    dt = round(dt1 + dt2, ndigits=10)

    return dt


def calc_astropy_time_from_unix_ns(t_unix_ns):
    r"""
    Constuct an astropy Time object corresponding to a UNIX timestamp in
    nanoseconds.

    Parameters
    ----------
    t_unix_ns : int
        A UNIX timestamp in nanoseconds.

    Returns
    -------
    Astropy Time object
        A Time object representing the given time.
    """

    # Get the nearest (earlier) UNIX time in whole seconds.
    t_unix_s = int(1.0e-9 * t_unix_ns)

    # The remaining nanoseconds from the whole second stamp.
    t_ns = t_unix_ns - 1_000_000_000 * t_unix_s

    # Use the Python time library to convert the UNIX time in seconds to a
    # struct_time containing the UTC calendar date.
    #
    # We cannot do this with Astropy, because on days with Leap Seconds
    # astropy's "unix" time is not a unix time, the Leap Second is smeared
    # throughout the day.
    t_ts = time.gmtime(t_unix_s)

    # Unpack the struct_time into an Astropy time object, add back the
    # remaining nanoseconds.
    t = Time(
        {
            "year": t_ts.tm_year,
            "month": t_ts.tm_mon,
            "day": t_ts.tm_mday,
            "hour": t_ts.tm_hour,
            "minute": t_ts.tm_min,
            "second": t_ts.tm_sec + 1.0e-9 * t_ns,
        },
        scale="utc",
        precision=9,
    )

    return t


def calc_astropy_time_from_inst_ns(t_inst_ns, time0_ns):
    r"""
    Constuct an astropy Time object corresponding to an Instrument time in nanoseconds. 
    Parameters
    ----------
    t_inst_ns : int
        An Instrument time in nanoseconds.

    Returns
    -------
    Astropy Time object
        A Time object representing the given time.
    """

    # First calculate t0, a good UNIX time
    t0 = calc_astropy_time_from_unix_ns(time0_ns)

    split_scale_ns = 100_000_000_000_000  # roughly 1 day

    dt_ns = t_inst_ns - time0_ns

    # Split dt into large and small parts to help astropy with precision.
    dt1_ns = split_scale_ns * (dt_ns // split_scale_ns)
    dt2_ns = dt_ns - dt1_ns

    # Now add the difference from t0 in TAI nanoseconds
    dt = TimeDelta(dt1_ns * units.ns, dt2_ns * units.ns, scale="tai")

    return t0 + dt


def calc_astropy_time_from_ut1_ns(ut1_ns):
    r"""
    Constuct an astropy Time object corresponding to an Instrument time in nanoseconds. 
    Parameters
    ----------
    t_inst_ns : int
        An Instrument time in nanoseconds.

    Returns
    -------
    Astropy Time object
        A Time object representing the given time.
    """

    # First calculate t0, the UT1 0 time in ERA definition.
    t0 = Time(2_451_545, format="jd", scale="ut1")

    split_scale_ns = 100_000_000_000_000  # roughly 1 day
    ut1_1_ns = split_scale_ns * (ut1_ns // split_scale_ns)
    ut1_2_ns = ut1_ns - ut1_1_ns

    # Now add the difference from t0 in TAI nanoseconds
    dt = TimeDelta(ut1_1_ns * units.ns, ut1_2_ns * units.ns, scale="ut1")

    return t0 + dt


def calc_tai_ns_from_dt(dt):
    r"""
    Compute the number of TAI nanoseconds elapsed over a time interval, rounded
    to the nearest nanosecond. Should be accurate (up to the precision of the
    given dt) so long as dt ~< 200 years.

    Parameters
    ----------
    dt : astropy TimeDelta object
        The input time interval

    Returns
    -------
    int
        The number of nanoseconds (rounded to the nearest nanosecond) for the
        time interval dt
    """

    return calc_ns_from_dt(dt.tai)


def calc_ut1_ns_from_dt(dt):
    r"""
    Compute the number of UT1 nanoseconds elapsed over a time interval, rounded
    to the nearest nanosecond. Should be accurate (up to the precision of the
    given dt) so long as dt ~< 200 years.

    Parameters
    ----------
    dt : astropy TimeDelta object
        The input time interval

    Returns
    -------
    int
        The number of nanoseconds (rounded to the nearest nanosecond) for the
        time interval dt
    """

    return calc_ns_from_dt(dt.ut1)


def calc_ns_from_dt(dt):
    r"""
    Compute the number of nanoseconds elapsed over a time interval, rounded
    to the nearest nanosecond. Should be accurate (up to the precision of the
    given dt) so long as dt ~< 200 years.

    Parameters
    ----------
    dt : astropy TimeDelta object
        The input time interval

    Returns
    -------
    int
        The number of nanoseconds (rounded to the nearest nanosecond) for the
        time interval dt
    """

    # The time is internally represented as the sum of two JD values in float.
    # Convert each of these to nanoseconds, in floating point.
    ns1_f = 86400 * 1e9 * dt.jd1
    ns2_f = 86400 * 1e9 * dt.jd2

    # Round the first component to the nearest integer nanosecond.
    ns1 = np.round(ns1_f).astype(int)

    # Compute the floating point remainder nanoseconds from rounding the first
    # part
    dns = ns1_f - ns1

    # Compute the integer nanoseconds from the second part, including the
    # difference from the first rounding.
    ns2 = np.round(ns2_f + dns).astype(int)

    if np.ndim(dt) == 0:
        return int(ns1 + ns2)

    return ns1 + ns2


def calc_ut1_ns_from_t(t):

    # Our UT1 epoch, from the definition of ERA
    t0_ut1 = Time(2_451_545, scale="ut1", format="jd")

    return calc_ut1_ns_from_dt(t.ut1 - t0_ut1)


def build_EOP_table(times, time0_ns, iers):
    r"""
    Construst the list of entries for the Earth Orientation Parameter (EOP)
    Table.

    Each EOP table entry is a dict with 4 entries:

        - "t_inst_ns" : int, the instrument timestamp in nanoseconds
        - "delta_UT1_inst" : float, the difference between UT1 and instrument
        time, in seconds.
        - "xp_as" : float, Polar Motion x' coordinate, in arcseconds.
        - "yp_as" : float, Polar Motion y' coordinate, in arcseconds.

    Parameters
    ----------
    times : list of astropy Time objects
        The times at at which table entries will be generated.
    time0_ns : int
        The timestamp in nanoseconds for frame0 of the telescope.
    iers : astropy IERS object

    Returns
    -------
    EOPTable: List of EOP dict entries.
    """

    # Make a Time object for the time0 value, so leap seconds can be calculated
    # later.
    t0 = calc_astropy_time_from_unix_ns(time0_ns)

    # Compute the number of leap seconds (TAI-UTC) at t0
    dtai0 = calc_delta_tai_utc(t0)

    # Initialize empty table
    eop_table = []

    for t in times:

        # Compute number of leap seconds at t.
        dtai = calc_delta_tai_utc(t)

        # Compute number of TAI nanoseconds elapsed since t0.
        dt_ns = calc_tai_ns_from_dt(t - t0)

        # Instrument time is the UNIX timestamp PLUS the TAI time elapsed
        # since start up.
        t_inst_ns = time0_ns + dt_ns

        # Get the UTC -> UT1 conversion offset at t. This value is
        # discontinuous over a leap second.
        # First argument is Time object, or jd1
        # Second argument is ignored if first is Time
        # Third argument is whether to return Status as third return value
        delta_ut1_utc, _ = iers.ut1_utc(t, None, True)

        # The Instrument -> UT1 conversion is UT1-UTC PLUS any elapsed leap
        # seconds since t0 (startup). This ensures
        # UT1 = t_inst_ns + delta_UT1_inst is a continuous function over a leap
        # second.
        delta_ut1_inst = delta_ut1_utc.to_value("second") - (dtai - dtai0)

        # Get Polar motion x & y from IERS Table.
        # First argument is Time object, or jd1
        # Second argument is ignored if first is Time
        # Third argument is whether to return Status as third return value
        x, y, status = iers.pm_xy(t, None, True)

        # Build the EOP entry! Remove numpy-ness.
        eop = dict(
            t_inst_ns=t_inst_ns,
            delta_UT1_inst=delta_ut1_inst.item(),
            xp_as=x.to_value("arcsecond").item(),
            yp_as=y.to_value("arcsecond").item(),
        )

        # Append to the table.
        eop_table.append(eop)

    # Done!
    return eop_table
