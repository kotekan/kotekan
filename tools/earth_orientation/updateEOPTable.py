#!/usr/bin/env python3
import time
import requests
from astropy.time import Time
import astropy.units as units
import astropy.utils.iers
import numpy as np
import matplotlib.pyplot as plt


def read_time0_ns(base_url, port):
    r"""
    Read the "time0_ns" parameter from a running Kotekan instance. 

    time0_ns is the UNIX timestamp (in nanoseconds) of the first frame in
    Kotekan, and serves as the base time for all future timestamps.

    Parameters
    ----------
    base_url : String
        The URL at which to find the kotekan instance, no trailing "/". For
        instance "http://localhost".
    port : int
        The port at which to find the kotekan instance. For instance 12048.

    Returns
    -------
    time0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    url = "{0:s}:{1:d}/telescope/time0_ns".format(base_url, port)

    resp = requests.get(url)

    return resp.json()["time0_ns"]


def broadcast_eop_table(base_url, port, eop_table):
    r"""
    Send a new EOP table to a running Kotekan instance.

    Parameters
    ----------
    base_url : String
        The URL at which to find the kotekan instance, no trailing "/". For
        instance "http://localhost".
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    eop_table : List of dicts, each an EOP table entry
        The EOP table. A list of entries, each a dict with entries
        "time_inst_ns", "delta_UT1_inst", "x_pm", and "y_pm"

    Returns
    -------
    time0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    url = "{0:s}:{1:d}/earth_rotation_data".format(base_url, port)

    payload = {"earth_orientation_parameter_table": eop_table}

    resp = requests.post(url, json=payload)

    return resp


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
    dt = round(dt1 + dt2, 10)

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
    )

    return t


def calc_unix_ns_from_t(t):
    r"""
    Compute the UNIX timestamp in nanoseconds from given time t.

    Parameters
    ----------
    t : astropy Time object
        The input time

    Returns
    -------
    int
        The corresponding UNIX timestamp in nanoseconds.
    """

    # Get time in UTC broken into calendar date.
    ymdhms = t.utc.ymdhms

    # Get the time at the beginning (0h) of the UTC day.  The astropy UNIX
    # time conversion is not accurate in the middle of a day the day before a
    # leap second.
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

    # Number of nanoseconds elapsed since t0.
    ns_from_0 = round((t - t0h).tai.to_value("ns"))

    # Return the sum of the unix timestamp from the start of the day and the
    # number of nanoseconds elapsed since then.
    return int(t0h.unix) * 1_000_000_000 + ns_from_0


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

    # Get the time in the TAI scale.
    tai = dt.tai

    # The time is internally represented as the sum of two JD values in float.
    # Convert each of these to nanoseconds, in floating point.
    ns1_f = 86400 * 1e9 * tai.jd1
    ns2_f = 86400 * 1e9 * tai.jd2

    # Round the first component to the nearest integer nanosecond.
    ns1 = round(ns1_f)

    # Compute the floating point remainder nanoseconds from rounding the first
    # part
    dns = ns1_f - ns1

    # Compute the integer nanoseconds from the second part, including the
    # difference from the first rounding.
    ns2 = round(ns2_f + dns)

    # Return the sum.
    return ns1 + ns2


def build_EOP_table(times, time0_ns, iers):
    r"""
    Construst the list of entries for the Earth Orientation Parameter (EOP)
    Table.

    Each EOP table entry is a dict with 4 entries:

        - "time_inst_ns" : int, the instrument timestamp in nanoseconds
        - "delta_UT1_inst" : float, the difference between UT1 and instrument
        time, in seconds.
        - "x_pm" : float, Polar Motion x' coordinate, in arcseconds.
        - "y_pm" : float, Polar Motion y' coordinate, in arcseconds.

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
        t_inst = time0_ns + dt_ns

        # Get the UTC -> UT1 conversion offset at t. This value is
        # discontinuous over a leap second.
        # First argument is Time object, or jd1
        # Second argument is ignored if first is Time
        # Third argument is whether to return Status as third return value
        delta_ut1_utc, _ = iers.ut1_utc(t, None, True)

        # The Instrument -> UT1 conversion is UT1-UTC PLUS any elapsed leap
        # seconds since t0 (startup). This ensures
        # UT1 = t_inst + delta_UT1_inst is a continuous function over a leap
        # second.
        delta_ut1_inst = delta_ut1_utc.to_value("second") - (dtai - dtai0)

        # Get Polar motion x & y from IERS Table.
        # First argument is Time object, or jd1
        # Second argument is ignored if first is Time
        # Third argument is whether to return Status as third return value
        x, y, status = iers.pm_xy(t, None, True)

        # Build the EOP entry!
        eop = dict(
            time_inst_ns=t_inst,
            delta_UT1_inst=delta_ut1_inst,
            x_pm=x.to_value("arcsecond"),
            y_pm=y.to_value("arcsecond"),
        )

        # Append to the table.
        eop_table.append(eop)

    # Done!
    return eop_table


def build_time_array(
    t_ref,
    n_intervals_before,
    n_intervals_after,
    interval_length_days,
    snap_to_grid=False,
):
    r"""
    Construct an array of times for the entries in the EOP Table.

    The table will have n_intervals_before + n_intervals_after + 2 entries.

    Parameters
    ----------
    t_ref : astropy Time object
        Reference time to align the bins. If snap_to_grid is False, the central
        time for the current interval.
    n_intervals_before : int
        Number of intervals to add before the current interval.
    n_intervals_after : int
        Number of intervals to add after the current interval.
    interval_length_days : float
        Length of intervals in days (86400 seconds).
    snap_to_grid : boolean
        Whether to snap the intervals to be at e.g. whole days.
    """

    # Interval length with units
    dt = interval_length_days * units.day

    # Array of all entry offsets from t0 (the beginning time of the current
    # interval).
    dts = dt * np.arange(-n_intervals_before, n_intervals_after + 2)

    if snap_to_grid:
        # We'll "grid" in MJD, which is an integer at 0h.
        mjd_ref = t_ref.mjd

        # Compute the MJD value for the beginning of the interval containing
        # t_ref. If length is 1.0, this just takes the floor of mjd_ref,
        # returning the most recent midnight UTC (assuming t_ref is UTC).
        # If length is 0,5, will return the most recent midnight or noon UTC
        # (if t_ref is UTC).
        mjd0 = int(mjd_ref / interval_length_days) * interval_length_days

        # Make the time object for this mjd.
        t0 = Time(mjd0, format="mjd", scale=t_ref.scale)

    else:
        # If not snapping, then t_ref is center of current interval and
        # t0 is half a dt earlier.
        t0 = t_ref - 0.5 * dt

    # constuct list of times.
    times = t0 + dts

    return times


if __name__ == "__main__":

    kotekan_url = "http://localhost"
    kotekan_port = 12048

    # Get the Frame0 time from kotekan
    t0_ns = read_time0_ns(kotekan_url, kotekan_port)

    # Build the list of times to generate EOP entries for
    t_ref = Time.now()

    # Add 2 intervals before the current one, 3 after.  Intervals are 1.0 days
    # long, intervals are snapped to whole days.
    ts = build_time_array(t_ref, 2, 3, 1.0, snap_to_grid=True)
    print("t_ref (now):", t_ref, t_ref.mjd)
    print("times in table:", ts.iso)
    print("times in table (mjd):", ts.mjd)

    # Build the table, use astropy's automatic IERS table
    # TODO: Make sure this table is up to date (force download, or set short
    # expiry time)
    iers = astropy.utils.iers.IERS_Auto.open()
    eop_table = build_EOP_table(ts, t0_ns, iers)
    iers.close()

    print(eop_table)

    # Send table to Kotekan
    broadcast_eop_table(kotekan_url, kotekan_port, eop_table)
