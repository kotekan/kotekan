#!/usr/bin/env python3
"""
/*********************************************************************************
* Earth Orientation Parameter Tools 
* File: eop_utils.py
* Purpose: Provide helper functions for converting between Astropy, Instrument, and UT1 times, for parsing EOP objects, and making REST calls to Kotekan and fpga_master
* Python Version: 3.12
* Dependencies: requests, astropy, numpy
* Authors: Geoffrey Ryan
*********************************************************************************/
"""
import json
from pathlib import Path
import sys
import time
import requests
from astropy.time import Time, TimeDelta
import astropy.units as units
import astropy.utils.iers
import astropy.utils.data
import numpy as np

# Ensure Astropy will download new IERS data when needed
astropy.utils.iers.conf.auto_download = True
# Set the Astropy IERS Refresh time to the minimum allowed (10 days)
astropy.utils.iers.conf.auto_max_age = 10.0


def make_rest_get_request(host, port, endpoint, timeout, protocol="http://"):
    r"""
    Make a REST GET request to the specified endpoint and return the response.

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    endpoint : String
        The endpoint to query, no leading "/".  For instance "get-frame0-time".
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    resp : Response
        A requests Response object.

    Raises
    ------
    Exceptions from requests.
    RuntimeError if no exception was raised by `requests` but response was not 200 OK
    """

    url = "{0:s}{1:s}:{2:d}/{3:s}".format(protocol, host, port, endpoint)

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if resp.status_code != 200:
        raise RuntimeError(
            "GET request was not OK, received: {} with reason {}".format(
                resp, resp.reason
            )
        )

    return resp


def make_rest_post_request(
    host, port, endpoint, json_payload, timeout, protocol="http://"
):
    r"""
    Make a REST POST request to the specified endpoint with the given payload and
    return the response.

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    endpoint : String
        The endpoint to query, no leading "/".  For instance "get-frame0-time".
    json_payload : JSON-able Object
        Payload for the POST, serializable as JSON (e.g. a dict)
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    resp : Response
        A requests Response object.

    Raises
    ------
    Exceptions from requests.
    RuntimeError if no exception was raised by `requests` but response was not 200 OK
    """

    url = "{0:s}{1:s}:{2:d}/{3:s}".format(protocol, host, port, endpoint)

    resp = requests.post(url, json=json_payload, timeout=timeout)
    resp.raise_for_status()
    if resp.status_code != 200:
        raise RuntimeError(
            "POST request was not OK, received: {} with reason {}".format(
                resp, resp.reason
            )
        )

    return resp


def is_kotekan_alive(host, port, timeout, protocol="http://"):
    r"""
    Check if kotekan is running at the given host:port by querying the
    "endpoints" GET endpoint. Does not raise an exception if the query fails.

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    bool
        True if kotekan responded correctly, False otherwise.

    Raises
    ------
    None
    """

    try:
        resp = make_rest_get_request(host, port, "endpoints", timeout, protocol)
        body = resp.json()
    except:
        body = {}

    if len(body) > 0 and "GET" in body.keys() and "POST" in body.keys():
        return True

    return False


def get_kotekan_endpoints(host, port, timeout, protocol="http://"):
    r"""

    Check the REST endpoints available on a Kotekan instance.
    
    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    endpoints : dict
        A dict of the received endpoints, with "GET" and "POST" keys.

    Raises
    ------
    Exceptions from requests.
    RuntimeError if no exception was raised by `requests` but response was not 200 OK
    """
    resp = make_rest_get_request(host, port, "endpoints", timeout, protocol)

    return resp.json()


def read_kotekan_frame0_ns(host, port, timeout, protocol="http://"):
    r"""
    Read the "time0_ns" parameter from a running Kotekan instance.

    time0_ns is the UNIX timestamp (in nanoseconds) of the first frame in
    F-Engine / Kotekan, and serves as the base time for all future timestamps.

    On fpga_master this is called "frame0_nano".

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    frame0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    resp = make_rest_get_request(host, port, "telescope/time0_ns", timeout, protocol)

    return resp.json()["time0_ns"]


def read_fpga_master_frame0_ns(
    host, port, timeout, protocol="http://", apply_rollover_correction=True
):
    r"""
    Read the "frame0_nano" parameter from fpga_master, possibly correcting
    for GPS 1024 week rollover differences.

    frame0_nano is the UNIX timestamp (in nanoseconds) of the first frame in
    F-Engine / Kotekan, and serves as the base time for all future timestamps.

    In kotekan this is called frame0_nano or time0_ns.

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)
    apply_rollover_correction : bool, optional
        Apply the GPS 1024 week rollover correction to the obtained value, using
        the fpga_master `start_ctime` value. Default True.

    Returns
    -------
    frame0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    resp = make_rest_get_request(host, port, "get-frame0-time", timeout, protocol)
    body_json = resp.json()
    frame0_nano = int(body_json["frame0_nano"])

    if apply_rollover_correction:

        # Get the start_ctime (float, when the F-Engine booted,
        # system UNIX time with fractional seconds)
        start_ctime = body_json["start_ctime"]

        # Convert the start time to an integer number of nanoseconds
        start_time_nano = int(start_ctime * 1e9)

        # Compute the GPS rollover period in nanoseconds
        DAYS_PER_WEEK = 7
        SECS_PER_DAY = 86400
        GIGA = 1_000_000_000
        rollover_dt_ns = 1024 * DAYS_PER_WEEK * SECS_PER_DAY * GIGA

        # Compute the UNIX time for the GPS epoch (Jan 6, 1980, 00:00:00 UTC)
        # This time is an exact second in UTC and so is an exact integer in UNIX time.
        gps0_ns = GIGA * int(Time("1980-01-06T00:00:00", scale="utc").unix)

        # Compute which rollover period the given frame0 time and the F-Engine
        # start_time are in.
        rollovers_at_start = (start_time_nano - gps0_ns) // rollover_dt_ns
        rollovers_in_sent_frame0 = (frame0_nano - gps0_ns) // rollover_dt_ns

        # adjust frame by the difference in rollover periods
        frame0_nano += rollover_dt_ns * (rollovers_at_start - rollovers_in_sent_frame0)

    return frame0_nano


def read_kotekan_eop_table(host, port, timeout, protocol="http://"):
    r"""
    Read the "eop_table" from a running Kotekan instance.

    The eop_table is a list of EOP objects, each containing a "t_inst_ns",
    "t_ut1_ns", "delta_UT1_inst", "ERA_deg", "xp_as", "yp_as".

    Parameters
    ----------
    host : String
        The hostname at which to find the kotekan instance, no trailing "/". For
        instance "localhost" or "127.0.0.1"
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    eop_table : [EOP, ...]
        List of EOP JSON objects

    Raises
    ------
    Exceptions from requests.
    """

    resp = make_rest_get_request(host, port, "telescope/eop_table", timeout, protocol)

    return resp.json()["eop_table"]


def broadcast_kotekan_eop_table(
    host, port, eop_endpoint, eop_table, timeout, protocol="http://"
):
    r"""
    Send a new EOP table to a running Kotekan instance.

    Parameters
    ----------
    base_url : String
        The URL at which to find the kotekan instance, no trailing "/". For
        instance "http://localhost".
    port : int
        The port at which to find the kotekan instance. For instance 12048.
    eop_endpoint : The endpoint for the EOP table config updater in Kotekan,
        no leading "/".
    eop_table : dict containing a List of dicts, each a BareEOP table entry
        The EOP table. A dict with a single key "earth_orientation_parameter_table",
        whose value is a list of BareEOP table update entries, each a dict with
        entries "t_inst_ns", "delta_UT1_inst", "xp_as", and "yp_as".
    timeout : float
        Timeout in seconds for the request.
    protocol : String, optional
        Prefix for the URL, for instance "http://" (the default)

    Returns
    -------
    time0_ns : int
        The UNIX timestamp in nanoseconds received from kotekan.

    Raises
    ------
    Exceptions from requests.
    """

    resp = make_rest_post_request(
        host, port, eop_endpoint, eop_table, timeout, protocol
    )

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

    # Now add the difference from t0 in TAI nanoseconds
    dt = TimeDelta((t_inst_ns - time0_ns) * units.ns, scale="tai")

    return t0 + dt


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
        precision=9,
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


def build_time_array(
    t_ref, n_intervals_before, n_intervals_after, interval_length_days, snap_to_grid,
):
    r"""
    Construct an array of times for the entries in the EOP Table.

    The table entries mark edges between intervals of time. The `current interval`
    is the interval containing the given t_ref. The table will contain at least two
    entries, marking the beginning and end of the current interval. The caller
    specifies the number of intervals to add before and after the current interval.

    The table will have n_intervals_before + n_intervals_after + 2 entries.

    If snap_to_grid is true, the intervals will be placed aligned with MJD = 0 UTC.
    Calls to this function with different t_ref will return identical
    time points when intervals overlap.

    Recommended operation for a telescope is to use `snap_to_grid = True` and
    `interval_length_days = 1.0`. This will always produce times at UTC midnight (0h)
    which is precisely the times at which the IERS tables apply. This function works
    correctly for intervals including a leap second.

    Parameters
    ----------
    t_ref : astropy Time object
        Reference time to align the bins. If snap_to_grid is False, the central
        time for the current interval.
    n_intervals_before : int
        Number of intervals to add before the current interval, >= 0.
    n_intervals_after : int
        Number of intervals to add after the current interval, >= 0.
    interval_length_days : float
        Length of intervals in UTC days (86399, 86400, or 86401 seconds).
    snap_to_grid : boolean
        Whether to snap the intervals to be at e.g. whole days.
    
    Returns
    -------
    times : Astropy Time() array
        A Time object containing an ordered array of times to construct EOP table
        entries for.

    Raises
    ------
    ValueError if arguments are invalid.
    """

    if n_intervals_before < 0:
        raise ValueError(
            "n_intervals_before must be positive or 0, received: {:d}".format(
                n_intervals_before
            )
        )

    if n_intervals_after < 0:
        raise ValueError(
            "n_intervals_after must be positive or 0, received: {:d}".format(
                n_intervals_after
            )
        )

    if interval_length_days <= 0.0:
        raise ValueError(
            "interval_length_days must be positive, received: {:g}".format(
                interval_length_days
            )
        )

    # We'll "grid" in UTC MJD, which is an integer at 0h UTC. 1 UTC day = 1 UTC mjd,
    # even on leap second days.
    mjd_ref = t_ref.utc.mjd

    # First, compute `mjd0`, the MJD for `t0`, the time which begins the current
    # interval. What exactly this time is depends on whether we're snapping to a grid.
    if snap_to_grid:

        # Compute the MJD value for the beginning of the interval containing
        # t_ref. If length is 1.0, this just takes the floor of mjd_ref,
        # returning the most recent midnight UTC.
        # If length is 0.5, will return the most recent midnight or noon UTC.
        mjd0 = int(mjd_ref / interval_length_days) * interval_length_days
    else:
        # If not snapping, then take the current interval to be centered on `t_ref` and
        # so place `t0` half a `dt` earlier.
        mjd0 = mjd_ref - 0.5 * interval_length_days

    # Array of all entry offsets in mjd from t0 (the beginning time of the current
    # interval).
    dt_mjd = interval_length_days * np.arange(
        -n_intervals_before, n_intervals_after + 2
    )

    # Constuct list of times. For precision separate mjd0 (which will be ~57000) from
    # the dt_mjd (which are likely ~integers or fractions thereof). The time
    # represented will be the sum of these, although internally the Time() object will
    # keep them separate when possible to preserve precision.
    times = Time(mjd0, dt_mjd, format="mjd", scale="utc", precision=9)

    t0 = times[n_intervals_before]
    t1 = times[n_intervals_before + 1]

    if not (t0 <= t_ref and t_ref < t1):
        raise RuntimeError(
            "build_time_array failed. The current interval [{}, {}] does not contain t_ref {}".format(
                t0.isot, t1.isot, t_ref.isot
            )
        )

    return times


def make_bare_EOP_from_full_EOP(eop):
    r"""
    Produce a BareEOP dict (suitable for sending to Kotekan or putting in a config
    file) from a full EOP dict (one received from Kotekan)
    
    Parameters
    ----------
    eop : EOP full entry dict
        A full Kotekan EOP table entry, containing the fields: t_inst_ns, t_ut1_ns,
        delta_UT1_inst, ERA_deg, xp_as, yp_as.

    Returns
    -------

    eop_bare : BareEOP dict
        An BareEOP table entry, contain the fields: t_inst_ns, delta_UT1_inst,
        xp_as, yp_as.
    """

    return dict(
        t_inst_ns=eop["t_inst_ns"],
        delta_UT1_inst=eop["delta_UT1_inst"],
        xp_as=eop["xp_as"],
        yp_as=eop["yp_as"],
    )


def merge_eop_tables(
    current_eop_table,
    new_eop_table,
    t_ref,
    time0_ns,
    num_intervals_before,
    merge_cushion_dt,
):
    r"""
    produce a new eop table from the current kotekan table and a proposed updated table.
    only include entries past the current interval to ensure continuity of parameters.
    drop early entries if they are no longer needed.

    Parameters
    ----------
    current_eop_table : list of eop entry dicts
        the current table in kotekan (fields: t_inst_ns, t_ut1_ns, delta_ut1_inst, era_deg,
        xp_as, yp_as)
    new_eop_table : list of BareEOP update entry dicts
        the proposed new eop table, may conflict with current table. (fields: 
        t_inst_ns, delta_ut1_inst, xp_as, yp_as)
    t_ref : Astropy Time object
        The reference (likely current) time used to determine the current interval.
    time0_ns : int
        The UNIX time (as integer nanoseconds) of telescope frame0.
    num_intervals_before : int
        Number of intervals desired in the final table before the current interval.
    merge_cushion_dt : Astropy TimeDelta
        Amount of buffer time to include when determining which current entries to drop.
    
    Returns
    -------
    final_eop_table : List of BareEOP entry dicts
        The final blended table.
    """

    # Find the pivot time (when to start replacing current table entries)
    # as an Instrument time (UNIX frame0 + TAI since)

    # First as an Astropy time.
    t_pivot = t_ref + merge_cushion_dt

    print(
        "Merging the current Kotekan table with the update. Using a pivot time of",
        t_pivot.isot,
    )

    # Now get the frame0 time as an Astropy object.
    t0 = calc_astropy_time_from_unix_ns(time0_ns)
    # Now calculate the difference between t_pivot and frame0
    # as TAI nanoseconds
    dt_pivot_ns = calc_tai_ns_from_dt(t_pivot - t0)
    # Now can get t_pivot as an instrument time, time0 + TAI dt
    t_pivot_inst_ns = time0_ns + dt_pivot_ns

    # Check input tables are sorted.
    for i in range(len(current_eop_table) - 1):
        if current_eop_table[i]["t_inst_ns"] >= current_eop_table[i + 1]["t_inst_ns"]:
            raise RuntimeError(
                "current_eop_table is not sorted. Can only merge sorted tables"
            )
    for i in range(len(new_eop_table) - 1):
        if new_eop_table[i]["t_inst_ns"] >= new_eop_table[i + 1]["t_inst_ns"]:
            raise RuntimeError(
                "new_eop_table is not sorted. Can only merge sorted tables"
            )

    # Extract array of times for easier searching.  If we're here, both arrays have
    # strictly increasing instrument times
    current_times = np.array([eop["t_inst_ns"] for eop in current_eop_table])
    new_times = np.array([eop["t_inst_ns"] for eop in new_eop_table])

    # For current_times, the pivot_idx will be the index of the first element strictly
    # greater than the pivot time.  For new_times, the pivot_idx is the index of the
    # first time strictly greater than the rightmost kept current_time.
    #
    #                 [------- keep ----------]
    #           tc0   tc1    tc2     tc3      tc4      tc5
    # current:  |  0  |   1  |   2   |   *3   |    4   |
    #                                   ^     |
    #                                   t_pivot (current_pivot_idx = 4)
    #                                         |
    #                                         | (new_pivot_idx = 3)
    #                                         v
    # new:               |   0   |   1   |  2    |  3    |   4   |
    #                    tn0     tn1     tn2     tn3     tn4     tn5
    #                                            [------keep-----]
    #
    #                 tc1    tc2     tc3      tc4
    # final:          |  0   |  1    |   2    |3 |  4    |   5   |
    #                                            tn3     tn4     tn5
    current_pivot_idx = np.searchsorted(current_times, t_pivot_inst_ns, side="right")

    # First entry index from the current table to keep
    # Last entry index from the current table to keep

    if current_pivot_idx <= 0:
        # The entire current table is in the future of pivot time.
        # For continuity keep the first entry, but discard all after
        print("The pivot time is before the current table. Keeping only entry 0")
        current_eop_to_keep = [current_eop_table[0]]
    elif current_pivot_idx < len(current_times):
        # Normal case, the pivot time is in the middle of the current table

        # First entry from current table to keep. Don't go out of bounds!
        left_idx = max(0, current_pivot_idx - 1 - num_intervals_before)
        # The pivot index already tells us the future entry still being used (endpoint
        # of the current interval)
        right_idx = current_pivot_idx

        print(
            "Keeping current table entries [{:d} - {:d}] (inclusive)".format(
                left_idx, right_idx
            )
        )
        # Start final table with slice from current table
        current_eop_to_keep = current_eop_table[left_idx : right_idx + 1]
    else:
        # The entire current table is in the past of the pivot time!
        # This is bad for telescope operations: the table has expired.
        # For continuity, keep the last entry in the table (which we are extrapolating
        # from) and add an entry at the pivot time, which *should* be some comfortable
        # distance in the future. This puts the current time in a short interval, which
        # will expire at the pivot time at which point we can start moving to the
        # new (correct, hopefully) table.

        eop_pivot = current_eop_table[-1].copy()
        eop_pivot["t_inst_ns"] = t_pivot_inst_ns

        print(
            "The pivot time is after the current table expires. Keeping only the last entry and cloning it to the pivot time"
        )
        current_eop_to_keep = [current_eop_table[-1], eop_pivot]

    # Initialize the final EOP update table.
    final_eop_table = [make_bare_EOP_from_full_EOP(eop) for eop in current_eop_to_keep]

    # Easy now.  Find the first entry in the new table that's ahead of the end of
    # the preliminary final table.
    new_pivot_idx = np.searchsorted(new_times, final_eop_table[-1]["t_inst_ns"])

    # If there are any entries (in normal operations there should be!) add them to
    # the end of the table
    if new_pivot_idx < len(new_eop_table):
        print(
            "Adding {:d} new entries to the EOP table".format(
                len(new_eop_table) - new_pivot_idx
            )
        )
        final_eop_table.extend(new_eop_table[new_pivot_idx:])

    # Done!
    return final_eop_table


def print_eop_table(eop_table):
    r"""
    Print the given EOP table to the stdout.

    Parameters
    ----------
    eop_table : List of EOP entry dicts
    
    Returns
    -------
    None
    """

    print("\n#### BEGIN EOP TABLE ####\n")
    # JSON for prettier printing and consistent formatting
    eop_json = json.dumps({"earth_orientation_parameter_table": eop_table}, indent=4)
    print(eop_json)
    print("####  END  EOP TABLE ####\n")


def output_json_eop_table(eop_table, filename):
    r"""
    Write the given EOP table to the file `filename` in JSON format.

    Parameters
    ----------
    eop_table : List of EOP entry dicts
    filename : String
        Name of file to write the table to. Must be writable. If exists, will be
        overwritten.
    
    Returns
    -------
    None
    """

    filepath = Path(filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {"earth_orientation_parameter_table": eop_table},
            f,
            indent=4,
            ensure_ascii=True,
        )

    print("Wrote EOP table to file: {}".format(filepath))


def parse_hostport_list(broadcast_list, default_host, default_port):
    r"""
    Parse the broadcast list (list of hosts and ports) into a list of
    hostport pairs.  

    The broadcast list is a list of hosts (strings) and ports (positive
    integers). If a port appears after a host, the two form a host-port pair.
    If a host or port appear alone, the default host or default port is used
    to make a pair.

    Ex.
    [ host1 port1 host2 host3 port2 port3 ]
    becomes
    [ (host1, port1), (host2, default_port), (host3, port2),
     (default_host, port3) ]

    Parameters
    ----------
    broadcast_list : List of str
        The list of hosts and ports
    default_host : str
        Default host to use
    default_port : int
        Default port to use
    
    Returns
    -------
    hostports : List of (str, int)
        The list of host-port pairs

    Raises
    ------
    ValueError if a negative port is received
    """

    hostports = []

    current_host = None

    for word in broadcast_list:
        isPort = False
        try:
            port = int(word)
            isPort = True
        except ValueError:
            isPort = False

        if isPort:
            if port < 0:
                raise ValueError("Bad port value:", port)
            host = current_host if current_host is not None else default_host
            hostports.append((host, port))
            current_host = None
        else:
            if current_host is None:
                current_host = word
            else:
                hostports.append((current_host, default_port))
                current_host = word

    if current_host is not None:
        hostports.append((current_host, default_port))

    if len(hostports) == 0:
        hostports.append((default_host, default_port))

    return hostports
