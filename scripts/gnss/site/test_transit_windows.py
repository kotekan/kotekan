#!/usr/bin/env python3
"""transit_windows.py's own check: the Sun and source geometry, the run finder and the names.
No TLE is read and nothing is fetched.

    python3 scripts/gnss/site/test_transit_windows.py
"""
import datetime as dt
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import transit_windows as tw  # noqa: E402

_fails = []


def check(ok, what):
    if not ok:
        _fails.append(what)
        print("  FAIL", what)


def utc(s):
    return (
        dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


def test_tau_a_transits_the_boresight():
    """Tau A crossed CHORD's boresight (az 180, el 62.70) at 11:03:28 UTC on 2026-10-29."""
    t = utc("2026-10-29T11:03:28")
    az, el = tw.radec_altaz(*tw.precess(83.633, 22.014, t), np.array([t]))
    check(
        abs(az[0] - 180.0) < 0.2 and abs(el[0] - 62.70) < 0.15,
        "Tau A at az %.2f el %.2f" % (az[0], el[0]),
    )
    t1 = t - 86164.09
    az1, _ = tw.radec_altaz(*tw.precess(83.633, 22.014, t1), np.array([t1]))
    check(abs(az1[0] - az[0]) < 0.05, "one sidereal day earlier the same azimuth")


def test_sun():
    """2026 northern solstice ~2026-06-21 08:24 UTC: Sun declination +23.44."""
    ra, dec = tw.sun_radec(np.array([utc("2026-06-21T08:24:00")]))
    check(
        abs(dec[0] - 23.44) < 0.02 and abs(ra[0] - 90.0) < 0.1,
        "solstice Sun RA %.2f Dec %.2f" % (ra[0], dec[0]),
    )


def test_runs_and_names():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    check(tw.runs(m) == [(1, 2), (4, 4), (7, 7)], "runs %s" % tw.runs(m))
    check(tw.short_name("GPS BIIR-5  (PRN 22)", "G") == "G22", "GPS name")
    check(tw.short_name("BEIDOU-3 M1 (C19)", "C") == "C19", "BeiDou name")
    check(tw.short_name("GSAT0101 (GALILEO-PFM)", "E") == "GSAT0101", "Galileo name")
    check(tw.short_name("HULIANWANG DIGUI-26", "W") == "DIGUI-26", "GuoWang name")


def test_boresight_unit_vector():
    b = tw.unit(180.0, 62.70)
    check(
        abs(np.linalg.norm(b) - 1) < 1e-12
        and b[1] < 0
        and abs(math.degrees(math.asin(b[2])) - 62.7) < 1e-9,
        "boresight points south at el 62.7",
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
