#!/usr/bin/env python3
"""GPS satellite visibility / almanac helper for the beam-mapping capture.

Lists which PRNs are above an observer's horizon -- right now, or (with --hours)
the union that rises above the mask over a future window. Uses the same skyfield
+ Celestrak TLE machinery as gps_beamtrack.py.

Intended use: planning the capture and as a sanity check. Note that for GPS the
simplest capture config just lists all 32 PRNs statically and lets the
correlator self-acquire whatever transits -- over a day essentially every PRN
appears, so a visible subset buys little. This tool makes that concrete (a 24 h
scan typically returns nearly all 32) and produces a paste-ready ``prns:`` line
if you do want to subset (e.g. a compute-limited or short run).

Examples
--------
    # PRNs above 10 deg right now
    python gps_visibility.py --lat 43.66 --lon -79.40 --mask 10

    # PRNs that rise above 5 deg at any point in the next 24 h
    python gps_visibility.py --lat 43.66 --lon -79.40 --hours 24
"""

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

from gps_beamtrack import DEFAULT_TLE_URL, load_gps_satellites


def altaz_over(sats_by_prn, lat, lon, alt_m, times):
    """Return {prn: (alt_deg[], az_deg[])} sampled at the given UTC datetimes."""
    from skyfield.api import load, wgs84

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    t = ts.from_datetimes(list(times))
    out = {}
    for prn, sat in sats_by_prn.items():
        topo = (sat - observer).at(t).altaz()
        out[prn] = (np.atleast_1d(topo[0].degrees), np.atleast_1d(topo[1].degrees))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True, help="observer latitude, deg")
    ap.add_argument("--lon", type=float, required=True, help="observer longitude, deg")
    ap.add_argument("--alt", type=float, default=0.0, help="observer height, m")
    ap.add_argument("--mask", type=float, default=5.0, help="horizon mask elevation, deg")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="scan a future window of this many hours (0 = now only)")
    ap.add_argument("--step", type=float, default=10.0, help="window sample step, minutes")
    ap.add_argument("--tle", default=DEFAULT_TLE_URL, help="TLE file path or URL")
    args = ap.parse_args(argv)

    sats = load_gps_satellites(args.tle)
    now = datetime.now(timezone.utc)
    if args.hours > 0:
        n = int(args.hours * 60.0 / args.step) + 1
        times = [now + timedelta(minutes=args.step * i) for i in range(n)]
    else:
        times = [now]

    aa = altaz_over(sats, args.lat, args.lon, args.alt, times)

    if len(times) == 1:
        vis = sorted(((p, a[0], z[0]) for p, (a, z) in aa.items() if a[0] >= args.mask),
                     key=lambda x: -x[1])
        print("# %d PRNs above %.0f deg at %s" % (len(vis), args.mask, now.isoformat()))
        for p, a, z in vis:
            print("PRN %2d: alt %5.1f  az %6.1f" % (p, a, z))
        prns = [p for p, _, _ in vis]
    else:
        peak = {p: float(np.max(a)) for p, (a, z) in aa.items()}
        prns = sorted(p for p in peak if peak[p] >= args.mask)
        print("# %d of %d PRNs rise above %.0f deg within %g h" %
              (len(prns), len(sats), args.mask, args.hours))
        for p in prns:
            print("PRN %2d: peak alt %5.1f" % (p, peak[p]))

    print("prns: [" + ", ".join(str(p) for p in prns) + "]")


if __name__ == "__main__":
    main()
