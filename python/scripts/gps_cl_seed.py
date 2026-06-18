#!/usr/bin/env python3
"""Compute the L2C CL pilot time-assist seed (code phase) for the correlator.

The CL code (767250 chips, 1.5 s period) is aligned to GPS time, so its phase at
the receiver is

    ((t_GPS - tau) mod 1.5 s) * 511500   chips,   tau = range / c

i.e. GPS time minus the propagation delay (the signal arriving now left the
satellite ~70 ms ago), reduced into the 1.5 s code period. That value is the
GpsReplicaCorrelator's ``code_phase_seed_chips`` for signal=GPS_L2C_CL -- and it
only has to be accurate to within +-half the coherent window, because the
per-block FFT refines the residual.

Range comes from skyfield + Celestrak TLEs (shared with gps_beamtrack); GPS time
from the host clock. The GPS-UTC offset (18 s) is a whole number of 1.5 s
periods, so leap seconds don't change the CL phase, but it's kept for a correct
continuous GPS time.

Scope: this is the *instantaneous* seed. Driving a long capture continuously
also needs the correlator's sample-0 GPS time (a capture timestamp) and periodic
re-seeding as code Doppler walks the phase -- a systems step, not done here.
"""

import argparse
from datetime import datetime, timezone

from gps_beamtrack import DEFAULT_TLE_URL, load_gps_satellites

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0        # leap seconds (constant since 2017; a multiple of 1.5 s)
CL_CHIP_RATE = 511500.0     # L2 CL component chip rate, chips/s
CL_PERIOD = 1.5             # s
CL_LEN = 767250             # chips
C = 299792458.0             # m/s


def gps_seconds(when):
    """Continuous GPS seconds since the GPS epoch for a UTC datetime."""
    return (when - GPS_EPOCH).total_seconds() + GPS_MINUS_UTC


def cl_seed_chips(sat, observer, ts, when):
    """CL component-chip phase at reception time `when` for a skyfield satellite
    and observer. Returns a float in [0, CL_LEN)."""
    t = ts.from_datetime(when)
    rng_m = (sat - observer).at(t).distance().m
    tau = rng_m / C
    return ((gps_seconds(when) - tau) % CL_PERIOD) * CL_CHIP_RATE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True, help="observer latitude, deg")
    ap.add_argument("--lon", type=float, required=True, help="observer longitude, deg")
    ap.add_argument("--alt", type=float, default=0.0, help="observer height, m")
    ap.add_argument("--prn", type=int, action="append", required=True,
                    help="PRN(s) to seed (repeatable)")
    ap.add_argument("--at", default=None, help="UTC time (ISO 8601); default now")
    ap.add_argument("--tle", default=DEFAULT_TLE_URL, help="TLE file path or URL")
    args = ap.parse_args(argv)

    from skyfield.api import load, wgs84

    sats = load_gps_satellites(args.tle)
    ts = load.timescale()
    observer = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt)
    when = (datetime.fromisoformat(args.at).replace(tzinfo=timezone.utc)
            if args.at else datetime.now(timezone.utc))

    print("# CL seed (code_phase_seed_chips) at %s" % when.isoformat(timespec="seconds"))
    for prn in args.prn:
        sat = sats.get(prn)
        if sat is None:
            print("PRN %2d: no TLE" % prn)
            continue
        t = ts.from_datetime(when)
        rng_km = (sat - observer).at(t).distance().km
        seed = cl_seed_chips(sat, observer, ts, when)
        print("PRN %2d: code_phase_seed_chips=%.1f  (range %.0f km, tau %.2f ms)"
              % (prn, seed, rng_km, rng_km * 1e3 / C * 1e3))


if __name__ == "__main__":
    main()
