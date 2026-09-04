#!/usr/bin/env python3
"""THE ACHROMATIC TEST: is a C/N0 dip an OBSTRUCTION or MULTIPATH?

Two-ray multipath is CHROMATIC -- the fringe nulls satisfy 2h*sin(el) = m*lambda, so they sit at
DIFFERENT elevations in two bands (L1 1575.42 vs L2C 1227.6 -> lambda ratio 1.283, so an L1 null
at sin(el) maps to L2C at 1.283*sin(el)). A physical BLOCKAGE (the house) is ACHROMATIC: the same
dip at the SAME elevation in both bands.

This plot settled it on 2026-07-15: the dominant west-sector feature -- a deep dip at ~57 deg --
is at the SAME elevation in both bands => it is the HOUSE, not the 0.45 m reflector that a
single-band two-ray fit had been over-fitted to. See [[site-obstruction-vs-multipath]].

Reads the SAME gps_status_logger JSONL as gps_cn0_map.py and reuses its loader + sky machinery,
so the elevation axis and the re-anchor excision are identical by construction.

Usage:
  python3 chromatic_check.py --l1 /tmp/gpswipe/status_log.jsonl \
      --l2c /tmp/gps_l2c_gpu/status_log.jsonl --outdir <dir> [--az-lo 240 --az-hi 330] [--trim-h 2]
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_cn0_map import load, pedestal  # noqa: E402
from gps_beamtrack import load_gps_satellites, DEFAULT_TLE_URL  # noqa: E402


def sky(prn, t, lat, lon, alt_m):
    from skyfield.api import load as sky_load, wgs84
    ts = sky_load.timescale()
    obs = wgs84.latlon(lat, lon, elevation_m=alt_m)
    sats = load_gps_satellites(DEFAULT_TLE_URL)
    el = np.full(len(prn), np.nan)
    az = np.full(len(prn), np.nan)
    for p_ in np.unique(prn):
        sat = sats.get(int(p_))
        if sat is None:
            continue
        sel = prn == p_
        times = ts.from_datetimes([datetime.fromtimestamp(u, tz=timezone.utc) for u in t[sel]])
        topo = (sat - obs).at(times).altaz()
        el[sel] = topo[0].degrees
        az[sel] = topo[1].degrees
    return el, az


def band(path, args, label):
    prn, t, x, xc, dop = load(path, "G")[:5]
    o = np.argsort(t)
    prn, t, x, xc = prn[o], t[o], x[o], xc[o]
    if args.trim_h > 0:   # drop the run's settling head (a clock ramp would fake a dip)
        m = t > t[0] + args.trim_h * 3600
        prn, t, x, xc = prn[m], t[m], x[m], xc[m]
    el, az = sky(prn, t, args.lat, args.lon, args.alt)
    ok = np.isfinite(el)
    prn, t, x, xc, el, az = prn[ok], t[ok], x[ok], xc[ok], el[ok], az[ok]
    # INCOHERENT observable: multipath's amplitude flicker lands here for GPS (the coherent
    # estimator hides it) -- see [[cn0-estimator-boc-bias]] / [[multipath-and-quality-metrics]].
    low = el < -5.0
    ped, _glob, _sig1, real_ped = pedestal(x, low, t, 600.0)
    sig = x - ped
    sect = (az >= args.az_lo) & (az <= args.az_hi) & (el >= 5.0)
    print("%s: %d rows, %d in az %g-%g%s"
          % (label, len(x), int(sect.sum()), args.az_lo, args.az_hi,
             "" if real_ped else "   [!] pedestal fell back to a signal-biased percentile "
                                "(no below-horizon probes) -- shape still comparable, zero is not"))
    return el[sect], sig[sect]


def profile(el, sig, nbin=24):
    """median C/N0-ish density vs elevation (dB, arbitrary common zero per band)."""
    edges = np.linspace(5, 90, nbin + 1)
    ctr, val = [], []
    for i in range(nbin):
        m = (el >= edges[i]) & (el < edges[i + 1])
        if m.sum() < 8:
            continue
        v = np.median(sig[m])
        if v <= 0:
            continue
        ctr.append(0.5 * (edges[i] + edges[i + 1]))
        val.append(10 * np.log10(v))
    return np.array(ctr), np.array(val)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1", default="/tmp/gpswipe/status_log.jsonl")
    ap.add_argument("--l2c", default="/tmp/gps_l2c_gpu/status_log.jsonl")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--az-lo", type=float, default=240.0, help="sector start (deg, the house is W)")
    ap.add_argument("--az-hi", type=float, default=330.0)
    ap.add_argument("--trim-h", type=float, default=2.0, help="drop this much run head (h)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1, s1 = band(args.l1, args, "L1 (1575.42)")
    e2, s2 = band(args.l2c, args, "L2C (1227.6)")
    c1, v1 = profile(e1, s1)
    c2, v2 = profile(e2, s2)
    if not len(c1) or not len(c2):
        print("not enough data in the sector")
        return 1
    v1 = v1 - np.nanmedian(v1)   # common zero: only the SHAPE vs elevation is the test
    v2 = v2 - np.nanmedian(v2)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(c1, v1, "-o", color="#4d9de0", lw=2, ms=4, label="L1 C/A (1575.42 MHz)")
    ax.plot(c2, v2, "-s", color="#e8923c", lw=2, ms=4, label="L2C (1227.6 MHz)")
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel("elevation (deg)")
    ax.set_ylabel("incoherent C/N0 anomaly (dB, per-band median removed)")
    ax.set_title("Achromatic test, az %g-%g deg: dips at the SAME elevation = OBSTRUCTION\n"
                 "(two-ray multipath is chromatic: an L1 null moves to sin(el)*1.283 at L2C)"
                 % (args.az_lo, args.az_hi))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    os.makedirs(args.outdir, exist_ok=True)
    f = os.path.join(args.outdir, "chromatic_L1_vs_L2C.png")
    fig.tight_layout()
    fig.savefig(f, dpi=120)
    print("  " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
