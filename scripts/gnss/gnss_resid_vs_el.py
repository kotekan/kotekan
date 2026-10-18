#!/usr/bin/env python3
"""Step 0b read-back (docs/CHORD_JOINT_TRACKING.md 6.3): tracker code residual (code_resid_m, receiver clock removed) against elevation, per band.
Fit resid = a + b/sin(el) + c*sin(el): b ~ an unmodelled zenith tropo (+2.3 m expected), c ~ a station up error dU.
usage: gnss_resid_vs_el.py [--atmos tropo|both] OBS_DIR YYYYMMDD band [band ...]   (every 5th row, el >= 5)"""
import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "python",
        "scripts",
        "gnss",
    ),
)
import gnss_atmos as A  # noqa: E402

SITE = (
    49.32001414,
    -119.62262691,
    545.0,
)  # config/gnss_chains_chord.yaml common lat/lon/alt
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("obs_dir")
ap.add_argument("day", help="YYYYMMDD of the obs files")
ap.add_argument("bands", nargs="+")
ap.add_argument(
    "--atmos",
    choices=("none", "tropo", "both"),
    default="none",
    help="add the model back before binning",
)
ap.add_argument("--lat", type=float, default=SITE[0])
ap.add_argument("--lon", type=float, default=SITE[1])
ap.add_argument("--alt", type=float, default=SITE[2])
ap.add_argument(
    "--iono",
    default="1.3970e-08,2.2352e-08,-5.9605e-08,-1.1921e-07,1.0854e05,1.1469e05,-1.3107e05,-1.9661e05",
    help="GPSA,GPSB (8 values) for --atmos both",
)
a = ap.parse_args()
odir, day = a.obs_dir, a.day
ATMOS = "" if a.atmos == "none" else a.atmos
LAT, LON, ALT = a.lat, a.lon, a.alt
_io = [float(x) for x in a.iono.split(",")]
ALPHA, BETA = _io[:4], _io[4:8]
FREQ = {
    "gps_l5": 1176.45e6,
    "gal_e5a": 1176.45e6,
    "gal_e5b": 1207.14e6,
    "bds_b2a": 1176.45e6,
    "bds_b3i": 1268.52e6,
    "gps_l2c": 1227.6e6,
}


def atmos(band, el, az, t):
    d = 0.0
    if ATMOS in ("tropo", "both"):
        d += A.tropo_delay_m(el, LAT, ALT)
    if ATMOS == "both":
        tow = (t - 315964800.0 + 18.0) % 604800.0
        d += A.klobuchar_delay_m(
            ALPHA, BETA, LAT, LON, el, az, tow, FREQ.get(band, A.F_L1)
        )
    return d


for band in a.bands:
    rows = []
    for f in glob.glob("%s/%s_%s.jsonl" % (odir, band, day)):
        for i, raw in enumerate(open(f, "rb")):
            if i % 5:
                continue
            if b'"code_resid_m"' not in raw or b'"el"' not in raw:
                continue
            try:
                r = json.loads(raw)
            except Exception:
                continue
            if r.get("code_resid_src") != "trk" or not r.get("code_resid_clk"):
                continue
            el = r.get("el")
            cr = r.get("code_resid_m")
            if el is None or cr is None or el < 5.0:
                continue
            # cr = measured - model: an unmodelled delay makes it negative; add the model back
            rows.append(
                (el, cr + atmos(band, el, r.get("az", 0.0), r["t"]), r["prn"], r["t"])
            )
    # code_resid_m carries the receiver clock (code_resid_clk): remove the per-10-s median across
    # satellites, keeping buckets with >= 4 PRNs, then the elevation dependence is what is left.
    buckets = {}
    for el, cr, p, t in rows:
        buckets.setdefault(int(t // 10), {})[p] = (el, cr)
    rows = []
    for b, d in buckets.items():
        if len(d) < 4:
            continue
        med = sorted(v[1] for v in d.values())[len(d) // 2]
        for p, (el, cr) in d.items():
            if abs(cr - med) < 100.0:
                rows.append((el, cr - med, p))
    if len(rows) < 50:
        print("%s: %d rows after clock removal, skipped" % (band, len(rows)))
        continue
    # elevation bins
    bins = {}
    for el, cr, p in rows:
        bins.setdefault(int(el // 10) * 10, []).append(cr)
    print(
        "== %s [ATMOS=%s]: %d rows, %d PRNs"
        % (band, ATMOS or "none", len(rows), len({p for _, _, p in rows}))
    )
    print("   el-bin   n     median   p25     p75")
    for b in sorted(bins):
        v = sorted(bins[b])
        n = len(v)
        print(
            "   %2d-%2d %6d  %+7.2f %+7.2f %+7.2f"
            % (b, b + 10, n, v[n // 2], v[n // 4], v[3 * n // 4])
        )
    # least squares a + b/sin + c*sin on the raw rows (robust-ish: iterate with a 3-sigma clip)
    import numpy as np  # noqa: E402

    el = np.radians(np.array([r[0] for r in rows]))
    y = np.array([r[1] for r in rows])
    A = np.column_stack([np.ones_like(el), 1.0 / np.sin(el), np.sin(el)])
    m = np.ones(len(y), bool)
    for _ in range(3):
        x, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        res = y - A @ x
        s = np.std(res[m])
        m = np.abs(res) < 3 * s
    print(
        "   fit: a %+.2f m, b %+.2f m per 1/sin(el) (tropo-like), c %+.2f m per sin(el) (dU-like); rms %.2f m (%d used)"
        % (x[0], x[1], x[2], s, m.sum())
    )
