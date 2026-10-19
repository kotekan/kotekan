"""gnss_atmos: the troposphere's size and shape, the Klobuchar structure and its header parse."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_atmos as A  # noqa: E402

FAIL = []


N_CHECK = [0]


def check(name, cond, detail=""):
    print("%-60s %s %s" % (name, "ok" if cond else "FAIL", detail))
    N_CHECK[0] += 1
    if not cond:
        FAIL.append(name)


# Troposphere: ~2.3 m at sea level mid-latitude, ~2.2 m at the site (545 m), mapping 1 at
# zenith, ~5.6 at 10 deg, ~10 at 5 deg, monotonic in elevation.
z0 = A.tropo_zenith_m(45.0, 0.0)
check("zenith tropo at sea level 2.25-2.40 m", 2.25 < z0 < 2.40, "%.3f" % z0)
zs = A.tropo_zenith_m(49.32, 545.0)
check("zenith tropo at the site 2.10-2.30 m", 2.10 < zs < 2.30, "%.3f" % zs)
check("zenith tropo falls with height", zs < z0)
check(
    "mapping 1 at zenith",
    abs(A.tropo_mapping(90.0) - 1.0) < 2e-3,
    "%.4f" % A.tropo_mapping(90.0),
)
check(
    "mapping ~5.6 at 10 deg",
    5.4 < A.tropo_mapping(10.0) < 5.9,
    "%.3f" % A.tropo_mapping(10.0),
)
check(
    "mapping ~10 at 5 deg",
    9.5 < A.tropo_mapping(5.0) < 10.8,
    "%.3f" % A.tropo_mapping(5.0),
)
ms = [A.tropo_mapping(e) for e in range(5, 91, 5)]
check("mapping monotonic", all(a > b for a, b in zip(ms, ms[1:])))
check(
    "slant = zenith x mapping",
    abs(A.tropo_delay_m(30.0, 49.32, 545.0) - zs * A.tropo_mapping(30.0)) < 1e-9,
)

# Klobuchar: the IS-GPS-200 structure. Night floor 5 ns x F; day above night; the slant
# factor grows toward the horizon; the frequency scaling is exactly (f_L1/f)^2; the
# clamps on PER and AMP hold.
alpha = [1.3970e-08, 2.2352e-08, -5.9605e-08, -1.1921e-07]  # a real header (2026-08-27)
beta = [1.0854e05, 1.1469e05, -1.3107e05, -1.9661e05]
site = (49.32, -119.62)
# local time at the pierce point ~ 02:00 for a night value: lon -119.6 deg -> lam ~ -0.664 sc
# -> t_local = tow + 43200*lam; pick tow so that t_local is 2 h
tow_night = (2 * 3600.0 - 43200.0 * (-119.62 / 180.0)) % 86400.0
tow_day = (14 * 3600.0 - 43200.0 * (-119.62 / 180.0)) % 86400.0
night = A.klobuchar_delay_m(alpha, beta, *site, 90.0, 0.0, tow_night)
day = A.klobuchar_delay_m(alpha, beta, *site, 90.0, 0.0, tow_day)
f_night = 1.0 + 16.0 * (0.53 - 0.5) ** 3
check(
    "night zenith = 5 ns x F x c at L1",
    abs(night - A.C_LIGHT * 5e-9 * f_night) < 1e-6,
    "%.4f m" % night,
)
check(
    "day zenith above the night floor", day > night + 1.0, "%.3f vs %.3f" % (day, night)
)
lo = A.klobuchar_delay_m(alpha, beta, *site, 10.0, 180.0, tow_day)
check("slant at 10 deg > zenith (day)", lo > day, "%.3f vs %.3f" % (lo, day))
l5 = A.klobuchar_delay_m(alpha, beta, *site, 90.0, 0.0, tow_day, f_hz=1176.45e6)
check(
    "L5 scales by (f_L1/f_L5)^2 = 1.793",
    abs(l5 / day - (1575.42 / 1176.45) ** 2) < 1e-9,
    "%.4f" % (l5 / day),
)
# clamps: a negative-amplitude set is the night floor by day; a short period is 72000 s
neg = [-1e-7, 0.0, 0.0, 0.0]
check(
    "AMP < 0 clamps to the night value",
    abs(A.klobuchar_delay_m(neg, beta, *site, 90.0, 0.0, tow_day) - night) < 1e-6,
)
short = [1.0, 0.0, 0.0, 0.0]  # PER = 1 s -> 72000 s
ref = A.klobuchar_delay_m(alpha, [72000.0, 0.0, 0.0, 0.0], *site, 90.0, 0.0, tow_day)
check(
    "PER < 72000 clamps to 72000",
    abs(A.klobuchar_delay_m(alpha, short, *site, 90.0, 0.0, tow_day) - ref) < 1e-6,
)

# Header parse: the RINEX 3.04 lines of a real file.
hdr = [
    "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n",
    "GPSA   1.3970E-08  2.2352E-08 -5.9605E-08 -1.1921E-07       IONOSPHERIC CORR    \n",
    "GPSB   1.0854E+05  1.1469E+05 -1.3107E+05 -1.9661E+05       IONOSPHERIC CORR    \n",
    "GAL    1.1875E+02  4.6875E-01  1.6541E-02  0.0000E+00       IONOSPHERIC CORR    \n",
    "GPSA   9.9999E-09  0.0000E+00  0.0000E+00  0.0000E+00       IONOSPHERIC CORR    \n",
    "                                                            END OF HEADER       \n",
    "GPSA   0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00       IONOSPHERIC CORR    \n",
]
ic = A.parse_iono_corr(hdr)
check("GPSA parsed", ic.get("GPSA") == alpha, str(ic.get("GPSA")))
check("GPSB parsed", ic.get("GPSB") == beta)
check(
    "GAL parsed (NeQuick, stored only)",
    ic.get("GAL") == [118.75, 0.46875, 0.016541, 0.0],
)
check("first occurrence wins, nothing past END OF HEADER", ic["GPSA"][0] == 1.3970e-08)

# iono_params: the first file with the lines wins and refreshes the cache; the cache serves,
# with its age, when no file has them; nothing serves None.
import gzip  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

_td = tempfile.mkdtemp()
_with = os.path.join(_td, "a_MN.rnx.gz")
_without = os.path.join(_td, "b_MN.rnx.gz")
with gzip.open(_with, "wt") as f:
    f.write("".join(hdr[:6]))
with gzip.open(_without, "wt") as f:
    f.write(hdr[0] + hdr[5])
r1 = A.iono_params([_without, _with], cache_dir=_td)
check(
    "iono_params: first file with GPSA/GPSB wins",
    r1 and r1["alpha"] == alpha and r1["src"] == "a_MN.rnx.gz",
)
check(
    "iono_params: the cache was written",
    os.path.exists(os.path.join(_td, "iono_corr.json")),
)
r2 = A.iono_params([_without], cache_dir=_td)
check(
    "iono_params: no file -> the cached set, aged",
    r2
    and r2["beta"] == beta
    and r2["src"].startswith("cache:")
    and r2["age_days"] < 1.0,
)
check(
    "iono_params: nothing anywhere -> None",
    A.iono_params([_without], cache_dir="") is None,
)

# predict_all carries the delays as separate fields, off by default, on request.
import glob  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gnss_ephemeris as E  # noqa: E402

FIX = os.path.join(
    os.environ.get("GNSS_FIXTURES", "/home/kvand/gnss/fixtures"), "brdc_merge_golden"
)
navs = sorted(glob.glob(os.path.join(FIX, "*_MN.rnx.gz")))
check("brdc_merge_golden fixture present (a missing fixture is a failure)", bool(navs))
if navs:
    eph = E.parse_rinex_nav(navs[:1])
    when = datetime(2026, 8, 27, 19, 30, tzinfo=timezone.utc)
    lat, lon, alt = 49.32, -119.62, 545.0
    off = E.predict_all(eph, lat, lon, alt, when, mask_deg=5.0)
    on = E.predict_all(
        eph,
        lat,
        lon,
        alt,
        when,
        mask_deg=5.0,
        tropo=True,
        iono=(alpha, beta, 1176.45e6),
    )
    check(
        "predict_all: satellites above 5 deg at the fixture epoch",
        len(off) >= 4,
        "%d" % len(off),
    )
    check(
        "predict_all: delays 0.0 by default",
        all(v["tropo_m"] == 0.0 and v["iono_m"] == 0.0 for v in off.values()),
    )
    same = all(
        off[k]["range_m"] == on[k]["range_m"]
        and off[k]["sat_clk_s"] == on[k]["sat_clk_s"]
        for k in off
    )
    check("predict_all: geometry and clock untouched by the atmosphere", same)
    t = E.gpst_of_utc(when) % 604800.0
    ok_t = all(
        abs(v["tropo_m"] - A.tropo_delay_m(v["el"], lat, alt)) < 1e-9
        for v in on.values()
    )
    ok_i = all(
        abs(
            v["iono_m"]
            - A.klobuchar_delay_m(alpha, beta, lat, lon, v["el"], v["az"], t, 1176.45e6)
        )
        < 1e-9
        for v in on.values()
    )
    check("predict_all: tropo_m is the Saastamoinen slant delay per row", ok_t)
    check("predict_all: iono_m is the Klobuchar delay at the carrier per row", ok_i)
    lo = min(on.values(), key=lambda v: v["el"])
    check(
        "predict_all: tropo larger at the lowest satellite than at zenith value",
        lo["tropo_m"] > A.tropo_zenith_m(lat, alt),
    )

if FAIL:
    print("FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
    sys.exit(1)
print("ALL PASS (%d checks)" % N_CHECK[0])
