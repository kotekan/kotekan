#!/usr/bin/env python3
"""Satellite transits through CHORD's beam over the next hour, and the quiet windows between them.

For choosing when to measure the F-engine's quiet-sky gain baseline: a transit is a GNSS or
GuoWang satellite within --transit-deg of the boresight; a quiet window is a stretch of at
least --min-quiet-s with no GNSS satellite within --quiet-deg, no GuoWang satellite within
--guowang-quiet-deg, the Sun further than --sun-deg and no bright radio source within
--source-deg.

    transit_windows.py [--start 2026-10-30T18:00] [--hours 1] [--by-band] [--fetch]

TLEs: GPS, Galileo, BeiDou and GLONASS from ~/.cache/kotekan_gps/tle_<group>.txt (kept fresh by
tle_refresh.py's cron), GuoWang from the newest ~/.cache/guowang/hulianwang_*.tle plus --extra-tle
(the uncatalogued GuoWang sets fitted on 2026-10-29). --fetch downloads the GuoWang group again
when the newest cache is older than 2 h (CelesTrak serves a group once per 2 h per client).
GNSS TLEs a few days old place a satellite to ~0.05 deg; GuoWang TLEs drift ~1 s along track per
day, so --fetch before a capture. The Sun and the radio sources use low-precision formulas
(~0.01 deg, with first-order precession from J2000 for the sources): enough to flag them.

Needs skyfield (the gnss venv: /home/kvand/gnss/venv/bin/python).
"""
import argparse
import datetime as dt
import glob
import math
import os
import sys
import time
import urllib.request

import numpy as np

SITE = dict(lat=49.3199845, lon=-119.62262691, h=528.4)
GNSS_TLE = os.path.expanduser("~/.cache/kotekan_gps")
GROUPS = [("gps-ops", "G"), ("galileo", "E"), ("beidou", "C"), ("glo-ops", "R")]
GW_CACHE = os.path.expanduser("~/.cache/guowang")
GW_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=hulianwang&FORMAT=tle"
GW_HIDDEN = "/home/kvand/gnss/builds/scratch_bak_20261023_gnssmain/taua1029/hidden_guowang_20261029.tle"
SOURCES = {
    "Tau A": (83.633, 22.014),
    "Cyg A": (299.868, 40.734),
    "Cas A": (350.850, 58.815),
    "Vir A": (187.706, 12.391),
}
# The F-engine lobes each constellation lights (GuoWang: 1202 and 1258-1279 MHz).
BANDS = [
    ("L5/E5a/B2a 1166-1187", "GEC"),
    ("E5b/B2b/L3 1197-1217", "ECRW"),
    ("L2 1217-1238", "G"),
    ("GLO L2 1238-1258", "R"),
    ("B3I/E6 1258-1284", "ECW"),
]


def parse_time(s):
    if not s:
        return math.floor(time.time())
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
        try:
            return (
                dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc).timestamp()
            )
        except ValueError:
            pass
    raise SystemExit("unreadable time %r (YYYY-MM-DDTHH:MM, UTC)" % s)


def hms(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S")


def read_tles(path):
    """(name, line1, line2) triples of a CelesTrak FORMAT=tle file."""
    lines = [x.rstrip() for x in open(path) if x.strip()]
    out = []
    for i in range(len(lines) - 2):
        if (
            lines[i + 1].startswith("1 ")
            and lines[i + 2].startswith("2 ")
            and not lines[i].startswith(("1 ", "2 "))
        ):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
    return out


def short_name(name, sys_):
    """G22, C06, ... where the CelesTrak name carries the PRN; the plain name otherwise."""
    if sys_ == "G" and "PRN" in name:
        return "G%02d" % int(name.split("PRN")[1].strip(" )"))
    if sys_ == "C" and name.endswith(")") and "(C" in name:
        return name[name.rindex("(") + 1 : -1]
    if sys_ == "W":
        return name.replace("HULIANWANG ", "")
    if sys_ in "ER" and "(" in name:
        return name[: name.index("(")].strip()
    return name


def newest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


def fetch_guowang():
    p = newest(os.path.join(GW_CACHE, "hulianwang_*.tle"))
    if p and time.time() - os.path.getmtime(p) < 7200.0:
        return p
    os.makedirs(GW_CACHE, exist_ok=True)
    with urllib.request.urlopen(GW_URL, timeout=30) as r:
        data = r.read().decode()
    if data.count("\n1 ") < 10:
        raise SystemExit(
            "CelesTrak answered without TLEs (once per 2 h per client): %r" % data[:200]
        )
    out = os.path.join(
        GW_CACHE, "hulianwang_%s.tle" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    with open(out, "w") as fh:
        fh.write(data)
    return out


def unit(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)], -1)


def precess(ra_deg, dec_deg, t):
    """J2000 RA/Dec to the date at unix time t, first order (good to ~0.01 deg over decades)."""
    yrs = (t / 86400.0 + 2440587.5 - 2451545.0) / 365.25
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    dra = (3.075 + 1.336 * math.sin(ra) * math.tan(dec)) * yrs * 15.0 / 3600.0
    ddec = 20.04 * math.cos(ra) * yrs / 3600.0
    return ra_deg + dra, dec_deg + ddec


def radec_altaz(ra_deg, dec_deg, t):
    """Alt/az (deg) of an RA/Dec of date at unix times t (low precision: ~0.01 deg)."""
    n = t / 86400.0 + 2440587.5 - 2451545.0
    lst = np.radians((280.46061837 + 360.98564736629 * n + SITE["lon"]) % 360.0)
    ha = lst - np.radians(ra_deg)
    lat, dec = math.radians(SITE["lat"]), np.radians(dec_deg)
    el = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(
        -np.cos(dec) * np.sin(ha),
        np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(ha),
    )
    return np.degrees(az) % 360.0, np.degrees(el)


def sun_radec(t):
    n = t / 86400.0 + 2440587.5 - 2451545.0
    g = np.radians(357.528 + 0.9856003 * n)
    lam = np.radians(
        280.460 + 0.9856474 * n + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    )
    eps = np.radians(23.439 - 4e-7 * n)
    return (
        np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))) % 360.0,
        np.degrees(np.arcsin(np.sin(eps) * np.sin(lam))),
    )


def separations(sats, t, bore):
    """Boresight separation (deg) per satellite and second: array (n_sat, n_t)."""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)
    tt = ts.from_datetimes([dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in t])
    site = wgs84.latlon(SITE["lat"], SITE["lon"], elevation_m=SITE["h"])
    out = np.empty((len(sats), len(t)))
    for i, (name, l1, l2) in enumerate(sats):
        alt, az, _ = (EarthSatellite(l1, l2, name, ts) - site).at(tt).altaz()
        out[i] = np.degrees(
            np.arccos(np.clip(unit(az.degrees, alt.degrees) @ bore, -1.0, 1.0))
        )
    return out


def runs(mask):
    """(first, last) index pairs of the True runs of a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", help="UTC, YYYY-MM-DDTHH:MM (default: now)")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--az", type=float, default=180.0, help="boresight azimuth, deg")
    ap.add_argument(
        "--el",
        type=float,
        default=62.70,
        help="boresight elevation, deg (Tau A pointing)",
    )
    ap.add_argument(
        "--transit-deg",
        type=float,
        default=5.0,
        help="a GNSS transit: closer than this",
    )
    ap.add_argument(
        "--quiet-deg", type=float, default=8.0, help="quiet: no GNSS satellite closer"
    )
    ap.add_argument("--guowang-transit-deg", type=float, default=4.0)
    ap.add_argument("--guowang-quiet-deg", type=float, default=8.0)
    ap.add_argument("--sun-deg", type=float, default=20.0)
    ap.add_argument("--source-deg", type=float, default=3.0)
    ap.add_argument("--min-quiet-s", type=float, default=60.0)
    ap.add_argument(
        "--extra-tle",
        nargs="*",
        default=[GW_HIDDEN] if os.path.exists(GW_HIDDEN) else [],
        help="more GuoWang TLE files (default: the 2026-10-29 uncatalogued sets)",
    )
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="refresh the GuoWang TLEs (at most every 2 h)",
    )
    ap.add_argument(
        "--by-band", action="store_true", help="also quiet windows per F-engine lobe"
    )
    args = ap.parse_args(argv)

    t0 = parse_time(args.start)
    t = t0 + np.arange(0.0, args.hours * 3600.0, 1.0)
    bore = unit(args.az, args.el)
    sats, sysof, ages = [], [], []
    for group, s in GROUPS:
        p = os.path.join(GNSS_TLE, "tle_%s.txt" % group)
        if os.path.exists(p):
            tl = read_tles(p)
            sats += tl
            sysof += [s] * len(tl)
            ages.append("%s %.1f d" % (group, (t0 - os.path.getmtime(p)) / 86400.0))
    gw = (
        fetch_guowang()
        if args.fetch
        else newest(os.path.join(GW_CACHE, "hulianwang_*.tle"))
    )
    for p in ([gw] if gw else []) + list(args.extra_tle):
        tl = [x for x in read_tles(p) if "DIGUI" in x[0]]
        sats += tl
        sysof += ["W"] * len(tl)
        ages.append(
            "%s %.1f d" % (os.path.basename(p), (t0 - os.path.getmtime(p)) / 86400.0)
        )
    if not sats:
        raise SystemExit("no TLEs found under %s or %s" % (GNSS_TLE, GW_CACHE))
    sysof = np.array(sysof)
    sep = separations(sats, t, bore)
    gw_rows = sysof == "W"

    sky = np.zeros(t.size, bool)
    notes = []
    ra, dec = sun_radec(t)
    saz, sel = radec_altaz(ra, dec, t)
    ssep = np.degrees(np.arccos(np.clip(unit(saz, sel) @ bore, -1, 1)))
    sky |= ssep < args.sun_deg
    notes.append(
        "Sun %.0f..%.0f deg from boresight (el %.0f..%.0f)"
        % (ssep.min(), ssep.max(), sel.min(), sel.max())
    )
    src_rows = []
    for name, (r, d) in SOURCES.items():
        a, e = radec_altaz(*precess(r, d, t0), t)
        s = np.degrees(np.arccos(np.clip(unit(a, e) @ bore, -1, 1)))
        near = s < args.source_deg
        sky |= near
        for i0, i1 in runs(near):
            k = i0 + int(np.argmin(s[i0 : i1 + 1]))
            src_rows.append((t[i0], t[i1], name, s[k], t[k]))

    print(
        "CHORD boresight az %.2f el %.2f; %s .. %s UTC (%s)"
        % (
            args.az,
            args.el,
            dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            hms(t[-1] + 1),
            ", ".join(ages),
        )
    )
    print(
        "transit: GNSS within %.1f deg, GuoWang within %.1f deg; quiet: no GNSS within %.1f, no GuoWang within "
        "%.1f, Sun beyond %.0f, no source within %.0f, for >= %.0f s; %s"
        % (
            args.transit_deg,
            args.guowang_transit_deg,
            args.quiet_deg,
            args.guowang_quiet_deg,
            args.sun_deg,
            args.source_deg,
            args.min_quiet_s,
            notes[0],
        )
    )
    rows = []
    for i in range(len(sats)):
        w = sysof[i] == "W"
        lim = args.guowang_transit_deg if w else args.transit_deg
        qlim = args.guowang_quiet_deg if w else args.quiet_deg
        # every pass inside the quiet radius: a transit if it comes within lim, else "near"
        for i0, i1 in runs(sep[i] < max(lim, qlim)):
            k = i0 + int(np.argmin(sep[i, i0 : i1 + 1]))
            if sep[i, k] < lim:
                a, b = runs(sep[i, i0 : i1 + 1] < lim)[0]
                a, b, kind = i0 + a, i0 + b, "closest"
            else:
                a, b, kind = i0, i1, "near   "
            bands = [b_ for b_, members in BANDS if sysof[i] in members]
            rows.append(
                (
                    t[a],
                    t[b],
                    short_name(sats[i][0], sysof[i]),
                    sep[i, k],
                    t[k],
                    bands,
                    kind,
                )
            )
    for r in src_rows:
        rows.append(
            (r[0], r[1], r[2], r[3], r[4], ["all (continuum source)"], "closest")
        )
    print("\nTRANSITS ('near' = inside the quiet radius only)")
    if not rows:
        print("  none")
    for a, b, name, s, tk, bands, kind in sorted(rows):
        print(
            "  %s-%s  %-13s %s %5.2f deg at %s  %s"
            % (hms(a), hms(b + 1), name, kind, s, hms(tk), ", ".join(bands))
        )

    def quiet_mask(members):
        rows_ = np.isin(sysof, list(members.replace("W", ""))) & ~gw_rows
        m = ~sky.copy()
        if rows_.any():
            m &= sep[rows_].min(0) >= args.quiet_deg
        if "W" in members and gw_rows.any():
            m &= sep[gw_rows].min(0) >= args.guowang_quiet_deg
        return m

    def show(mask, indent="  "):
        n = 0
        for i0, i1 in runs(mask):
            if i1 - i0 + 1 >= args.min_quiet_s:
                print(
                    "%s%s-%s  %5.1f min"
                    % (indent, hms(t[i0]), hms(t[i1] + 1), (i1 - i0 + 1) / 60.0)
                )
                n += i1 - i0 + 1
        if n == 0:
            print(indent + "none")
        return n

    print("\nQUIET WINDOWS (all lobes)")
    n = show(quiet_mask("GECRW"))
    print("  quiet %.0f %% of the window" % (100.0 * n / t.size))
    if args.by_band:
        for name, members in BANDS:
            print(
                "\nQUIET in %s (%s)"
                % (
                    name,
                    ", ".join(
                        {
                            "G": "GPS",
                            "E": "Galileo",
                            "C": "BeiDou",
                            "R": "GLONASS",
                            "W": "GuoWang",
                        }[m]
                        for m in members
                    ),
                )
            )
            n = show(quiet_mask(members))
            print("  quiet %.0f %% of the window" % (100.0 * n / t.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
