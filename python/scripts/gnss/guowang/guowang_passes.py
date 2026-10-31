#!/usr/bin/env python3
"""GuoWang (CelesTrak GROUP=hulianwang, "HULIANWANG DIGUI-nn") pass planner for CHORD (#218).

The 1268.52 / 1202.025 MHz bursts are DIGUI satellites crossing the dish beam (az 180, el 62.70,
FWHM ~2.7 deg at 1268 MHz) at ~0.3 deg/s. This lists every pass whose closest approach to the
boresight is within --max-sep, with the predicted Doppler and Doppler rate at both carriers, and
groups the passes into in-plane trains (consecutive satellites of one orbital plane, ~12.7 min
apart, the middle one closest and strongest).

    passes      every pass within --max-sep of boresight in [--start, --start + --days)
    capture     proposed start times for a --length s capture centred on each train's best pass
    validate    observed burst list (burstspec RESULTS file) against the predictions

TLEs: --tle FILE, else the newest cache in --cache-dir. --fetch downloads the group from CelesTrak
into the cache, but only when the newest cache is older than 2 h: CelesTrak serves a GROUP once per
2 h per client and answers sooner requests with an error, so a test download burns the slot.
Refresh the TLEs the day before a capture; 1-2 day old TLEs are good to ~1 s along track.

    python guowang_passes.py passes --start 2026-10-29 --days 7
    python guowang_passes.py capture --start 2026-10-29 --days 7 --length 80
    python guowang_passes.py validate --bursts .../RESULTS_burstspec_20261028.txt --tle .../g_hulianwang.tle

Needs skyfield (the gnss venv: /home/kvand/gnss/venv/bin/python).
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys
import time
import urllib.request

import numpy as np

C = 299792458.0
F_B3 = 1268.52e6
F_L3 = 1202.025e6
SITE = dict(lat=49.3199845, lon=-119.62262691, h=528.4)
BORESIGHT = dict(az=180.0, el=62.70)
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP=hulianwang&FORMAT=tle"
CACHE_DIR = os.path.expanduser("~/.cache/guowang")
FALLBACK_TLE = (
    "/home/kvand/gnss/builds/scratch_bak_20261023_gnssmain/emit1027/g_hulianwang.tle"
)
# Satellites seen NOT to transmit while inside the dish beam (10-26 subset archive + rail recorder;
# burstspec1028 / tier2_1028): both from the first polar launch, 2024-240 (DIGUI-01..10). Later polar
# satellites (DIGUI-59, -83, -115) do transmit. capture skips them unless --include-silent.
SILENT = {
    "HULIANWANG DIGUI-01": "silent 10-26 13:20Z at 0.1-0.8 deg",
    "HULIANWANG DIGUI-03": "silent 10-26 01:49Z at 0.6-0.8 deg",
}
SUSPECT_BATCHES = {
    "2024-240": "first polar launch: DIGUI-01 and -03 silent in the beam"
}


# --------------------------------------------------------------------------- TLEs


def newest_cache(cache_dir):
    files = sorted(glob.glob(os.path.join(cache_dir, "hulianwang_*.tle")))
    return files[-1] if files else None


def fetch_tle(cache_dir, min_age_s=7200.0):
    """Download the group into cache_dir/hulianwang_YYYYmmddTHHMMSSZ.tle unless the newest cache is
    younger than min_age_s. Returns the path used."""
    os.makedirs(cache_dir, exist_ok=True)
    last = newest_cache(cache_dir)
    if last is not None and time.time() - os.path.getmtime(last) < min_age_s:
        age = (time.time() - os.path.getmtime(last)) / 60.0
        print(
            "# cache %s is %.0f min old (< %.0f min): not fetching"
            % (last, age, min_age_s / 60),
            file=sys.stderr,
        )
        return last
    with urllib.request.urlopen(CELESTRAK, timeout=60) as r:
        txt = r.read().decode("ascii", "replace")
    if "1 " not in txt or "2 " not in txt:
        raise RuntimeError("CelesTrak returned no TLEs: %r" % txt[:200])
    path = os.path.join(
        cache_dir, time.strftime("hulianwang_%Y%m%dT%H%M%SZ.tle", time.gmtime())
    )
    with open(path, "w") as f:
        f.write(txt)
    print("# fetched %s" % path, file=sys.stderr)
    return path


def read_tle(path, ts, name_filter="DIGUI"):
    """EarthSatellite list, one per catalogue number, names containing name_filter ('' = all)."""
    from skyfield.api import EarthSatellite

    lines = [l.rstrip() for l in open(path).read().splitlines() if l.strip()]
    sats, seen = [], set()
    i = 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
            i += 3
            if l1[2:7] in seen or (name_filter and name_filter not in name):
                continue
            seen.add(l1[2:7])
            sats.append(EarthSatellite(l1, l2, name, ts))
        else:
            i += 1
    return sats


# --------------------------------------------------------------------------- geometry


def ts_unix(ts, unix):
    """skyfield Time from unix seconds. ts.utc(1970, 1, 1, 0, 0, unix) is NOT this: skyfield takes
    the leap-second offset of the date given (1970: 10 s), so it lands 27 s early in 2026."""
    unix = np.asarray(unix, float)
    day = np.floor(unix / 86400.0)
    return ts.utc(1970, 1, 1 + day, 0, 0, unix - day * 86400.0)


def enu_unit(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)], -1)


class Geometry:
    def __init__(self, ts, site=SITE, az=BORESIGHT["az"], el=BORESIGHT["el"]):
        from skyfield.api import wgs84

        self.ts = ts
        self.site = wgs84.latlon(site["lat"], site["lon"], elevation_m=site["h"])
        self.b = enu_unit(az, el)

    def topo(self, sat, unix):
        """(unit ENU vectors [n,3], range m [n]) at unix seconds (array)."""
        t = ts_unix(self.ts, unix)
        alt, az, dist = (sat - self.site).at(t).altaz()
        return enu_unit(az.degrees, alt.degrees), dist.m

    def sep(self, u):
        return np.degrees(np.arccos(np.clip(u @ self.b, -1.0, 1.0)))


def orbit_normal(sat, ts, unix):
    """Unit orbit normal (GCRS) at one time: the plane label."""
    t = ts_unix(ts, unix)
    g = sat.at(t)
    h = np.cross(g.position.km, g.velocity.km_per_s)
    return h / np.linalg.norm(h)


def batch(sat):
    """COSPAR launch of a satellite ('24240A' -> '2024-240')."""
    d = sat.model.intldesg.strip()
    return (
        ("19" if int(d[:2]) >= 57 else "20") + d[:2] + "-" + d[2:5]
        if len(d) >= 5
        else "?"
    )


def flag(p):
    """'' / 'SILENT' / 'suspect' for a pass, from SILENT and SUSPECT_BATCHES."""
    if p["name"] in SILENT:
        return "SILENT"
    return "suspect" if p["batch"] in SUSPECT_BATCHES else ""


def cluster_planes(sats, ts, unix, tol_deg=2.0):
    """Plane index per satellite: orbit normals within tol_deg share a plane (greedy)."""
    normals = np.array([orbit_normal(s, ts, unix) for s in sats])
    plane = -np.ones(len(sats), int)
    centres = []
    for i, n in enumerate(normals):
        for k, c in enumerate(centres):
            if np.degrees(np.arccos(np.clip(n @ c, -1, 1))) < tol_deg:
                plane[i] = k
                break
        else:
            plane[i] = len(centres)
            centres.append(n)
    incl = [np.degrees(np.arccos(c[2])) for c in centres]
    return plane, incl


def find_passes(sats, geo, t0, t1, coarse=10.0, near_deg=10.0, fine=0.25):
    """Local minima of the boresight separation below near_deg, refined on a fine grid.

    Returns a list of dicts sorted by time of closest approach (tca, unix s)."""
    out = []
    tc = np.arange(t0, t1 + coarse, coarse)
    for si, s in enumerate(sats):
        u, _ = geo.topo(s, tc)
        d = geo.sep(u)
        # local minima on the coarse grid (endpoints included), below near_deg + margin
        lm = np.where(
            (d < near_deg + 5.0)
            & (d <= np.r_[np.inf, d[:-1]])
            & (d <= np.r_[d[1:], np.inf])
        )[0]
        for k in lm:
            tf = np.arange(tc[k] - 2 * coarse, tc[k] + 2 * coarse + fine, fine)
            uf, rf = geo.topo(s, tf)
            df = geo.sep(uf)
            j = int(np.argmin(df))
            if df[j] > near_deg or j in (0, len(tf) - 1):
                continue
            # parabolic refinement of the minimum time
            y0, y1, y2 = df[j - 1] ** 2, df[j] ** 2, df[j + 1] ** 2
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den > 0 else 0.0
            tca = tf[j] + frac * fine
            out.append(
                dict(
                    sat=si,
                    name=s.name,
                    tca=tca,
                    incl=float(np.degrees(s.model.inclo)),
                    batch=batch(s),
                )
            )
    # derived quantities at tca: separation, az/el, rates, Doppler
    for p in out:
        s = sats[p["sat"]]
        h = 1.0
        tt = p["tca"] + np.array([-2 * h, -h, 0.0, h, 2 * h])
        u, r = geo.topo(s, tt)
        sep = geo.sep(u)
        e, n, z = u[2]
        p["sep"] = float(sep[2])
        p["el"] = float(np.degrees(np.arcsin(z)))
        p["az"] = float(np.degrees(np.arctan2(e, n)) % 360.0)
        p["range_km"] = float(r[2] / 1e3)
        du = (u[3] - u[1]) / (2 * h)
        p["rate_dps"] = float(np.degrees(np.linalg.norm(du)))
        # heading in the (l=E, m=N) direction-cosine plane, deg from +m toward +l (nearfield1028's)
        p["heading"] = float(np.degrees(np.arctan2(du[0], du[1])) % 360.0)
        rr = (r[3] - r[1]) / (2 * h)  # m/s
        ra = (r[4] - 2 * r[2] + r[0]) / (2 * h) ** 2  # m/s^2 (step 2h, quieter)
        p["range_rate"] = float(rr)
        for tag, f in (("b3", F_B3), ("l3", F_L3)):
            p["dop_" + tag] = float(-f * rr / C)
            p["dopdot_" + tag] = float(-f * ra / C)
    out.sort(key=lambda p: p["tca"])
    return out


def assign_trains(passes, plane, gap_max_s=20 * 60.0):
    """Train = consecutive passes of one plane separated by < gap_max_s. Adds plane, train, k, n,
    best (True on the train's closest pass)."""
    by_plane = {}
    for p in passes:
        p["plane"] = int(plane[p["sat"]])
        by_plane.setdefault(p["plane"], []).append(p)
    tid = 0
    trains = []
    for pl in sorted(by_plane):
        ps = sorted(by_plane[pl], key=lambda p: p["tca"])
        cur = [ps[0]]
        for p in ps[1:]:
            if p["tca"] - cur[-1]["tca"] < gap_max_s:
                cur.append(p)
            else:
                trains.append(cur)
                cur = [p]
        trains.append(cur)
    trains.sort(key=lambda tr: tr[0]["tca"])
    for tid, tr in enumerate(trains):
        best = min(tr, key=lambda p: p["sep"])
        for k, p in enumerate(tr):
            p["train"] = tid
            p["k"] = k + 1
            p["n"] = len(tr)
            p["best"] = p is best
            p["spacing_min"] = (
                (p["tca"] - tr[k - 1]["tca"]) / 60.0 if k > 0 else float("nan")
            )
    return trains


# --------------------------------------------------------------------------- output


def iso(unix, frac=False):
    t = dt.datetime.fromtimestamp(unix, dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S") + (
        ".%01d" % int((unix % 1) * 10) if frac else ""
    )


HDR = (
    "tca (UTC)              sat            sep   el    az    rate  head  range  "
    "dop1268   dd1268  dop1202   dd1202  plane train k/n  dt(min)  batch"
)
UNITS = (
    "                                      deg   deg   deg   d/s   deg   km     "
    "Hz        Hz/s    Hz        Hz/s"
)


def fmt_pass(p):
    return (
        "%s  %-13s %5.2f %5.1f %5.1f %5.3f %5.0f %6.0f %+8.0f %+7.1f %+8.0f %+7.1f  %4d %5d %d/%d %s%s  %s %s"
        % (
            iso(p["tca"], True),
            p["name"].replace("HULIANWANG ", ""),
            p["sep"],
            p["el"],
            p["az"],
            p["rate_dps"],
            p["heading"],
            p["range_km"],
            p["dop_b3"],
            p["dopdot_b3"],
            p["dop_l3"],
            p["dopdot_l3"],
            p["plane"],
            p["train"],
            p["k"],
            p["n"],
            "  %5.1f" % p["spacing_min"] if np.isfinite(p["spacing_min"]) else "     -",
            "  *" if p["best"] else "   ",
            p["batch"],
            flag(p),
        )
    )


def parse_time(s):
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return (
                dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc).timestamp()
            )
        except ValueError:
            pass
    raise ValueError("bad time %r (use YYYY-mm-dd[THH:MM[:SS]], UTC)" % s)


def setup(args):
    from skyfield.api import load

    ts = load.timescale()
    path = args.tle
    if path is None:
        path = (
            fetch_tle(args.cache_dir)
            if args.fetch
            else (newest_cache(args.cache_dir) or FALLBACK_TLE)
        )
    sats = read_tle(path, ts, args.name_filter)
    epochs = np.array([s.epoch.utc_datetime().timestamp() for s in sats])
    print(
        "# TLE %s: %d objects matching %r, epochs %s .. %s"
        % (path, len(sats), args.name_filter, iso(epochs.min()), iso(epochs.max()))
    )
    geo = Geometry(ts, az=args.az, el=args.el)
    print(
        "# site %.7f N %.8f E %.1f m; boresight az %.2f el %.2f"
        % (SITE["lat"], SITE["lon"], SITE["h"], args.az, args.el)
    )
    return ts, sats, geo, epochs


def compute(args, t0, t1):
    ts, sats, geo, epochs = setup(args)
    age = (t0 - epochs.max()) / 86400.0
    if age > 3:
        print(
            "# WARNING: the window starts %.1f days after the newest TLE epoch: refresh (--fetch)"
            % age
        )
    tw = time.time()
    passes = find_passes(sats, geo, t0, t1, coarse=args.coarse, near_deg=args.near)
    plane, incl = cluster_planes(sats, ts, t0)
    trains = assign_trains(passes, plane) if passes else []
    print(
        "# %d passes within %.0f deg in %s .. %s (%.1f s); %d planes, inclinations %s"
        % (
            len(passes),
            args.near,
            iso(t0),
            iso(t1),
            time.time() - tw,
            len(incl),
            ", ".join(sorted({"%.1f" % i for i in incl})),
        )
    )
    return sats, passes, trains, plane, incl


# --------------------------------------------------------------------------- modes


def cmd_passes(args):
    t0 = parse_time(args.start)
    t1 = t0 + args.days * 86400.0
    sats, passes, trains, plane, incl = compute(args, t0, t1)
    sel = [p for p in passes if p["sep"] <= args.max_sep]
    print(
        "# %d passes within %.1f deg of boresight; '*' = the closest pass of its train"
        % (len(sel), args.max_sep)
    )
    print(
        "# dop = -f * range_rate / c (geometric; the transmitter's own offset is unknown); dd = its rate"
    )
    print(HDR)
    print(UNITS)
    for p in sel:
        print(fmt_pass(p))


def cmd_capture(args):
    t0 = parse_time(args.start)
    t1 = t0 + args.days * 86400.0
    sats, passes, trains, plane, incl = compute(args, t0, t1)
    L = args.length
    rows = []
    for tr in trains:
        live = [p for p in tr if args.include_silent or flag(p) != "SILENT"]
        if not live:
            continue
        best = min(live, key=lambda p: p["sep"])
        if best["sep"] > args.max_sep:
            continue
        start = np.floor(best["tca"] - L / 2.0)
        others = [
            p
            for p in passes
            if p is not best
            and start - 30 <= p["tca"] <= start + L + 30
            and p["sep"] <= args.near
        ]
        rows.append((best, start, tr, others))
    rows.sort(key=lambda r: r[0]["tca"])
    print(
        "# capture windows: %.0f s centred on the closest pass of each train with best sep <= %.1f deg (%d)"
        % (L, args.max_sep, len(rows))
    )
    print(
        "# the burst (> 2x) lasts ~25-50 s around tca at sep < ~2 deg; the train's other passes are ~12.7 min away"
    )
    print(
        "# known-silent satellites (%s) are %s; '?' = a suspect batch (%s)"
        % (
            ", ".join(k.replace("HULIANWANG ", "") for k in SILENT),
            "kept" if args.include_silent else "skipped",
            "; ".join("%s: %s" % kv for kv in SUSPECT_BATCHES.items()),
        )
    )
    print(
        "start (UTC)          end       sat           incl  sep   el    rate  dop1268  dd1268  dop1202  dd1202  train(n)  batch     others in window"
    )
    for best, start, tr, others in rows:
        print(
            "%s  %s  %-13s %4.1f %5.2f %5.1f %5.3f %+8.0f %+7.1f %+8.0f %+7.1f  %4d(%d)  %-8s%s  %s"
            % (
                iso(start),
                iso(start + L)[11:],
                best["name"].replace("HULIANWANG ", ""),
                best["incl"],
                best["sep"],
                best["el"],
                best["rate_dps"],
                best["dop_b3"],
                best["dopdot_b3"],
                best["dop_l3"],
                best["dopdot_l3"],
                best["train"],
                best["n"],
                best["batch"],
                "?" if flag(best) else " ",
                "; ".join(
                    "%s %.1fdeg @%+.0fs%s"
                    % (
                        o["name"].replace("HULIANWANG ", ""),
                        o["sep"],
                        o["tca"] - best["tca"],
                        " " + flag(o) if flag(o) else "",
                    )
                    for o in others
                )
                or "-",
            )
        )
    if args.per_day:
        print("\n# best %d window(s) per UTC day (smallest sep)" % args.per_day)
        days = {}
        for r in rows:
            days.setdefault(iso(r[0]["tca"])[:10], []).append(r)
        for d in sorted(days):
            for best, start, tr, others in sorted(days[d], key=lambda r: r[0]["sep"])[
                : args.per_day
            ]:
                print(
                    "%s  start %s  %-13s i %.1f %s%s sep %.2f el %.1f dop1268 %+.0f Hz dd %+.1f Hz/s"
                    % (
                        d,
                        iso(start)[11:],
                        best["name"].replace("HULIANWANG ", ""),
                        best["incl"],
                        best["batch"],
                        "?" if flag(best) else " ",
                        best["sep"],
                        best["el"],
                        best["dop_b3"],
                        best["dopdot_b3"],
                    )
                )


# observed bursts: the rail table of RESULTS_burstspec (lines "10-26 00:14:27  47  20.6 ...")
_RAIL = re.compile(
    r"^\s*(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\s+(\d+)\s+([\d.]+)\s+\S+\s*\|.*\|\s*(\S+)\s*$"
)
# rail coverage of that table (RESULTS_burstspec Q1, "Rail, 1 s"): 10-26 00:00-18:32 without
# 11:25-12:05 (dead chains), and 10-27 17:08-24:00
RAIL_COVERAGE = [
    ("2026-10-26T00:00:00", "2026-10-26T11:25:00"),
    ("2026-10-26T12:05:00", "2026-10-26T18:32:00"),
    ("2026-10-27T17:08:00", "2026-10-28T00:00:00"),
]


def read_bursts(path, year=2026):
    out = []
    for line in open(path):
        m = _RAIL.match(line)
        if not m:
            continue
        mo, d, H, M, S = (int(x) for x in m.groups()[:5])
        t = dt.datetime(year, mo, d, H, M, S, tzinfo=dt.timezone.utc).timestamp()
        out.append(
            dict(t=t, dur=int(m.group(6)), peak=float(m.group(7)), kind=m.group(8))
        )
    return out


def cmd_validate(args):
    bursts = read_bursts(args.bursts)
    if not bursts:
        sys.exit("no bursts parsed from %s" % args.bursts)
    cov = [(parse_time(a), parse_time(b)) for a, b in RAIL_COVERAGE]
    t0, t1 = cov[0][0] - 600, cov[-1][1] + 600
    sats, passes, trains, plane, incl = compute(args, t0, t1)
    incl_of = {i: float(np.degrees(s.model.inclo)) for i, s in enumerate(sats)}
    W = args.match_window
    print(
        "# %d observed bursts (rail table, >= 15 s, > 2x) from %s"
        % (len(bursts), args.bursts)
    )
    print(
        "# match: a pass with |tca - burst peak| <= %.0f s (sep <= %.0f deg)"
        % (W, args.near)
    )
    print("\nBURST -> PREDICTED PASS")
    print(
        "burst peak (UTC)      x     type      | sat            dt(s)  sep   el    rate  head  incl  dop1268  dd1268  train k/n"
    )
    seps = []
    for b in bursts:
        cand = [p for p in passes if abs(p["tca"] - b["t"]) <= W]
        p = min(cand, key=lambda p: p["sep"]) if cand else None
        b["pass"] = p
        if p is None:
            print("%s  %5.1f  %-9s | -" % (iso(b["t"]), b["peak"], b["kind"]))
            continue
        p["burst"] = b
        seps.append((p["sep"], b["peak"]))
        print(
            "%s  %5.1f  %-9s | %-13s %+5.0f %5.2f %5.1f %5.3f %5.0f %5.1f %+8.0f %+7.1f %4d %d/%d"
            % (
                iso(b["t"]),
                b["peak"],
                b["kind"],
                p["name"].replace("HULIANWANG ", ""),
                p["tca"] - b["t"],
                p["sep"],
                p["el"],
                p["rate_dps"],
                p["heading"],
                incl_of[p["sat"]],
                p["dop_b3"],
                p["dopdot_b3"],
                p["train"],
                p["k"],
                p["n"],
            )
        )
    nm = sum(1 for b in bursts if b["pass"] is not None)
    print(
        "\n%d of %d bursts (%.0f %%) have a predicted pass within %.0f s"
        % (nm, len(bursts), 100.0 * nm / len(bursts), W)
    )
    dts = np.array([b["pass"]["tca"] - b["t"] for b in bursts if b["pass"] is not None])
    if len(dts):
        print(
            "  tca - burst peak: median %+.1f s, p10/p90 %+.1f/%+.1f s"
            % (np.median(dts), *np.percentile(dts, [10, 90]))
        )
    if seps:
        s = np.array(seps)
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10)):
            m = (s[:, 0] >= lo) & (s[:, 0] < hi)
            if m.any():
                print(
                    "  matched bursts with sep %d-%d deg: %2d, peak x median %.1f (range %.1f-%.1f)"
                    % (
                        lo,
                        hi,
                        m.sum(),
                        np.median(s[m, 1]),
                        s[m, 1].min(),
                        s[m, 1].max(),
                    )
                )
        print("  Spearman rank corr(sep, peak x): %.2f" % _spearman(s[:, 0], s[:, 1]))
    # predicted -> burst, inside the rail coverage
    inside = [p for p in passes if any(a <= p["tca"] <= b for a, b in cov)]
    print(
        "\nPREDICTED PASS -> BURST (inside the rail coverage, %.1f h)"
        % (sum(b - a for a, b in cov) / 3600.0)
    )
    print("sep bin   passes  with a burst  fraction   by inclination (with/total)")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10)):
        ps = [p for p in inside if lo <= p["sep"] < hi]
        if not ps:
            continue
        hit = [p for p in ps if "burst" in p]
        byinc = {}
        for p in ps:
            k = "%.0f" % incl_of[p["sat"]]
            a = byinc.setdefault(k, [0, 0])
            a[1] += 1
            a[0] += "burst" in p
        print(
            "%2d-%2d deg  %5d  %8d      %5.2f      %s"
            % (
                lo,
                hi,
                len(ps),
                len(hit),
                len(hit) / len(ps),
                "  ".join("i%s: %d/%d" % (k, *v) for k, v in sorted(byinc.items())),
            )
        )
    if args.list_misses:
        print("\nPASSES <= %.1f deg WITHOUT A BURST (inside coverage)" % args.max_sep)
        print(HDR)
        for p in inside:
            if p["sep"] <= args.max_sep and "burst" not in p:
                print(fmt_pass(p))
    print(
        "\nPER SATELLITE (inside coverage): passes <= %.1f deg / of them with a burst; bursts matched at any sep"
        % args.max_sep
    )
    hist = {}
    for p in inside:
        h = hist.setdefault(p["name"], [0, 0, 0, incl_of[p["sat"]], p["batch"]])
        if p["sep"] <= args.max_sep:
            h[0] += 1
            h[1] += "burst" in p
        h[2] += "burst" in p
    rows = sorted(hist.items(), key=lambda kv: (-kv[1][2], kv[0]))
    for k, v in rows:
        if v[0] or v[2]:
            print(
                "  %-12s i%.0f %s  %d/%d  %d"
                % (k.replace("HULIANWANG ", ""), v[3], v[4], v[1], v[0], v[2])
            )
    # nearfield1028 tle_match_hlw.py named satellites: does the planner put them at those bursts?
    print(
        "\nNEARFIELD1028 TLE MATCHES (satellite named by the track fit) vs the planner's pass at that burst"
    )
    for when, name in (
        ("2026-10-26T00:14:27", "DIGUI-48"),
        ("2026-10-26T00:26:53", "DIGUI-47"),
        ("2026-10-26T06:17:04", "DIGUI-151"),
        ("2026-10-26T07:17:24", "DIGUI-59"),
        ("2026-10-26T10:23:30", "DIGUI-156"),
    ):
        tb = parse_time(when)
        ps = [p for p in passes if name in p["name"] and abs(p["tca"] - tb) < 600]
        if ps:
            p = min(ps, key=lambda p: abs(p["tca"] - tb))
            print(
                "  %s %-10s planner: tca %+5.0f s, sep %.2f deg, el %.1f, heading %.0f"
                % (when, name, p["tca"] - tb, p["sep"], p["el"], p["heading"])
            )
        else:
            print(
                "  %s %-10s planner: no pass within %.0f deg / 10 min"
                % (when, name, args.near)
            )


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["passes", "capture", "validate"])
    ap.add_argument(
        "--start",
        default=time.strftime("%Y-%m-%d", time.gmtime()),
        help="UTC, YYYY-mm-dd[THH:MM[:SS]]",
    )
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument(
        "--tle", help="TLE file (default: newest cache, else %s)" % FALLBACK_TLE
    )
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="download the group into the cache if it is > 2 h old",
    )
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument(
        "--name-filter",
        default="DIGUI",
        help="substring of the TLE name ('' = every object in the file)",
    )
    ap.add_argument("--az", type=float, default=BORESIGHT["az"])
    ap.add_argument("--el", type=float, default=BORESIGHT["el"])
    ap.add_argument(
        "--max-sep",
        type=float,
        default=3.0,
        help="deg: passes listed / trains proposed",
    )
    ap.add_argument(
        "--near",
        type=float,
        default=10.0,
        help="deg: passes found (for trains and matching)",
    )
    ap.add_argument("--coarse", type=float, default=10.0, help="s: coarse search grid")
    ap.add_argument("--length", type=float, default=80.0, help="capture: s")
    ap.add_argument(
        "--per-day",
        type=int,
        default=0,
        help="capture: also list the N best windows per UTC day",
    )
    ap.add_argument("--bursts", help="validate: RESULTS_burstspec file (rail table)")
    ap.add_argument("--match-window", type=float, default=30.0, help="validate: s")
    ap.add_argument("--list-misses", action="store_true")
    ap.add_argument(
        "--include-silent",
        action="store_true",
        help="capture: keep the known-silent satellites",
    )
    args = ap.parse_args(argv)
    {"passes": cmd_passes, "capture": cmd_capture, "validate": cmd_validate}[args.mode](
        args
    )


if __name__ == "__main__":
    main()
