#!/usr/bin/env python3
"""GuoWang ephemeris table for the broker (#227): the satellites' TLEs propagated off the broker.

GuoWang broadcasts no ephemeris, so its geometry comes from CelesTrak TLEs (group hulianwang)
through SGP4. The broker runs on a free-threaded Python with no SGP4, so this job (the gnss
venv, skyfield) propagates the satellites whose 1202 MHz code is known and writes, per pass that
comes within --sep-deg of the dish boresight, their ITRS position and velocity on a --step-s
grid. gnss_ephemeris.load_gw_table reads the file and interpolates (cubic Hermite: a 10 s step
is good to well under a metre), so predict_all, seeding, the PRN map and the element steering
see GuoWang satellites as constellation W, PRN = G2' phase + 1. Some satellites share a G2'
phase; where their segments overlap, the satellite closer to the boresight keeps the time
(one_per_prn), so the broker sees one satellite per PRN at a time.

The file also carries the passes within --transit-deg of the boresight of every DIGUI satellite,
code known or not, and of the satellites in --extra-tle ("transits", keyed by catalogue number).
Every broker chain adds them to its transit sky (sky.transit_sky): a GuoWang satellite in the
beam rails the shared quantiser like any other bright satellite.

    gw_ephem_table.py [--hours 3] [--sep-deg 25] [--step-s 10] [--transit-deg 10]
                      [--extra-tle PATH] [--out PATH] [--fetch]

Run every ~10 min (a pass is a few minutes; the file always covers the next --hours). The write
is atomic; the broker keeps the last file it read. Passes by satellites whose code is not yet in
the database are listed too ("unknown"), so the beam-dump schedule can target them.

--sweep tables satellites whose code is predicted by the reuse rules but not yet measured
("implied"). With no theta the chain cannot find their code epoch, theta spreading over ~30 us
between satellites while a slot pulls in +-0.5 chip, so their passes are cut into steps
(sweep), each at one theta of --sweep-theta-us; a step on which the chain finds the satellite
gives its theta (builds/gw1202/gw_sweep_harvest.py), which the sweep file then carries.
Which theta a step tries (--sweep-order prior, the default): each satellite's next step takes
the theta it has tried least (a step counts as tried when the harvest's steps.jsonl says the
chain evaluated it in >= 3 cycles; steps already handed out and not yet judged count too), and
among those the one where the known thetas are densest, so a new satellite tries the middle of
the population first. Each step's theta is kept in assign.json beside the sweep file, so every
run of this job gives a step the same theta. --sweep-order cycle is the older stateless cycle.
"""

import argparse
import calendar
import json
import math
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
GNSS = os.path.join(HERE, "..", "..", "..", "python", "scripts", "gnss")
sys.path.insert(0, os.path.join(GNSS, "guowang"))
sys.path.insert(0, GNSS)

import gnss_ephemeris  # noqa: E402
import guowang_codes  # noqa: E402
import guowang_passes as gp  # noqa: E402

SCHEMA = "chord-gnss-gw-ephem/0"
DEFAULT_OUT = "/var/tmp/gnss-guowang/gw_ephem.json"
# TLEs of the inferred uncatalogued GuoWang satellites, read for the transits when present.
HIDDEN_TLE = os.path.join(gp.CACHE_DIR, "hidden_digui.tle")


def short_name(name):
    """'HULIANWANG DIGUI-79' -> 'DIGUI-79', the key of the code database."""
    return name.replace("HULIANWANG", "").strip()


def separation_deg(sat, site, t, bore, el_min=-90.0):
    """Boresight separation (deg) of `sat` at each time of the skyfield Time `t`, 180 where it
    is below el_min. Share one `t` across satellites: skyfield keeps its Earth-orientation
    terms, the bulk of the cost."""
    alt, az, _ = (sat - site).at(t).altaz()
    cosang = gp.enu_unit(az.degrees, alt.degrees) @ bore
    return [
        math.degrees(math.acos(max(-1.0, min(1.0, c)))) if e >= el_min else 180.0
        for c, e in zip(cosang, alt.degrees)
    ]


def segments(unix, sep, sep_deg, pad_s):
    """[(t0, t1, min_sep, t_min)] where sep <= sep_deg, padded by pad_s and merged."""
    out = []
    i, n = 0, len(unix)
    while i < n:
        if sep[i] > sep_deg:
            i += 1
            continue
        j = i
        while j + 1 < n and sep[j + 1] <= sep_deg:
            j += 1
        k = min(range(i, j + 1), key=lambda m: sep[m])
        t0, t1 = unix[i] - pad_s, unix[j] + pad_s
        if out and t0 <= out[-1][1]:
            p = out[-1]
            best = (p[2], p[3]) if p[2] <= sep[k] else (sep[k], unix[k])
            out[-1] = (p[0], max(p[1], t1), best[0], best[1])
        else:
            out.append((t0, t1, sep[k], unix[k]))
        i = j + 1
    return out


def state_grid(sat, ts, t0_gpst, n, step):
    """ITRS position (m) and velocity (m/s) at t0_gpst + k*step, k < n."""
    from skyfield.framelib import itrs

    utc0 = t0_gpst - gnss_ephemeris.GPS_UTC_LEAP + gnss_ephemeris.GPS_EPOCH.timestamp()
    t = gp.ts_unix(ts, [utc0 + k * step for k in range(n)])
    pos, vel = sat.at(t).frame_xyz_and_velocity(itrs)
    p, v = pos.m, vel.m_per_s
    return (
        [[round(p[0][k], 3), round(p[1][k], 3), round(p[2][k], 3)] for k in range(n)],
        [[round(v[0][k], 6), round(v[1][k], 6), round(v[2][k], 6)] for k in range(n)],
    )


def load_db(path=guowang_codes.GUOWANG_1202_DB):
    """{name: (G2' phase, theta_s or None)} and theta_unknown_s, from the satellite database."""
    with open(path) as fh:
        d = json.load(fh)
    live = load_live_pages()
    sats = {
        n: (
            s["g2_phase"],
            s["theta_us"] * 1e-6 if "theta_us" in s else None,
            live.get(n, s.get("nav_pages")),
        )
        for n, s in d["satellites"].items()
        if s.get("g2_phase") is not None
    }
    return sats, d["theta_unknown_us"] * 1e-6


LIVE_PAGES = "/home/kvand/gnss/builds/gw1202/nav_pages_live.json"
LIVE_PAGES_MAX_AGE_S = 6 * 3600.0


def load_live_pages(path=LIVE_PAGES, max_age_s=LIVE_PAGES_MAX_AGE_S):
    """{name: nav_pages} decoded live by the track-beam loop (gw_nav_live), newer than
    max_age_s; they override the library's. Missing or unreadable file: nothing."""
    try:
        with open(path) as fh:
            live = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    now = time.time()
    for n, e in live.items():
        try:
            t = calendar.timegm(time.strptime(e["utc"], "%Y-%m-%dT%H:%M:%SZ"))
        except (KeyError, ValueError):
            continue
        if now - t <= max_age_s and len(e.get("nav_pages") or []) == 4:
            out[n] = list(e["nav_pages"])
    return out


def build(
    sats,
    ts,
    t_start,
    t_end,
    db,
    sep_deg=25.0,
    step=10.0,
    pad_s=60.0,
    coarse=10.0,
    theta_unknown_s=0.0,
    el_min=-90.0,
):
    """(records, unknown): table records for the satellites in `db` (name -> (G2' phase, theta_s
    or None)) that pass within sep_deg of the boresight above el_min in [t_start, t_end] (unix), and the passes
    of the others. A record's clk_s is -theta: the code starts theta after each UTC 2 ms
    boundary, so the broker's code-phase model takes it as the satellite clock; theta_unknown_s
    where the satellite's is not measured."""
    from skyfield.api import wgs84

    site = wgs84.latlon(gp.SITE["lat"], gp.SITE["lon"], elevation_m=gp.SITE["h"])
    bore = gp.enu_unit(gp.BORESIGHT["az"], gp.BORESIGHT["el"])
    unix = [t_start + k * coarse for k in range(int((t_end - t_start) / coarse) + 1)]
    t = gp.ts_unix(ts, unix)
    records, unknown = [], []
    for sat in sats:
        name = short_name(sat.name)
        sep = separation_deg(sat, site, t, bore, el_min)
        segs = segments(unix, sep, sep_deg, pad_s)
        if not segs:
            continue
        cat = int(sat.model.satnum)
        if name not in db:
            for t0, t1, smin, tmin in segs:
                unknown.append(
                    dict(
                        name=name, catnum=cat, tca_unix=tmin, min_sep_deg=round(smin, 2)
                    )
                )
            continue
        tle_epoch_gpst = gnss_ephemeris.gpst_of_utc(sat.epoch.utc_datetime())
        for t0, t1, smin, tmin in segs:
            g0 = gnss_ephemeris.gpst_of_utc(t0)
            g0 = math.floor(g0 / step) * step
            n = int(math.ceil((gnss_ephemeris.gpst_of_utc(t1) - g0) / step)) + 1
            pos, vel = state_grid(sat, ts, g0, n, step)
            phase, theta = db[name][:2]
            pages = db[name][2] if len(db[name]) > 2 else None
            records.append(
                dict(
                    prn=int(phase) + 1,
                    g2_phase=int(phase),
                    clk_s=-(theta if theta is not None else theta_unknown_s),
                    theta_measured=theta is not None,
                    name=name,
                    catnum=cat,
                    t0_gpst=g0,
                    t1_gpst=g0 + (n - 1) * step,
                    dt=step,
                    toe_gpst=gnss_ephemeris.gpst_of_utc(tmin),
                    **({"nav_pages": list(pages)} if pages else {}),
                    tca_unix=tmin,
                    min_sep_deg=round(smin, 3),
                    tle_epoch_gpst=tle_epoch_gpst,
                    pos=pos,
                    vel=vel,
                )
            )
    records.sort(key=lambda r: r["t0_gpst"])
    unknown.sort(key=lambda u: u["tca_unix"])
    return records, unknown


def transits(sats, ts, t_start, t_end, radius_deg=10.0, step=30.0, coarse=10.0):
    """Every pass of `sats` within radius_deg of the boresight in [t_start, t_end] (unix), in
    the layout of build's records without a code or clock, for the broker's transit sky. A
    10 s scan puts a point inside a 10 deg radius on every pass that comes within 9 deg of the
    boresight, even a 500 km satellite's (0.8 deg/s there), and each segment is padded by one
    grid step, so it covers the time inside radius_deg between scan points."""
    from skyfield.api import wgs84

    site = wgs84.latlon(gp.SITE["lat"], gp.SITE["lon"], elevation_m=gp.SITE["h"])
    bore = gp.enu_unit(gp.BORESIGHT["az"], gp.BORESIGHT["el"])
    unix = [t_start + k * coarse for k in range(int((t_end - t_start) / coarse) + 1)]
    t = gp.ts_unix(ts, unix)
    out = []
    for sat in sats:
        sep = separation_deg(sat, site, t, bore)
        for t0, t1, smin, tmin in segments(unix, sep, radius_deg, step):
            g0 = math.floor(gnss_ephemeris.gpst_of_utc(t0) / step) * step
            n = int(math.ceil((gnss_ephemeris.gpst_of_utc(t1) - g0) / step)) + 1
            pos, vel = state_grid(sat, ts, g0, n, step)
            out.append(
                dict(
                    name=short_name(sat.name),
                    catnum=int(sat.model.satnum),
                    t0_gpst=g0,
                    t1_gpst=g0 + (n - 1) * step,
                    dt=step,
                    toe_gpst=gnss_ephemeris.gpst_of_utc(tmin),
                    min_sep_deg=round(smin, 2),
                    pos=pos,
                    vel=vel,
                )
            )
    out.sort(key=lambda r: r["t0_gpst"])
    return out


def one_per_prn(records):
    """(records, names cut): where satellites that share a prn (a G2' phase) have overlapping
    segments, each grid point goes to the one closer to the boresight, and a record keeps its
    runs of two points or more, so no two records of one prn overlap in time. A handover
    leaves one grid step with neither, so the broker drops the first satellite's seed before
    it seeds the second. Every record of one build shares the grid (t0_gpst is a multiple of
    dt)."""
    lat, lon = gp.SITE["lat"], gp.SITE["lon"]
    rx = gnss_ephemeris._ecef_of_llh(lat, lon, gp.SITE["h"])
    bore = gp.enu_unit(gp.BORESIGHT["az"], gp.BORESIGHT["el"])
    by_prn = {}
    for r in records:
        by_prn.setdefault(r["prn"], []).append(r)
    out, cut = [], set()
    for recs in by_prn.values():
        if len(recs) == 1:
            out += recs
            continue
        # grid index -> (separation, index in recs) of the closest satellite
        owner = {}
        for i, r in enumerate(recs):
            k0 = round(r["t0_gpst"] / r["dt"])
            for j, p in enumerate(r["pos"]):
                az, el, _ = gnss_ephemeris._azel(rx, p, lat, lon)
                c = float(gp.enu_unit(az, el) @ bore)
                sep = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                if k0 + j not in owner or sep < owner[k0 + j][0]:
                    owner[k0 + j] = (sep, i)
        for i, r in enumerate(recs):
            k0 = round(r["t0_gpst"] / r["dt"])
            n = len(r["pos"])
            mine = [owner[k0 + j][1] == i for j in range(n)]
            if all(mine):
                out.append(r)
                continue
            cut.add(r["name"])
            j = 0
            while j < n:
                if not mine[j]:
                    j += 1
                    continue
                e = j
                while e + 1 < n and mine[e + 1]:
                    e += 1
                if e > j:
                    out.append(
                        dict(
                            r,
                            t0_gpst=r["t0_gpst"] + j * r["dt"],
                            t1_gpst=r["t0_gpst"] + e * r["dt"],
                            pos=r["pos"][j : e + 1],
                            vel=r["vel"][j : e + 1],
                        )
                    )
                j = e + 1
    out.sort(key=lambda r: r["t0_gpst"])
    return out, sorted(cut)


def load_sweep(path):
    """{name: (G2' phase, theta_s or None)} from a --sweep file: satellites whose code is
    predicted (code-reuse rules) but not measured, with a candidate theta once a sweep found one."""
    with open(path) as fh:
        d = json.load(fh)
    return {
        n: (
            s["g2_phase"],
            s["theta_us"] * 1e-6 if s.get("theta_us") is not None else None,
        )
        for n, s in d["satellites"].items()
    }


def prior_order(thetas_s, prior_us, bw_us=1.5):
    """The thetas (s), densest first under a Gaussian kernel (bw_us) over the known thetas
    prior_us (us); ties go to the one nearer the median, then the lower. No prior: unchanged."""
    if not prior_us:
        return list(thetas_s)
    med = sorted(prior_us)[len(prior_us) // 2]

    def dens(t):
        return sum(math.exp(-0.5 * ((t * 1e6 - p) / bw_us) ** 2) for p in prior_us)

    return sorted(thetas_s, key=lambda t: (-round(dens(t), 9), abs(t * 1e6 - med), t))


class SweepOrder:
    """Per-satellite theta choice for the sweep steps (see the module doc). order_s: the thetas,
    preferred first; tried: {name: [theta_us of tried steps]}; judged: {(name, step)} the harvest
    has judged; assign: {name: {str(step): theta_us}} from earlier runs; now_gpst: GPST seconds
    (steps are on the GPST grid). A step handed out but not judged is pending for pending_steps
    steps after it, then dropped (it never ran). A past or current step with no assignment (it
    ran before this order took over, or its satellite just joined) keeps cycle_s(name, step), the
    stateless cycle's theta, which is what an earlier run published for it."""

    def __init__(self, order_s, tried, judged, assign, now_gpst, step_s, cycle_s=None, keep_s=2 * 86400.0, pending_steps=120):
        self.order = [round(t * 1e6, 4) for t in order_s]
        self.rank = {t: k for k, t in enumerate(self.order)}
        self.tried = {n: [round(t, 4) for t in ts] for n, ts in tried.items()}
        self.judged = judged
        self.now_step = math.floor(now_gpst / step_s)
        old = self.now_step - keep_s / step_s
        self.assign = {n: {k: v for k, v in m.items() if int(k) >= old} for n, m in assign.items()}
        self.pending_steps = pending_steps
        self.cycle_s = cycle_s

    def theta_s(self, name, step):
        m = self.assign.setdefault(name, {})
        k = str(step)
        if k not in m and self.cycle_s is not None and step <= self.now_step + 1:
            m[k] = round(self.cycle_s(name, step) * 1e6, 4)
        if k not in m:
            uses = {}
            for t in self.tried.get(name, ()):
                uses[t] = uses.get(t, 0) + 1
            for ks, t in m.items():
                if (name, int(ks)) not in self.judged and int(ks) >= self.now_step - self.pending_steps:
                    uses[t] = uses.get(t, 0) + 1
            m[k] = min(self.order, key=lambda t: (uses.get(t, 0), self.rank[t]))
        return m[k] * 1e-6


def load_sweep_history(steps_path):
    """({name: [theta_us tried]}, {(name, step) judged}) from the harvest's steps.jsonl."""
    tried, judged = {}, set()
    if not os.path.exists(steps_path):
        return tried, judged
    with open(steps_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("judged"):
                continue
            if r.get("step") is not None:  # early rows carry no step
                judged.add((r["name"], int(r["step"])))
            if r.get("n_eval", 0) >= 3:
                tried.setdefault(r["name"], []).append(float(r["theta_us"]))
    return tried, judged


def sweep(records, names, thetas_s, step_s, chooser=None):
    """The records of the satellites in `names` cut into steps of step_s on the GPST grid, each
    step at its own clock: theta = thetas_s[(step index + crc32(name)) % len(thetas_s)], so every
    run of this job gives a step the same theta, and two satellites on one schedule are offset.
    A step keeps its grid points in [0, step_s - dt], so one grid step between steps has no
    record: the broker drops the seed there and seeds the next step afresh at its theta (a held
    seed would slew to the new code phase at 0.025 chips/s). A step needs two grid points.
    step_s must be a multiple of the records' dt. chooser(name, step) -> theta_s, when given,
    replaces the cycle (SweepOrder.theta_s); steps are visited in time order per record."""
    out = []
    for r in records:
        if r["name"] not in names:
            out.append(r)
            continue
        dt = r["dt"]
        off = zlib.crc32(r["name"].encode())
        steps = {}
        for j in range(len(r["pos"])):
            t = r["t0_gpst"] + j * dt
            s = math.floor(t / step_s + 1e-9)
            if t - s * step_s <= step_s - dt + 1e-6:
                steps.setdefault(s, []).append(j)
        for s, js in sorted(steps.items()):
            if len(js) < 2:
                continue
            theta = chooser(r["name"], s) if chooser else thetas_s[(s + off) % len(thetas_s)]
            out.append(
                dict(
                    r,
                    t0_gpst=r["t0_gpst"] + js[0] * dt,
                    t1_gpst=r["t0_gpst"] + js[-1] * dt,
                    pos=r["pos"][js[0] : js[-1] + 1],
                    vel=r["vel"][js[0] : js[-1] + 1],
                    clk_s=-theta,
                    sweep_step=s,
                    theta_try_us=round(theta * 1e6, 4),
                )
            )
    out.sort(key=lambda r: r["t0_gpst"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hours", type=float, default=3.0, help="window from now")
    ap.add_argument(
        "--sep-deg", type=float, default=25.0, help="pass radius about boresight"
    )
    ap.add_argument(
        "--el-min",
        type=float,
        default=-90.0,
        help="deg: pass segments only above this elevation (the transits are not cut)",
    )
    ap.add_argument("--step-s", type=float, default=10.0, help="table grid")
    ap.add_argument(
        "--pad-s", type=float, default=60.0, help="segment padding each side"
    )
    ap.add_argument("--coarse-s", type=float, default=30.0, help="separation scan step")
    ap.add_argument(
        "--transit-deg",
        type=float,
        default=10.0,
        help="transit radius about boresight, every satellite; above the widest gate "
        "the broker's transit sky feeds (the assemblers' 8 deg formation freeze)",
    )
    ap.add_argument("--transit-step-s", type=float, default=30.0, help="transit grid")
    ap.add_argument(
        "--extra-tle",
        default=HIDDEN_TLE if os.path.exists(HIDDEN_TLE) else None,
        help="more GuoWang TLEs, for the transits only (default: %s if present)"
        % HIDDEN_TLE,
    )
    ap.add_argument(
        "--tle", help="TLE file (default: the newest cached hulianwang group)"
    )
    ap.add_argument(
        "--fetch", action="store_true", help="refresh the cache if older than 2 h"
    )
    ap.add_argument(
        "--sweep",
        help='JSON of satellites with a predicted code, tabled too ("implied" in their '
        "records): at their theta_us where it is given, else in --sweep-step-s steps "
        "across --sweep-theta-us (sweep)",
    )
    ap.add_argument("--sweep-step-s", type=float, default=30.0)
    ap.add_argument(
        "--sweep-theta-us",
        default="-18,12,0.375",
        help="LO,HI,STEP: the thetas a sweep tries (0.375 us = 0.77 chip)",
    )
    ap.add_argument(
        "--sweep-order",
        choices=("prior", "cycle"),
        default="prior",
        help="prior: per satellite, the least-tried theta, densest among the known thetas "
        "first (state in assign.json beside --sweep); cycle: the stateless cycle",
    )
    ap.add_argument("--cache-dir", default=gp.CACHE_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    from skyfield.api import load

    ts = load.timescale(builtin=True)
    tle = a.tle or (
        gp.fetch_tle(a.cache_dir) if a.fetch else gp.newest_cache(a.cache_dir)
    )
    if not tle:
        sys.exit("no TLE: give --tle or --fetch")
    sats = gp.read_tle(tle, ts, "DIGUI")
    db, theta_unknown_s = load_db()
    # A missing or unreadable sweep file costs the sweep, not the table.
    implied = {}
    if a.sweep:
        try:
            implied = {n: v for n, v in load_sweep(a.sweep).items() if n not in db}
        except Exception as e:
            print("%s: unreadable (%s); no sweep" % (a.sweep, e))
    now = time.time()
    records, unknown = build(
        sats,
        ts,
        now - 600.0,
        now + a.hours * 3600.0,
        dict(db, **implied),
        a.sep_deg,
        a.step_s,
        a.pad_s,
        a.coarse_s,
        theta_unknown_s,
        a.el_min,
    )
    records, cut = one_per_prn(records)
    for r in records:
        if r["name"] in implied:
            r["implied"] = True
    lo, hi, st = (float(x) for x in a.sweep_theta_us.split(","))
    thetas = [(lo + k * st) * 1e-6 for k in range(int(round((hi - lo) / st)) + 1)]
    swept = {n for n, (_, th) in implied.items() if th is None}
    # The prior order is optional: any failure falls back to the stateless cycle.
    chooser, order_note = None, "cycle"
    if a.sweep and a.sweep_order == "prior" and swept:
        try:
            sdir = os.path.dirname(os.path.abspath(a.sweep))
            prior = [v[1] * 1e6 for v in db.values() if v[1] is not None]
            prior += [v[1] * 1e6 for v in implied.values() if v[1] is not None]
            tried, judged = load_sweep_history(os.path.join(sdir, "steps.jsonl"))
            assign_path = os.path.join(sdir, "assign.json")
            assign = {}
            if os.path.exists(assign_path):
                with open(assign_path) as fh:
                    assign = json.load(fh)
            order = SweepOrder(
                prior_order(thetas, prior),
                tried,
                judged,
                assign,
                gnss_ephemeris.gpst_of_utc(now),
                a.sweep_step_s,
                cycle_s=lambda n, s: thetas[(s + zlib.crc32(n.encode())) % len(thetas)],
            )
            chooser = order.theta_s
            order_note = "prior (%d known thetas, densest %+.3f us)" % (len(prior), order.order[0])
        except Exception as e:
            print("sweep order: prior unavailable (%s); stateless cycle" % e)
    records = sweep(records, swept, thetas, a.sweep_step_s, chooser)
    if chooser is not None:
        try:
            tmp = assign_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(order.assign, fh, separators=(",", ":"), sort_keys=True)
            os.replace(tmp, assign_path)
        except Exception as e:
            print("sweep order: assign.json not written (%s)" % e)
    # The extra TLEs are optional: an unreadable file costs its satellites, not the table.
    extra = []
    if a.extra_tle:
        try:
            extra = gp.read_tle(a.extra_tle, ts, "")
        except Exception as e:
            print("%s: unreadable (%s); no extra satellites" % (a.extra_tle, e))
    catalogued = {s.model.satnum for s in sats}
    extra = [s for s in extra if s.model.satnum not in catalogued]
    tr = transits(
        sats + extra,
        ts,
        now - 600.0,
        now + a.hours * 3600.0,
        a.transit_deg,
        a.transit_step_s,
    )
    doc = dict(
        schema=SCHEMA,
        created_unix=now,
        tle=tle,
        tle_mtime_unix=os.path.getmtime(tle),
        sep_deg=a.sep_deg,
        el_min_deg=a.el_min,
        step_s=a.step_s,
        known=sorted(db),
        implied=sorted(implied),
        sweep=dict(
            file=a.sweep,
            step_s=a.sweep_step_s,
            theta_us=[lo, hi, st],
            order=order_note,
            names=sorted(swept),
        ),
        records=records,
        unknown=unknown,
        extra_tle=a.extra_tle,
        transit_deg=a.transit_deg,
        transits=tr,
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(
        "%s: %d pass segment(s) of %d known satellite(s), %d sweep step(s) of %d implied "
        "one(s) and %d at a candidate theta, %d pass(es) of unknown ones, "
        "shared-prn overlap cut from %s, %d transit(s) within %g deg of %d satellite(s) "
        "(%d extra), TLE %s"
        % (
            a.out,
            sum(1 for r in records if r["name"] in db),
            len({r["name"] for r in records if r["name"] in db}),
            sum(1 for r in records if "sweep_step" in r),
            len({r["name"] for r in records if "sweep_step" in r}),
            len(
                {
                    r["name"]
                    for r in records
                    if r.get("implied") and "sweep_step" not in r
                }
            ),
            len(unknown),
            ", ".join(cut) or "none",
            len(tr),
            a.transit_deg,
            len(sats) + len(extra),
            len(extra),
            tle,
        )
    )


if __name__ == "__main__":
    main()
