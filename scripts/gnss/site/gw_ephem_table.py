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
"""

import argparse
import json
import math
import os
import sys
import time

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


def separation_deg(sat, site, t, bore):
    """Boresight separation (deg) of `sat` at each time of the skyfield Time `t`. Share one `t`
    across satellites: skyfield keeps its Earth-orientation terms, the bulk of the cost."""
    alt, az, _ = (sat - site).at(t).altaz()
    cosang = gp.enu_unit(az.degrees, alt.degrees) @ bore
    return [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cosang]


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
    sats = {
        n: (s["g2_phase"], s["theta_us"] * 1e-6 if "theta_us" in s else None)
        for n, s in d["satellites"].items()
        if s.get("g2_phase") is not None
    }
    return sats, d["theta_unknown_us"] * 1e-6


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
):
    """(records, unknown): table records for the satellites in `db` (name -> (G2' phase, theta_s
    or None)) that pass within sep_deg of the boresight in [t_start, t_end] (unix), and the passes
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
        sep = separation_deg(sat, site, t, bore)
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
            phase, theta = db[name]
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hours", type=float, default=3.0, help="window from now")
    ap.add_argument(
        "--sep-deg", type=float, default=25.0, help="pass radius about boresight"
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
    now = time.time()
    records, unknown = build(
        sats,
        ts,
        now - 600.0,
        now + a.hours * 3600.0,
        db,
        a.sep_deg,
        a.step_s,
        a.pad_s,
        a.coarse_s,
        theta_unknown_s,
    )
    records, cut = one_per_prn(records)
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
        step_s=a.step_s,
        known=sorted(db),
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
        "%s: %d pass segment(s) of %d known satellite(s), %d pass(es) of unknown ones, "
        "shared-prn overlap cut from %s, %d transit(s) within %g deg of %d satellite(s) "
        "(%d extra), TLE %s"
        % (
            a.out,
            len(records),
            len({r["name"] for r in records}),
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
