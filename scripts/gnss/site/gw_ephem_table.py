#!/usr/bin/env python3
"""GuoWang ephemeris table for the broker (#227): the satellites' TLEs propagated off the broker.

GuoWang broadcasts no ephemeris, so its geometry comes from CelesTrak TLEs (group hulianwang)
through SGP4. The broker runs on a free-threaded Python with no SGP4, so this job (the gnss
venv, skyfield) propagates the satellites whose 1202 MHz code is known and writes, per pass that
comes within --sep-deg of the dish boresight, their ITRS position and velocity on a --step-s
grid. gnss_ephemeris.load_gw_table reads the file and interpolates (cubic Hermite: a 10 s step
is good to well under a metre), so predict_all, seeding, the PRN map and the element steering
see GuoWang satellites as constellation W, PRN = G2' phase + 1.

    gw_ephem_table.py [--hours 3] [--sep-deg 25] [--step-s 10] [--out PATH] [--fetch]

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


def short_name(name):
    """'HULIANWANG DIGUI-79' -> 'DIGUI-79', the key of the code database."""
    return name.replace("HULIANWANG", "").strip()


def separation_deg(sat, site, ts, unix, bore):
    """Boresight separation (deg) of `sat` at each unix time."""
    alt, az, _ = (sat - site).at(gp.ts_unix(ts, unix)).altaz()
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


def build(
    sats, ts, t_start, t_end, db, sep_deg=25.0, step=10.0, pad_s=60.0, coarse=10.0
):
    """(records, unknown): table records for the satellites in `db` (name -> G2' phase) that pass
    within sep_deg of the boresight in [t_start, t_end] (unix), and the passes of the others."""
    from skyfield.api import wgs84

    site = wgs84.latlon(gp.SITE["lat"], gp.SITE["lon"], elevation_m=gp.SITE["h"])
    bore = gp.enu_unit(gp.BORESIGHT["az"], gp.BORESIGHT["el"])
    unix = [t_start + k * coarse for k in range(int((t_end - t_start) / coarse) + 1)]
    records, unknown = [], []
    for sat in sats:
        name = short_name(sat.name)
        sep = separation_deg(sat, site, ts, unix, bore)
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
            records.append(
                dict(
                    prn=int(db[name]) + 1,
                    g2_phase=int(db[name]),
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
    db = dict(guowang_codes.GUOWANG_1202_G2P_PHASE)
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
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(
        "%s: %d pass segment(s) of %d known satellite(s), %d pass(es) of unknown ones, TLE %s"
        % (a.out, len(records), len({r["name"] for r in records}), len(unknown), tle)
    )


if __name__ == "__main__":
    main()
