#!/usr/bin/env python3
"""Current TLEs for the 27 GuoWang satellites inferred on 10-29 (names DIGUI-H<plane>-<k>).

Bursts in CHORD's beam with no catalogued satellite placed three sets of nine satellites in the
catalogued 50 deg planes at RAAN ~24.5, ~54.1 and ~84.4, between the catalogued ones
(taua1029/make_hidden_tle.py: plane 24.5 fitted to the bursts, timing ~+-5 s; the other two at
the midpoints of their catalogued neighbours, ~+-40 s). Their 10-29 elements drift along track
when propagated, but the constellation keeps its in-plane spacing, so each one is placed from
its two catalogued in-plane neighbours in the newest cache instead: argument of latitude, RAAN
and inclination are each neighbour's plus the offset measured at the 10-29 fit epoch (against
the catalogue the 10-29 sets were made from), averaged over the two; the mean motion is theirs.
Circular, no drag, as on 10-29; epoch = the newer neighbour's.

    gw_hidden_tle.py [--tle FILE] [--out FILE] [--check]          gnss venv (skyfield)

Writes --out atomically, default ~/.cache/guowang/hidden_digui.tle (never hulianwang_*.tle:
newest_cache would take it for the catalogue). Catalogue numbers 99001-99027 in the order
H24-1..9, H54-1..9, H84-1..9, outside the NORAD range. Prints each satellite's neighbours and its
along-track offset from the 10-29 set at the fit epoch; --check also lists the closest passes at
the bursts the 10-29 sets explained. Rerun after each TLE refresh.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "..", "python", "scripts", "gnss", "guowang")
)
import guowang_passes as gp  # noqa: E402

# the catalogue of 10-29 (hulianwang_20261028T211401Z) followed by the 27 inferred sets
REF = "/home/kvand/gnss/builds/scratch_bak_20261023_gnssmain/taua1029/guowang_plus_hidden_20261029.tle"
OUT = os.path.join(gp.CACHE_DIR, "hidden_digui.tle")
CATNUM0 = 99001
# bursts no catalogued satellite explains (taua1029/midpoint_check.py), and the 10-30 baseband
# dump's DIGUI-H24-9 (taua1029/map_data.py)
BURSTS = [
    "2026-10-26T13:43:15",
    "2026-10-26T15:27:15",
    "2026-10-26T15:39:59",
    "2026-10-26T15:52:44",
    "2026-10-26T16:00:29",
    "2026-10-26T16:13:03",
    "2026-10-26T18:09:44",
    "2026-10-29T10:31:41",
    "2026-10-29T10:44:19",
    "2026-10-29T10:56:57",
    "2026-10-30T10:36:08",
]


def elements(sat, jd):
    """(RAAN deg, argument of latitude deg, inclination deg, mean motion rad/min) at jd: the
    SGP4 mean elements with their secular rates."""
    m = sat.model
    dtm = (jd - m.jdsatepoch - m.jdsatepochF) * 1440.0
    return (
        np.degrees(m.nodeo + m.nodedot * dtm) % 360.0,
        np.degrees(m.argpo + m.argpdot * dtm + m.mo + m.mdot * dtm) % 360.0,
        np.degrees(m.inclo),
        m.no_kozai,
    )


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def circmean(deg):
    return np.degrees(np.angle(np.mean(np.exp(1j * np.radians(deg))))) % 360.0


def neighbours(h, cat, jd):
    """[(catnum, du, draan, dinc)]: the catalogued in-plane satellites just behind and just ahead
    of h in argument of latitude at jd, and h's element offsets from each."""
    r, u, i, _ = elements(h, jd)
    plane = []
    for s in cat:
        rs, us, is_, _ = elements(s, jd)
        if abs(wrap(rs - r)) < 1.5 and abs(is_ - i) < 1.0:
            plane.append((wrap(u - us), wrap(r - rs), i - is_, int(s.model.satnum)))
    behind = min((p for p in plane if p[0] > 0), key=lambda p: p[0])
    ahead = max((p for p in plane if p[0] < 0), key=lambda p: p[0])
    return [(p[3], p[0], p[1], p[2]) for p in (behind, ahead)]


def place(refs, now):
    """(jd, RAAN, u, inc, mean motion) from the neighbours in `now` (catnum -> satellite)."""
    have = [(now[c], du, dr, di) for c, du, dr, di in refs if c in now]
    if not have:
        return None
    jd = max(s.model.jdsatepoch + s.model.jdsatepochF for s, _, _, _ in have)
    est = [(elements(s, jd), du, dr, di) for s, du, dr, di in have]
    return (
        jd,
        circmean([e[0] + dr for e, du, dr, di in est]),
        circmean([e[1] + du for e, du, dr, di in est]),
        float(np.mean([e[2] + di for e, du, dr, di in est])),
        float(np.mean([e[3] for e, du, dr, di in est])),
    )


def tle(catnum, jd, raan, u, inc, n):
    from sgp4 import exporter
    from sgp4.api import WGS72, Satrec

    s = Satrec()
    s.sgp4init(
        WGS72,
        "i",
        catnum,
        jd - 2433281.5,
        0.0,
        0.0,
        0.0,
        1e-5,
        0.0,
        np.radians(inc),
        np.radians(u),
        n,
        np.radians(raan),
    )
    return exporter.export_tle(s)


def along_track_s(a, b, t):
    """b's position minus a's along a's velocity at t, in seconds of a's motion."""
    pa, pb = a.at(t), b.at(t)
    v = pa.velocity.km_per_s
    return float(np.dot(pb.position.km - pa.position.km, v) / np.dot(v, v))


def check(old, new, ts):
    """The closest pass of each set within 90 s of each burst."""
    geo = gp.Geometry(ts)
    print(
        "burst (UTC)           10-29 set: sat  tca-burst  sep    regenerated: sat  tca-burst  sep"
    )
    for b in BURSTS:
        t = gp.parse_time(b)
        row = []
        for sats in (old, new):
            ps = gp.find_passes(sats, geo, t - 300.0, t + 300.0, near_deg=10.0)
            ps = [p for p in ps if abs(p["tca"] - t) <= 90.0]
            if ps:
                p = min(ps, key=lambda p: p["sep"])
                row.append(
                    "%-12s %+6.1f s %5.2f"
                    % (p["name"].replace("HULIANWANG ", ""), p["tca"] - t, p["sep"])
                )
            else:
                row.append("%-12s %8s %5s" % ("-", "", ""))
        print("%s   %s    %s" % (b.replace("T", " "), row[0], row[1]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tle", help="catalogue (default: the newest cached hulianwang group)"
    )
    ap.add_argument("--ref", default=REF, help="10-29 catalogue + inferred sets")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true", help="also match the bursts")
    a = ap.parse_args()
    from skyfield.api import EarthSatellite, load

    ts = load.timescale(builtin=True)
    path = a.tle or gp.newest_cache(gp.CACHE_DIR)
    ref = gp.read_tle(a.ref, ts, "DIGUI")
    old = [s for s in ref if "DIGUI-H" in s.name]
    cat0 = [s for s in ref if "DIGUI-H" not in s.name]
    now = {int(s.model.satnum): s for s in gp.read_tle(path, ts, "DIGUI")}
    jd_fit = old[0].model.jdsatepoch + old[0].model.jdsatepochF
    print(
        "# 10-29 sets %s (fit epoch %s); catalogue now %s"
        % (a.ref, old[0].epoch.utc_iso(), path)
    )
    out, new = [], []
    for k, h in enumerate(old):
        refs = neighbours(h, cat0, jd_fit)
        p = place(refs, now)
        if p is None:
            print("%s: no neighbour in %s, left out" % (h.name, path))
            continue
        l1, l2 = tle(CATNUM0 + k, *p)
        s = EarthSatellite(l1, l2, h.name.strip(), ts)
        new.append(s)
        out.append("%-24s\n%s\n%s\n" % (h.name.strip(), l1, l2))
        print(
            "%-22s %d  neighbours %s  epoch %s  along-track vs the 10-29 set at the fit epoch %+.1f s"
            % (
                h.name.strip(),
                CATNUM0 + k,
                " ".join(
                    "%d (%+.2f deg)" % (c, du) for c, du, _, _ in refs if c in now
                ),
                s.epoch.utc_strftime("%m-%d %H:%M"),
                along_track_s(h, s, old[0].epoch),
            )
        )
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("".join(out))
    os.replace(tmp, a.out)
    print(
        "# wrote %s (%d satellites) at %s"
        % (a.out, len(out), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    )
    if a.check:
        check(old, new, ts)


if __name__ == "__main__":
    main()
