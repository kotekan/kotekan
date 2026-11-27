#!/usr/bin/env python3
"""Pointing records for satellites no broadcast product carries, from their TLEs, for the broker.

Some satellites transmit in our bands but are in no broadcast-ephemeris product: a GLONASS spare
(COSMOS 2433, which CHORD reads near the L2 channel k = 8), the BeiDou-3S experimental M2-S. To
search for them, track them and read their own navigation messages, the broker needs a
prediction of where they are. This writes one, a RINEX 3.04 navigation file the broker reads
after the BRDC sources (--extra-nav, gnss_brdc_supply.EXTRA_NAV):

* GLONASS (R): the SGP4 state at each epoch in the Earth-fixed frame, with no lunisolar
  acceleration, health 0 and the frequency channel given (the carrier is k x 437.5 kHz).
* BeiDou (C): the osculating Kepler elements at each epoch, J2's node rate as OMEGA DOT, no
  harmonic corrections.

Each satellite gets an epoch every --step-s over [hour - 1 h, hour + 7 h). The orbit is the
TLE's, good to kilometres, and the clock is unknown (zero). The header carries
gnss_ephemeris.POINTING_MARK, so the records read back with an unknown clock (clk_sigma_s): the
broker seeds them from its search, never from the model, and keeps them out of its clock solves.
These records are pointing only. They are never broadcast data and are not to be served as such.

A satellite with its own decoded ephemeris (decoded_eph.Reader over --decoded) is left out, so
the broker falls back to the decoded orbit (it prefers any record in the BRDC store). Once the
decoded issues age out, the satellite's pointing records come back for its next pass.

    gnss_oddsat_pointing.py --out pointing.rnx --sats R25:32275:8,R28:32275:7,C58:40748
        [--decoded DIR_OR_FILE,...] [--every-s 300] [--once]

--sats: SYS PRN : NORAD catalogue number [: GLONASS frequency channel]. Several PRNs may share
one satellite (twin slots for an unknown channel, or candidate codes). The TLEs come from
CelesTrak by catalogue number, cached in ~/.cache/kotekan_gps/oddsat_tle for TLE_MAX_AGE_S and
used from the cache when the network fails.
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday

import gnss_ephemeris as ge

TLE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps", "oddsat_tle")
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=%d&FORMAT=TLE"
TLE_MAX_AGE_S = 12 * 3600.0
J2 = 1.08262668e-3
RE_M = 6378137.0
FDMA_SPACING_HZ = 437500.0


def parse_sats(spec):
    """[(sys, prn, norad, k or None)] from "R25:32275:8,C58:40748"."""
    out = []
    for item in spec.split(","):
        f = item.strip().split(":")
        if not f[0]:
            continue
        sysc, prn, norad = f[0][0].upper(), int(f[0][1:]), int(f[1])
        k = int(f[2]) if len(f) > 2 and f[2] != "" else None
        if sysc not in "RC" or (sysc == "R" and k is None):
            raise SystemExit(
                "--sats %s: R needs its frequency channel, only R and C" % item
            )
        out.append((sysc, prn, norad, k))
    return out


def tle(norad, now=None):
    """The (line1, line2) for a catalogue number: the cache while fresh, else CelesTrak, else
    the stale cache."""
    now = time.time() if now is None else now
    path = os.path.join(TLE_DIR, "%d.txt" % norad)
    have = None
    try:
        with open(path) as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        have = (lines[-2], lines[-1])
        if now - os.path.getmtime(path) < TLE_MAX_AGE_S:
            return have
    except (OSError, IndexError):
        pass
    try:
        req = urllib.request.Request(
            TLE_URL % norad, headers={"User-Agent": "gnss-tle"}
        )
        text = urllib.request.urlopen(req, timeout=20).read().decode()
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        l1 = next(ln for ln in lines if ln.startswith("1 "))
        l2 = next(ln for ln in lines if ln.startswith("2 "))
        os.makedirs(TLE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
        return l1, l2
    except Exception as e:  # noqa: BLE001 -- the network fails in many ways
        if have is None:
            raise RuntimeError("no TLE for %d: %s" % (norad, e)) from e
        print(
            "TLE %d: refresh failed (%s); the cached one is used" % (norad, e),
            flush=True,
        )
        return have


def _gmst(t_utc):
    """Greenwich mean sidereal angle (rad) at unix time t_utc (UT1 ~ UTC)."""
    jd = t_utc / 86400.0 + 2440587.5
    d = jd - 2451545.0
    tt = d / 36525.0
    deg = (
        280.46061837 + 360.98564736629 * d + 0.000387933 * tt * tt - tt**3 / 38710000.0
    )
    return math.radians(deg % 360.0)


def ecef_state(sat, t_utc):
    """SGP4 position (m) and Earth-fixed velocity (m/s) at unix time t_utc, TEME rotated by the
    mean sidereal angle (no polar motion: metres against a TLE's kilometres)."""
    dt = datetime.fromtimestamp(t_utc, timezone.utc)
    jd, fr = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    )
    err, r, v = sat.sgp4(jd, fr)
    if err:
        raise RuntimeError("sgp4 error %d" % err)
    g = _gmst(t_utc)
    c, s = math.cos(g), math.sin(g)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    pos = rot @ (np.array(r) * 1e3)
    vel = rot @ (np.array(v) * 1e3)
    w = ge.OMEGA_E[
        "C"
    ]  # the frame rotation; PZ-90 and CGCS2000 differ in the 10th digit
    vel = vel - np.cross([0.0, 0.0, w], pos)
    return pos, vel


def kepler(pos, vel, mu, om_e):
    """Osculating elements of an Earth-fixed state, in the inertial frame aligned with the
    Earth-fixed one at that instant: (a, e, i, node longitude, argument of perigee, mean
    anomaly), the node longitude being Earth-fixed."""
    r = np.asarray(pos, float)
    v = np.asarray(vel, float) + np.cross([0.0, 0.0, om_e], r)
    rn, v2 = np.linalg.norm(r), v @ v
    h = np.cross(r, v)
    a = 1.0 / (2.0 / rn - v2 / mu)
    ev = ((v2 - mu / rn) * r - (r @ v) * v) / mu
    ecc = np.linalg.norm(ev)
    inc = math.acos(h[2] / np.linalg.norm(h))
    node = np.cross([0.0, 0.0, 1.0], h)
    raan = math.atan2(node[1], node[0])
    nn = np.linalg.norm(node)
    argp = math.acos(np.clip(node @ ev / (nn * ecc), -1.0, 1.0))
    if ev[2] < 0:
        argp = 2 * math.pi - argp
    nu = math.acos(np.clip(ev @ r / (ecc * rn), -1.0, 1.0))
    if r @ v < 0:
        nu = 2 * math.pi - nu
    ea = 2.0 * math.atan2(
        math.sqrt(1 - ecc) * math.sin(nu / 2), math.sqrt(1 + ecc) * math.cos(nu / 2)
    )
    m0 = ea - ecc * math.sin(ea)
    return a, ecc, inc, raan, argp, m0


def _num(x):
    return "%19.12E" % x


def _orbit_line(vals):
    return "    " + "".join(_num(v) for v in vals) + "\n"


def glonass_record(prn, k, sat, tb_utc):
    """The RINEX R record (4 lines) at tb (UTC, unix seconds)."""
    pos, vel = ecef_state(sat, tb_utc)
    dt = datetime.fromtimestamp(tb_utc, timezone.utc)
    utc_sow = (dt - ge.GPS_EPOCH).total_seconds() % 604800.0
    head = "R%02d %04d %02d %02d %02d %02d %02d" % (
        prn,
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
    )
    lines = [head + _num(0.0) + _num(0.0) + _num(utc_sow) + "\n"]
    for ax, extra in ((0, 0.0), (1, float(k)), (2, 0.0)):
        lines.append(_orbit_line([pos[ax] / 1e3, vel[ax] / 1e3, 0.0, extra]))
    return "".join(lines)


def beidou_record(prn, sat, toe_utc):
    """The RINEX C record (8 lines) with its toe at toe_utc (unix seconds; BDT = UTC + leap -
    14 s, so the toe sits on BDT's own second count)."""
    t_gpst = ge.gpst_of_utc(toe_utc)
    pos, vel = ecef_state(sat, toe_utc)
    mu, om_e = ge.MU["C"], ge.OMEGA_E["C"]
    a, ecc, inc, raan, argp, m0 = kepler(pos, vel, mu, om_e)
    bdt = t_gpst + ge.BDT_GPST  # BDT as seconds since the GPS epoch
    week = int(bdt // 604800.0) - 1356
    sow = bdt - (week + 1356) * 604800.0
    n0 = math.sqrt(mu / a**3)
    omegadot = -1.5 * n0 * J2 * (RE_M / a) ** 2 * math.cos(inc) / (1 - ecc * ecc) ** 2
    omega0 = (
        raan + om_e * sow
    )  # the broadcast node: OMEGA_k = OMEGA0 + (OMEGADOT - om_e) tk - om_e toe
    omega0 = (omega0 + math.pi) % (2 * math.pi) - math.pi
    epoch = ge.GPS_EPOCH + timedelta(seconds=bdt)
    head = "C%02d %04d %02d %02d %02d %02d %02d" % (
        prn,
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second,
    )
    lines = [head + _num(0.0) + _num(0.0) + _num(0.0) + "\n"]
    lines.append(_orbit_line([0.0, 0.0, 0.0, m0]))  # AODE, Crs, dn, M0
    lines.append(_orbit_line([0.0, ecc, 0.0, math.sqrt(a)]))  # Cuc, e, Cus, sqrtA
    lines.append(_orbit_line([sow, 0.0, omega0, 0.0]))  # toe, Cic, OMEGA0, Cis
    lines.append(_orbit_line([inc, 0.0, argp, omegadot]))  # i0, Crc, omega, OMEGA DOT
    lines.append(
        _orbit_line([0.0, 0.0, float(week), 0.0])
    )  # IDOT, spare, BDT week, spare
    lines.append(_orbit_line([0.0, 0.0, 0.0, 0.0]))  # accuracy, SatH1, TGD1, TGD2
    lines.append(_orbit_line([sow, 0.0, 0.0, 0.0]))  # transmission time, AODC
    return "".join(lines)


def header(stamp):
    """The RINEX header, its comment carrying POINTING_MARK."""
    return (
        "%-60sRINEX VERSION / TYPE\n"
        % "     3.04           N: GNSS NAV DATA    M: MIXED"
        + "%-20s%-20s%-20sPGM / RUN BY / DATE\n" % ("gnss_oddsat_pointing", "", stamp)
        + "%-60sCOMMENT\n" % ("TLE-DERIVED POINTING RECORDS, " + ge.POINTING_MARK)
        + "%-60sEND OF HEADER\n" % ""
    )


def epochs(now, step_s):
    """[hour - 1 h, hour + 7 h) in steps of step_s, the hour being now's."""
    h0 = math.floor(now / 3600.0) * 3600.0
    return [h0 - 3600.0 + i * step_s for i in range(int(8 * 3600 / step_s))]


def decoded_sats(paths, now):
    """{(sys, prn)} with a decoded ephemeris current at now."""
    if not paths:
        return set()
    import decoded_eph

    try:
        g = ge.gpst_of_utc(now)
        return {(e[0], e[1]) for e in decoded_eph.Reader(paths).entries(g)}
    except (
        Exception
    ) as e:  # noqa: BLE001 -- a bad cache file must not stop the pointing
        print("decoded ephemerides unreadable (%s)" % e, flush=True)
        return set()


def build(sats, now, step_s, skip=()):
    """The file's text, and the (sys, prn) it holds."""
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d %H%M%S UTC")
    out, held = [header(stamp)], []
    for sysc, prn, norad, k in sats:
        if (sysc, prn) in skip:
            continue
        l1, l2 = tle(norad, now)
        sat = Satrec.twoline2rv(l1, l2)
        for t in epochs(now, step_s):
            out.append(
                glonass_record(prn, k, sat, t)
                if sysc == "R"
                else beidou_record(prn, sat, t)
            )
        held.append((sysc, prn))
    return "".join(out), held


def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--sats", required=True)
    ap.add_argument(
        "--decoded",
        default="",
        help="decoded-ephemeris files or cache directories, comma-separated",
    )
    ap.add_argument("--step-s", type=float, default=600.0)
    ap.add_argument("--every-s", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    sats = parse_sats(a.sats)
    decoded = [p for p in a.decoded.split(",") if p]
    last = None
    while True:
        now = time.time()
        skip = decoded_sats(decoded, now) & {(s, p) for s, p, _, _ in sats}
        key = (math.floor(now / 3600.0), tuple(sorted(skip)))
        if key != last:
            try:
                text, held = build(sats, now, a.step_s, skip)
                write(a.out, text)
                last = key
                print(
                    "wrote %s: %s%s"
                    % (
                        a.out,
                        " ".join("%s%02d" % h for h in held) or "nothing",
                        (
                            "; decoded, left out: "
                            + " ".join("%s%02d" % s for s in sorted(skip))
                            if skip
                            else ""
                        ),
                    ),
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 -- keep the last file and retry
                print("pointing not written: %s" % e, flush=True)
        if a.once:
            return 0
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(main())
