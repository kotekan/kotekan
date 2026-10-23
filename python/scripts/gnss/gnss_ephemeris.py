#!/usr/bin/env python3
"""Broadcast-ephemeris prediction: parse the IGS multi-GNSS BRDC RINEX 3 nav file and predict
per-satellite position, range, range rate and satellite clock, the precision a TLE/almanac
lacks (TLE: km orbits, no usable clock; BRDC: ~1-2 m orbits, ~5 ns clocks, refreshed about
every 2 h).

This is the precision layer under the broker's dead reckoning: with BRDC and a receiver clock
known to ~300 ns, every visible satellite's code phase is predictable to about the same, so the
search serves as bootstrap, fallback and integrity monitor (gnss_broker/deadreckon solves the
clock).

Supports GPS (G), Galileo (E), BeiDou (C; BDT = GPST - 14 s) and GLONASS (R; a PZ-90.11 state
vector integrated by Runge-Kutta rather than a Keplerian set). BeiDou GEO records get the
MEO/IGSO equations; the ICD's GEO rotation is not applied. Pure stdlib and math.

Usage:
    from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all
    paths = fetch_brdc()                         # cached nav files, best-first
    eph = parse_rinex_nav(paths)                 # {(sys, prn): [records]}
    pred = predict_all(eph, lat, lon, alt, t_utc)  # {(sys,prn): dict(az, el, range_m,
                                                   #   range_rate_mps, sat_clk_s, ...)}
"""
import gzip
import math

import gnss_atmos
import warnings
from datetime import datetime, timezone

C_LIGHT = 299792458.0
GPS_UTC_LEAP = 18.0  # GPST - UTC, s; update at the next leap second
BDT_GPST = -14.0  # BDT = GPST - 14 s
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Gravitational parameter / earth rotation per constellation ICD (S: WGS84, the MT9 frame).
MU = {
    "G": 3.986005e14,
    "E": 3.986004418e14,
    "C": 3.986004418e14,
    "R": 3.986004418e14,
    "S": 3.986004418e14,
}
OMEGA_E = {
    "G": 7.2921151467e-5,
    "E": 7.2921151467e-5,
    "C": 7.292115e-5,
    "R": 7.292115e-5,
    "S": 7.2921151467e-5,
}
# PZ-90.11 (GLONASS ICD): the oblateness term of the equations of motion. PZ-90.11 and ITRF
# agree to centimetres, so no frame transform is applied to the result.
A_E_PZ90 = 6378136.0
J2_PZ90 = 1082.62575e-6
GLONASS_STEP_S = 60.0  # RK4 step; the ICD's recommended maximum
GLONASS_INTENDED_S = 900.0  # a record is meant for +-15 min about its epoch tb
# The clock-grade window for an R record. The state-vector propagation, judged against the
# next record (docs/CHORD_MULTIBAND.md section 6), errs 18 m line-of-sight median (0.6 chip at
# 10.23 Mcps, p90 1.5 chips) at 2 h and 117 m at 4 h, where the Keplerian systems hold for
# the default 4 h; 2.5 h keeps a seed inside the spectrum fit's capture while leaving the
# hourly merge's 35-95 min record age nearly an hour of slack.
GLONASS_MAX_AGE_S = 9000.0
# A GEO's MT9 state vector (RINEX S record, 256 s cadence in the BRDC) propagated as a two-body
# orbit (sbas_state) errs ~85 m median against the next records at 60-90 min of age, ~160 m at
# 90-120 min and ~210 m (0.7 L5 chip) at 2 h; a chain seeds from a record 1-2 h old.
SBAS_MAX_AGE_S = 7200.0
SBAS_STEP_S = 60.0  # RK4 step of the two-body propagation (sbas_state), as GLONASS's
# How sbas_state carries an S record away from its epoch: "twobody" integrates the state vector
# as a point-mass orbit, "constaccel" is the MT9 polynomial pos + vel t + acc t^2 / 2 (the only
# model before 2026-10-23, kept for comparison). The function's `model` keyword overrides.
SBAS_MODEL = "twobody"
# GLONASS L3OC per-satellite timing against the FDMA broadcast clock (#210). The broadcast
# clock (-tau_n) is fitted to the FDMA L1/L2 signals; some satellites transmit the CDMA L3OC
# code with a large, stable offset from it: the two GLONASS-K2 (SVN 803, 804) ~135-138 m EARLY,
# the three newest K1+/K1B (807, 808, 813) ~41-44 m LATE, the rest within +-3 m. Measured as
# P - model on IGS stations' C3Q pseudoranges with this module's model and the FDMA clock, the
# GPS-L5 station clock removed (fixtures/jointstate/glo_l3_station.py; CHWK and WILL full days
# 2026-10-09/15/18, CHWK, WILL and BREW 2026-10-12 04-12Z), each station-day referred to its
# median over the constellation: the constellation-common part is the receiver's inter-system
# bias, which a chain's own clock or, for the model-primary glo_l3oc chain on the joint clock,
# the per-satellite trims absorb; a common level moves nothing here. Station-days agree to 1 m
# on the K2 pair and 4-8 m on the K1 three (10-12 and BREW read low: the FDMA clock's
# day-to-day). CAS DCB publishes no GLONASS C3 lines, so this is the only
# source. Metres of pseudorange residual, P - model; the clock term is -value / c. Keyed by
# SVN because the slot assignments change (803 moved from slot R26 to R08 on 2026-09-25); the
# slot history is igs_satellite_metadata.snx's SATELLITE/PRN block for these SVNs. A slot
# with no entry is at the datum (0.0). 1.3-4.7 chips at 10.23 Mcps: without this term the
# seed lands on a grating lobe of the sparse comb (95.9 m) and the tracker parks there.
GLO_L3_BIAS_M = {803: -135.2, 804: -138.2, 807: +41.0, 808: +43.1, 813: +41.9}
# (slot, SVN, valid from, valid to or None), UTC; the SNX's epochs. Re-measuring: the station
# tool adds this term back for R, so its output stays the FDMA offset, not the remainder.
GLO_SLOT_SVN = (
    (26, 803, "2025-02-01T04:31", "2026-09-25T06:15"),
    (8, 803, "2026-09-25T06:30", None),
    (27, 804, "2025-03-02T00:00", "2025-12-24T05:22"),
    (6, 804, "2025-12-24T05:22:30", None),
    (19, 807, "2025-05-16T21:45", None),
    (20, 808, "2025-10-16T01:00", None),
    (23, 813, "2026-09-17T06:00", None),
)


def _gpst_of_iso(s):
    return (datetime.fromisoformat(s).replace(tzinfo=timezone.utc) - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP


_GLO_SLOT_SVN_GPST = tuple(
    (slot, svn, _gpst_of_iso(t0), None if t1 is None else _gpst_of_iso(t1))
    for slot, svn, t0, t1 in GLO_SLOT_SVN
)


def glonass_svn(slot, t_gpst):
    """The SVN in GLONASS `slot` at `t_gpst` per GLO_SLOT_SVN, or None when the history has no
    entry (only the SVNs with an L3OC bias are listed)."""
    for s, svn, t0, t1 in _GLO_SLOT_SVN_GPST:
        if s == slot and t_gpst >= t0 and (t1 is None or t_gpst < t1):
            return svn
    return None


def glonass_l3_bias_s(slot, t_gpst):
    """Seconds to ADD to the FDMA broadcast clock so it refers to the L3OC code of the satellite
    in `slot` at `t_gpst` (GLO_L3_BIAS_M, keyed by SVN through GLO_SLOT_SVN): a satellite whose
    L3OC code arrives late (P - model positive) gets a negative term. 0.0 off the table."""
    svn = glonass_svn(slot, t_gpst)
    if svn is None:
        return 0.0
    return -GLO_L3_BIAS_M.get(svn, 0.0) / C_LIGHT

F_REL = -4.442807633e-10  # relativistic clock coefficient (s/sqrt(m))


# ---------------------------------------------------------------------------------------
# The BRDC supply (fetch, cache, merge) is gnss_brdc_supply. fetch_brdc is re-exported for
# callers that reach it through this module (the broker, gnss_beam_cube, gen_fleet).
from gnss_brdc_supply import fetch_brdc  # noqa: F401  (re-exported)


def _f(s):
    """RINEX float field ('D' exponents, blanks)."""
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


def parse_rinex_nav(paths):
    """RINEX 3 mixed nav -> {(sys, prn): [eph dicts sorted by toe_gpst]}. G/E/C/R.

    `paths` is a single path or a list of paths (fetch_brdc returns a list). A list is merged
    per PRN: the union of every source's records, deduplicated by toe, so a satellite missing
    or stale in one product (a frozen daily, the sparse station-hourly merge) is filled from
    another. A file that fails to parse is skipped. The first path wins a same-toe tie, so
    pass sources best-first."""
    if isinstance(paths, (list, tuple)):
        merged = {}
        for p in paths:
            try:
                one = _parse_rinex_nav_file(p)
            except Exception:
                continue
            for key, recs in one.items():
                b = merged.setdefault(key, {})
                for e in recs:
                    b.setdefault(
                        round(e["toe_gpst"], 3), e
                    )  # first (best) source wins same toe
        return {
            k: sorted(v.values(), key=lambda e: e["toe_gpst"])
            for k, v in merged.items()
        }
    return _parse_rinex_nav_file(paths)


def glonass_freq_channels(paths=None):
    """{slot: frequency-channel number k} for GLONASS, read from the same BRDC files everything
    else uses. `paths` defaults to fetch_brdc().

    GLONASS is FDMA: every satellite transmits the same 511-chip code and is separated by
    carrier, satellite k sitting at 1246.0 + k*0.4375 MHz on L2. A receiver needs k per
    satellite before it can search, and k is not in the TLE or the satellite name and is
    reassigned as satellites are replaced, so a hardcoded table goes stale.

    parse_rinex_nav() also yields k (e["k"]), but parses every system's full records; this
    reads the one integer and nothing else.

    RINEX 3 GLONASS record = 4 lines: epoch/clock, then X/X'/X''/health, Y/Y'/Y''/freq num,
    Z/Z'/Z''/age; k is the 4th field of the second orbit line. Returns {} if nothing parses.
    k is not unique across the constellation: antipodal satellites share a channel (they are
    never simultaneously visible), so expect ~14 distinct values over ~24 satellites.
    """
    if paths is None:
        paths = fetch_brdc()
    if isinstance(paths, str):
        paths = [paths]
    out = {}
    for path in paths:
        try:
            op = gzip.open if str(path).endswith(".gz") else open
            with op(path, "rt", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue
        i = 0
        while i < len(lines) and "END OF HEADER" not in lines[i]:
            i += 1
        i += 1
        while i + 3 < len(lines):
            line = lines[i]
            if not line.startswith("R"):
                i += 1
                continue
            try:
                slot = int(line[1:3])
                k = int(round(_f(lines[i + 2][61:80])))
            except Exception:
                i += 1
                continue
            i += 4
            if -7 <= k <= 6 and 1 <= slot <= 32:
                out.setdefault(slot, k)  # first (best) source wins
    return out


def _parse_rinex_nav_file(path):
    """Parse one RINEX 3 mixed-nav file -> {(sys, prn): [eph dicts sorted by toe_gpst]}."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        i += 1
    i += 1
    out = {}
    # broadcast orbit line counts per system (RINEX 3): G/E/C use 7 orbit lines, R uses 3.
    while i < len(lines):
        line = lines[i]
        sysc = line[0] if line else ""
        if sysc not in "GECRS":
            i += 1
            continue
        norb = 3 if sysc in "RS" else 7
        try:
            prn = int(line[1:3])
            yy, mo, dd, hh, mi, ss = (
                int(line[4:8]),
                int(line[9:11]),
                int(line[12:14]),
                int(line[15:17]),
                int(line[18:20]),
                int(line[21:23]),
            )
            af = [_f(line[23:42]), _f(line[42:61]), _f(line[61:80])]
            orb = []
            for k in range(1, norb + 1):
                l2 = lines[i + k]
                orb += [_f(l2[4:23]), _f(l2[23:42]), _f(l2[42:61]), _f(l2[61:80])]
        except Exception:
            i += 1
            continue
        i += norb + 1
        toc = datetime(
            yy, mo, dd, hh, mi, ss, tzinfo=timezone.utc
        )  # in the system's own time
        toc_gpst = (toc - GPS_EPOCH).total_seconds()
        if sysc == "C":
            toc_gpst -= BDT_GPST  # BDT epoch -> GPST seconds-since-GPS-epoch
        if sysc == "S":
            # SBAS GEO: epoch in GPS time, orbit lines X/X'/X''/health, Y/Y'/Y''/URA, Z/Z'/Z''/IODN
            # in km (RINEX 3.04 table A14); propagated by sbas_state.
            e = _sbas_record(prn, af, orb, toc_gpst)
            if e is not None:
                out.setdefault((sysc, e["prn"]), []).append(e)  # keyed by the PRN (S31 -> 131)
            continue
        if sysc == "R":
            e = _glonass_record(prn, af, orb, toc_gpst + GPS_UTC_LEAP)
            if e is not None:
                out.setdefault((sysc, prn), []).append(e)
            continue
        e = dict(
            sys=sysc,
            prn=prn,
            af0=af[0],
            af1=af[1],
            af2=af[2],
            crs=orb[1],
            dn=orb[2],
            m0=orb[3],
            cuc=orb[4],
            ecc=orb[5],
            cus=orb[6],
            sqrta=orb[7],
            toe_sow=orb[8],
            cic=orb[9],
            omega0=orb[10],
            cis=orb[11],
            i0=orb[12],
            crc=orb[13],
            omega=orb[14],
            omegadot=orb[15],
            # health = BROADCAST ORBIT 6 field 2 (orb[21]): SV health (G/E), SatH1 (C).
            # Field 2 of orbit line 7 (orb[25]) is a different quantity in every system
            # (GPS fit interval, BeiDou AODC), and predict_all's health gate reads this one.
            idot=orb[16],
            week=orb[18],
            health=orb[21],
            toc_gpst=toc_gpst,
            # Fields the orbit model does not need: subframe 1's URA/IODC/TGD/L2 flags and
            # subframes 2/3's IODE (read by group_delay_s and sky._lnav_brdc_xcheck).
            # ORBIT 1 = [IODE, Crs, dn, M0] (0-3); ORBIT 5 = [IDOT, L2 codes, week,
            # L2P flag] (16-19); ORBIT 6 = [accuracy, health, TGD, IODC] (20-23). The names
            # are GPS's; Galileo's l2_codes is its data-sources word and tgd/iodc are
            # BGD(E1,E5a)/BGD(E1,E5b), BeiDou's tgd/iodc are TGD1/TGD2 (see group_delay_s).
            iode=orb[0],
            l2_codes=orb[17],
            l2p_flag=orb[19],
            accuracy=orb[20],
            tgd=orb[22],
            iodc=orb[23],
        )
        # toe as absolute GPST: the RINEX week field is the system's own week count (GPS and
        # Galileo: the GPS-aligned continuous week; BeiDou: the BDT week, epoch 2006-01-01).
        if sysc in "GE":
            e["toe_gpst"] = e["week"] * 604800.0 + e["toe_sow"]
        else:
            e["toe_gpst"] = (e["week"] + 1356.0) * 604800.0 + e["toe_sow"] - BDT_GPST
        if e["sqrta"] > 1000.0:  # sanity: reject empty/garbled records
            out.setdefault((sysc, prn), []).append(e)
        # (SBAS, QZSS and NavIC records are skipped by the sysc filter.)
    for k in out:
        out[k].sort(key=lambda e: e["toe_gpst"])
    return out


def _sbas_record(prn, af, orb, toe_gpst):
    """SBAS S record -> eph dict: ECEF (WGS84) position, velocity and acceleration in metres at
    the record epoch (GPS time), the clock aGf0 + aGf1 (t - toc). PRNs are the RINEX S number
    + 100 (S31 = PRN 131, WAAS). A GEO's state carried by sbas_state is good to half an L5 chip
    for ~100 min from its epoch; SBAS_MAX_AGE_S bounds it."""
    # RINEX writes the state in km, km/s, km/s^2. The rapid BRDC (BRDC00WRD_R) carries the
    # GAGAN records (S27, S28) in metres, 1000x too far, which a horizon test then passes
    # (2026-10-20: the broker listed PRNs 127/128 as up from DRAO). A geocentric radius above
    # 1e6 km is not an orbit, so it is read as metres; outside 30,000-50,000 km (GEO/IGSO) the
    # record is dropped.
    scale = 1e3
    r_km = math.sqrt(orb[0] ** 2 + orb[4] ** 2 + orb[8] ** 2)
    if r_km > 1.0e6:
        scale = 1.0
        r_km *= 1e-3
    if not 3.0e4 <= r_km <= 5.0e4:
        return None
    return dict(
        sys="S",
        prn=prn + 100,
        af0=af[0],
        af1=af[1],
        frame_time=af[2],
        pos=(orb[0] * scale, orb[4] * scale, orb[8] * scale),
        vel=(orb[1] * scale, orb[5] * scale, orb[9] * scale),
        acc=(orb[2] * scale, orb[6] * scale, orb[10] * scale),
        health=orb[3],
        ura=orb[7],
        iodn=orb[11],
        toc_gpst=toe_gpst,
        toe_gpst=toe_gpst,
    )


def _twobody_deriv(s):
    """d/dt of the inertial state (x, y, z, vx, vy, vz) under point-mass gravity (WGS84 GM)."""
    x, y, z, vx, vy, vz = s
    r2 = x * x + y * y + z * z
    g = -MU["S"] / (r2 * math.sqrt(r2))
    return (vx, vy, vz, g * x, g * y, g * z)


def sbas_state(e, t_gpst, model=None):
    """ECEF (WGS84) position (m) and velocity (m/s) of an SBAS GEO at t_gpst from its MT9 state
    vector (RINEX S record), by `model` (SBAS_MODEL when None).

    "twobody": the record's (position, velocity) is taken into an inertial frame whose axes
    coincide with ECEF at the record epoch (the velocity gains omega x r), integrated as a
    point-mass orbit r'' = -GM r / |r|^3 by 4th-order Runge-Kutta at SBAS_STEP_S or finer, and
    rotated back to ECEF by the Earth rotation over the elapsed time. The record's acceleration
    is unused: it is the ECEF acceleration at MT9's 1.25e-5 m/s^2 LSB (DO-229), so its rounding
    alone moves the quadratic term by up to 0.5 * 6.25e-6 * dt^2 = 40 m per axis at 1 h and
    160 m at 2 h, while the orbit's own acceleration follows from (r, v) and GM to the 0.0004 m/s
    LSB of the velocity. What the point mass leaves out (J2, the Moon and the Sun, ~1e-5 m/s^2
    at GEO) is the remainder. Against the later records of the WAAS GEOs (sbas_eph_check.py,
    the rapid BRDCs of 2026-10-21/22, 5418 record pairs): median position error 85 m (per PRN
    82-88) at 60-90 min of age and 163 m (159-180) at 90-120 min, where the polynomial errs
    195 m (160-246) and 411 m (320-556); the median reaches half an L5 chip (146 m) at ~100 min
    against ~67 min. Under ~15 min the polynomial is the slightly better model (1.2 m median
    against 1.6: its acceleration carries the J2 and lunisolar terms); the hourly BRDC lands
    about an hour late, so 1-2 h of age is the norm for the sbas_l5 seed, one L5 chip is
    293 m and the tracker pulls in +-0.5 chip.

    "constaccel": the MT9 polynomial pos + vel dt + acc dt^2 / 2, the ICD's extrapolation for
    minutes about the epoch.

    Past SBAS_MAX_AGE_S from the epoch a warning is raised, as glonass_state does past its
    intended window; predict_all's clock-grade window stops there."""
    if model is None:
        model = SBAS_MODEL
    dt = t_gpst - e["toe_gpst"]
    if abs(dt) > SBAS_MAX_AGE_S:
        warnings.warn("SBAS record propagated beyond 2 h of its epoch", stacklevel=2)
    if model == "constaccel":
        pos = tuple(
            e["pos"][i] + e["vel"][i] * dt + 0.5 * e["acc"][i] * dt * dt for i in range(3)
        )
        vel = tuple(e["vel"][i] + e["acc"][i] * dt for i in range(3))
        return pos, vel
    if model != "twobody":
        raise ValueError("sbas_state: unknown model %r" % (model,))
    w = OMEGA_E["S"]
    x, y, z = e["pos"]
    vx, vy, vz = e["vel"]
    # Inertial axes = ECEF axes at the epoch: the position carries over, the velocity gains the
    # frame's own motion omega x r = (-w y, w x, 0).
    s = [x, y, z, vx - w * y, vy + w * x, vz]
    n = max(1, int(math.ceil(abs(dt) / SBAS_STEP_S)))
    h = dt / n
    for _ in range(n):
        k1 = _twobody_deriv(s)
        k2 = _twobody_deriv([s[i] + 0.5 * h * k1[i] for i in range(6)])
        k3 = _twobody_deriv([s[i] + 0.5 * h * k2[i] for i in range(6)])
        k4 = _twobody_deriv([s[i] + h * k3[i] for i in range(6)])
        s = [
            s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(6)
        ]
    # ECEF has turned eastward by w dt since the epoch: take omega x r back out of the velocity
    # and rotate both vectors about z by -w dt.
    x, y, z, vx, vy, vz = s
    vx, vy = vx + w * y, vy - w * x
    c, sn = math.cos(w * dt), math.sin(w * dt)
    return (c * x + sn * y, c * y - sn * x, z), (c * vx + sn * vy, c * vy - sn * vx, vz)


def _glonass_record(prn, af, orb, tb_gpst):
    """GLONASS R record -> eph dict. The three orbit lines carry X/X'/X''/health, Y/Y'/Y''/k and
    Z/Z'/Z''/age in PZ-90.11 km; stored here in metres. The epoch is tb in UTC, so the caller
    passes it as GPST. af0/af1 are the RINEX fields (-tau_n, +gamma_n), the same sign convention
    as the G/E/C clock polynomial, so the clock reads af0 + af1 * (t - toc) for every system."""
    pos = (orb[0] * 1e3, orb[4] * 1e3, orb[8] * 1e3)
    if (
        math.sqrt(sum(p * p for p in pos)) < 1.0e6
    ):  # sanity: reject empty/garbled records
        return None
    return dict(
        sys="R",
        prn=prn,
        af0=af[0],
        af1=af[1],
        frame_time=af[2],
        pos=pos,
        vel=(orb[1] * 1e3, orb[5] * 1e3, orb[9] * 1e3),
        acc=(orb[2] * 1e3, orb[6] * 1e3, orb[10] * 1e3),
        health=orb[3],
        k=int(round(orb[7])),
        age=orb[11],
        toc_gpst=tb_gpst,
        toe_gpst=tb_gpst,
    )


def _glonass_deriv(s, acc):
    """d/dt of the state (x, y, z, vx, vy, vz) in the rotating PZ-90.11 frame: central + J2
    gravity, centrifugal and Coriolis terms, and the record's lunisolar acceleration held
    constant (GLONASS ICD equations of motion)."""
    x, y, z, vx, vy, vz = s
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    mu = MU["R"]
    w = OMEGA_E["R"]
    g = -mu / (r2 * r)
    j = -1.5 * J2_PZ90 * mu * A_E_PZ90 * A_E_PZ90 / (r2 * r2 * r)
    z5 = 5.0 * z * z / r2
    return (
        vx,
        vy,
        vz,
        g * x + j * x * (1.0 - z5) + w * w * x + 2.0 * w * vy + acc[0],
        g * y + j * y * (1.0 - z5) + w * w * y - 2.0 * w * vx + acc[1],
        g * z + j * z * (3.0 - z5) + acc[2],
    )


def glonass_state(e, t_gpst):
    """PZ-90.11 ECEF position (m) and velocity (m/s) at t_gpst: 4th-order Runge-Kutta from the
    record's state at tb, at GLONASS_STEP_S or finer. A record is meant for +-15 min about tb;
    further is allowed with a warning, since the oblateness-only model drifts past that."""
    dt = t_gpst - e["toe_gpst"]
    if abs(dt) > GLONASS_INTENDED_S:
        warnings.warn(
            "GLONASS record propagated beyond 15 min of its epoch", stacklevel=2
        )
    n = max(1, int(math.ceil(abs(dt) / GLONASS_STEP_S)))
    h = dt / n
    acc = e["acc"]
    s = e["pos"] + e["vel"]
    for _ in range(n):
        k1 = _glonass_deriv(s, acc)
        k2 = _glonass_deriv([s[i] + 0.5 * h * k1[i] for i in range(6)], acc)
        k3 = _glonass_deriv([s[i] + 0.5 * h * k2[i] for i in range(6)], acc)
        k4 = _glonass_deriv([s[i] + h * k3[i] for i in range(6)], acc)
        s = [
            s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(6)
        ]
    return tuple(s[:3]), tuple(s[3:])


def sat_pos_clk(e, t_gpst):
    """ICD propagation -> ECEF position (m), velocity (m/s), clock offset (s): Kepler for
    G/E/C, the integrated state vector for R.
    t_gpst = seconds since the GPS epoch (continuous, no week rollover)."""
    if e["sys"] == "S":
        pos, vel = sbas_state(e, t_gpst)
        return pos, vel, e["af0"] + e["af1"] * (t_gpst - e["toc_gpst"])
    if e["sys"] == "R":
        pos, vel = glonass_state(e, t_gpst)
        # -tau_n + gamma_n * (t - tb); the ICD folds the relativistic term into tau_n.
        return pos, vel, e["af0"] + e["af1"] * (t_gpst - e["toc_gpst"])
    mu = MU[e["sys"]]
    om_e = OMEGA_E[e["sys"]]
    a = e["sqrta"] ** 2
    n = math.sqrt(mu / a ** 3) + e["dn"]
    tk = t_gpst - e["toe_gpst"]
    m = e["m0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + e["ecc"] * math.sin(ek)
    sv = math.sqrt(1 - e["ecc"] ** 2)
    vk = math.atan2(sv * math.sin(ek), math.cos(ek) - e["ecc"])
    phi = vk + e["omega"]
    s2p, c2p = math.sin(2 * phi), math.cos(2 * phi)
    du = e["cus"] * s2p + e["cuc"] * c2p
    dr = e["crs"] * s2p + e["crc"] * c2p
    di = e["cis"] * s2p + e["cic"] * c2p
    u = phi + du
    r = a * (1 - e["ecc"] * math.cos(ek)) + dr
    ik = e["i0"] + di + e["idot"] * tk
    om = e["omega0"] + (e["omegadot"] - om_e) * tk - om_e * e["toe_sow"]
    x, y = r * math.cos(u), r * math.sin(u)
    so, co = math.sin(om), math.cos(om)
    si, ci = math.sin(ik), math.cos(ik)
    X = x * co - y * ci * so
    Y = x * so + y * ci * co
    Z = y * si
    # Velocity by central difference over +-0.5 s. A forward difference estimates v(t + h/2), a
    # 0.25 s time-tag offset: at a 0.37 Hz/s Doppler rate that is 0.024 m/s, 0.09 Hz at L5, six
    # times the ~15 mHz a 0.1 rad phase budget over a 1 s window allows. The central form's
    # leading error is (h^2/6)*jerk, ~3e-6 m/s at GPS altitude.
    dt = 0.5
    e2 = dict(e)
    pp = _pos_only(e2, t_gpst + dt)
    pm = _pos_only(e2, t_gpst - dt)
    V = tuple((pp[k] - pm[k]) / (2.0 * dt) for k in range(3))
    # clock: polynomial + relativistic
    tc = t_gpst - e["toc_gpst"]
    clk = (
        e["af0"]
        + e["af1"] * tc
        + e["af2"] * tc * tc
        + F_REL * e["ecc"] * e["sqrta"] * math.sin(ek)
    )
    return (X, Y, Z), V, clk


# -- Broadcast group delay ----------------------------------------------------------------
# Carrier frequencies (Hz) and the gamma = (f_ref/f_signal)^2 scalings the ICDs use.
F_E1 = F_L1 = 1575.42e6
F_L2 = 1227.60e6
F_E5A = F_L5 = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L2 = (F_L1 / F_L2) ** 2  # 1.64694
GAMMA_E1E5A = (F_E1 / F_E5A) ** 2  # 1.79329
GAMMA_E1E5B = (F_E1 / F_E5B) ** 2  # 1.70325

# RINEX 3 GAL dataSources bits (BROADCAST ORBIT 5 field 2, parsed into e["l2_codes"]):
# bit 8 = clock referenced to (E5a, E1)  [F/NAV];  bit 9 = referenced to (E5b, E1)  [I/NAV].
_GAL_CLK_E5A = 0x100
_GAL_CLK_E5B = 0x200


def group_delay_s(e, signal, dcb=None):
    """Seconds to add to the broadcast clock so it refers to `signal`'s own code phase.

    The broadcast clock is fitted to some other signal combination (GPS LNAV to L1/L2, Galileo
    F/NAV to E1/E5a and I/NAV to E1/E5b, BeiDou D1/D2 to B3I), so a single-frequency user of
    any other signal owes the corresponding group-delay differential.

    A constellation-common part is degenerate with the receiver clock, so a chain that solves
    its own clock never sees it. What this buys is the per-satellite spread (GPS TGD spans tens
    of ns, BeiDou TGD +-45 ns: tenths of a chip at 10.23 Mchip/s), which otherwise lands in
    the satellite's code bias. It does not move a constellation offset.

    The Galileo BRDC mixes F/NAV and I/NAV records and best_eph picks by toe alone, so the clock
    reference a satellite gets can change at a refresh. Both cross-type conversions follow the
    ICD identity t_E1 = t_IF_a - BGD_a = t_IF_b - BGD_b, so a record-type change does not step
    the satellite's clock by the BGD difference.

    Returns 0.0 for a signal not handled below (Galileo E6 and GLONASS L2OC among them) and
    when the term is unavailable rather than guessing, notably BeiDou B2a and B1C, whose
    TGD_B2ap and TGD_B1Cp are in B-CNAV2 and B-CNAV1 and absent from RINEX 3 (which carries
    only TGD1 B1I/B3I and TGD2 B2I/B3I).

    GLONASS L3OC has no broadcast term and no DCB line; its per-satellite offset from the FDMA
    clock is this module's own measured table (glonass_l3_bias_s, #210), looked up by the slot
    and the record's epoch, so a slot reassignment changes the term with the satellite.
    """
    if not signal:
        return 0.0
    sig = str(signal).lower()
    sysc = e.get("sys")
    if sysc == "R":
        if "l3" in sig:
            return glonass_l3_bias_s(int(e.get("prn") or 0), float(e.get("toe_gpst") or 0.0))
        return 0.0
    # Measured DCBs (gnss_dcb.signal_bias_s) take precedence when loaded. They cover GPS L5 and
    # L2C, Galileo E5a/E5b and BeiDou B2a/B2b (B2a's broadcast term is not in RINEX 3 at all)
    # and return None otherwise, so a missing pair falls through to the broadcast term for that
    # satellite alone. The product is zero-mean per constellation, so it carries the
    # per-satellite spread and no constellation-common level; only the broadcast term has one.
    if dcb:
        try:
            import gnss_dcb as _dcbmod

            _inav = (
                bool(int(e.get("l2_codes") or 0) & _GAL_CLK_E5B)
                if sysc == "E"
                else False
            )
            _v = _dcbmod.signal_bias_s(
                dcb, sysc, int(e.get("prn") or 0), sig, gal_inav=_inav
            )
            if _v is not None and math.isfinite(_v):
                return _v
        except Exception:
            pass
    tgd = float(e.get("tgd") or 0.0)  # G: TGD | E: BGD(E1,E5a) | C: TGD1(B1I,B3I)
    tgd2 = float(e.get("iodc") or 0.0)  # E: BGD(E1,E5b) | C: TGD2(B2I,B3I) | G: IODC
    if sysc == "G":
        # L5: IS-GPS-705's term is -TGD + ISC_L5I5 (the form the DCB path computes). ISC_L5I5
        # is CNAV-only and not in RINEX 3, so it is taken as 0. Scaling TGD as 1/f^2
        # (gamma_15 * TGD) would assume ISC_L5I5 = (1 - gamma_15) * TGD, which the ICD does not.
        if "l5" in sig:
            return -tgd
        # L2C: IS-GPS-200's term is -TGD + ISC_L2C, and ISC_L2C is CNAV-only too. TGD is
        # defined on L1P/L2P, so -gamma_12 * TGD is the L2P term exactly, and L2C on the same
        # carrier differs from it only by its code bias against P(Y). Per satellite it fits the
        # CAS product's L2C biases to about 1 ns rms, where -TGD leaves about 5 ns.
        if "l2c" in sig:
            return -GAMMA_L1L2 * tgd
        if "l1" in sig:
            return -tgd
        return 0.0
    if sysc == "E":
        ds = int(e.get("l2_codes") or 0)
        ref_a = bool(
            ds & _GAL_CLK_E5A
        )  # F/NAV: clock is the E1/E5a iono-free combination
        ref_b = bool(
            ds & _GAL_CLK_E5B
        )  # I/NAV: clock is the E1/E5b iono-free combination
        if not (ref_a or ref_b):
            ref_a = True  # unflagged records: treat as F/NAV, the ICD default
        if "e5a" in sig:
            if ref_a:
                return -GAMMA_E1E5A * tgd
            # I/NAV record used for E5a: t_IF_a = t_IF_b + BGD_a - BGD_b, then apply E5a's own.
            return -(GAMMA_E1E5A - 1.0) * tgd - tgd2
        if "e5b" in sig:
            if ref_b:
                return -GAMMA_E1E5B * tgd2
            # F/NAV record used for E5b, the mirror of the above.
            return -tgd - (GAMMA_E1E5B - 1.0) * tgd2
        if "e1" in sig:
            return -(tgd if ref_a else tgd2)
        return 0.0
    if sysc == "C":
        # The D1/D2 clock is B3I-referenced. B2b shares B2I's carrier (1207.14 MHz), so TGD2
        # applies to it; B2a (1176.45 MHz) needs TGD_B2ap, which RINEX 3 does not carry, so it
        # gets 0 rather than a borrowed TGD2. TGD1 is B1I's term; B1C needs TGD_B1Cp from
        # B-CNAV1, also absent from RINEX 3, so B1C gets 0 for the same reason.
        if "b2b" in sig:
            return -tgd2
        if "b1i" in sig:
            return -tgd
        return 0.0
    return 0.0


def _pos_only(e, t_gpst):
    if e["sys"] == "S":
        return sbas_state(e, t_gpst)[0]
    if e["sys"] == "R":
        return glonass_state(e, t_gpst)[0]
    mu = MU[e["sys"]]
    om_e = OMEGA_E[e["sys"]]
    a = e["sqrta"] ** 2
    n = math.sqrt(mu / a ** 3) + e["dn"]
    tk = t_gpst - e["toe_gpst"]
    m = e["m0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + e["ecc"] * math.sin(ek)
    sv = math.sqrt(1 - e["ecc"] ** 2)
    vk = math.atan2(sv * math.sin(ek), math.cos(ek) - e["ecc"])
    phi = vk + e["omega"]
    s2p, c2p = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + e["cus"] * s2p + e["cuc"] * c2p
    r = a * (1 - e["ecc"] * math.cos(ek)) + e["crs"] * s2p + e["crc"] * c2p
    ik = e["i0"] + e["cis"] * s2p + e["cic"] * c2p + e["idot"] * tk
    om = e["omega0"] + (e["omegadot"] - om_e) * tk - om_e * e["toe_sow"]
    x, y = r * math.cos(u), r * math.sin(u)
    so, co = math.sin(om), math.cos(om)
    si, ci = math.sin(ik), math.cos(ik)
    return (x * co - y * ci * so, x * so + y * ci * co, y * si)


def _ecef_of_llh(lat, lon, alt):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    sl, cl = math.sin(math.radians(lat)), math.cos(math.radians(lat))
    so, co = math.sin(math.radians(lon)), math.cos(math.radians(lon))
    N = a / math.sqrt(1 - e2 * sl * sl)
    return ((N + alt) * cl * co, (N + alt) * cl * so, (N * (1 - e2) + alt) * sl)


def _azel(rx, sat, lat, lon):
    dx = [s - r for s, r in zip(sat, rx)]
    sl, cl = math.sin(math.radians(lat)), math.cos(math.radians(lat))
    so, co = math.sin(math.radians(lon)), math.cos(math.radians(lon))
    e = -so * dx[0] + co * dx[1]
    n = -sl * co * dx[0] - sl * so * dx[1] + cl * dx[2]
    u = cl * co * dx[0] + cl * so * dx[1] + sl * dx[2]
    rng = math.sqrt(sum(d * d for d in dx))
    return math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.asin(u / rng)), rng


def gpst_of_utc(t_utc):
    """datetime UTC (or unix seconds) -> continuous GPST seconds since the GPS epoch."""
    if isinstance(t_utc, (int, float)):
        t_utc = datetime.fromtimestamp(t_utc, tz=timezone.utc)
    return (t_utc - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP


def best_eph(records, t_gpst, max_age=14400.0):
    """The record whose toe is nearest t, within max_age (default 4 h); None if all stale."""
    cand = [e for e in records if abs(t_gpst - e["toe_gpst"]) < max_age]
    return min(cand, key=lambda e: abs(t_gpst - e["toe_gpst"])) if cand else None


def predict_all(
    eph,
    lat,
    lon,
    alt,
    t_utc,
    mask_deg=0.0,
    max_age=14400.0,
    signal=None,
    dcb=None,
    health_gate=True,
    tropo=False,
    iono=None,
):
    """Per visible satellite: az/el, geometric range (m) with the Earth-rotation (Sagnac)
    correction, range rate (m/s) and satellite clock (s). The receiver clock is not included:
    solving it from measured minus predicted is the time bootstrap.

    Atmosphere, as separate one-way delays (m) to add to range_m, so a consumer that wants
    geometry only keeps it: `tropo=True` fills tropo_m with the Saastamoinen slant delay from
    the site (gnss_atmos); `iono=(alpha, beta, f_hz)` fills iono_m with the Klobuchar delay at
    carrier f_hz. Both are 0.0 when off.

    max_age (s) is the toe validity window (default 4 h). Geometry-only callers (gnss_observables,
    sky.transit_sky, the beam cube) widen it so a stale or offline-cached ephemeris still yields
    az/el through a network gap; Keplerian propagation stays sub-degree and within a few km for
    many hours past toe. Dead reckoning, which needs the clock, keeps the default; toe_age_s is
    per satellite so a consumer can still gate on freshness. GLONASS records are state vectors
    integrated from tb, so a window at or under the default is cut to GLONASS_MAX_AGE_S for R.

    `signal` (with `dcb`) refers sat_clk_s to that signal's code (group_delay_s; tgd_s is the
    correction applied). signal=None leaves the broadcast clock as it is.

    health_gate=False keeps unhealthy records (BeiDou is never gated, see below), for geometry
    only (sky.transit_sky): an unhealthy satellite still transmits and still captures the
    array. Seeds never set it."""
    t = gpst_of_utc(t_utc)
    rx = _ecef_of_llh(lat, lon, alt)
    out = {}
    for key, recs in eph.items():
        # A clock-grade window (the default or tighter) is cut to GLONASS_MAX_AGE_S for R;
        # geometry callers that widened it keep their window.
        e = best_eph(
            recs,
            t,
            min(max_age, GLONASS_MAX_AGE_S)
            if key[0] == "R" and max_age <= 14400.0
            else min(max_age, SBAS_MAX_AGE_S)
            if key[0] == "S" and max_age <= 14400.0
            else max_age,
        )
        if e is None:
            continue
        if health_gate and e.get("health", 0) not in (0, 0.0) and key[0] not in "CS":
            # SBAS is exempt too: the RINEX S health is MT17's status bitmask (ranging /
            # corrections / integrity flags; WAAS reads 31 with the signal on the air),
            # and a lane direction (#190) needs the transmission, not a ranging licence.
            # BeiDou is exempt: SatH1 describes B1I, so a BDS-3 satellite can carry SatH1 = 1
            # while the signal a chain tracks is fine, and the gate would drop it.
            # TODO: try the strict gate for C on sky; if no healthy satellite is lost, remove
            # the exemption.
            continue
        # light-time + Sagnac: two iterations are plenty
        tau = 0.075
        for _ in range(2):
            pos, vel, clk = sat_pos_clk(e, t - tau)
            # rotate the sat position into the ECEF frame at reception time
            th = OMEGA_E[e["sys"]] * tau
            px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
            py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
            pos_rx = (px, py, pos[2])
            az, el, rng = _azel(rx, pos_rx, lat, lon)
            tau = rng / C_LIGHT
        if el < mask_deg:
            continue
        # vel is in the transmit-epoch frame, like pos: rotate it by the same angle so the
        # range rate is taken in the reception frame.
        vel_rx = (
            vel[0] * math.cos(th) + vel[1] * math.sin(th),
            -vel[0] * math.sin(th) + vel[1] * math.cos(th),
            vel[2],
        )
        # The light time tau = rng/c changes by rr/c per second, which moves the transmit epoch
        # (scaling vel_rx by 1 - rr/c) and the Sagnac angle (d(pos_rx)/d(th) = (py, -px, 0)).
        # With a = u.vel_rx and b = u.(py, -px, 0) for the unit line of sight u,
        # rr = a*(1 - rr/c) + omega*b*rr/c, solved for rr. The terms are up to a few mm/s.
        a = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel_rx)) / rng
        b = (px * rx[1] - py * rx[0]) / rng
        rr = a / (1.0 + (a - OMEGA_E[e["sys"]] * b) / C_LIGHT)
        # Refer the broadcast clock to `signal` (group_delay_s). Only dead reckoning passes
        # one; signal=None leaves the clock as broadcast.
        _gd = group_delay_s(e, signal, dcb)
        out[key] = dict(
            az=az,
            el=el,
            range_m=rng,
            range_rate_mps=rr,
            sat_clk_s=clk + _gd,
            tgd_s=_gd,
            toe_age_s=t - e["toe_gpst"],
            tropo_m=gnss_atmos.tropo_delay_m(el, lat, alt) if tropo else 0.0,
            iono_m=(
                gnss_atmos.klobuchar_delay_m(
                    iono[0], iono[1], lat, lon, el, az, t % 604800.0, iono[2]
                )
                if iono
                else 0.0
            ),
        )
    return out
