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
import warnings
from datetime import datetime, timezone

C_LIGHT = 299792458.0
GPS_UTC_LEAP = 18.0  # GPST - UTC, s; update at the next leap second
BDT_GPST = -14.0  # BDT = GPST - 14 s
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Gravitational parameter / earth rotation per constellation ICD.
MU = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14, "R": 3.986004418e14}
OMEGA_E = {
    "G": 7.2921151467e-5,
    "E": 7.2921151467e-5,
    "C": 7.292115e-5,
    "R": 7.292115e-5,
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
        if sysc not in "GECR":
            i += 1
            continue
        norb = 3 if sysc == "R" else 7
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
F_E5A = F_L5 = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L5 = (F_L1 / F_L5) ** 2  # 1.79329
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

    Returns 0.0 for a signal not handled below (GPS L2C and Galileo E6 among them) and when the
    term is unavailable rather than guessing, notably BeiDou B2a, whose TGD_B2ap is in B-CNAV2
    and absent from RINEX 3 (which carries only TGD1 B1I/B3I and TGD2 B2I/B3I).
    """
    if not signal:
        return 0.0
    sig = str(signal).lower()
    sysc = e.get("sys")
    # Measured DCBs (gnss_dcb.signal_bias_s) take precedence when loaded. They cover GPS L5,
    # Galileo E5a/E5b and BeiDou B2a/B2b (B2a's broadcast term is not in RINEX 3 at all) and
    # return None otherwise, so a missing pair falls through to the broadcast term for that
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
        # L5: TGD scaled to L5 as 1/f^2 (gamma_15 * TGD). IS-GPS-705's term is -TGD + ISC_L5I5
        # (the form the DCB path computes), but ISC_L5I5 is CNAV-only and not in RINEX 3; the
        # scaling matches it only if the satellite's group delays go as 1/f^2.
        if "l5" in sig:
            return -GAMMA_L1L5 * tgd
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
        # gets 0 rather than a borrowed TGD2.
        if "b2b" in sig:
            return -tgd2
        if "b1" in sig:
            return -tgd
        return 0.0
    return 0.0


def _pos_only(e, t_gpst):
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
):
    """Per visible satellite: az/el, geometric range (m) with the Earth-rotation (Sagnac)
    correction, range rate (m/s) and satellite clock (s). The receiver clock is not included:
    solving it from measured minus predicted is the time bootstrap.

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
            else max_age,
        )
        if e is None:
            continue
        if health_gate and e.get("health", 0) not in (0, 0.0) and key[0] != "C":
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
        rr = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel)) / rng
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
        )
    return out
