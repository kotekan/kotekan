#!/usr/bin/env python3
"""Broadcast-ephemeris dead-reckoning: fetch + parse the IGS multi-GNSS BRDC RINEX-3 nav
file and predict per-satellite position / range / range-rate / SATELLITE CLOCK -- the two
quantities the TLE/almanac class fundamentally lacks at useful precision (TLE: km orbits,
no usable clock; BRDC: ~1-2 m orbits, ~5 ns clocks, refreshed ~2-hourly).

This is the precision layer under dead-reckoned operations (2026-07-13): with it +
receiver time to <~300 ns, code phase is predictable to <0.3 chips for every visible
satellite -- the search demotes to bootstrap (one-time receiver-clock solve), fallback,
and integrity monitor. The receiver-clock solve itself: measured-vs-predicted code phase
over ALL tracked sats, robust median (each sat is an independent clock measurement;
outliers flag stale ephemerides / maneuvers / false locks).

Supports GPS (G), Galileo (E), BeiDou-3 (C; BDT = GPST - 14 s handled; GEOs excluded
upstream by the BEIDOU-3 name filter). Pure stdlib + math -- no georinex dependency.

Usage:
    from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all
    path = fetch_brdc()                          # cached daily file
    eph = parse_rinex_nav(path)                  # {(sys, prn): [records]}
    pred = predict_all(eph, lat, lon, alt, t_utc)  # {(sys,prn): dict(az, el, range_m,
                                                   #   range_rate_mps, sat_clk_s, ...)}
Self-test: python3 gnss_ephemeris.py            (fetch + predict, prints visible sats)
"""
import json
import gzip
import math
import os
import re
import threading
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

C_LIGHT = 299792458.0
GPS_UTC_LEAP = 18.0          # GPST - UTC (2026; bump at the next leap second)
BDT_GPST = -14.0             # BDT = GPST - 14 s
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Gravitational parameter / earth rotation per constellation ICD.
MU = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14}
OMEGA_E = {"G": 7.2921151467e-5, "E": 7.2921151467e-5, "C": 7.292115e-5}
F_REL = -4.442807633e-10     # relativistic clock coefficient (s/sqrt(m))


# ---------------------------------------------------------------------------------------
# THE SUPPLY LAYER lives in gnss_brdc_supply (split 2026-08-27). It is imported rather than
# merged back so that a caller wanting orbits does not drag in the fetch machinery.
#
# ⚠️ READS ARE FORWARDED, WRITES ARE NOT -- and that asymmetry is deliberate rather than an
# oversight. `from x import *` would COPY the bindings, so a rebind in the supply module
# (LOG_HOOK is rebound, not mutated) would leave this module serving a stale value forever:
# exactly the silent-divergence class this codebase has spent a day removing. The module
# __getattr__ below therefore forwards every read LIVE. It cannot forward writes (PEP 562
# defines no module __setattr__), so `gnss_ephemeris.LOG_HOOK = f` would set an attribute
# here that nothing reads. Writers target gnss_brdc_supply by name; there are five, and they
# all say so at the call site.
import gnss_brdc_supply as _supply
from gnss_brdc_supply import fetch_brdc  # noqa: F401  (physics calls it as a default)


def __getattr__(name):
    """Forward every supply name LIVE, so the two modules can never disagree about state."""
    try:
        return getattr(_supply, name)
    except AttributeError:
        raise AttributeError(
            "module %r has no attribute %r (and neither does gnss_brdc_supply)"
            % (__name__, name))


def __dir__():
    return sorted(set(list(globals()) + dir(_supply)))

def _f(s):
    """RINEX float field ('D' exponents, blanks)."""
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


def parse_rinex_nav(paths):
    """RINEX 3 mixed nav -> {(sys, prn): [eph dicts sorted by toe_gpst]}. G/E/C only.

    `paths` is a single path OR a list of paths (fetch_brdc returns a list). A list is merged
    PER-PRN: the union of every source's records, deduped by toe, so the freshest valid
    ephemeris for each PRN is available no matter which source carries it. This is what lets a
    frozen daily product (BKG stuck at 08:00 UTC) be transparently backfilled by a sibling
    source (the CDDIS daily, or the station-hourly merge) that still has the sat -- the root of
    the C31/C39 'el --' gap: C39 was absent from the sparse hourly the old single-file fallback
    picked, but present fresh in the CDDIS daily sitting unused in cache. sources are passed
    best-first, so a daily's record wins an identical-toe tie over the same record elsewhere."""
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
                    b.setdefault(round(e["toe_gpst"], 3), e)  # first (best) source wins same toe
        return {k: sorted(v.values(), key=lambda e: e["toe_gpst"]) for k, v in merged.items()}
    return _parse_rinex_nav_file(paths)


def glonass_freq_channels(paths=None):
    """{slot: frequency-channel number k} for GLONASS, read from the SAME BRDC files everything
    else uses. `paths` defaults to fetch_brdc().

    ★ WHY THIS EXISTS AS ITS OWN PARSER. GLONASS is FDMA: every satellite transmits the SAME
    511-chip code and is separated by CARRIER, satellite k sitting at 1246.0 + k*0.4375 MHz on
    L2. So a GLONASS receiver needs k per satellite before it can even search -- and k is not in
    the TLE, not in the satellite name, and NOT a constant (it is reassigned as satellites are
    replaced, so a hardcoded table silently goes stale and points the search at the wrong
    carrier).

    ★ AND WHY IT IS NOT parse_rinex_nav(). That parser filters `sysc in "GEC"` and skips GLONASS
    entirely, correctly: GLONASS broadcasts a position/velocity/acceleration state vector to be
    integrated by Runge-Kutta, not a Keplerian element set, so none of the orbit code applies.
    But k is just an INTEGER sitting in those same records -- no orbit maths needed to read it.
    This function takes only that one field and ignores everything else, so we get the frequency
    plan from an authoritative live source without owning a GLONASS propagator.

    RINEX 3 GLONASS record = 4 lines: epoch/clock, then X/X'/X''/health, Y/Y'/Y''/**freq num**,
    Z/Z'/Z''/age. k is the 4th field of the SECOND orbit line. Returns {} if nothing parses.
    ⚠️ k is NOT unique across the constellation: antipodal satellites share a channel (they are
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
                out.setdefault(slot, k)  # first (freshest source) wins
    return out


def _parse_rinex_nav_file(path):
    """Parse ONE RINEX 3 mixed-nav file -> {(sys, prn): [eph dicts sorted by toe_gpst]}."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        i += 1
    i += 1
    out = {}
    # broadcast orbit line counts per system (RINEX 3): G/E/C all use 7 orbit lines.
    while i < len(lines):
        line = lines[i]
        sysc = line[0] if line else ""
        if sysc not in "GEC":
            i += 1
            continue
        try:
            prn = int(line[1:3])
            yy, mo, dd, hh, mi, ss = (int(line[4:8]), int(line[9:11]), int(line[12:14]),
                                      int(line[15:17]), int(line[18:20]), int(line[21:23]))
            af = [_f(line[23:42]), _f(line[42:61]), _f(line[61:80])]
            orb = []
            for k in range(1, 8):
                l2 = lines[i + k]
                orb += [_f(l2[4:23]), _f(l2[23:42]), _f(l2[42:61]), _f(l2[61:80])]
        except Exception:
            i += 1
            continue
        i += 8
        toc = datetime(yy, mo, dd, hh, mi, ss, tzinfo=timezone.utc)  # in the SYSTEM time
        toc_gpst = (toc - GPS_EPOCH).total_seconds()
        if sysc == "C":
            toc_gpst -= BDT_GPST  # BDT epoch -> GPST seconds-since-GPS-epoch
        e = dict(sys=sysc, prn=prn, af0=af[0], af1=af[1], af2=af[2],
                 crs=orb[1], dn=orb[2], m0=orb[3],
                 cuc=orb[4], ecc=orb[5], cus=orb[6], sqrta=orb[7],
                 toe_sow=orb[8], cic=orb[9], omega0=orb[10], cis=orb[11],
                 i0=orb[12], crc=orb[13], omega=orb[14], omegadot=orb[15],
                 # health = BROADCAST ORBIT 6 field 2 (orb[21]): SV health (G/E) / SatH1 (C).
                 # It was orb[25] -- field 2 of orbit line 7 -- which is a DIFFERENT quantity in
                 # every system, and the mistake was invisible in two of the three:
                 #   G: Fit interval  -> reads 4.0 on every healthy sat -> the strict G/E gate in
                 #      predict_all rejected THE ENTIRE GPS CONSTELLATION (measured 2026-07-16:
                 #      predict_all returned 0 GPS of 32 parsed, so every GPS observable row was
                 #      logged with NO el/az/range -> no CMC, no IPP, and TEC from L1+L2C (both
                 #      GPS!) was impossible). This is the real cause of the "el 1% populated"
                 #      symptom that the BRDC staleness fix (6976642e) only partly addressed.
                 #   E: blank -> 0.0 -> Galileo passed the gate BY ACCIDENT.
                 #   C: AODC -> ~1 -> BeiDou was let through by the SatH1 exemption below, which
                 #      was written for orb[21] all along but never actually read it.
                 idot=orb[16], week=orb[18], health=orb[21], toc_gpst=toc_gpst,
                 # Fields the ORBIT model does not need but the LNAV BIT ENCODER does
                 # (gps_lnav_encode.py): subframe 1 carries URA/IODC/TGD/L2 flags and
                 # subframes 2/3 carry IODE, none of which were kept before 2026-07-25.
                 # RINEX has them all; dropping them silently is what made "encode the
                 # ephemeris back into subframe bits" look impossible from BRDC alone.
                 # ORBIT 1 = [IODE, Crs, dn, M0] (0-3); ORBIT 5 = [IDOT, L2 codes, week,
                 # L2P flag] (16-19); ORBIT 6 = [accuracy, health, TGD, IODC] (20-23);
                 # ORBIT 7 = [transmission time, fit interval] (24-25). health=orb[21]
                 # above is the anchor that fixes this indexing.
                 iode=orb[0], l2_codes=orb[17], l2p_flag=orb[19], accuracy=orb[20],
                 tgd=orb[22], iodc=orb[23], fit=orb[25] if len(orb) > 25 else 0.0)
        # toe as absolute GPST: the RINEX week field is the system's own week count --
        # GPS/GAL: GPS-aligned continuous week; BDS: BDT week (epoch 2006-01-01) + 14 s.
        if sysc in "GE":
            e["toe_gpst"] = e["week"] * 604800.0 + e["toe_sow"]
        else:
            e["toe_gpst"] = (e["week"] + 1356.0) * 604800.0 + e["toe_sow"] - BDT_GPST
        if e["sqrta"] > 1000.0:  # sanity: reject empty/garbled records
            out.setdefault((sysc, prn), []).append(e)
        # (GLONASS and SBAS records have different layouts; skipped by the sysc filter.)
    for k in out:
        out[k].sort(key=lambda e: e["toe_gpst"])
    return out


def sat_pos_clk(e, t_gpst):
    """ICD Kepler propagation -> ECEF position (m), velocity (m/s), clock offset (s).
    t_gpst = seconds since the GPS epoch (continuous, no week rollover)."""
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
    # VELOCITY BY CENTRAL DIFFERENCE. This was a FORWARD difference over 0.5 s, commented
    # "adequate: range-rate to ~mm/s". It is not, and the error is not noise -- a forward
    # difference does not estimate v(t), it estimates v(t + h/2). That is a 0.25 s TIME-TAG
    # OFFSET, and a time offset in Doppler is dop_rate * 0.25.
    #
    # Along the line of sight a 0.37 Hz/s Doppler rate is ~0.094 m/s^2 of range acceleration,
    # so the bias is 0.094 * 0.25 = 0.024 m/s -> 0.09 Hz at 1176.45 MHz. The cross-window phase
    # budget (task #52) needs the carrier to ~15 mHz for 0.1 rad over one 1.049 s window, so
    # this ONE first-order scheme was 6x the entire budget.
    #
    # Central differencing costs one extra position evaluation and is second-order: the leading
    # error becomes (h^2/6)*jerk, and for a GPS-altitude orbit (omega^3 r ~ 8e-5 m/s^3) that is
    # ~3e-6 m/s -- 13 microHz, four orders below the budget. Do NOT shrink h to compensate for
    # the old scheme: the bias is first-order in h, so halving h only halves it, while the
    # central form removes it outright.
    dt = 0.5
    e2 = dict(e)
    pp = _pos_only(e2, t_gpst + dt)
    pm = _pos_only(e2, t_gpst - dt)
    V = tuple((pp[k] - pm[k]) / (2.0 * dt) for k in range(3))
    # clock: polynomial + relativistic
    tc = t_gpst - e["toc_gpst"]
    clk = e["af0"] + e["af1"] * tc + e["af2"] * tc * tc \
        + F_REL * e["ecc"] * e["sqrta"] * math.sin(ek)
    return (X, Y, Z), V, clk


# -- BROADCAST GROUP DELAY (task A0b) -----------------------------------------------------
# Carrier frequencies (Hz) and the gamma = (f_ref/f_signal)^2 scalings the ICDs use.
F_E1 = F_L1 = 1575.42e6
F_E5A = F_L5 = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L5 = (F_L1 / F_L5) ** 2          # 1.79329
GAMMA_E1E5A = (F_E1 / F_E5A) ** 2        # 1.79329
GAMMA_E1E5B = (F_E1 / F_E5B) ** 2        # 1.70325

# RINEX 3 GAL dataSources bits (BROADCAST ORBIT 5 field 2, parsed into e["l2_codes"]):
# bit 8 = clock referenced to (E5a, E1)  [F/NAV];  bit 9 = referenced to (E5b, E1)  [I/NAV].
_GAL_CLK_E5A = 0x100
_GAL_CLK_E5B = 0x200


def group_delay_s(e, signal, dcb=None):
    """Seconds to ADD to the broadcast clock so it refers to `signal`'s own code phase.

    THE BROADCAST CLOCK IS NOT REFERENCED TO OUR SIGNAL. Every constellation fits its clock
    to some other combination -- GPS LNAV to L1/L2, Galileo F/NAV to E1/E5a and I/NAV to
    E1/E5b, BeiDou legacy to B3I -- and a single-frequency user of anything else owes the
    corresponding differential. Until 2026-08-23 nothing here applied it: the decoders
    parsed TGD/BGD and the correction path ignored them (buglist A0b).

    SIZE, MEASURED on the 2026-08-23 BRDC before writing this: the CONSTELLATION-COMMON
    part is small -- GPS TGD median -8.4 ns -> +0.15 chips at L5, Galileo BGD -0.9 ns ->
    +0.02 chips -- and a common term is degenerate with the receiver clock anyway, so a
    chain that solves its own clock never sees it. What this actually buys is the PER-SAT
    spread: GPS TGD ranges -17.7..+6.5 ns and BDS TGD +-45 ns, i.e. +-0.3..0.5 chips
    landing straight in b_sat. Do NOT expect it to move a constellation offset.

    OUR GALILEO BRDC IS MIXED, and best_eph picks by freshness: 19 sats carried F/NAV
    (clock ref E5a,E1) and 11 carried I/NAV (clock ref E5b,E1) on 2026-08-23. So the
    reference a satellite gets is arbitrary AND CAN FLIP at a refresh. Both cross-type
    conversions are implemented from the ICD identity t_E1 = t_IF_a - BGD_a = t_IF_b -
    BGD_b; without them a record-type flip would step that satellite's clock by a few ns.

    Returns 0.0 when the term is unavailable rather than guessing -- notably BeiDou B2a,
    whose TGD_B2ap lives in B-CNAV2 and is absent from RINEX 3 (only TGD1 B1I/B3I and
    TGD2 B2I/B3I are broadcast there).
    """
    if not signal:
        return 0.0
    sig = str(signal).lower()
    sysc = e.get("sys")
    # MEASURED BIAS BEATS BROADCAST, when we have it (2026-08-23). The MGEX DCB product
    # covers every signal we track -- including BeiDou B2a, whose broadcast term is not in
    # RINEX 3 at all -- and is a global-network estimate rather than the satellite's own.
    # ⚠️ It is ZERO-MEAN PER CONSTELLATION by construction, so it supplies the PER-SAT
    # spread and never a constellation-common level; the broadcast term is the only source
    # of the latter, and for B2a there is none. Falls through to broadcast when a pair is
    # missing, so a partial product degrades per-satellite rather than all-or-nothing.
    if dcb:
        try:
            import gnss_dcb as _dcbmod
            _inav = bool(int(e.get("l2_codes") or 0) & _GAL_CLK_E5B) if sysc == "E" else False
            _v = _dcbmod.signal_bias_s(dcb, sysc, int(e.get("prn") or 0), sig,
                                       gal_inav=_inav)
            if _v is not None and math.isfinite(_v):
                return _v
        except Exception:
            pass
    tgd = float(e.get("tgd") or 0.0)        # G: TGD | E: BGD(E1,E5a) | C: TGD1(B1I,B3I)
    tgd2 = float(e.get("iodc") or 0.0)      # E: BGD(E1,E5b) | C: TGD2(B2I,B3I) | G: IODC(!)
    if sysc == "G":
        # IS-GPS-705: an L5 user owes gamma_15 * TGD (plus ISC_L5I5, which is CNAV-only and
        # not in RINEX 3 -- so this is the larger half of the term, not all of it).
        if "l5" in sig:
            return -GAMMA_L1L5 * tgd
        if "l1" in sig:
            return -tgd
        return 0.0
    if sysc == "E":
        ds = int(e.get("l2_codes") or 0)
        ref_a = bool(ds & _GAL_CLK_E5A)     # F/NAV: clock is the E1/E5a iono-free combination
        ref_b = bool(ds & _GAL_CLK_E5B)     # I/NAV: clock is the E1/E5b iono-free combination
        if not (ref_a or ref_b):
            ref_a = True                    # unflagged records: treat as F/NAV, the ICD default
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
        # BDS legacy D1/D2 clock is B3I-referenced. B2b shares B2I's carrier (1207.14 MHz),
        # so TGD2 is the right band term for it; B2a (1176.45) needs TGD_B2ap, which this
        # message does not carry -- return 0 and stay honest rather than borrow TGD2.
        if "b2b" in sig:
            return -tgd2
        if "b1" in sig:
            return -tgd
        return 0.0
    return 0.0


def _pos_only(e, t_gpst):
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
    """Freshest record whose toe is within max_age (4 h) of t; None if all stale."""
    cand = [e for e in records if abs(t_gpst - e["toe_gpst"]) < max_age]
    return min(cand, key=lambda e: abs(t_gpst - e["toe_gpst"])) if cand else None


def predict_all(eph, lat, lon, alt, t_utc, mask_deg=0.0, max_age=14400.0, signal=None,
                dcb=None):
    """Per visible sat: az/el, geometric range (m) with Earth-rotation (Sagnac)
    correction, range-rate (m/s), sat clock (s). Receiver clock NOT included --
    solving it from measured-vs-predicted IS the time bootstrap.

    max_age (s) is the toe validity window (default 4 h). GEOMETRY-only callers
    (observables az/el/range for the sky map / mapping factor) can widen it so a
    stale-in-memory or offline-cached ephemeris still yields az/el through a network
    gap -- Keplerian propagation is sub-degree/few-km for many hours past toe. The
    precise-clock callers (dead-reckon) keep the tight default; toe_age_s is per-sat
    so a consumer can still gate on freshness."""
    t = gpst_of_utc(t_utc)
    rx = _ecef_of_llh(lat, lon, alt)
    out = {}
    for key, recs in eph.items():
        e = best_eph(recs, t, max_age)
        if e is None:
            continue
        if e.get("health", 0) not in (0, 0.0) and key[0] != "C":
            # BDS exemption -- KEPT, but its original rationale was an artifact and is void:
            # it was written when health was misparsed from orb[25], which for BDS is AODC
            # (~1 on everything). The 2026-07-13 "healthy C21 reads health=1" evidence was
            # really "C21's AODC is 1" and proves nothing about SatH1. With the field fixed
            # (orb[21]) BDS SatH1 reads 0.0 on 36 of 37 sats, i.e. it looks as trustworthy
            # as G/E. Left permissive for now because a BDS-3 sat CAN legitimately carry
            # SatH1=1 (the word is B1I-referenced) while its B1C is fine -- dropping it would
            # silently lose that sat. The measured-vs-predicted integrity residuals referee.
            # TODO: re-test strict-for-C on sky now that the word is real; if no healthy sat
            # is lost, delete this exemption.
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
        # A0b: the broadcast clock refers to some OTHER signal combination; make it refer
        # to ours. `signal=None` (every non-broker caller) keeps the old, uncorrected value.
        _gd = group_delay_s(e, signal, dcb)
        out[key] = dict(az=az, el=el, range_m=rng, range_rate_mps=rr,
                        sat_clk_s=clk + _gd, tgd_s=_gd,
                        toe_age_s=t - e["toe_gpst"])
    return out


if __name__ == "__main__":
    lat, lon, alt = 43.968697, -79.252106, 260.0
    path = fetch_brdc()
    print("BRDC:", path)
    eph = parse_rinex_nav(path)
    print("parsed: %d sats (%s)" % (len(eph), "".join(sorted(set(k[0] for k in eph)))))
    pred = predict_all(eph, lat, lon, alt, datetime.now(timezone.utc), mask_deg=5.0)
    print("visible now: %d" % len(pred))
    for (s, p), v in sorted(pred.items(), key=lambda kv: -kv[1]["el"])[:12]:
        dop = -v["range_rate_mps"] / C_LIGHT * 1575.42e6
        print("  %s%02d el %4.1f az %5.1f  range %8.1f km  dop %+7.1f Hz  clk %+9.3f us  toe_age %4.0f min"
              % (s, p, v["el"], v["az"], v["range_m"] / 1e3, dop,
                 v["sat_clk_s"] * 1e6, v["toe_age_s"] / 60))
