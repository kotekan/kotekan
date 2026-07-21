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
import gzip
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

C_LIGHT = 299792458.0
GPS_UTC_LEAP = 18.0          # GPST - UTC (2026; bump at the next leap second)
BDT_GPST = -14.0             # BDT = GPST - 14 s
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Gravitational parameter / earth rotation per constellation ICD.
MU = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14}
OMEGA_E = {"G": 7.2921151467e-5, "E": 7.2921151467e-5, "C": 7.292115e-5}
F_REL = -4.442807633e-10     # relativistic clock coefficient (s/sqrt(m))

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")


def fetch_brdc(when=None, cache_dir=CACHE):
    """Fetch (with cache) the IGS multi-GNSS broadcast nav file for `when` (default now).
    Tries today's streamed file then falls back a day (files appear with some latency)."""
    os.makedirs(cache_dir, exist_ok=True)
    when = when or datetime.now(timezone.utc)
    # Process each DAY fully -- fresh cache, then (re-)download, then a STALE same-day cache --
    # BEFORE falling back to the previous day. This ordering is load-bearing: a current-day
    # file predicts even hours stale (its toe's still bracket "now"), but YESTERDAY's file
    # predicts NOTHING today (all toe's > best_eph's 4 h window -> predict_all returns 0 ->
    # the broker seeds/hints nothing -> the require_hint search goes dark on every band).
    # The old code returned a cached yesterday file (back==1) UNCONDITIONALLY the moment a
    # current-day re-fetch failed, handing the broker a dead-on-arrival ephemeris during a
    # BRDC-server outage (2026-07-21: whole node stopped acquiring while today's valid file
    # sat one line up in the cache). Fix: only fall to the previous day when THIS day yields
    # nothing at all -- not merely because its re-download failed.
    for back in (0, 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        locals_ = []
        for kind in ("S", "R"):
            name = "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, d.year, doy)
            locals_.append(os.path.join(cache_dir, name))
        # 1) a FRESH current-day cache (< 2 h) or ANY cached previous-day file (final/immutable)
        #    is usable as-is, no network needed.
        for local in locals_:
            if os.path.exists(local) and (back == 1
                                          or time.time() - os.path.getmtime(local) < 7200):
                return local
        # 2) (re-)download this day (current-day files GROW through the day, so refresh a stale
        #    cache to pull the newest toe's).
        for local in locals_:
            url = ("https://igs.bkg.bund.de/root_ftp/IGS/BRDC/%04d/%03d/%s"
                   % (d.year, doy, os.path.basename(local)))
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                open(local, "wb").write(data)
                return local
            except Exception:
                continue
        # 3) download failed but a STALE same-day cache exists -> use it (still predicts) rather
        #    than falling to the previous day's dead-for-today ephemeris.
        for local in locals_:
            if os.path.exists(local):
                return local
        # nothing for this day at all -> fall back to the previous day (early-UTC-morning case).
    raise RuntimeError("no BRDC file reachable (network?) and no cache")


def _f(s):
    """RINEX float field ('D' exponents, blanks)."""
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


def parse_rinex_nav(path):
    """RINEX 3 mixed nav -> {(sys, prn): [eph dicts sorted by toe_gpst]}. G/E/C only."""
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
                 idot=orb[16], week=orb[18], health=orb[21], toc_gpst=toc_gpst)
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
    # velocity by simple differentiation step (adequate: range-rate to ~mm/s)
    dt = 0.5
    e2 = dict(e)
    p2 = _pos_only(e2, t_gpst + dt)
    V = tuple((p2[k] - p) / dt for k, p in enumerate((X, Y, Z)))
    # clock: polynomial + relativistic
    tc = t_gpst - e["toc_gpst"]
    clk = e["af0"] + e["af1"] * tc + e["af2"] * tc * tc \
        + F_REL * e["ecc"] * e["sqrta"] * math.sin(ek)
    return (X, Y, Z), V, clk


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


def predict_all(eph, lat, lon, alt, t_utc, mask_deg=0.0, max_age=14400.0):
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
        out[key] = dict(az=az, el=el, range_m=rng, range_rate_mps=rr,
                        sat_clk_s=clk, toe_age_s=t - e["toe_gpst"])
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
