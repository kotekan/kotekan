"""Predict from the broker's own decoded ephemeris: the fallback while no BRDC is loaded.

Dead reckoning normally predicts with `gnss_ephemeris.predict_all` over the BRDC files
`fetch_brdc` supplies. While no BRDC ephemeris is loaded (no connectivity, a mirror outage), the
broker's nav decoders still hold full ephemerides, each cross-checked against BRDC
(sky._*_brdc_xcheck). This module turns them into the per-satellite output `predict_all`
produces, so dead reckoning keeps predicting (gnss_broker/deadreckon; decoded_eph_fallback in
gnss_broker/cli.py).

Why not convert decoded records to RINEX and reuse predict_all: the CNAV and B-CNAV2
ephemerides use the modernized orbit model (A_ref+dA, A_dot, dn0_dot, i0_dot), which the legacy
`sat_pos_clk` cannot represent. So each decoder's own `sv_position_*` (the propagator the BRDC
cross-check validates) is reused, and only the geometry wrapper is replicated here: the same
light-time/Sagnac/az-el/range-rate block as `predict_all`.

Scope:
  * Position -> az/el/range: the decoder's propagator.
  * Velocity -> range rate/Doppler: central difference of position (the propagators return no
    velocity), rotated into the reception frame with the position, plus the light-time rate,
    as in predict_all.
  * sat_clk_s: the decoder's clock polynomial plus the relativistic term (as sat_pos_clk)
    where CLOCK_FIELDS lists one (Galileo I/NAV and F/NAV, GPS LNAV), else None; no
    group-delay term. Geometry and Doppler do not need it; only absolute code-phase seeding
    does, and a tracking satellite's DLL holds code phase regardless. So the fallback keeps
    acquisition, hints and the sky alive; cold code seeding degrades where the clock is absent.

Only the broker's own satellites are covered (its decoders), which is the set it predicts.
Every entry is guarded separately so one bad ephemeris cannot take out the rest, and the caller
wraps the whole call so the fallback cannot worsen an outage.
"""

import glob
import importlib
import json
import math
import os

from gnss_ephemeris import (
    BDT_GPST,
    C_LIGHT,
    F_REL,
    MU,
    OMEGA_E,
    _azel,
    _ecef_of_llh,
    gpst_of_utc,
)

# Clock coefficient, clock reference and ephemeris reference (toe, for the relativistic term's
# Kepler solve) field names per decoder family. The decoded eph dicts differ in casing (Galileo
# af0/t0c/t0e); constellations whose decoder does not surface clock are absent here and get
# sat_clk_s = None.
CLOCK_FIELDS = {
    "GAL_E1B_INAV": ("af0", "af1", "af2", "t0c", "t0e"),
    "GAL_E5BI_INAV": ("af0", "af1", "af2", "t0c", "t0e"),
    "GAL_E5AI_FNAV": ("af0", "af1", "af2", "t0c", "t0e"),
    # GPS LNAV carries its clock in subframe 1 (af0/af1/af2 about toc), so unlike CNAV the L1
    # fallback can seed absolute code phase, not just geometry and Doppler.
    "GPS_L1_LNAV": ("af0", "af1", "af2", "toc", "toe"),
    # CNAV's clock is its type 30-37 block, merged by cnav_predictor when it is the same upload.
    "GPS_L5_CNAV": ("af0", "af1", "af2", "toc", "toe"),
    "GPS_L2C_CNAV": ("af0", "af1", "af2", "toc", "toe"),
    # B-CNAV2's clock is its type 30-34 block, merged by bcnav2_predictor for the same upload.
    "BDS_B2A_BCNAV2": ("a0", "a1", "a2", "t_oc", "t_oe"),
    # GuoWang's relay (guowang_csk.py): its clocks with the relay's time-scale offset added back
    "GW_CSK_RELAY": ("af0", "af1", "af2", "toc", "toe"),
    # The almanacs' clocks (nav_almanac.py), about t_oa
    "GAL_ALM": ("af0", "af1", "af2", "toc", "toe"),
    "BDS_ALM": ("af0", "af1", "af2", "toc", "toe"),
}
# Sources for a satellite no other entry covers: an entry from another source replaces theirs,
# and theirs never replaces another's. GuoWang's relay refits its orbits every minute, but as a
# plain Kepler set without secular terms, which drifts 10-20 m in 15 min where the satellite's
# own message does not.
LAST_RESORT = {"GW_CSK_RELAY"}
# Kepler fields the relativistic term reads; every decoder in CLOCK_FIELDS names them alike.
KEPLER_FIELDS = ("M0", "dn", "sqrtA", "e")
# Each decoded signal's propagator, (module, function): sv_pos(eph, t_sow) -> ECEF at transmit
# time, t_sow in the constellation's seconds of week. The broker's own table names the same.
SV_POS = {
    "GPS_L1_LNAV": ("gps_nav_decode", "sv_position_lnav"),
    "GPS_L2C_CNAV": ("gps_cnav", "sv_position_cnav"),
    "GPS_L5_CNAV": ("gps_cnav", "sv_position_cnav"),
    "GAL_E1B_INAV": ("galileo_inav", "sv_position_inav"),
    "GAL_E5BI_INAV": ("galileo_inav", "sv_position_inav"),
    "GAL_E5AI_FNAV": ("galileo_fnav", "sv_position_fnav"),
    "BDS_B1C_BCNAV1": ("beidou_bcnav1", "sv_position_bcnav1"),
    "BDS_B2A_BCNAV2": ("beidou_bcnav2", "sv_position_bcnav2"),
    "GW_CSK_RELAY": ("guowang_csk", "sv_position_relay"),
    "GW_1202_NAV": ("gw_nav_predictor", "sv_position_gw"),
    "GLO_L2OF": ("glonass_nav_predictor", "sv_position_glo"),
}
# Signals whose decoded records are good only near their epoch, by the most they are taken from
# it, s: a GLONASS record is a state vector integrated from tb (gnss_ephemeris.GLONASS_MAX_AGE_S).
MAX_AGE_S = {"GLO_L2OF": 9000.0}
# The almanacs the decoders read (nav_almanac.py), by the same propagator interface. Only a Reader
# made with almanac=True reads them, so nothing that wants an ephemeris predicts from one.
ALM_POS = {
    "GAL_ALM": ("nav_almanac", "sv_position_alm"),
    "BDS_ALM": ("nav_almanac", "sv_position_alm"),
    "GLO_ALM": ("nav_almanac", "sv_position_glo_alm"),
}
# Decoders whose clock is not the Kepler form above, (module, function): clk(eph, t_sow) -> s.
SV_CLK = {
    "GW_1202_NAV": ("gw_nav_predictor", "sv_clock_gw"),
    "GLO_ALM": ("nav_almanac", "sv_clock_glo_alm"),
    "GLO_L2OF": ("glonass_nav_predictor", "sv_clock_glo"),
}
# Constellations on BDT (GPST - 14 s): BeiDou, and GuoWang, whose every time field is BDT.
BDT_SYS = ("C", "W")


def _week_wrap(dt):
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800
    return dt


def toe_gpst_near(sys, toe_sow, gpst_now):
    """A decoded ephemeris's reference time (seconds of week) as continuous GPST, in the week
    that puts it within half a week of gpst_now: the `toe_gpst` of an entry. BeiDou's t_oe is
    BDT seconds of week (BDT = GPST - 14 s); GPS and Galileo count from the GPST week."""
    toe = toe_sow - BDT_GPST if sys in BDT_SYS else toe_sow
    toe_gpst = int(gpst_now // 604800) * 604800 + toe
    if toe_gpst - gpst_now > 302400:
        toe_gpst -= 604800
    elif toe_gpst - gpst_now < -302400:
        toe_gpst += 604800
    return toe_gpst


def _sat_clock(eph, sys, signal, t_sow, wrap=True):
    """Broadcast SV clock bias (s): af0 + af1*dt + af2*dt^2 about t0c plus the relativistic
    term F*e*sqrtA*sin(E), as sat_pos_clk computes them; dt and tk are week-wrapped unless
    wrap=False (t_sow counting on from the start of toe's week). None when the decoder does
    not carry clock terms for this signal."""
    if signal in SV_CLK:
        mod, fn = SV_CLK[signal]
        return getattr(importlib.import_module(mod), fn)(eph, t_sow)
    fields = CLOCK_FIELDS.get(signal)
    if not fields or not all(k in eph for k in fields + KEPLER_FIELDS):
        return None
    a0, a1, a2, toc, toe = (eph[k] for k in fields)
    dt, tk = t_sow - toc, t_sow - toe
    if wrap:
        dt, tk = _week_wrap(dt), _week_wrap(tk)
    a = eph["sqrtA"] ** 2
    n = math.sqrt(MU[sys] / a ** 3) + eph["dn"]
    m = eph["M0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + eph["e"] * math.sin(ek)
    rel = F_REL * eph["e"] * eph["sqrtA"] * math.sin(ek)
    return a0 + a1 * dt + a2 * dt * dt + rel


def predict_one(
    sv_pos, eph, sys, signal, rx, lat, lon, t_sow, toe_age_s, dt=0.5, wrap=True
):
    """One satellite's az/el/range/range-rate/sat_clk, mirroring predict_all's per-sat block.

    `t_sow` and everything `sv_pos`/`_sat_clock` see is the constellation's seconds of week: the
    decoder propagators (`sv_position_*`) fold `tk = t - t0e` per week and so need a
    seconds-of-week time, not continuous GPST. `sv_pos(eph, t_sow) -> ECEF (x,y,z)`.
    `toe_age_s` (continuous) is supplied by the caller for the report only. wrap=False passes
    t_sow counting on from the start of toe's week, unwrapped, for an issue more than half a
    week old. Returns predict_all's dict without tgd_s, or None on any failure."""
    try:
        kw = {} if wrap else {"wrap": False}
        om_e = OMEGA_E[sys]
        tau = 0.075
        pos_rx = az = el = rng = None
        for _ in range(2):  # light-time + Sagnac, two iterations (== predict_all)
            pos = sv_pos(eph, t_sow - tau, **kw)
            th = om_e * tau
            px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
            py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
            pos_rx = (px, py, pos[2])
            az, el, rng = _azel(rx, pos_rx, lat, lon)
            tau = rng / C_LIGHT
        # velocity by central difference of the un-rotated position, then rotated into the
        # reception frame by the position's angle (== predict_all), so range rate matches
        # predict_all's to the finite-difference error
        pa = sv_pos(eph, t_sow - tau + dt, **kw)
        pb = sv_pos(eph, t_sow - tau - dt, **kw)
        vel = [(a - b) / (2.0 * dt) for a, b in zip(pa, pb)]
        vel_rx = (
            vel[0] * math.cos(th) + vel[1] * math.sin(th),
            -vel[0] * math.sin(th) + vel[1] * math.cos(th),
            vel[2],
        )
        # plus the light-time rate, in predict_all's closed form
        a = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel_rx)) / rng
        b = (px * rx[1] - py * rx[0]) / rng
        rr = a / (1.0 + (a - om_e * b) / C_LIGHT)
        return dict(
            az=az,
            el=el,
            range_m=rng,
            range_rate_mps=rr,
            sat_clk_s=_sat_clock(eph, sys, signal, t_sow - tau, wrap),
            toe_age_s=toe_age_s,
        )
    except Exception:
        return None


def expand(paths):
    """The files named, with a directory (an ephemeris cache, eph_cache.py) standing for the
    *.json files in it."""
    out = []
    for path in paths:
        if os.path.isdir(path):
            out += sorted(glob.glob(os.path.join(path, "*.json")))
        else:
            out.append(path)
    return out


class Reader:
    """Decoded ephemerides from files and cache directories (expand), each file re-read when it
    changes: predict_from_decoders' entries for every satellite they hold. A file that cannot be
    read keeps what it last gave. With almanac=True, the almanacs (ALM_POS) instead."""

    def __init__(self, paths, almanac=False):
        self.paths = list(paths)
        self.table = ALM_POS if almanac else SV_POS
        self.files = {}  # path -> (mtime_ns, its rows)

    def entries(self, gpst_now, sys=None):
        out = []
        for path in expand(self.paths):
            try:
                mt = os.stat(path).st_mtime_ns
                if self.files.get(path, (None,))[0] != mt:
                    with open(path) as fh:
                        doc = json.load(fh)
                    rows = [
                        r
                        for r in doc.get("eph", [])
                        if r.get("signal") in self.table
                        and r.get("toe_field") in r.get("eph", {})
                    ]
                    self.files[path] = (mt, rows)
            except (OSError, ValueError):
                pass
            for r in self.files.get(path, (None, []))[1]:
                if sys is not None and r["sys"] != sys:
                    continue
                mod, fn = self.table[r["signal"]]
                toe = r.get("toe_gpst")
                if toe is None:
                    toe = toe_gpst_near(r["sys"], r["eph"][r["toe_field"]], gpst_now)
                if abs(gpst_now - toe) > MAX_AGE_S.get(r["signal"], float("inf")):
                    continue
                sv_pos = getattr(importlib.import_module(mod), fn)
                out.append(
                    (r["sys"], int(r["prn"]), r["signal"], r["eph"], sv_pos, toe)
                )
        return out


def load_files(paths):
    """{(sys, prn): [(signal, eph, toe_field)]} from nav_decode.py and guowang_csk.py --eph-out
    files and ephemeris cache directories: the satellites' own ephemerides, for a receiver with
    no BRDC. A file that cannot be read is skipped."""
    out = {}
    for path in expand(paths):
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        for r in doc.get("eph", []):
            if r.get("signal") in SV_POS and r.get("toe_field") in r.get("eph", {}):
                key = (r["sys"], int(r["prn"]))
                out.setdefault(key, []).append((r["signal"], r["eph"], r["toe_field"]))
    return out


def state(sys, signal, eph, t_gpst, dt=0.5):
    """(ECEF position, velocity, clock s or None) of a decoded ephemeris at continuous GPST
    t_gpst (transmit time), the velocity by central difference."""
    mod, fn = SV_POS[signal]
    sv_pos = getattr(importlib.import_module(mod), fn)
    ts = t_gpst % 604800.0
    if sys in BDT_SYS:
        ts = (ts + BDT_GPST) % 604800.0
    a, b = sv_pos(eph, ts - dt), sv_pos(eph, ts + dt)
    vel = [(q - p) / (2.0 * dt) for p, q in zip(a, b)]
    return list(sv_pos(eph, ts)), vel, _sat_clock(eph, sys, signal, ts)


def predict_from_decoders(entries, lat, lon, alt, t_utc, mask_deg=0.0, max_age=14400.0):
    """predict_all-shaped output from decoded ephemerides.

    `entries`: list of (sys, prn, signal, eph_dict, sv_pos_fn, toe_gpst). `toe_gpst` is the
    ephemeris reference in continuous GPST, used for the toe-age gate and the report; pass None
    to skip the gate (age unknown). The propagators themselves are fed seconds of week (see
    predict_one). Returns {(sys, prn): {...}}.

    Independent per-entry guards: one malformed ephemeris drops that satellite, never the set.
    Side-effect free, so a caller can wrap it in try/except without risk to the running loop."""
    t = gpst_of_utc(t_utc)
    t_sow = (
        t - (t // 604800) * 604800
    )  # continuous GPST -> seconds-of-week for the propagators
    rx = _ecef_of_llh(lat, lon, alt)
    # One entry per satellite: the issue nearest now, any source's before the relay's. A cache
    # holds a satellite's issues for days, so taking the last given would take any of them.
    best = {}
    for ent in entries:
        sys, prn, signal, toe_gpst = ent[0], ent[1], ent[2], ent[5]
        toe_age = (t - toe_gpst) if toe_gpst is not None else None
        if toe_age is not None and abs(toe_age) > min(
            max_age, MAX_AGE_S.get(signal, max_age)
        ):
            continue
        rank = (signal in LAST_RESORT, abs(toe_age) if toe_age is not None else 0.0)
        if (sys, prn) not in best or rank < best[(sys, prn)][0]:
            best[(sys, prn)] = (rank, ent)
    out = {}
    for _, (sys, prn, signal, eph, sv_pos, toe_gpst) in best.values():
        toe_age = (t - toe_gpst) if toe_gpst is not None else None
        # BeiDou runs on BDT (= GPST - 14 s) and its decoded t_oe is BDT-sow, so the propagator
        # must be fed BDT seconds-of-week. Galileo GST and GPS GPST share the sow origin here.
        ts = (t_sow + BDT_GPST) % 604800 if sys in BDT_SYS else t_sow
        # The propagators wrap t - toe to within half a week, so an older issue is fed the time
        # counting on from the start of its own week, unwrapped.
        wrap = toe_age is None or abs(toe_age) <= 302400
        if not wrap:
            ts = (toe_gpst + (BDT_GPST if sys in BDT_SYS else 0.0)) % 604800 + toe_age
        v = predict_one(sv_pos, eph, sys, signal, rx, lat, lon, ts, toe_age, wrap=wrap)
        if v is None or v["el"] < mask_deg:
            continue
        out[(sys, prn)] = v
    return out
