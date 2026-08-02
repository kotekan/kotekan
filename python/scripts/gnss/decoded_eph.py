"""Predict from ON-NODE DECODED ephemeris -- the BRDC fallback for network loss.

The broker's whole predict/seed path runs on `gnss_ephemeris.predict_all(eph, ...)`, which is
fed by a NETWORK fetch (`fetch_brdc`). When that fetch fails (no connectivity, mirror outage)
the broker goes dark. But each constellation's decoder already extracts a full, BRDC-validated
ephemeris on-node (the 60 s dpos cross-check proves sub-metre agreement). This module turns those
decoded ephemerides into the SAME per-satellite output `predict_all` produces, so the broker can
keep predicting off its own decode when the network drops.

WHY NOT convert decoded -> RINEX records and reuse predict_all directly: the CNAV / B-CNAV2
ephemerides use the MODERNIZED orbit model (A_ref+dA, A_dot, dn0_dot, i0_dot) that the legacy
`sat_pos_clk` cannot represent. So we reuse each decoder's OWN `sv_position_*` (the exact
propagator the dpos cross-check validated) and replicate only the geometry wrapper here --
byte-for-byte the light-time/Sagnac/az-el/range-rate block from `predict_all`.

SCOPE / HONESTY:
  * POSITION -> az/el/range: exact (validated propagator).
  * VELOCITY -> range-rate/Doppler: central finite-difference of position (the propagators return
    no velocity); ~mm/s accurate, i.e. sub-0.1 Hz at L1 -- ample for search Doppler hints.
  * sat_clk_s: from the decoder's clock terms WHERE it exposes them (Galileo I/NAV & F/NAV carry
    af0/af1/af2+t0c; CNAV/B-CNAV keep clock in separate message types and may not) -> None when
    absent. Geometry and Doppler do NOT need it; only ABSOLUTE code-phase seeding does, and an
    already-tracking sat's DLL holds code phase regardless. So the fallback keeps acquisition,
    hints and the sky alive through an outage; precise cold code-seeding degrades where the clock
    is absent -- flagged, not hidden.

Only a per-constellation broker's OWN sats are covered (its decoder), which is exactly the set it
predicts. Fail-safe by construction: every entry is independently guarded so one bad ephemeris
cannot take out the rest, and a caller wraps the whole thing so the fallback can never worsen an
outage.
"""

import math

from gnss_ephemeris import (BDT_GPST, C_LIGHT, OMEGA_E, _azel, _ecef_of_llh, gpst_of_utc)

# Clock reference / coefficient field names per decoder family. The decoded eph dicts differ in
# casing (Galileo af0/t0c); constellations whose decoder does not surface clock are absent here
# and get sat_clk_s = None. `toe` is the field carrying the ephemeris reference time-of-week, also
# used for the freshness gate; its name likewise differs per family.
CLOCK_FIELDS = {
    "GAL_E1B_INAV": ("af0", "af1", "af2", "t0c"),
    "GAL_E5AI_FNAV": ("af0", "af1", "af2", "t0c"),
}
TOE_FIELD = {
    "GAL_E1B_INAV": "t0e", "GAL_E5AI_FNAV": "t0e",
    "GPS_L2C_CNAV": "toe", "GPS_L5_CNAV": "toe",
    "BDS_B1C_BCNAV1": "t_oe", "BDS_B2A_BCNAV2": "t_oe",
}


def _sat_clock(eph, signal, t_sow):
    """Broadcast SV clock bias (s): af0 + af1*dt + af2*dt^2 about t0c, dt week-wrapped. None when
    the decoder does not carry clock terms for this signal."""
    fields = CLOCK_FIELDS.get(signal)
    if not fields or not all(k in eph for k in fields):
        return None
    a0, a1, a2, toc = (eph[fields[0]], eph[fields[1]], eph[fields[2]], eph[fields[3]])
    dt = t_sow - toc
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800
    return a0 + a1 * dt + a2 * dt * dt


def predict_one(sv_pos, eph, sys, signal, rx, lat, lon, t_sow, toe_age_s, dt=0.5):
    """One satellite's az/el/range/range-rate/sat_clk, mirroring predict_all's per-sat block.

    ★ FRAME: `t_sow` and everything `sv_pos`/`_sat_clock` see is the constellation's SECONDS-OF-
    WEEK, because the decoder propagators (`sv_position_inav/_fnav/_cnav/_bcnav1/_bcnav2`) week-
    wrap `tk = t - t0e` with a per-week fold and so REQUIRE a sow time, not continuous GPST.
    `sv_pos(eph, t_sow) -> ECEF (x,y,z)`. `toe_age_s` (continuous) is supplied by the caller for
    the report only. Returns the dict predict_all emits, or None on any failure."""
    try:
        om_e = OMEGA_E[sys]
        tau = 0.075
        pos_rx = az = el = rng = None
        for _ in range(2):  # light-time + Sagnac, two iterations (== predict_all)
            pos = sv_pos(eph, t_sow - tau)
            th = om_e * tau
            px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
            py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
            pos_rx = (px, py, pos[2])
            az, el, rng = _azel(rx, pos_rx, lat, lon)
            tau = rng / C_LIGHT
        # velocity by central difference of the (un-rotated) position, exactly the frame
        # predict_all's analytic `vel` lives in -> range-rate matches to the finite-diff error.
        pa = sv_pos(eph, t_sow - tau + dt)
        pb = sv_pos(eph, t_sow - tau - dt)
        vel = [(a - b) / (2.0 * dt) for a, b in zip(pa, pb)]
        rr = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel)) / rng
        return dict(az=az, el=el, range_m=rng, range_rate_mps=rr,
                    sat_clk_s=_sat_clock(eph, signal, t_sow - tau),
                    toe_age_s=toe_age_s)
    except Exception:
        return None


def predict_from_decoders(entries, lat, lon, alt, t_utc, mask_deg=0.0, max_age=14400.0):
    """predict_all-shaped output from decoded ephemerides.

    `entries`: list of (sys, prn, signal, eph_dict, sv_pos_fn, toe_gpst). `toe_gpst` is the
    ephemeris reference in the CONTINUOUS GPST frame, used for the toe-age gate + report; pass
    None to skip the gate (age unknown). The propagators themselves are fed SECONDS-OF-WEEK
    (see predict_one). Returns {(sys, prn): {...}}.

    Independent per-entry guards: one malformed ephemeris drops that satellite, never the set.
    Side-effect-free, so a caller can try/except it around the network fetch with zero risk to
    the running loop."""
    t = gpst_of_utc(t_utc)
    t_sow = t - (t // 604800) * 604800   # continuous GPST -> seconds-of-week for the propagators
    rx = _ecef_of_llh(lat, lon, alt)
    out = {}
    for sys, prn, signal, eph, sv_pos, toe_gpst in entries:
        toe_age = (t - toe_gpst) if toe_gpst is not None else None
        if toe_age is not None and abs(toe_age) > max_age:
            continue
        # BeiDou runs on BDT (= GPST - 14 s) and its decoded t_oe is BDT-sow, so the propagator
        # must be fed BDT seconds-of-week. Galileo GST and GPS GPST share the sow origin here.
        ts = (t_sow + BDT_GPST) % 604800 if sys == "C" else t_sow
        v = predict_one(sv_pos, eph, sys, signal, rx, lat, lon, ts, toe_age)
        if v is None or v["el"] < mask_deg:
            continue
        out[(sys, prn)] = v
    return out
