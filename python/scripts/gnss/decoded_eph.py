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
    velocity), rotated into the reception frame with the position; a few mm/s (the light-time
    rate is left out, as in predict_all), ample for search Doppler hints.
  * sat_clk_s: the decoder's clock polynomial plus the relativistic term (as sat_pos_clk)
    where CLOCK_FIELDS lists one (Galileo I/NAV and F/NAV, GPS LNAV), else None; no
    group-delay term. Geometry and Doppler do not need it; only absolute code-phase seeding
    does, and a tracking satellite's DLL holds code phase regardless. So the fallback keeps
    acquisition, hints and the sky alive; cold code seeding degrades where the clock is absent.

Only the broker's own satellites are covered (its decoders), which is the set it predicts.
Every entry is guarded separately so one bad ephemeris cannot take out the rest, and the caller
wraps the whole call so the fallback cannot worsen an outage.
"""

import math

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
    "GAL_E5AI_FNAV": ("af0", "af1", "af2", "t0c", "t0e"),
    # GPS LNAV carries its clock in subframe 1 (af0/af1/af2 about toc), so unlike CNAV the L1
    # fallback can seed absolute code phase, not just geometry and Doppler.
    "GPS_L1_LNAV": ("af0", "af1", "af2", "toc", "toe"),
}
# Kepler fields the relativistic term reads; every decoder in CLOCK_FIELDS names them alike.
KEPLER_FIELDS = ("M0", "dn", "sqrtA", "e")


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
    toe = toe_sow - BDT_GPST if sys == "C" else toe_sow
    toe_gpst = int(gpst_now // 604800) * 604800 + toe
    if toe_gpst - gpst_now > 302400:
        toe_gpst -= 604800
    elif toe_gpst - gpst_now < -302400:
        toe_gpst += 604800
    return toe_gpst


def _sat_clock(eph, sys, signal, t_sow):
    """Broadcast SV clock bias (s): af0 + af1*dt + af2*dt^2 about t0c plus the relativistic
    term F*e*sqrtA*sin(E), as sat_pos_clk computes them; dt and tk are week-wrapped. None when
    the decoder does not carry clock terms for this signal."""
    fields = CLOCK_FIELDS.get(signal)
    if not fields or not all(k in eph for k in fields + KEPLER_FIELDS):
        return None
    a0, a1, a2, toc, toe = (eph[k] for k in fields)
    dt = _week_wrap(t_sow - toc)
    tk = _week_wrap(t_sow - toe)
    a = eph["sqrtA"] ** 2
    n = math.sqrt(MU[sys] / a ** 3) + eph["dn"]
    m = eph["M0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + eph["e"] * math.sin(ek)
    rel = F_REL * eph["e"] * eph["sqrtA"] * math.sin(ek)
    return a0 + a1 * dt + a2 * dt * dt + rel


def predict_one(sv_pos, eph, sys, signal, rx, lat, lon, t_sow, toe_age_s, dt=0.5):
    """One satellite's az/el/range/range-rate/sat_clk, mirroring predict_all's per-sat block.

    `t_sow` and everything `sv_pos`/`_sat_clock` see is the constellation's seconds of week: the
    decoder propagators (`sv_position_*`) fold `tk = t - t0e` per week and so need a
    seconds-of-week time, not continuous GPST. `sv_pos(eph, t_sow) -> ECEF (x,y,z)`.
    `toe_age_s` (continuous) is supplied by the caller for the report only. Returns
    predict_all's dict without tgd_s, or None on any failure."""
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
        # velocity by central difference of the un-rotated position, then rotated into the
        # reception frame by the position's angle (== predict_all), so range rate matches
        # predict_all's to the finite-difference error
        pa = sv_pos(eph, t_sow - tau + dt)
        pb = sv_pos(eph, t_sow - tau - dt)
        vel = [(a - b) / (2.0 * dt) for a, b in zip(pa, pb)]
        vel_rx = (
            vel[0] * math.cos(th) + vel[1] * math.sin(th),
            -vel[0] * math.sin(th) + vel[1] * math.cos(th),
            vel[2],
        )
        rr = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel_rx)) / rng
        return dict(
            az=az,
            el=el,
            range_m=rng,
            range_rate_mps=rr,
            sat_clk_s=_sat_clock(eph, sys, signal, t_sow - tau),
            toe_age_s=toe_age_s,
        )
    except Exception:
        return None


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
