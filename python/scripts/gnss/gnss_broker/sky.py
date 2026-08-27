"""Sky: BRDC prediction, visibility, and the broadcast-vs-decoded ephemeris cross-checks.

Extracted verbatim from gps_distributed_broker.py (task #27 M1).

THIS IS THE SHARED LAYER. Every function here is indexed by constellation and takes the
ephemeris state as an argument -- none of it is per-signal. In the unified broker one Sky
serves every chain, which is the whole point: a CNAV decode on L5 produces the ephemeris
E5a needs, and today that is thrown away at the process boundary
(docs/CHORD_BROKER_REFACTOR.md 1.2).
"""
import math
import re
from datetime import timedelta

from .transport import _log, _log_rl, _now

C_LIGHT = 299792458.0  # m/s


def brdc_predict(state, lat, lon, alt_m, sysc, min_prn, t_utc, f_carrier_hz):
    """BRDC-almanac analog of gps_beamtrack.predict_dopplers for ONE constellation:
    {prn: (doppler_hz, doppler_rate_hz_s, elevation_deg, range_m, sat_clk_s)}.

    Same tuple layout/sign convention as the TLE path (+Doppler = approaching), extended
    with the broadcast SATELLITE CLOCK (the TLE class has none) -- the nh time-assist
    convention is t_sv = t_gpst - range/c + clk (proven to 0.01 chip, c31_convention.py).
    All elevations are returned (down to -90): the noise probes pick the deepest-BELOW-
    horizon sats. Rate = range-rate differencing over a 4 s epoch pair, like the
    dead-reckon block. Full-sky predict_all is ~1.2 ms, so per-cycle cost is trivial
    (the skyfield path it replaces was heavier).
    """
    ge = state["mod"]
    # Hand the ephemeris library this chain's tagged logger, once. `_log` reads a thread-local
    # tag, so one global install still labels every chain's line correctly -- and the chain that
    # pays for a fetch is the one that reports what it got.
    if getattr(ge, "LOG_HOOK", "unset") is None:
        ge.LOG_HOOK = _log
    now = _now()
    if state["eph"] is None or now - state["eph_t"] > 7200.0:
        # Current-day BRDC files GROW; fetch_brdc re-fetches a cache older than 2 h. Pass
        # t_utc so a replay (--almanac-epoch) gets the DAY-MATCHED file -- today's file
        # cannot predict another epoch (best_eph 4 h window).
        try:
            state["eph"] = ge.parse_rinex_nav(ge.fetch_brdc(t_utc))
            state["eph_t"] = now
            _log("brdc almanac: ephemeris refreshed (%d sats)" % len(state["eph"]))
        except Exception as e:
            state["eph_t"] = now - 7200.0 + 600.0  # coast on the old set, retry in 10 min
            if state["eph"] is None:
                raise
            _log("brdc almanac: refresh failed (%s); coasting on the previous set "
                 "(sats thin out as toe ages past 4 h)" % e)
    dt = 4.0
    pa = ge.predict_all(state["eph"], lat, lon, alt_m, t_utc, mask_deg=-90.0)
    pb = ge.predict_all(state["eph"], lat, lon, alt_m, t_utc + timedelta(seconds=dt),
                        mask_deg=-90.0)
    C = C_LIGHT
    out = {}
    for (s, prn), v in pa.items():
        if s != sysc or prn < min_prn:
            continue
        dop = -v["range_rate_mps"] / C * f_carrier_hz
        v2 = pb.get((s, prn))
        rate = (-(v2["range_rate_mps"] - v["range_rate_mps"]) / dt / C * f_carrier_hz
                if v2 else 0.0)
        out[prn] = (dop, rate, v["el"], v["range_m"], v["sat_clk_s"])
    # PREDICTION-COLLAPSE GUARD (2026-07-19): the daily BRDC's C/E nav records can LAG the
    # GPS ones by hours, so an entire constellation's newest toe crosses best_eph's window
    # at ONE instant -- measured 11:59:57Z: 13 seeded sats dropped 'set below horizon' in
    # one cycle (incl. C31 at el 70) and the hint-gated search went DARK for the remainder
    # of the 2 h refresh period. If the predicted-sat count collapses below half its
    # running peak: force an early re-fetch (the CDN file has usually grown past the gap
    # by then; retry at 10 min if not) and BRIDGE with the last good prediction set --
    # a coasted el is drop-gate-grade for hours, and a stale dop hint degrades the search
    # no worse than the no-hint darkness it replaces.
    # ⚠️⚠️ THE COLLAPSE BOOKKEEPING IS PER-CONSTELLATION AND THE STATE DICT IS NOT.
    # `state` is the SHARED BRDC store -- receiver.brdc() hands every chain the same object so
    # one parse serves G/E/C (that part is deliberate and right). But `peak_n` and `last_good`
    # are statements about ONE constellation's sky, and writing them at `state` scope made both
    # of them cross-constellation (2026-08-27, KV: "why are there zero probe sats on BeiDou"):
    #
    #   peak_n  became a MAX OVER CONSTELLATIONS. BeiDou has ~13 sats in its eph window against
    #           GPS's ~24, so `len(out) < 0.5 * peak` was true permanently: bds_b2a fired
    #           PREDICTION COLLAPSE 9 ms after its own almanac init, against a peak it had
    #           never set, and never came out of it.
    #   last_good became WHICHEVER CHAIN PREDICTED LAST. So the bridge -- the thing that exists
    #           to coast one constellation's sky through a thin refresh -- handed BeiDou GPS's
    #           and Galileo's satellites, complete with their elevations. Measured in the log:
    #           `bds_b2a: noise probe PRN 1 seeded (elev -88)` and `bds_b2b: noise probe PRN 3
    #           seeded (elev -32)` on chains whose min_prn is 19, 1.5 s after the Galileo chains
    #           seeded the same PRN 3. Sub-19 PRNs cannot come out of BeiDou's own filtered
    #           output. The probe selector then picked those foreign PRNs, the ones that
    #           happened to land in 19-42 survived --probe-require-slot, and bds_b2a sat at 2
    #           probes -> UNANCHORED -> nothing admitted, nothing trimmed.
    #
    # Same shape as the four fallbacks audited on 2026-08-26 (docs/CHORD_PEER_RELATIVE_AUDIT.md):
    # a fallback reproduces the primary path's OUTPUT while dropping one of its invariants --
    # here the constellation identity -- and runs exactly when the primary is already degraded.
    # Scope is (sysc, min_prn) because min_prn is part of what defines this chain's population.
    sc = state.setdefault("scope", {}).setdefault((sysc, min_prn), {})
    peak = sc.get("peak_n", 0)
    if len(out) >= peak:
        sc["peak_n"] = peak = len(out)
    if peak >= 4 and len(out) < 0.5 * peak:
        # THE REFETCH IS A SHARED SIDE EFFECT and keeps a shared cooldown: eph_t governs the one
        # store, so three constellations must not each force their own re-download. The
        # ANNOUNCEMENT is per-scope, because "BeiDou collapsed" is not news GPS can deliver.
        if now - state.get("collapse_refetch_t", 0.0) > 600.0:
            state["collapse_refetch_t"] = now
            state["eph_t"] = now - 7201.0  # re-fetch on the next cycle
        lg = sc.get("last_good")
        if lg:
            # ⚠️ A BRIDGED SKY IS A STALE SKY AND IT MUST SAY SO EVERY TIME IT IS SERVED.
            # This used to log only inside the 600 s refetch guard, so the FIRST cycle of a
            # collapse announced itself and the next several hundred were silent -- while every
            # one of them returned a frozen elevation set that the drop gate, the probe
            # selector and the search hints all read as live. bds_b2a bridged for 25 minutes on
            # one line of log. Rate-limited, but on its own key, and it carries the AGE.
            sc.setdefault("bridge_since", now)
            _log_rl("brdc-bridge",
                    "brdc almanac: PREDICTION COLLAPSE (%d of peak %d %s sats%s in the eph "
                    "window) -> BRIDGING on a sky frozen %.0f s ago (%d sats); early refresh "
                    "forced. Elevations are STALE -- probes, drops and hints all ride this."
                    % (len(out), peak, sysc,
                       " with PRN >= %d" % min_prn if min_prn > 1 else "",
                       now - sc["bridge_since"], len(lg)),
                    every_s=120.0)
            return lg
    else:
        if sc.pop("bridge_since", None) is not None:
            _log("brdc almanac: %s prediction RECOVERED (%d sats) -- off the bridge, live sky"
                 % (sysc, len(out)))
        sc["last_good"] = out
    return out


def _dh_dpos(xc):
    """Pull the numeric BRDC-agreement residual (metres) out of an xcheck suffix string, or
    None. The xcheck functions return a human suffix (' | BRDC dpos=0.05 m ...'); the
    decode-health export wants the number. Parsing the string we already control keeps the five
    shared, tested xcheck functions untouched (they ride the live seed loop)."""
    if not xc:
        return None
    m = re.search(r"dpos=([-+]?\d+(?:\.\d+)?)", xc)
    return float(m.group(1)) if m else None


def _cnav_brdc_xcheck(brdc_alm, sys, prn, cnav_eph, log):
    """S4 ephemeris cross-check: ECEF position residual between the live-decoded CNAV ephemeris
    and the downloaded BRDC (LNAV) ephemeris, both propagated to the SAME absolute instant (the
    CNAV toe). Returns a short suffix for the health log. Two independent messages/encodings of
    one orbit -> a small residual validates the decode; a huge one flags a decode/convention
    fault or a week mismatch (shown via the toe delta). Kepler propagation only, no Viterbi."""
    try:
        import gps_cnav as _C
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        toe = cnav_eph["toe"]
        # the CNAV toe is seconds-of-week; place it in the BRDC record's CONTINUOUS GPS frame via
        # that record's week start (toe_gpst - toe_sow), assuming the same GPS week (a big toe
        # delta in the log flags a week straddle).
        t_com = (be["toe_gpst"] - be["toe_sow"]) + toe
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _C.sv_position_cnav(cnav_eph, toe)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s)" % (dpos, be["toe_sow"] - toe)
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _cnav2_brdc_xcheck(brdc_alm, sys, prn, cnav2_eph, log):
    """GPS L1C-D CNAV-2 ephemeris cross-check -- the _cnav_brdc_xcheck analogue (CNAV-2 SF2 is the
    same CNAV Keplerian set, so sv_position reuses gps_cnav). Ready for the live field map; until
    CNV2_EPH_FIELDS is filled the ephemeris is None so this is never reached."""
    try:
        import gps_cnav2 as _C2
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        toe = cnav2_eph["toe"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + toe
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _C2.sv_position_cnav2(cnav2_eph, toe)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s)" % (dpos, be["toe_sow"] - toe)
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _inav_brdc_xcheck(brdc_alm, sys, prn, inav_eph, log):
    """S5 ephemeris cross-check, the _cnav_brdc_xcheck analogue for Galileo I/NAV: ECEF
    position residual between the live-decoded I/NAV ephemeris and BRDC, both propagated to
    the I/NAV t0e. BRDC's Galileo record IS I/NAV-derived, so a right decode matches it to a
    few metres; a huge residual flags a decode/convention fault (the G2 or field-offset
    conventions galileo_inav flagged as ICD-owned) or a week straddle (shown via toe delta).
    Kepler only, no Viterbi."""
    try:
        import galileo_inav as _I
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        t0e = inav_eph["t0e"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + t0e
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _I.sv_position_inav(inav_eph, t0e)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODnav %d)" % (
            dpos, be["toe_sow"] - t0e, inav_eph.get("IODnav", -1))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _lnav_brdc_xcheck(brdc_alm, sys, prn, lnav_eph, log):
    """LNAV ephemeris cross-check, the _inav_brdc_xcheck analogue for GPS L1 C/A. Unlike the
    other constellations this is the STRONGEST possible check: BRDC's GPS record IS the LNAV
    message, and sv_position_lnav is the same legacy Keplerian math as ge.sat_pos_clk, so a
    correct on-node decode matches BRDC to ~centimetres (the only residual is a different
    upload set, shown via the toe/IODE deltas). A large dpos here means a WRONG field offset in
    LNAV_EPH_FIELDS -- this is the live validator that pack/unpack cannot be (see that table's
    note). Kepler only, no framing."""
    try:
        import gps_nav_decode as _L
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        toe = lnav_eph["toe"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + toe
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _L.sv_position_lnav(lnav_eph, toe)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODE %d/%d)" % (
            dpos, be["toe_sow"] - toe, int(lnav_eph.get("IODE", -1)), int(be.get("iode", -1)))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _fnav_brdc_xcheck(brdc_alm, sys, prn, fnav_eph, log):
    """S5 ephemeris cross-check for Galileo E5a-I F/NAV, the _inav_brdc_xcheck analogue.
    F/NAV and I/NAV describe the SAME Galileo orbit through different framing, so a correct
    F/NAV decode matches BRDC to a few metres just as I/NAV does; a huge residual flags a
    decode/convention fault (the sync / interleaver / field-offset conventions galileo_fnav
    flagged as ICD-owned, pending live symbols) or a week straddle. Kepler only."""
    try:
        import galileo_fnav as _F
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        t0e = fnav_eph["t0e"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + t0e
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _F.sv_position_fnav(fnav_eph, t0e)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODnav %d)" % (
            dpos, be["toe_sow"] - t0e, fnav_eph.get("IODnav", -1))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav2_brdc_xcheck(brdc_alm, sys, prn, bcnav2_eph, log):
    """S5 ephemeris cross-check for BeiDou B2a B-CNAV2, the _fnav_brdc_xcheck analogue on the
    BDS broker (sys='C'). B-CNAV2 and BRDC describe the same BDS-3 orbit through different
    framing, so a correct decode matches BRDC to a few metres; a huge residual flags a decode
    fault (the NB-LDPC / field-offset / GEO-vs-MEO conventions beidou_bcnav2 flagged as
    ICD-owned, pending live symbols). Kepler/CNAV only."""
    try:
        import beidou_bcnav2 as _B
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        t0e = bcnav2_eph["t_oe"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + t0e
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _B.sv_position_bcnav2(bcnav2_eph, t0e)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODE %d, SatType %d)" % (
            dpos, be["toe_sow"] - t0e, bcnav2_eph.get("IODE", -1),
            int(round(bcnav2_eph.get("SatType", -1))))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav3_brdc_xcheck(brdc_alm, sys, prn, bcnav3_eph, log):
    """S5 ephemeris cross-check for BeiDou B2b B-CNAV3, the _bcnav2_brdc_xcheck analogue. Ready
    for Phase 2 (needs beidou_bcnav3.sv_position_bcnav3 + a filled BCNV3_EPH_FIELDS); until then
    the ephemeris is None so this is never reached. Kepler/CNAV-style."""
    try:
        import beidou_bcnav3 as _B
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        t0e = bcnav3_eph["t_oe"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + t0e
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _B.sv_position_bcnav3(bcnav3_eph, t0e)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODE %d, SatType %d)" % (
            dpos, be["toe_sow"] - t0e, bcnav3_eph.get("IODE", -1),
            int(round(bcnav3_eph.get("SatType", -1))))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav1_brdc_xcheck(brdc_alm, sys, prn, bcnav1_eph, log):
    """S5 ephemeris cross-check for BeiDou B1C B-CNAV1, the _bcnav2_brdc_xcheck analogue on the
    L1 BDS broker (sys='C'). B-CNAV1 SF2 and BRDC describe the same BDS-3 orbit; a correct
    decode matches BRDC to a few metres. Same CNAV propagation as B-CNAV2."""
    try:
        import beidou_bcnav1 as _B
        ge = brdc_alm["mod"]
        recs = brdc_alm["eph"].get((sys, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, ge.gpst_of_utc(_now()))
        if be is None:
            return " | BRDC:stale"
        t0e = bcnav1_eph["t_oe"]
        t_com = (be["toe_gpst"] - be["toe_sow"]) + t0e
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _B.sv_position_bcnav1(bcnav1_eph, t0e)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, IODE %d, SatType %d)" % (
            dpos, be["toe_sow"] - t0e, bcnav1_eph.get("IODE", -1),
            int(round(bcnav1_eph.get("SatType", -1))))
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def visible_prns(lat, lon, alt_m, mask_deg, look_ahead_s):
    """PRNs above the elevation mask now (and look_ahead_s ahead). Needs skyfield."""
    try:
        from gps_beamtrack import load_gps_sats, sat_elevation  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        _log("visibility gating unavailable (%s); not gating" % e)
        return None
    sats = load_gps_sats()
    up = set()
    for prn, sat in sats.items():
        if sat_elevation(sat, lat, lon, alt_m, look_ahead_s) >= mask_deg:
            up.add(prn)
    return up


# ---- BORESIGHT GEOMETRY (the quantiser-railing veto) ---------------------------------------
# CHORD's dishes sit 8.59 deg SOUTH of zenith (dec +40.73): boresight is az 180.0, el 81.41.
# ⚠️ `telescope.dish_coelev_deg` in the generated configs reads -27.3 and does NOT give this.
BORESIGHT_AZ_DEG = 180.0
BORESIGHT_EL_DEG = 81.41


def _unit(el_deg, az_deg):
    e, a = math.radians(el_deg), math.radians(az_deg)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


def boresight_sep_deg(el_deg, az_deg, el0=BORESIGHT_EL_DEG, az0=BORESIGHT_AZ_DEG):
    """Angular separation from boresight, degrees."""
    u, b = _unit(el_deg, az_deg), _unit(el0, az0)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(u, b))))))


def nearest_boresight(pd, el0=BORESIGHT_EL_DEG, az0=BORESIGHT_AZ_DEG):
    """(separation_deg, (sys, prn)) for the visible satellite closest to boresight, or None.

    ⚠️ POOLED OVER EVERY CONSTELLATION ON PURPOSE, and `pd` carries all three. One satellite
    near boresight rails the 4+4b quantiser for EVERY chain -- they share the nibbles -- so a
    per-constellation answer would clear the chain carrying the bright satellite and leave the
    others contaminated while looking like clean controls. Measured 2026-08-26: with a sat
    inside 3 deg the tracked population drops from 7 satellites per epoch to 4, and the
    survivors read 2-3 dB HIGH.
    """
    best = None
    for key, v in (pd or {}).items():
        try:
            el, az = v["el"], v["az"]
        except (TypeError, KeyError):
            continue
        if el is None or az is None or el <= 0.0:
            continue
        d = boresight_sep_deg(el, az, el0, az0)
        if best is None or d < best[0]:
            best = (d, key)
    return best
