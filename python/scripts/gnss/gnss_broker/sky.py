"""Sky: BRDC prediction, visibility, and the broadcast-vs-decoded ephemeris cross-checks.

Every function here is indexed by constellation and takes the ephemeris state as an
argument; none of it is per-signal. The BRDC store is shared (Receiver.brdc hands every chain
in the process the same dict), so one parse serves every constellation.
"""
import math
import re
import threading
from datetime import timedelta

from .transport import _log, _log_rl, _now

C_LIGHT = 299792458.0  # m/s

# brdc_predict's re-parse gate on the shared BRDC store, the shorter gate of a constellation on
# the collapse bridge, and the wait after a failed refresh before the next attempt.
_SKY_REFRESH_S = 900.0
_SKY_BRIDGE_REFRESH_S = 60.0
_SKY_RETRY_S = 600.0


def brdc_predict(state, lat, lon, alt_m, sysc, min_prn, t_utc, f_carrier_hz):
    """BRDC analogue of gps_beamtrack.predict_dopplers for one constellation:
    {prn: (doppler_hz, doppler_rate_hz_s, elevation_deg, range_m, sat_clk_s, az_deg,
    az_rate_deg_s, el_rate_deg_s, t_unix)}.

    The first four elements have the TLE path's layout and sign convention (+Doppler =
    approaching); the rest are BRDC-only. sat_clk_s is the broadcast satellite clock, for the
    nh time-assist convention t_sv = t_gpst - range/c + clk. All elevations are returned
    (down to -90): the noise probes pick the deepest below-horizon satellites. The rates are
    differenced over a 4 s epoch pair (t_utc, t_utc + 4 s).
    """
    ge = state["mod"]
    # Install the broker's logger as the BRDC supply's LOG_HOOK, once. `_log` reads a
    # thread-local tag, so one install labels each line with the chain whose thread logged it;
    # lines from the supply's brdc-refresh thread carry no chain tag.
    # LOG_HOOK lives on gnss_brdc_supply; gnss_ephemeris re-exports only fetch_brdc.
    import gnss_brdc_supply as _supply

    if getattr(_supply, "LOG_HOOK", "unset") is None:
        _supply.LOG_HOOK = _log
    now = _now()
    sc = state.setdefault("scope", {}).setdefault((sysc, min_prn), {})
    # Re-parse every _SKY_REFRESH_S, as often as the product updates. The sky is served by the
    # station-hourly merge, which updates every hour, and a broadcast record is valid for 4 h
    # from its toe. With a slower gate the toes age between refreshes: satellites drop out of
    # the prediction one at a time as they cross the window, with nothing in the log to say
    # the sky is thinning rather than setting, and the survivors are predicted from ever-older
    # records, a growing model error the fast loop absorbs into a deeper and deeper trim.
    #
    # This is cheap. fetch_brdc keeps its own caches (1 h for the daily, 30 min for the
    # station-hourly merge) and refreshes them on a daemon thread, so a call costs a parse,
    # not a fetch. The gate only decides how soon a newer file is noticed.
    #
    # A constellation on the collapse bridge (below) re-parses every _SKY_BRIDGE_REFRESH_S:
    # the collapse guard's early re-parse reads the cached files and fetch_brdc only starts
    # the download, so the files it lands are picked up at the next look.
    gate = _SKY_BRIDGE_REFRESH_S if "bridge_since" in sc else _SKY_REFRESH_S
    # One chain refreshes the shared store at a time; the others predict on the set in hand
    # rather than parse the same files again. Only a store with no set yet waits for the lock.
    lock = state.setdefault("lock", threading.Lock())
    if lock.acquire(blocking=state["eph"] is None):
        try:
            if state["eph"] is None or (
                now - state["eph_t"] > gate and now >= state.get("eph_retry_t", 0.0)
            ):
                _eph_refresh(state, ge, now, t_utc)
        finally:
            lock.release()
    eph = state["eph"]
    dt = 4.0
    pa = ge.predict_all(eph, lat, lon, alt_m, t_utc, mask_deg=-90.0)
    pb = ge.predict_all(
        eph, lat, lon, alt_m, t_utc + timedelta(seconds=dt), mask_deg=-90.0
    )
    C = C_LIGHT
    out = {}
    for (s, prn), v in pa.items():
        if s != sysc or prn < min_prn:
            continue
        dop = -v["range_rate_mps"] / C * f_carrier_hz
        v2 = pb.get((s, prn))
        rate = (
            -(v2["range_rate_mps"] - v["range_rate_mps"]) / dt / C * f_carrier_hz
            if v2
            else 0.0
        )
        # Elements are appended, never inserted: consumers index this tuple positionally
        # (almanac.py rebuilds it as ctx.pred) and nothing unpacks it strictly, so growing it
        # at the end is safe and reordering it would be silent corruption. az (element 5) is
        # what the broker serves on get_sky.
        #
        # Elements 6-8 (az rate and el rate in deg/s, and the unix epoch az/el describe) are for
        # the element steering: geometry is posted every ~30 s, so the assemblers extrapolate
        # along the track between posts (gnssElemSteer.hpp). The epoch travels with the tuple so
        # a bridged (frozen) sky extrapolates from when it was true rather than being stamped
        # as current.
        if v2:
            _daz = (v2["az"] - v["az"] + 540.0) % 360.0 - 180.0
            _az_rate, _el_rate = _daz / dt, (v2["el"] - v["el"]) / dt
        else:
            _az_rate = _el_rate = 0.0
        out[prn] = (
            dop,
            rate,
            v["el"],
            v["range_m"],
            v["sat_clk_s"],
            v["az"],
            _az_rate,
            _el_rate,
            t_utc.timestamp(),
        )
    # Prediction-collapse guard. The daily BRDC's C/E nav records can lag the GPS ones by hours,
    # so a whole constellation's newest toe can leave best_eph's window at one instant: every
    # satellite of it drops in one cycle and the hint-gated search goes dark. If the predicted
    # count falls below half its running peak, force an early re-parse and bridge with the last
    # good prediction: a coasted el is drop-gate-grade for hours, and a stale Doppler hint
    # degrades the search no worse than no hint at all. The re-parse reads the cached files and
    # starts a background download, which the bridge's short gate (above) picks up.
    #
    # The bookkeeping is per scope; the store is not. `state` is the shared BRDC store
    # (Receiver.brdc hands every chain the same dict), but peak_n and last_good describe one
    # constellation's sky. Kept at store scope, peak_n would be a max over constellations,
    # holding BeiDou (fewer satellites in its eph window than GPS) below half of it for good,
    # and last_good would bridge one constellation with another's satellites. Scope is
    # (sysc, min_prn) because min_prn is part of what defines this chain's population.
    peak = sc.get("peak_n", 0)
    if len(out) >= peak:
        sc["peak_n"] = peak = len(out)
    if peak >= 4 and len(out) < 0.5 * peak:
        # The early re-parse is a shared side effect with a shared 600 s cooldown: eph_t
        # governs the one store, so the constellations must not each force their own. The
        # announcement is per scope.
        if now - state.get("collapse_refetch_t", 0.0) > 600.0:
            state["collapse_refetch_t"] = now
            state["eph_t"] = now - _SKY_REFRESH_S - 1.0  # re-parse on the next call
        lg = sc.get("last_good")
        if lg:
            # A bridged sky is stale and says so every time it is served: the drop gate, the
            # probe selector and the search hints all read its frozen elevations as live.
            # Rate-limited on its own key, outside the refresh cooldown, and it carries the age.
            sc.setdefault("bridge_since", now)
            _log_rl(
                "brdc-bridge",
                "brdc almanac: PREDICTION COLLAPSE (%d of peak %d %s sats%s in the eph "
                "window) -> BRIDGING on a sky frozen %.0f s ago (%d sats); early refresh "
                "forced at most once per 600 s. Elevations are STALE -- probes, drops and "
                "hints all ride this."
                % (
                    len(out),
                    peak,
                    sysc,
                    " with PRN >= %d" % min_prn if min_prn > 1 else "",
                    now - sc["bridge_since"],
                    len(lg),
                ),
                every_s=120.0,
            )
            return lg
    else:
        if sc.pop("bridge_since", None) is not None:
            _log(
                "brdc almanac: %s prediction RECOVERED (%d sats) -- off the bridge, live sky"
                % (sysc, len(out))
            )
        sc["last_good"] = out
    return out


def _eph_refresh(state, ge, now, t_utc):
    """Re-parse the shared BRDC store, for brdc_predict under the store's lock. A failure keeps
    the current set and holds the next attempt off for _SKY_RETRY_S; with no set yet it raises.
    """
    # Current-day BRDC files grow; fetch_brdc re-fetches a daily older than 1 h. Pass
    # t_utc so a replay gets the day-matched file: today's file cannot predict another
    # epoch (best_eph's 4 h window).
    try:
        _old_eph = state["eph"]
        # block only while there is no sky at all; with one in hand the fetch must not
        # stall the pass (gnss_brdc_supply: the refresh runs on a thread).
        state["eph"] = ge.parse_rinex_nav(ge.fetch_brdc(t_utc, block=_old_eph is None))
        state["eph_t"] = now
        _n = len(state["eph"])
        # Log only when the count changed: a line every 15 min regardless would bury a
        # real thinning.
        if _n != state.get("eph_n"):
            _log(
                "brdc almanac: ephemeris refreshed (%d sats%s)"
                % (
                    _n,
                    "" if state.get("eph_n") is None else ", was %d" % state["eph_n"],
                )
            )
        state["eph_n"] = _n
    except Exception as e:
        # Coast on the old set and retry in _SKY_RETRY_S, not on the next pass: the store is
        # shared, and a failure that persists would otherwise re-parse on every chain's every
        # pass.
        state["eph_retry_t"] = now + _SKY_RETRY_S
        if state["eph"] is None:
            raise
        _log(
            "brdc almanac: refresh failed (%s); coasting on the previous set "
            "(sats thin out as toe ages past 4 h)" % e
        )


def _dh_dpos(xc):
    """The numeric BRDC-agreement residual (metres) in an xcheck suffix, or None. The xcheck
    functions return log text (' | BRDC dpos=0.05 m ...'); the decode-health export wants the
    number, so it is parsed back out of that text."""
    if not xc:
        return None
    m = re.search(r"dpos=([-+]?\d+(?:\.\d+)?)", xc)
    return float(m.group(1)) if m else None


def _cnav_brdc_xcheck(brdc_alm, sys, prn, cnav_eph, log):
    """Ephemeris cross-check: ECEF position residual between the live-decoded CNAV ephemeris
    and the downloaded BRDC (LNAV) ephemeris, both propagated to the same absolute instant (the
    CNAV toe). Returns a short suffix for the health log. Two independent encodings of one
    orbit, so a small residual validates the decode; a huge one flags a decode or convention
    fault, or a week mismatch (shown by the toe delta). Kepler propagation only."""
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
        # the CNAV toe is seconds-of-week; place it in the BRDC record's continuous GPS frame via
        # that record's week start (toe_gpst - toe_sow), assuming the same GPS week (a big toe
        # delta in the log flags a week straddle).
        t_com = (be["toe_gpst"] - be["toe_sow"]) + toe
        bx, by, bz = ge.sat_pos_clk(be, t_com)[0]
        cx, cy, cz = _C.sv_position_cnav(cnav_eph, toe)
        dpos = math.sqrt((bx - cx) ** 2 + (by - cy) ** 2 + (bz - cz) ** 2)
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s)" % (dpos, be["toe_sow"] - toe)
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _inav_brdc_xcheck(brdc_alm, sys, prn, inav_eph, log):
    """The _cnav_brdc_xcheck analogue for Galileo I/NAV: ECEF position residual between the
    live-decoded I/NAV ephemeris and BRDC, both propagated to the I/NAV t0e. BRDC's Galileo
    record is I/NAV-derived, so a correct decode matches it to a few metres; a huge residual
    flags a decode or convention fault (the G2 inversion and interleaver orientation that
    galileo_inav isolates as ICD-owned constants) or a week straddle (shown by the toe delta).
    Kepler only."""
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
            dpos,
            be["toe_sow"] - t0e,
            inav_eph.get("IODnav", -1),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _lnav_brdc_xcheck(brdc_alm, sys, prn, lnav_eph, log):
    """LNAV ephemeris cross-check, the _inav_brdc_xcheck analogue for GPS L1 C/A, and the
    strongest of them: BRDC's GPS record is the LNAV message, and sv_position_lnav is the same
    legacy Keplerian math as ge.sat_pos_clk, so a correct on-node decode matches BRDC to
    centimetres (the only residual is a different upload set, shown by the toe/IODE deltas).
    A large dpos here means a wrong field offset in LNAV_EPH_FIELDS: this is the live
    validator a pack/unpack round-trip cannot be (see that table's note). Kepler only."""
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
            dpos,
            be["toe_sow"] - toe,
            int(lnav_eph.get("IODE", -1)),
            int(be.get("iode", -1)),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _fnav_brdc_xcheck(brdc_alm, sys, prn, fnav_eph, log):
    """Ephemeris cross-check for Galileo E5a-I F/NAV, the _inav_brdc_xcheck analogue. F/NAV
    and I/NAV describe the same Galileo orbit through different framing, so a correct F/NAV
    decode matches BRDC to a few metres just as I/NAV does; a huge residual flags a decode or
    convention fault (the sync, interleaver and field-offset conventions galileo_fnav
    isolates as ICD-owned constants) or a week straddle. Kepler only."""
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
            dpos,
            be["toe_sow"] - t0e,
            fnav_eph.get("IODnav", -1),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav2_brdc_xcheck(brdc_alm, sys, prn, bcnav2_eph, log):
    """Ephemeris cross-check for BeiDou B2a B-CNAV2, the _fnav_brdc_xcheck analogue for a
    BeiDou chain (sys='C'). B-CNAV2 and BRDC describe the same BDS-3 orbit through different
    framing, so a correct decode matches BRDC to a few metres; a huge residual flags a decode
    fault (the preamble, bit-order and field-offset conventions beidou_bcnav2 isolates as
    ICD-owned). Kepler/CNAV only."""
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
            dpos,
            be["toe_sow"] - t0e,
            bcnav2_eph.get("IODE", -1),
            int(round(bcnav2_eph.get("SatType", -1))),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav3_brdc_xcheck(brdc_alm, sys, prn, bcnav3_eph, log):
    """Ephemeris cross-check for BeiDou B2b B-CNAV3, the _bcnav2_brdc_xcheck analogue.
    Message Type 10 carries the same BDS-3 CNAV orbit block as B-CNAV1 SF2, propagated with the
    B-CNAV2 math (beidou_bcnav3.sv_position_bcnav3), so a correct decode matches BRDC to a few
    metres. The nav-bits stage appends it to the bcnav3 decode-health line once a Type 10
    ephemeris is in hand. Kepler/CNAV-style."""
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
            dpos,
            be["toe_sow"] - t0e,
            bcnav3_eph.get("IODE", -1),
            int(round(bcnav3_eph.get("SatType", -1))),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


def _bcnav1_brdc_xcheck(brdc_alm, sys, prn, bcnav1_eph, log):
    """Ephemeris cross-check for BeiDou B1C B-CNAV1, the _bcnav2_brdc_xcheck analogue for a B1C
    chain (sys='C'). B-CNAV1 SF2 and BRDC describe the same BDS-3 orbit; a correct decode
    matches BRDC to a few metres. Same CNAV propagation as B-CNAV2."""
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
            dpos,
            be["toe_sow"] - t0e,
            bcnav1_eph.get("IODE", -1),
            int(round(bcnav1_eph.get("SatType", -1))),
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


# ---- Boresight geometry (the quantiser-railing veto) ---------------------------------------
# Where the dishes point, read from the fleet's running config by load_boresight():
# `telescope.dish_coelev_deg`, the tilt from zenith along the meridian (negative = south). It
# is the value the F-engine and recv1 label their data with, and recv1's ConfigTracker refuses
# a node whose telescope block differs from its own, so every node serves the same number.
# None until read; while unknown (no node answered yet, or --boresight none with the dishes
# tipped for site work) nearest_boresight() reports no satellite near boresight, which turns
# off everything keyed on it -- the D3 transit veto, the dead-reckon clock's transit freeze,
# the code-rate clock's transit hold, the shadow joint feed's transit hold and the "_bore" post
# that freezes the assemblers' model formation.
BORESIGHT_AZ_DEG = None
BORESIGHT_EL_DEG = None
BORESIGHT_KNOWN = False
BORESIGHT_SOURCE = None
_bore_lock = threading.Lock()
_bore_init_lock = threading.Lock()
_bore_watch = None
_bore_inited = False


def boresight_from_coelev(coelev_deg):
    """(az, el) of the boresight for a meridian tilt of `coelev_deg` from zenith (negative =
    south): -27.3 -> (180, 62.7)."""
    c = float(coelev_deg)
    return (180.0 if c < 0.0 else 0.0), 90.0 - abs(c)


def set_boresight(az_deg, el_deg, source):
    """Install a boresight for the whole process (every chain shares the quantiser), or clear
    it with az_deg None."""
    global BORESIGHT_AZ_DEG, BORESIGHT_EL_DEG, BORESIGHT_KNOWN, BORESIGHT_SOURCE
    with _bore_lock:
        if az_deg is None:
            BORESIGHT_AZ_DEG = BORESIGHT_EL_DEG = None
            BORESIGHT_KNOWN = False
        else:
            BORESIGHT_AZ_DEG, BORESIGHT_EL_DEG = float(az_deg), float(el_deg)
            BORESIGHT_KNOWN = True
        BORESIGHT_SOURCE = source


def read_pointing(bases, get):
    """(coelev_deg, detail) from the nodes' running config (GET <base>/config), or (None,
    detail). Every base is asked; the ones that answer must agree, since a disagreement means
    the fleet is mid-change or mislabelled and no single number describes the dishes."""
    vals, errs = {}, []
    for b in bases:
        try:
            v = (get("%s/config" % b.rstrip("/")).get("telescope") or {}).get(
                "dish_coelev_deg"
            )
        except Exception as e:  # a node being down is the normal outage case
            errs.append("%s: %s" % (b, e))
            continue
        if v is None:
            errs.append("%s: no telescope.dish_coelev_deg" % b)
            continue
        vals[b] = float(v)
    if not vals:
        return None, "no node answered (%s)" % ("; ".join(errs) or "none listed")
    distinct = sorted(set(vals.values()))
    if len(distinct) > 1:
        return (
            None,
            "nodes DISAGREE: %s"
            % ", ".join("%s=%+.2f" % (b, v) for b, v in vals.items()),
        )
    return (
        distinct[0],
        "%d/%d node(s) agree%s"
        % (
            len(vals),
            len(bases),
            ("; unanswered: %s" % "; ".join(errs)) if errs else "",
        ),
    )


def load_boresight(bases, get, quiet_if_same=False):
    """Read the pointing (read_pointing) and install it. A failed read keeps whatever boresight
    is installed, so a node outage does not switch the transit guards off. True when a
    boresight is installed afterwards."""
    coelev, detail = read_pointing(bases, get)
    if coelev is None:
        _log_rl(
            "boresight-read",
            "BORESIGHT: pointing not read (%s) -- %s"
            % (
                detail,
                "keeping az %.2f el %.2f" % (BORESIGHT_AZ_DEG, BORESIGHT_EL_DEG)
                if BORESIGHT_KNOWN
                else "UNKNOWN: every transit guard is OFF until a node answers",
            ),
            every_s=300.0,
        )
        return BORESIGHT_KNOWN
    az, el = boresight_from_coelev(coelev)
    changed = not (
        BORESIGHT_KNOWN
        and abs(az - BORESIGHT_AZ_DEG) < 1e-6
        and abs(el - BORESIGHT_EL_DEG) < 1e-6
    )
    if changed and BORESIGHT_KNOWN:
        _log(
            "*** BORESIGHT CHANGED: az %.2f el %.2f -> az %.2f el %.2f (dish_coelev_deg %+.2f; %s)"
            % (BORESIGHT_AZ_DEG, BORESIGHT_EL_DEG, az, el, coelev, detail)
        )
    elif changed or not quiet_if_same:
        _log(
            "BORESIGHT az %.2f el %.2f from the fleet's telescope.dish_coelev_deg %+.2f (%s)"
            % (az, el, coelev, detail)
        )
    set_boresight(az, el, "dish_coelev_deg %+.2f" % coelev)
    return True


def watch_boresight(bases, get, every_s=600.0):
    """Re-read the pointing every `every_s` in one daemon thread per process (idempotent): a
    re-pointing restarts the nodes but not the broker. The first read is the caller's."""
    global _bore_watch
    with _bore_lock:
        if _bore_watch is not None:
            return
        bases = list(bases)

        def _run():
            ev = threading.Event()
            while True:
                # an unknown boresight is retried every minute, a known one every every_s
                ev.wait(every_s if BORESIGHT_KNOWN else 60.0)
                try:
                    load_boresight(bases, get, quiet_if_same=True)
                except Exception as e:  # never let the watcher die silently
                    _log_rl("boresight-watch", "BORESIGHT watch: %s" % e, every_s=300.0)

        _bore_watch = threading.Thread(target=_run, name="boresight-watch", daemon=True)
        _bore_watch.start()


def _unit(el_deg, az_deg):
    e, a = math.radians(el_deg), math.radians(az_deg)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


def init_boresight(setting, bases, get, watch=True):
    """Apply --boresight once per process (the first chain to start; the rest find it done).
    'none' clears it, an (az, el) tuple installs it as given, 'telescope' reads the fleet
    (load_boresight) and, with `watch`, starts the re-read thread."""
    global _bore_inited
    with _bore_init_lock:
        if _bore_inited:
            return
        _bore_inited = True
        if setting == "none":
            set_boresight(None, None, "--boresight none")
            _log(
                "BORESIGHT none: the dishes are not pointed; every transit guard is off"
            )
        elif isinstance(setting, tuple):
            set_boresight(setting[0], setting[1], "--boresight %g,%g" % setting)
            _log("BORESIGHT az %.2f el %.2f (fixed by --boresight)" % setting)
        else:
            load_boresight(bases, get)
            if watch:
                watch_boresight(bases, get)


def boresight_sep_deg(el_deg, az_deg, el0=None, az0=None):
    """Angular separation from boresight (the installed one unless el0/az0 are given),
    degrees."""
    el0 = BORESIGHT_EL_DEG if el0 is None else el0
    az0 = BORESIGHT_AZ_DEG if az0 is None else az0
    u, b = _unit(el_deg, az_deg), _unit(el0, az0)
    return math.degrees(
        math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(u, b)))))
    )


def nearest_boresight(pd, el0=None, az0=None):
    """(separation_deg, (sys, prn)) for the visible satellite closest to boresight, or None.

    Pooled over every constellation on purpose; `pd` (the transit sky) carries all of them.
    One satellite near boresight rails the 4+4b quantiser for every chain, since they share
    the nibbles, so a per-constellation answer would clear the chain carrying the bright
    satellite and leave the others contaminated while looking like clean controls.
    """
    if not BORESIGHT_KNOWN:
        return None
    el0 = BORESIGHT_EL_DEG if el0 is None else el0
    az0 = BORESIGHT_AZ_DEG if az0 is None else az0
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


# The transit sky's toe validity: geometry only, so a day past toe is still sub-degree.
TRANSIT_MAX_AGE_S = 86400.0


def transit_sky(ge, eph, lat, lon, alt_m, t_utc):
    """Geometry only: {(sys, prn): {az, el, health, toe_age_s}} over every constellation in
    `eph`, with predict_all's health gate off and a day of toe validity.

    This is the sky the shared-model transit freeze (almanac `_bore`), the dead-reckon clock
    freeze and the railing veto read. The strict sky drops an unhealthy or stale-ephemeris
    satellite, but that satellite still transmits, and passing near boresight it still
    captures the shared element model. Never for seeds, probes or hints: those keep
    predict_all's defaults, and nothing here carries a clock. Objects in no BRDC at all stay
    invisible."""
    pa = ge.predict_all(
        eph,
        lat,
        lon,
        alt_m,
        t_utc,
        mask_deg=-90.0,
        max_age=TRANSIT_MAX_AGE_S,
        health_gate=False,
    )
    t = ge.gpst_of_utc(t_utc)
    out = {}
    for key, v in pa.items():
        e = ge.best_eph(eph.get(key, ()), t, TRANSIT_MAX_AGE_S)
        out[key] = {
            "az": v["az"],
            "el": v["el"],
            "toe_age_s": v["toe_age_s"],
            "health": (e or {}).get("health", 0),
        }
    return out
