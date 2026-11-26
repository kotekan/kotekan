"""Galileo HAS corrections for the PVT self-survey's code residuals.

gnss_pvt.measurements_from_obs gives each satellite's code residual against the broker's model:
the BRDC record whose toe is nearest (gnss_ephemeris.best_eph over the cached merge), its clock
referred to the chain's signal by the broadcast group delays (group_delay_s; the broker runs
without --dcb-bias). HAS (has_predictor, in the E6-B decoder's payload file) corrects one
broadcast issue, the one whose IOD it names: GPS LNAV, Galileo I/NAV. The BRDC lags the sky by
up to an hour and HAS names the issue on the air, so the issue is taken from the decoders' own
ephemerides first (I/NAV, else F/NAV with its clock moved to I/NAV's), then from the BRDC; an
IOD repeats within days, so only an issue within ISSUE_WINDOW_S matches. The HAS model of a
satellite is that issue's range with the orbit correction (radial, in-track and cross-track,
added to the position) on the line of sight, its clock plus the clock correction, and the
tracked signal's code bias added to the measurement (HAS SIS ICD 1.0 section 7, with the signs
gnss-sdr applies). With a model range - c * clock, a residual becomes

    resid + broker model - (HAS model - code bias)

Each satellite takes the newest orbit message whose set (mask and IOD set IDs) has a clock
message, both within their validity intervals, and the newest valid bias message on that mask.
A satellite is left out when its orbit, clock or code bias is missing or its issue is not held.
"""

import json
import os

import numpy as np

import decoded_eph
import eph_cache
import gnss_ephemeris as ge
from gnss_brdc_supply import cached_brdc

# The chain (a measurement's group, the broker's signal) -> the HAS signal it tracks, its pilot.
HAS_SIGNAL = {
    "gal_e5a": "E5a-Q",
    "gal_e5b": "E5b-Q",
    "gal_e6": "E6-C",
    "gps_l5": "L5 Q",
    "gps_l2c": "L2 CL",
}
C_LIGHT = 299792458.0
DECODER = "GAL_E6B_CNAV"
ISSUE_WINDOW_S = 4 * 3600.0
# Decoded Galileo ephemerides by preference: I/NAV's clock is the one HAS corrects.
DECODED = ("GAL_E5BI_INAV", "GAL_E1B_INAV", "GAL_E5AI_FNAV")


def message_gpst(toh, t_rx):
    """GPS seconds of a HAS message: its time of hour in the hour of t_rx (host UTC when it was
    recovered), or the hour before when that would put it after t_rx."""
    g = ge.gpst_of_utc(t_rx)
    t = g - g % 3600.0 + toh
    return t - 3600.0 if t > g + 30.0 else t


def inav_clock_s(e, clk):
    """A Galileo record's clock referred to I/NAV's E1/E5b combination: an F/NAV (E1/E5a) or
    unflagged record's, by the ICD identity t_E1 = t_IF_a - BGD_a = t_IF_b - BGD_b."""
    if e["sys"] != "E" or int(e.get("l2_codes") or 0) & ge._GAL_CLK_E5B:
        return clk
    return clk - float(e.get("tgd") or 0.0) + float(e.get("iodc") or 0.0)


def _geometry(state, rx, t, sysc):
    """(range m, clock s, unit vector receiver to satellite, satellite position and velocity at
    transmission) seen from rx at t (GPST), state(t_tx) -> (position, velocity, clock s): the
    light time iterated and the position turned by the Earth's rotation over it, as
    gnss_ephemeris._sat_geometry does."""
    tau = 0.075
    for _ in range(3):
        pos, vel, clk = state(t - tau)
        th = ge.OMEGA_E[sysc] * tau
        c, s = np.cos(th), np.sin(th)
        p = np.array([c * pos[0] + s * pos[1], -s * pos[0] + c * pos[1], pos[2]])
        rng = float(np.linalg.norm(p - rx))
        tau = rng / C_LIGHT
    return rng, clk, (p - rx) / rng, np.asarray(pos), np.asarray(vel)


def _brdc_state(e):
    return lambda t: ge.sat_pos_clk(e, t)


def issue_state(sysc, prn, iod, t, eph, decoded):
    """state(t_tx) of the HAS issue: decoded first (Galileo), then BRDC; None if not held. The
    clock is I/NAV's for Galileo."""
    if sysc == "E":
        near = [
            r
            for r in decoded
            if (r[0], r[1]) == (sysc, prn) and abs(t - r[5]) < ISSUE_WINDOW_S
        ]
        for sig in DECODED:
            for _, _, signal, d, _, _ in near:
                if signal != sig or int(d.get("IODnav", -1)) != iod:
                    continue
                if signal != "GAL_E5AI_FNAV":
                    return lambda tt, d=d, s=signal: decoded_eph.state(sysc, s, d, tt)
                recs = eph.get((sysc, prn))
                e_b = ge.best_eph(recs, t, ISSUE_WINDOW_S) if recs else None
                if e_b is None:
                    continue  # no BGDs for the clock's move to I/NAV
                bgd = float(e_b.get("iodc") or 0.0) - float(e_b.get("tgd") or 0.0)

                def st(tt, d=d):
                    pos, vel, clk = decoded_eph.state(sysc, "GAL_E5AI_FNAV", d, tt)
                    return pos, vel, clk + bgd

                return st
    recs = [
        e
        for e in eph.get((sysc, prn), ())
        if int(e.get("iode", -1)) == iod and abs(t - e["toe_gpst"]) < ISSUE_WINDOW_S
    ]
    if not recs:
        return None
    e = min(recs, key=lambda e: abs(t - e["toe_gpst"]))
    return lambda tt: (lambda p, v, c: (p, v, inav_clock_s(e, c)))(
        *ge.sat_pos_clk(e, tt)
    )


def orbit_offset_m(pos, vel, orb):
    """The HAS orbit correction as an ECEF offset: radial, in-track along the velocity,
    cross-track along r x v."""
    ea = vel / np.linalg.norm(vel)
    ec = np.cross(pos, vel)
    ec /= np.linalg.norm(ec)
    er = np.cross(ea, ec)
    return er * orb["radial_m"] + ea * orb["in_track_m"] + ec * orb["cross_track_m"]


def _valid(msg, kind, t):
    valid = msg.get(kind + "_validity_s", msg.get("clock_subset_validity_s", 0))
    return abs(t - message_gpst(msg["toh"], msg["t_rx"])) <= valid


def terms(has, key, sig, t):
    """(orbit, clock m, code bias m) for one satellite and signal at t (GPST), or None."""
    for orb_msg in has.get("orbit", ()):
        orb = orb_msg["sats"].get(key, {}).get("orbit")
        if not orb or None in orb.values() or not _valid(orb_msg, "orbit", t):
            continue
        pair = (orb_msg["mask_id"], orb_msg["iod_set"])
        for clk_msg in has.get("clock", ()):
            dclk = clk_msg["sats"].get(key, {}).get("clock_m")
            if (clk_msg["mask_id"], clk_msg["iod_set"]) != pair or dclk is None:
                continue
            if not _valid(clk_msg, "clock", t):
                continue
            for bias_msg in has.get("bias", ()):
                cb = (bias_msg["sats"].get(key, {}).get("code_bias_m") or {}).get(sig)
                if bias_msg["mask_id"] == pair[0] and cb is not None:
                    if _valid(bias_msg, "code_bias", t):
                        return orb, dclk, cb
            return None
    return None


def correction_m(m, has, eph, decoded, rx):
    """Metres to add to measurement m's resid_m to put it against the HAS model, or None."""
    sig = HAS_SIGNAL.get(m["group"])
    recs = eph.get((m["sys"], int(m["prn"])))
    if not (sig and recs):
        return None
    t = ge.gpst_of_utc(m["t"])
    got = terms(has, "%s%02d" % (m["sys"], int(m["prn"])), sig, t)
    if got is None:
        return None
    orb, dclk, cb = got
    e_b = ge.best_eph(recs, t)
    state = issue_state(m["sys"], int(m["prn"]), orb["iod"], t, eph, decoded)
    if e_b is None or state is None:
        return None
    rng_b, clk_b, _, _, _ = _geometry(_brdc_state(e_b), rx, t, m["sys"])
    model_b = rng_b - C_LIGHT * (clk_b + ge.group_delay_s(e_b, m["group"], None))
    rng_h, clk_h, u, pos, vel = _geometry(state, rx, t, m["sys"])
    model_h = rng_h + float(u @ orbit_offset_m(pos, vel, orb)) - C_LIGHT * clk_h - dclk
    return model_b - (model_h - cb)


class Corrector:
    """The newest HAS corrections, from the first nav_decode payload file matching payload_glob
    that carries them, against the cached BRDC the broker merges; both re-read when they
    change."""

    def __init__(self, payload_glob, decoded_paths=(eph_cache.DEFAULT_DIR,)):
        self.glob = payload_glob
        self._has = (None, None, {})  # (path, mtime, corrections)
        self._eph = (None, {})  # (cache files and mtimes, parsed)
        self.decoded = decoded_eph.Reader(decoded_paths)
        # after apply: the satellites among the measurements with current corrections, and
        # those whose corrections name an issue not held yet (for GPS, the BRDC lags uploads)
        self.named, self.waiting = set(), set()

    def corrections(self):
        import glob

        for p in sorted(glob.glob(self.glob)):
            try:
                mt = os.path.getmtime(p)
                if self._has[:2] == (p, mt):
                    return self._has[2]
                with open(p) as f:
                    has = json.load(f).get("signals", {}).get(DECODER, {}).get("has")
            except (OSError, ValueError):
                continue
            if has:
                self._has = (p, mt, has)
                return has
        return {}

    def eph(self):
        paths = cached_brdc()
        key = tuple((p, os.path.getmtime(p)) for p in paths)
        if key != self._eph[0]:
            self._eph = (key, ge.parse_rinex_nav(paths) if paths else {})
        return self._eph[1]

    def apply(self, meas, lat, lon, alt):
        """The measurements HAS corrects, each a copy with resid_m moved to the HAS model and
        the change in has_m; the rest left out."""
        self.named, self.waiting = set(), set()
        has = self.corrections()
        if not has:
            return []
        eph = self.eph()
        decoded = (
            self.decoded.entries(ge.gpst_of_utc(meas[0]["t"]), sys="E") if meas else []
        )
        rx = np.asarray(ge._ecef_of_llh(lat, lon, alt))
        out = []
        for m in meas:
            try:
                d = correction_m(m, has, eph, decoded, rx)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                d = None
            if d is not None:
                out.append(dict(m, resid_m=m["resid_m"] + d, has_m=d))
            self._note(m, has, eph, decoded)
        return out

    def _note(self, m, has, eph, decoded):
        """Add m's satellite to `named` when HAS has current corrections for it, and to
        `waiting` when their issue is not held."""
        key = "%s%02d" % (m["sys"], int(m["prn"]))
        t = ge.gpst_of_utc(m["t"])
        got = terms(has, key, HAS_SIGNAL.get(m["group"]), t)
        if got is None:
            return
        self.named.add(key)
        try:
            held = issue_state(m["sys"], int(m["prn"]), got[0]["iod"], t, eph, decoded)
        except (KeyError, TypeError, ValueError):
            held = None
        if held is None:
            self.waiting.add(key)
