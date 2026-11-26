"""WAAS clock-ephemeris corrections (SBAS L5, DFMC MT32) for the PVT self-survey's code residuals.

The SBAS L5 decoder (sbas_l5_predictor, in nav_decode's payload file) keeps each satellite
slot's newest MT32 (nav_fields.SBAS_MT32): a correction to one broadcast issue, named by its
IODN (for GPS the LNAV IODC), of the satellite's ECEF position (dx, dy, dz) and clock (dB, m),
each with a rate from the time of day t_D. WAAS sends its L5 service in test (MT0 "do not use"
among the messages), with slots for GPS 1-32. Only LNAV names its issue by IODC, so the issue
comes from the cached BRDC; the decoders' own GPS ephemerides are CNAV, which carry no IODC. A
correction is used for MAX_AGE_S from its reception (SouthPAN SDD SBAS-STN-0001 5.3.2).

The WAAS model of a satellite is that issue's range with the position correction on the line of
sight, less its clock and dB (the clock moved as SBAS's long-term correction moves it). With a
model range - c * clock, a residual becomes resid + broker model - WAAS model.
"""

import glob
import json
import os

import numpy as np

import gnss_ephemeris as ge
from gnss_brdc_supply import cached_brdc
from gnss_has import C_LIGHT, ISSUE_WINDOW_S, _brdc_state, _geometry

SIGNAL = "SBAS_L5_DFMC"
MAX_AGE_S = 100.0
FIELDS = ("slot", "IODN", "δx", "δy", "δz", "δB", "δx rate", "δy rate", "δz rate")
FIELDS += ("δB rate", "t_D")


def corrections(payload_glob):
    """{GPS PRN: MT32 fields plus "t" (host time of reception) and "geo" (the SBAS PRN)}: each
    GPS slot's newest MT32 across the SBAS satellites in the payload files."""
    out = {}
    for path in sorted(glob.glob(payload_glob)):
        try:
            with open(path) as f:
                prns = json.load(f)["signals"][SIGNAL]["prns"]
        except (OSError, ValueError, KeyError):
            continue
        for geo, e in prns.items():
            for key, m in (e.get("msgs") or {}).items():
                if not key.startswith("type 32 slot ") or not m[1]:
                    continue
                v = {r[0]: r[1] for r in m[3]}
                if any(k not in v for k in FIELDS):
                    continue
                slot = int(v["slot"])
                if 1 <= slot <= 32 and (slot not in out or m[1] > out[slot]["t"]):
                    out[slot] = dict(v, t=m[1], geo=int(geo))
    return out


def correction_m(m, c, eph, rx):
    """Metres to add to GPS measurement m's resid_m to put it against the WAAS model, or None
    when its correction is stale or its issue is not in the BRDC."""
    recs = eph.get(("G", int(m["prn"])))
    if not recs or m["t"] - c["t"] > MAX_AGE_S:
        return None
    t = ge.gpst_of_utc(m["t"])
    e_b = ge.best_eph(recs, t)
    issue = [
        e
        for e in recs
        if int(e.get("iodc", -1)) == int(c["IODN"])
        and abs(t - e["toe_gpst"]) < ISSUE_WINDOW_S
    ]
    if e_b is None or not issue:
        return None
    e_w = min(issue, key=lambda e: abs(t - e["toe_gpst"]))
    dt = (t - c["t_D"] + 43200.0) % 86400.0 - 43200.0
    rng_b, clk_b, _, _, _ = _geometry(_brdc_state(e_b), rx, t, "G")
    model_b = rng_b - C_LIGHT * (clk_b + ge.group_delay_s(e_b, m["group"], None))
    rng_w, clk_w, u, _, _ = _geometry(_brdc_state(e_w), rx, t, "G")
    dr = np.array([c["δx"], c["δy"], c["δz"]])
    dr += dt * np.array([c["δx rate"], c["δy rate"], c["δz rate"]])
    model_w = (
        rng_w
        + float(u @ dr)
        - C_LIGHT * (clk_w + ge.group_delay_s(e_w, m["group"], None))
        - (c["δB"] + dt * c["δB rate"])
    )
    return model_b - model_w


class Corrector:
    """The newest WAAS corrections from the nav_decode payload files matching payload_glob,
    against the cached BRDC the broker merges (re-read when it changes)."""

    def __init__(self, payload_glob):
        self.glob = payload_glob
        self._eph = (None, {})  # (cache files and mtimes, parsed)

    def eph(self):
        paths = cached_brdc()
        key = tuple((p, os.path.getmtime(p)) for p in paths)
        if key != self._eph[0]:
            self._eph = (key, ge.parse_rinex_nav(paths) if paths else {})
        return self._eph[1]

    def apply(self, meas, lat, lon, alt):
        """The GPS measurements WAAS corrects, each a copy with resid_m moved to the WAAS model
        and the change in sbas_m; the rest left out."""
        corr = corrections(self.glob)
        if not corr:
            return []
        eph = self.eph()
        rx = np.asarray(ge._ecef_of_llh(lat, lon, alt))
        out = []
        for m in meas:
            c = corr.get(int(m["prn"])) if m["sys"] == "G" else None
            if c is None:
                continue
            try:
                d = correction_m(m, c, eph, rx)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                d = None
            if d is not None:
                out.append(dict(m, resid_m=m["resid_m"] + d, sbas_m=d))
        return out
