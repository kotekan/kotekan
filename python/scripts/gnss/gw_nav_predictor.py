"""Streaming GuoWang 1202.025 MHz low-rate decoder for nav_decode: each sender's own ephemeris and
clock, and its subframes as time marks.

guowang/guowang1202_nav.py is the codec (subframes, an issue's words, the user algorithm); this is
the per-code state around it, as fnav_predictor is for F/NAV. GnssNavSymbols (gw_symbols) serves
each code's newest few seconds of symbols, so every new window is decoded whole and subframes are
kept by their BDT second of week. The shell the orbit needs is the sender's own, from page 1's
satellite key, so nothing here reads a TLE or the host clock, as gw_nav_table.py does.

nav_obs interface as the other decoders': {utc_ref, rec_dt, br, phase, bits=[[idx, +-1], ...]}.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "guowang"))
import guowang1202_nav as gn  # noqa: E402
from gw_nav_table import is_fill  # noqa: E402

KEEP_S = 1200.0  # issues and marks kept behind the newest subframe, s


def key_of(payload):
    """(shell, sender ID) from page 1's satellite key, 512 x shell + ID: payload bit 2 is the
    shell (1 the 50 deg shell, 0 the polar one), bits 4-11 the almanac ID."""
    return (50 if int(payload[2]) else 86), gn._u(payload, 4, 8)


def sv_position_gw(eph, t_sow):
    """ECEF position at BDT second of week t_sow, by the user algorithm."""
    return gn.Ephemeris(shell=eph["shell"]).position(eph, t_sow)


def sv_clock_gw(eph, t_sow):
    """The sender's clock bias at BDT second of week t_sow (af0 about its cycle's reference)."""
    dt = t_sow - (eph["toc_count"] + 1) * 12.0
    dt -= 604800.0 * round(dt / 604800.0)
    return eph["af0"] + eph["af1"] * dt


class _PrnState:
    def __init__(self):
        self.last_obs = None
        # pages by issue; the shell served is page 1's
        self.eph = gn.Ephemeris(shell=50)
        self.shell = None
        self.sender = None
        # BDT SOW -> (page, host-labelled UTC of the subframe's first symbol edge, BDT week on
        # page 1 or None)
        self.marks = {}
        self.n_sf = 0


class GwNavPredictor:
    def __init__(self):
        self._p = {}

    def ingest(self, prn, obs):
        st = self._p.setdefault(prn, _PrnState())
        try:
            utc_ref = float(obs["utc_ref"])
            rec_dt = float(obs["rec_dt"])
            phase = int(obs["phase"])
            br = int(obs["br"])
            pairs = obs["bits"]
        except (KeyError, TypeError, ValueError):
            return
        key = (utc_ref, phase, br)
        if key == st.last_obs or rec_dt <= 0 or br <= 0 or not pairs:
            return
        st.last_obs = key
        sym = np.array([float(b[1]) for b in pairs])
        newest = None
        for sf in gn.decode_subframes(sym):
            newest = sf.sow
            if sf.sow in st.marks:
                continue
            wn = None
            if sf.page == 1:
                st.shell, st.sender = key_of(sf.payload)
                wn = gn._u(sf.payload, 12, 13)
            t_host = utc_ref + (int(pairs[sf.index][0]) * br - phase) * rec_dt
            st.marks[sf.sow] = (sf.page, t_host, wn)
            st.n_sf += 1
            if not is_fill(sf):
                st.eph.add(sf)
        if newest is None:
            return
        for sow in [s for s in st.marks if (newest - s) % 604800 > KEEP_S]:
            del st.marks[sow]
        for toe in [t for t in st.eph.pages if (newest - t) % 604800 > KEEP_S]:
            del st.eph.pages[toe]

    def ephemeris(self, prn):
        """The newest complete issue's words, with the sender's shell and ID, or None."""
        st = self._p.get(prn)
        if st is None or st.shell is None:
            return None
        got = st.eph.latest()
        if got is None:
            return None
        w = dict(got[1])
        w.update(shell=st.shell, sender=st.sender)
        return w

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_sf,
            "words": st.n_sf,
            "have": sorted({m[0] for m in st.marks.values()}),
            "eph": self.ephemeris(prn) is not None,
        }
