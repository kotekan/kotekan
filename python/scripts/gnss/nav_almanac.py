"""The almanacs the receiver's own decoders read: every satellite's orbit from any one satellite
of its constellation, for the search hints of satellites the receiver holds no ephemeris for
(sky._with_decoded).

    Galileo  F/NAV pages 5 and 6 (E5a), I/NAV words 7 to 10 (E5b, E1): three satellites a pair
             of pages or two pairs of words, the whole constellation in 10 min
    BeiDou   B-CNAV2 type 40 (B2a), the midi almanac: one satellite a message

GPS CNAV carries no almanac at present (types 12, 31 and 37 are not broadcast), so GPS has none
here. BeiDou's GEO satellites (almanac type 1) are skipped: they have no B2a.

Each almanac becomes an ephemeris cache entry (eph_cache.py), signal GAL_ALM or BDS_ALM, which
nav_decode writes to a file of its own beside its ephemerides; decoded_eph.Reader(almanac=True)
reads them back. The eph dict is a plain Kepler set (sqrtA, e, i0, OMEGA0, OMEGA_dot, omega, M0)
and clock (af0, af1) about toe = toc = t_oa in the system's seconds of week, with af2 and dn
zero, so decoded_eph's clock and relativistic term apply unchanged.

Layouts: Galileo OS SIS ICD 2.1 and BDS-SIS-ICD-B2a 1.0, as nav_fields.py lists them, checked
on live messages (11-24) against BRDC to the fields' resolution. F/NAV page 5's last 4 bits are
the top of its second satellite's Omega0, whose other 12 bits open page 6.
"""

import math

from gnss_ephemeris import BDT_GPST, MU, OMEGA_E
from nav_fields import _read

SIGNALS = ("GAL_E5AI_FNAV", "GAL_E5BI_INAV", "GAL_E1B_INAV", "BDS_B2A_BCNAV2")
SC = math.pi * 2.0**-15  # a 16-bit angle: semicircles, 2^-15
GAL_SQRTA0 = math.sqrt(29600e3)  # the nominal orbit's sqrt(A), m^0.5
GAL_I0 = 56.0 / 180.0 * math.pi
BDS_I0 = 0.30 * math.pi  # MEO and IGSO
GST_WEEK0 = 1024  # GPS week of GST week 0
BDT_WEEK0 = 1356  # GPS week of BDT week 0
FNAV_PAGE = 238  # bits of an F/NAV page as fnav_predictor stores it


def _gal(b, wna, t0a, sv, om0, omd, m0, af0, af1, hs):
    """A Galileo almanac from the bits b (one message, or two joined): the SVID at sv, its
    dA^0.5, e, omega and di after it, and the other fields at the offsets given."""
    eph = {
        "sys": "E",
        "sqrtA": GAL_SQRTA0 + _read(b, sv + 6, 13, "s") * 2.0**-9,
        "e": _read(b, sv + 19, 11, "u") * 2.0**-16,
        "omega": _read(b, sv + 30, 16, "s") * SC,
        "i0": GAL_I0 + _read(b, sv + 46, 11, "s") * math.pi * 2.0**-14,
        "OMEGA0": _read(b, om0, 16, "s") * SC,
        "OMEGA_dot": _read(b, omd, 11, "s") * math.pi * 2.0**-33,
        "M0": _read(b, m0, 16, "s") * SC,
        "af0": _read(b, af0, 16, "s") * 2.0**-19,
        "af1": _read(b, af1, 13, "s") * 2.0**-38,
        "health": _read(b, hs, 2, "u"),
    }
    return "E", _read(b, sv, 6, "u"), wna, t0a, eph


def _week_toa(b):
    """(WN_a, t_oa s) of an F/NAV page 5 or an I/NAV word 7 or 9."""
    return _read(b, 10, 2, "u"), _read(b, 12, 10, "u") * 600


def fnav_sv1(p5):
    return _gal(p5, *_week_toa(p5), 22, 79, 95, 106, 122, 138, 151)


def fnav_sv2_sv3(p5, p6):
    """The second and third satellites of a page 5 and the page 6 after it."""
    P = FNAV_PAGE
    b = list(p5) + list(p6)
    om0 = (210, 4, P + 10, 12)
    sv2 = _gal(b, *_week_toa(p5), 153, om0, P + 22, P + 33, P + 49, P + 65, P + 78)
    sv3 = _gal(p6, *_week_toa(p5), 80, 137, 153, 164, 180, 196, 209)
    return [sv2, sv3]


def inav_svs(w7, w8, w9, w10):
    """The almanacs of one run of words 7 to 10, any of them None: satellite 1 needs words 7
    and 8, satellite 2 words 8 and 9, satellite 3 words 9 and 10. The health read is E5b's.
    """
    W = 128
    out = []
    if w7 is not None and w8 is not None:
        b = list(w7) + list(w8)
        out.append(_gal(b, *_week_toa(w7), 22, 79, 95, 106, W + 10, W + 26, W + 39))
    if w8 is not None and w9 is not None:
        b = list(w8) + list(w9)
        out.append(
            _gal(b, *_week_toa(w9), 43, 100, 116, W + 22, W + 38, W + 54, W + 67)
        )
    if w9 is not None and w10 is not None:
        b = list(w9) + list(w10)
        out.append(
            _gal(b, *_week_toa(w9), 71, W + 10, W + 26, W + 37, W + 53, W + 69, W + 82)
        )
    return out


def bcnav2_type40(f):
    """A B-CNAV2 type 40 frame's midi almanac, or None for a GEO's."""
    if _read(f, 75, 2, "u") == 1:
        return None
    eph = {
        "sys": "C",
        "sqrtA": _read(f, 120, 17, "u") * 2.0**-4,
        "e": _read(f, 98, 11, "u") * 2.0**-16,
        "omega": _read(f, 164, 16, "s") * SC,
        "i0": BDS_I0 + _read(f, 109, 11, "s") * math.pi * 2.0**-14,
        "OMEGA0": _read(f, 137, 16, "s") * SC,
        "OMEGA_dot": _read(f, 153, 11, "s") * math.pi * 2.0**-33,
        "M0": _read(f, 180, 16, "s") * SC,
        "af0": _read(f, 196, 11, "s") * 2.0**-20,
        "af1": _read(f, 207, 10, "s") * 2.0**-37,
        "health": _read(f, 217, 8, "u"),
    }
    return (
        "C",
        _read(f, 69, 6, "u"),
        _read(f, 77, 13, "u"),
        _read(f, 90, 8, "u") * 4096,
        eph,
    )


def toa_gpst(sysc, wna, toa, gpst_ref):
    """An almanac's reference time as continuous GPST. BeiDou's WN_a is the full BDT week;
    Galileo's is the GST week mod 4, taken within two weeks of gpst_ref."""
    if sysc == "C":
        return (wna + BDT_WEEK0) * 604800.0 + toa - BDT_GPST
    w = int(gpst_ref // 604800)
    w += (wna - (w - GST_WEEK0)) % 4
    t = w * 604800.0 + toa
    if t - gpst_ref > 2 * 604800.0:
        t -= 4 * 604800.0
    return t


def sv_position_alm(eph, t, wrap=True):
    """ECEF (x, y, z) m of an almanac at t, the system's seconds of week (tk = t - toe wrapped
    to within half a week unless wrap=False): the GPS almanac algorithm (IS-GPS-200
    20.3.3.5.2.1), Kepler with no harmonic terms, in the system's constants."""
    mu, om_e = MU[eph["sys"]], OMEGA_E[eph["sys"]]
    a = eph["sqrtA"] ** 2
    tk = t - eph["toe"]
    if wrap:
        tk = (tk + 302400) % 604800 - 302400
    m = eph["M0"] + math.sqrt(mu / a**3) * tk
    e = eph["e"]
    ek = m
    for _ in range(10):
        ek = m + e * math.sin(ek)
    u = math.atan2(math.sqrt(1 - e * e) * math.sin(ek), math.cos(ek) - e) + eph["omega"]
    r = a * (1 - e * math.cos(ek))
    om = eph["OMEGA0"] + (eph["OMEGA_dot"] - om_e) * tk - om_e * eph["toe"]
    x, y = r * math.cos(u), r * math.sin(u)
    ci = math.cos(eph["i0"])
    return (
        x * math.cos(om) - y * ci * math.sin(om),
        x * math.sin(om) + y * ci * math.cos(om),
        y * math.sin(eph["i0"]),
    )


class Collector:
    """The almanacs in one decoder's messages. scan() after every ingest: a decoder holds only
    each satellite's newest message of each type, and an almanac can span two of them. take()
    hands over what was read since, as ephemeris cache entries."""

    def __init__(self, signal):
        self.signal = signal
        self.seen = {}  # (transmitting prn, message type) -> the decode time last read
        self.last = {}  # (transmitting prn, message type) -> (bits, time)
        self.got = []  # (sys, prn, wna, toa, eph)

    def _messages(self, st):
        """(type, bits, time) of the almanac messages a decoder's PRN state holds: time is the
        page's start for F/NAV, the decode time for I/NAV, which pairs words by it."""
        if self.signal == "GAL_E5AI_FNAV":
            return [(k, v[1], v[3]) for k, v in st.pages.items() if k in (5, 6)]
        if self.signal == "BDS_B2A_BCNAV2":
            v = st.frames.get(40)
            return [(40, v[0], v[2])] if v else []
        return [(k, v[1], v[2]) for k, v in st.words.items() if k in (7, 8, 9, 10)]

    def _pair(self, prn, k, bits, t, gap):
        """The newest message of type k from prn, if it came gap[0] to gap[1] s before the
        message (bits, t) and carries its IODa."""
        prev = self.last.get((prn, k))
        if prev is None or _read(prev[0], 6, 4, "u") != _read(bits, 6, 4, "u"):
            return None
        return prev[0] if gap[0] <= t - prev[1] <= gap[1] else None

    def scan(self, dec):
        for prn, st in list(dec._p.items()):
            new = []
            for k, bits, t in self._messages(st):
                if self.seen.get((prn, k)) != t:
                    self.seen[(prn, k)] = t
                    new.append((t, k, bits))
            for t, k, bits in sorted(new, key=lambda m: (m[0], m[1])):
                got = []
                if k == 40:
                    got = [bcnav2_type40(bits)]
                elif k == 5:
                    got = [fnav_sv1(bits)]
                elif k == 6:
                    p5 = self._pair(prn, 5, bits, t, (40.0, 60.0))
                    got = fnav_sv2_sv3(p5, bits) if p5 is not None else []
                elif k == 8:
                    got = inav_svs(
                        self._pair(prn, 7, bits, t, (0.0, 10.0)), bits, None, None
                    )
                elif k == 9:
                    got = inav_svs(
                        None, self._pair(prn, 8, bits, t, (0.0, 45.0)), bits, None
                    )
                elif k == 10:
                    got = inav_svs(
                        None, None, self._pair(prn, 9, bits, t, (0.0, 10.0)), bits
                    )
                self.last[(prn, k)] = (bits, t)
                self.got += [a for a in got if a is not None and a[1] > 0]

    def take(self, gpst_ref):
        """The almanacs read since the last take, as ephemeris cache entries placed in GPST from
        their own weeks (Galileo's mod 4 within two weeks of gpst_ref)."""
        out = []
        for sysc, prn, wna, toa, eph in self.got:
            eph = dict(eph, toe=float(toa), toc=float(toa), af2=0.0, dn=0.0)
            out.append(
                {
                    "sys": sysc,
                    "prn": prn,
                    "signal": "GAL_ALM" if sysc == "E" else "BDS_ALM",
                    "toe_field": "toe",
                    "eph": eph,
                    "toe_gpst": toa_gpst(sysc, wna, toa, gpst_ref),
                    "week_from": "sky",
                }
            )
        self.got = []
        return out
