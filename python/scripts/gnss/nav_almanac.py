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

SIGNALS = (
    "GAL_E5AI_FNAV",
    "GAL_E5BI_INAV",
    "GAL_E1B_INAV",
    "BDS_B2A_BCNAV2",
    "GLO_L2OF",
)
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


# ---- GLONASS (ICD 5.1 4.5 and A.3.1.3): strings 6-15 of frames 1-4, 6-13 of frame 5 ----------
GLO_MU = 398600.4418e9
GLO_AE = 6378136.0
GLO_J2 = 1082625.75e-9
GLO_OMEGA_E = 7.2921150e-5
GLO_I0 = 63.0 / 180.0 * math.pi
GLO_T_DR = 43200.0  # the nominal draconic period, s
MSK_UTC_S = 3 * 3600.0  # GLONASS time is Moscow time, UTC + 3 h
PAIR_KEEP_S = 3600.0  # how long a GLONASS almanac pair read once waits for its repeat
GLO_EPOCH_DAY = (
    9496  # 1996-01-01 (the first 4-year span's day 1) in days from 1970-01-01
)


def _glo_u(b, hi, width):
    v = 0
    for i in range(hi, hi - width, -1):
        v = (v << 1) | int(b[i])
    return v


def _glo_sm(b, hi, width):
    m = _glo_u(b, hi - 1, width - 1)
    return -m if b[hi] else m


def glo_almanac(even, odd, n_a):
    """(sys, slot, n_a, t_lambda s of the Moscow day, eph) of one satellite's almanac from its
    string pair (bits b[i] = bit i, as glonass_nav_predictor keeps them) and N^A, the almanac's
    day of the 4-year span (string 5)."""
    sc = math.pi
    eph = {
        "sys": "R",
        "lam": _glo_sm(even, 62, 21) * 2.0**-20 * sc,
        "di": _glo_sm(even, 41, 18) * 2.0**-20 * sc,
        "e": _glo_u(even, 23, 15) * 2.0**-20,
        "tau": _glo_sm(even, 72, 10) * 2.0**-18,
        "omega": _glo_sm(odd, 80, 16) * 2.0**-15 * sc,
        "dT": _glo_sm(odd, 43, 22) * 2.0**-9,
        "dTdot": _glo_sm(odd, 21, 7) * 2.0**-14,
        "health": _glo_u(even, 80, 1),
    }
    h = _glo_u(odd, 14, 5)
    eph["k"] = h - 32 if h > 24 else h
    return "R", _glo_u(even, 77, 5), n_a, _glo_u(odd, 64, 21) * 2.0**-5, eph


def glo_toa_gpst(n_a, t_lambda, gpst_ref):
    """The GPST of an almanac's node passage: second t_lambda of the Moscow day N^A of the 4-year
    span, the span taken as the one that puts it nearest gpst_ref."""
    import datetime

    ref_utc = ge_utc(gpst_ref)
    best = None
    for y0 in range(1996, ref_utc.year + 1, 4):
        day = datetime.datetime(
            y0, 1, 1, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(days=n_a - 1, seconds=t_lambda - MSK_UTC_S)
        t = gpst_of(day)
        if best is None or abs(t - gpst_ref) < abs(best - gpst_ref):
            best = t
    return best


def ge_utc(gpst):
    import datetime

    from gnss_ephemeris import GPS_EPOCH, GPS_UTC_LEAP

    return GPS_EPOCH + datetime.timedelta(seconds=gpst - GPS_UTC_LEAP)


def gpst_of(t_utc):
    from gnss_ephemeris import gpst_of_utc

    return gpst_of_utc(t_utc)


def sv_position_glo_alm(eph, t, wrap=True):
    """ECEF (PZ-90) position of a GLONASS almanac at t, GPST seconds of week (t - toe wrapped to
    within half a week unless wrap=False), toe the node passage: the ICD's almanac algorithm
    (A.3.1.3) with the J2 secular drifts of the node and perigee and without its periodic terms,
    a few km off, which a Doppler hint does not see."""
    dt = t - eph["toe"]
    if wrap:
        dt = (dt + 302400) % 604800 - 302400
    t_dr0 = GLO_T_DR + eph["dT"]
    w = math.floor(dt / t_dr0)
    i = GLO_I0 + eph["di"]
    t_dr = t_dr0 + (2 * w + 1) * eph["dTdot"]
    n = 2 * math.pi / t_dr
    e, om = eph["e"], eph["omega"]
    # semi-major axis from the draconic period, by iteration (A.3.1.3 step 2)
    t_osc = t_dr
    a = 0.0
    for _ in range(10):
        a_new = (t_osc / (2 * math.pi)) ** (2.0 / 3.0) * GLO_MU ** (1.0 / 3.0)
        p = a_new * (1 - e * e)
        t_osc = t_dr / (
            1
            - 1.5
            * GLO_J2
            * (GLO_AE / p) ** 2
            * (
                (2 - 2.5 * math.sin(i) ** 2)
                * (1 - e * e) ** 1.5
                / (1 + e * math.cos(om)) ** 2
                + (1 + e * math.cos(om)) ** 3 / (1 - e * e)
            )
        )
        if abs(a_new - a) < 0.01:
            break
        a = a_new
    p = a * (1 - e * e)
    lam = (
        eph["lam"]
        - (GLO_OMEGA_E + 1.5 * GLO_J2 * n * (GLO_AE / p) ** 2 * math.cos(i)) * dt
    )
    om_k = om - 0.75 * GLO_J2 * n * (GLO_AE / p) ** 2 * (1 - 5 * math.cos(i) ** 2) * dt
    e0 = -2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(om / 2))
    l1 = om + e0 - e * math.sin(e0)
    l2 = l1 + n * (dt - t_dr0 * w - eph["dTdot"] * w * w)
    m = l2 - om_k
    ek = m
    for _ in range(10):
        ek = m + e * math.sin(ek)
    nu = math.atan2(math.sqrt(1 - e * e) * math.sin(ek), math.cos(ek) - e)
    u = nu + om_k
    r = a * (1 - e * math.cos(ek))
    return (
        r * (math.cos(lam) * math.cos(u) - math.sin(lam) * math.sin(u) * math.cos(i)),
        r * (math.sin(lam) * math.cos(u) + math.cos(lam) * math.sin(u) * math.cos(i)),
        r * math.sin(u) * math.sin(i),
    )


def sv_clock_glo_alm(eph, t):
    """A GLONASS almanac's coarse clock, s: -tau_n^A (to ~2 us; no drift term)."""
    return -eph["tau"]


class Collector:
    """The almanacs in one decoder's messages. scan() after every ingest: a decoder holds only
    each satellite's newest message of each type, and an almanac can span two of them. take()
    hands over what was read since, as ephemeris cache entries."""

    def __init__(self, signal):
        self.signal = signal
        self.seen = {}  # (transmitting prn, message type) -> the decode time last read
        self.last = {}  # (transmitting prn, message type) -> (bits, time)
        self.got = []  # (sys, prn, wna, toa, eph)
        self.pairs = {}  # GLONASS: a string pair's bits -> (prn, UTC start) first read

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

    def scan_glonass(self, dec):
        """GLONASS: the decoder keeps every string by its UTC start, so each odd almanac string
        (7-15) read since the last scan is joined to the even one 2 s before it, with N^A from
        string 5 of its frame (or the newest one seen). Frame 5's strings 14 and 15 are not
        almanac. Frames start on the 150 s superframe grid of UTC (Moscow midnight), on the
        satellite's clock (the decoder's utc_off), so a satellite's strings are read once they
        are placed in time. A pair is
        taken once it has been read again, from another satellite or superframe: a string the
        Hamming code "corrected" to the wrong word would otherwise put one slot's orbit and
        channel on another."""
        for prn, st in list(dec._p.items()):
            if st.utc_off is None:
                continue
            for ts, (m, b, _t) in sorted(st.strings.items()):
                if m == 5:
                    self.last[(prn, "n_a")] = _glo_u(b, 80, 11)
                if m < 7 or m % 2 == 0 or self.seen.get((prn, ts)):
                    continue
                f0 = ts - 2.0 * (m - 1)
                if int((f0 + st.utc_off) % 150.0 // 30.0) + 1 == 5 and m >= 14:
                    continue
                even = st.strings.get(ts - 2.0)
                if even is None or even[0] != m - 1:
                    continue
                s5 = st.strings.get(f0 + 8.0)
                n_a = (
                    _glo_u(s5[1], 80, 11)
                    if s5 and s5[0] == 5
                    else self.last.get((prn, "n_a"))
                )
                if n_a is None:
                    continue
                self.seen[(prn, ts)] = True
                first = self.pairs.setdefault(tuple(even[1]) + tuple(b), (prn, ts))
                if first == (prn, ts):
                    continue
                a = glo_almanac(even[1], b, n_a)
                if a[1] > 0:
                    self.got.append(a)
            for key in [k for k in self.seen if k[0] == prn and k[1] not in st.strings]:
                del self.seen[key]
        t_new = max((ts for _, ts in self.pairs.values()), default=0.0)
        for key in [k for k, v in self.pairs.items() if t_new - v[1] > PAIR_KEEP_S]:
            del self.pairs[key]

    def scan(self, dec):
        if self.signal == "GLO_L2OF":
            self.scan_glonass(dec)
            return
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
            if (
                sysc == "R"
            ):  # wna is N^A, toa the node passage's second of the Moscow day
                t = glo_toa_gpst(wna, toa, gpst_ref)
                eph = dict(eph, toe=t % 604800.0)
                signal = "GLO_ALM"
            else:
                t = toa_gpst(sysc, wna, toa, gpst_ref)
                eph = dict(eph, toe=float(toa), toc=float(toa), af2=0.0, dn=0.0)
                signal = "GAL_ALM" if sysc == "E" else "BDS_ALM"
            out.append(
                {
                    "sys": sysc,
                    "prn": prn,
                    "signal": signal,
                    "toe_field": "toe",
                    "eph": eph,
                    "toe_gpst": t,
                    "week_from": "sky",
                }
            )
        self.got = []
        return out
