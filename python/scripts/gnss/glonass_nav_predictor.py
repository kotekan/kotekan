"""GLONASS FDMA navigation message decode (L2OF; L1OF carries the same strings).

Ingests a combiner's bit_export nav_obs: one 20 ms bit per slot, the 10 ms meander already
removed by the tracker code (GLO_L2OF_MC). Emits are chained into a per-PRN bit history
(navobs_align.SymbolHistory). A string is 2 s: 85 data bits in relative code, bit 85 first, then
the 0.3 s time mark, 30 meander-free 10 ms symbols, which against the meander in the tracker code
leave 15 bit values of a fixed pattern, 8 of them nonzero. Strings are found by that pattern,
the relative code is undone, and each string is checked by its Hamming code (bits 8-1: one error
corrected, two rejected) and by its number against its place in the 30 s frame (string m starts
2(m - 1) s after a UTC 30 s boundary, GLONASS time being UTC + 3 h). Strings 1-4 of one frame
give the satellite's ephemeris: the PZ-90 state vector at tb, tau_n and gamma_n, taken when the
next frame repeats it and built as a gnss_ephemeris record (sys "R"), so sat_pos_clk propagates
it and the BRDC cross-check compares like with like.

Layout per the GLONASS ICD (edition 5.1): string fields in table 4.5, the Hamming code in 4.7,
the time mark in 3.3.2.2.
"""

import math
import time

import numpy as np

import gnss_ephemeris as ge
from navobs_align import HistoryState, SymbolHistory

DATA_BITS = 85
STRING_BITS = 100  # 85 data bits and the time mark's 15 bit periods, 2 s
TIME_MARK = "111110001101110101000010010110"  # 30 symbols of 10 ms
# The time mark against the tracker code's meander (+1 for the first 10 ms, -1 for the second):
# each 20 ms bit period sums its two symbols with opposite signs, so a pair (a, b) of logic
# levels reads b - a, and equal pairs read 0.
TM_BITS = np.array(
    [int(TIME_MARK[2 * i + 1]) - int(TIME_MARK[2 * i]) for i in range(15)], dtype=float
)
HIST_BITS = 1500  # 30 s: a frame's strings 1-4 and room
STRING_TTL_S = 300.0
PROP_S = 0.07  # a GLONASS signal's travel time, ~64-83 ms: receive time back to transmit time
UTC_MSK_S = 3 * 3600.0  # GLONASS time is Moscow time, UTC + 3 h


def _hamming_positions():
    """The Hamming position of each data bit b9..b85: the integers from 3 that are not powers
    of two, in order (the check bits b1..b7 sit at 1, 2, 4, ..., 64)."""
    pos, p = [], 2
    while len(pos) < DATA_BITS - 8:
        p += 1
        if p & (p - 1):
            pos.append(p)
    return pos


HAM_POS = _hamming_positions()
POS_TO_BIT = {p: 9 + i for i, p in enumerate(HAM_POS)}


def hamming_check(b):
    """Check and correct one string in place. b[1..85] are the string's bits (b[0] unused),
    b1..b8 the check bits beta1..beta8. Returns the number of bits corrected (0 or 1), or None
    for a string with two errors or one the code cannot place."""
    s = 0
    for j in range(7):
        c = b[j + 1]
        for i, p in enumerate(HAM_POS):
            if (p >> j) & 1:
                c ^= b[9 + i]
        s |= c << j
    parity = sum(b[1:]) & 1
    if s == 0:
        return 0 if parity == 0 else 1  # with parity 1, the error is beta8 itself
    if parity == 0:
        return None  # two errors
    if s & (s - 1) == 0:
        return 1  # one of beta1..beta7
    bit = POS_TO_BIT.get(s)
    if bit is None:
        return None
    b[bit] ^= 1
    return 1


def hamming_encode(b):
    """Fill b[1..8] for data bits b[9..85] (the self-test's transmitter)."""
    for j in range(7):
        c = 0
        for i, p in enumerate(HAM_POS):
            if (p >> j) & 1:
                c ^= b[9 + i]
        b[j + 1] = c
    b[8] = 0
    b[8] = sum(b[1:]) & 1


def _u(b, hi, lo):
    v = 0
    for i in range(hi, lo - 1, -1):
        v = (v << 1) | b[i]
    return v


def _sm(b, hi, lo):
    """Sign-magnitude: the top bit is the sign."""
    m = _u(b, hi - 1, lo)
    return -m if b[hi] else m


def string_number(b):
    return _u(b, 84, 81)


def ephemeris_from(s1, s2, s3, s4, prn, t_rx):
    """gnss_ephemeris R record from strings 1-4 of one frame. t_rx (unix) places tb, a time
    of the Moscow day, in the UTC day it falls nearest."""
    km = 1e3
    tb_s = _u(s2, 76, 70) * 900.0
    t_msk = t_rx + UTC_MSK_S
    day0 = math.floor(t_msk / 86400.0) * 86400.0
    t_tb = min(
        (day0 + d * 86400.0 + tb_s - UTC_MSK_S for d in (-1, 0, 1)),
        key=lambda t: abs(t - t_rx),
    )
    toe = ge.gpst_of_utc(t_tb)
    tk = _u(s1, 76, 72) * 3600.0 + _u(s1, 71, 66) * 60.0 + _u(s1, 65, 65) * 30.0
    pos = tuple(_sm(s, 35, 9) * 2.0**-11 * km for s in (s1, s2, s3))
    vel = tuple(_sm(s, 64, 41) * 2.0**-20 * km for s in (s1, s2, s3))
    acc = tuple(_sm(s, 40, 36) * 2.0**-30 * km for s in (s1, s2, s3))
    e = dict(
        sys="R",
        prn=prn,
        af0=-_sm(s4, 80, 59) * 2.0**-30,  # -tau_n, the RINEX convention
        af1=_sm(s3, 79, 69) * 2.0**-40,  # +gamma_n
        frame_time=tk,
        pos=pos,
        vel=vel,
        acc=acc,
        health=float((_u(s2, 80, 78) >> 2) | s3[65]),  # Bn's top bit, or l_n
        k=None,
        age=float(_u(s4, 53, 49)),  # E_n, days
        toc_gpst=toe,
        toe_gpst=toe,
        tb=_u(s2, 76, 70),
        slot=_u(s4, 15, 11),
        dtau=_sm(s4, 58, 54) * 2.0**-30,
        mod=_u(s4, 10, 9),
    )
    if math.sqrt(sum(p * p for p in pos)) < 1.0e7:
        return None
    return e


def _key(e):
    """What the next frame must repeat: everything but the frame time."""
    return (e["tb"], e["pos"], e["vel"], e["acc"], e["af0"], e["af1"], e["slot"])


def brdc_xcheck(brdc, sysc, prn, eph, log=None):
    """The decoded ephemeris against BRDC's record for the same tb, both propagated by
    sat_pos_clk to tb: BRDC carries the broadcast values, so a correct decode agrees to the
    message's own rounding."""
    try:
        recs = brdc["eph"].get((sysc, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, eph["toe_gpst"])
        if be is None:
            return " | BRDC:stale"
        t = eph["toe_gpst"]
        a, _, ca = ge.sat_pos_clk(be, t)
        b, _, cb = ge.sat_pos_clk(eph, t)
        return " | BRDC dpos=%.2f m dclk=%.1f ns (brdc tb %+.0f s)" % (
            math.dist(a, b),
            (cb - ca) * 1e9,
            be["toe_gpst"] - t,
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


class _PrnState(HistoryState):
    def __init__(self):
        super().__init__()
        self.sym_s = 0.02
        self.strings = {}  # UTC start of the string (s) -> (m, bits, t_decoded)
        self.n_str = 0  # distinct strings decoded
        self.n_corr = 0  # strings the Hamming code corrected
        self.n_tm = 0  # time marks found
        self.cand = None  # the last frame's ephemeris, awaiting the next frame's
        self.eph = None


class GloNavPredictor(SymbolHistory):
    hist_max = HIST_BITS

    def new_state(self):
        return _PrnState()

    def ingest(self, prn, obs):
        try:
            sym_s = int(obs["br"]) * float(obs["rec_dt"])
        except (KeyError, TypeError, ValueError):
            sym_s = 0.0
        super().ingest(prn, obs)
        st = self._p.get(prn)
        if st is not None and sym_s > 0:
            st.sym_s = sym_s

    def _decode(self, prn, st):
        n = len(st.hist)
        first = st.scan - st.s0
        last = n - STRING_BITS  # a start needs its whole string and time mark
        if last < first:
            return
        mag = np.abs(st.hist)
        sgn = np.sign(st.hist)
        nz = TM_BITS != 0
        now = time.time()
        for i in range(first, last + 1):
            tm = st.hist[i + DATA_BITS : i + STRING_BITS]
            # The 8 nonzero places agree with the pattern at one polarity, and the 7 zero places
            # hold little power.
            agree = sgn[i + DATA_BITS : i + STRING_BITS][nz] * TM_BITS[nz]
            if not (np.all(agree > 0) or np.all(agree < 0)):
                continue
            m_nz = mag[i + DATA_BITS : i + STRING_BITS][nz].mean()
            if np.abs(tm[~nz]).mean() > 0.5 * m_nz:
                continue
            st.n_tm += 1
            d = sgn[i : i + DATA_BITS]
            if np.any(d == 0):
                continue
            # Relative code: each bit is the change from the bit before; bit 85 is idle (0).
            b = [0] * (DATA_BITS + 1)
            for k in range(1, DATA_BITS):
                b[DATA_BITS - k] = int(d[k] != d[k - 1])
            ncorr = hamming_check(b)
            if ncorr is None:
                continue
            m = string_number(b)
            t_start = st.grid0 + (st.s0 + i) * st.sym_s - PROP_S
            ts = round(t_start / 2.0) * 2.0
            if m != int(ts % 30.0) // 2 + 1:
                continue
            if ts not in st.strings:
                st.n_str += 1
                st.n_corr += ncorr
            st.strings[ts] = (m, b, now)
            if m <= 4:
                self._assemble(prn, st, ts - 2.0 * (m - 1), t_start)
        st.scan = st.s0 + last + 1
        for k in [k for k, v in st.strings.items() if now - v[2] > STRING_TTL_S]:
            del st.strings[k]

    def _assemble(self, prn, st, f0, t_rx):
        got = [st.strings.get(f0 + 2.0 * j) for j in range(4)]
        if not all(g is not None and g[0] == j + 1 for j, g in enumerate(got)):
            return
        e = ephemeris_from(got[0][1], got[1][1], got[2][1], got[3][1], prn, t_rx)
        if e is None or e["slot"] != prn:
            return
        # Taken once the next frame repeats it: a string the Hamming code "corrected" to the
        # wrong word would otherwise land in the orbit.
        if st.cand is not None and _key(e) == _key(st.cand) and f0 != st.cand["_f0"]:
            st.eph = e
        st.cand = dict(e, _f0=f0)

    def ephemeris(self, prn):
        st = self._p.get(prn)
        return st.eph if st is not None else None

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "words": st.n_str,
            "corrected": st.n_corr,
            "marks": st.n_tm,
            "eph": st.eph is not None,
        }


def _string_bits(fields):
    """A string's bits b[1..85] from {(hi, lo): value} (sign-magnitude where negative)."""
    b = [0] * (DATA_BITS + 1)
    for (hi, lo), v in fields.items():
        w = hi - lo + 1
        if v < 0:
            v = (1 << (w - 1)) | (-v)
        for i in range(w):
            b[lo + i] = (v >> i) & 1
    hamming_encode(b)
    return b


def _air(b):
    """b[1..85] -> the 100 bit values on air after the tracker code: relative code (bit 85
    first), then the time mark."""
    r, prev = [], 0
    for k in range(DATA_BITS, 0, -1):
        prev ^= b[k] if k < DATA_BITS else 0
        r.append(1.0 - 2.0 * prev)
    return r + list(TM_BITS)


def _selftest():
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = [0] + [int(x) for x in rng.integers(0, 2, DATA_BITS)]
        b[DATA_BITS] = 0
        hamming_encode(b)
        assert hamming_check(list(b)) == 0
        for flip in rng.choice(np.arange(1, DATA_BITS + 1), 3, replace=False)[:1]:
            c = list(b)
            c[flip] ^= 1
            assert hamming_check(c) == 1 and c[9:] == b[9:], flip
        c = list(b)
        i, j = rng.choice(np.arange(9, DATA_BITS + 1), 2, replace=False)
        c[i] ^= 1
        c[j] ^= 1
        assert hamming_check(c) is None
    # A frame's strings 1-4 for slot 7, twice, starting on a UTC 30 s boundary.
    t_frame = math.floor(time.time() / 30.0) * 30.0 - 60.0
    tb = int(((t_frame + UTC_MSK_S) % 86400.0) // 900)
    fields = {
        1: {
            (84, 81): 1,
            (76, 72): 5,
            (71, 66): 7,
            (64, 41): -1234567,
            (40, 36): 3,
            (35, 9): 12345678,
        },
        2: {
            (84, 81): 2,
            (80, 78): 0,
            (76, 70): tb,
            (64, 41): 2345678,
            (40, 36): -2,
            (35, 9): -23456789,
        },
        3: {
            (84, 81): 3,
            (79, 69): -321,
            (64, 41): -345678,
            (40, 36): 1,
            (35, 9): 34567890,
        },
        4: {
            (84, 81): 4,
            (80, 59): -654321,
            (58, 54): 2,
            (53, 49): 1,
            (15, 11): 7,
            (10, 9): 1,
        },
    }
    air = []
    for f in range(2):
        for m in range(1, 16):
            fl = fields.get(m, {(84, 81): m})
            air += _air(_string_bits(fl))
    p = GloNavPredictor()
    # The time mark's zero places as weak bits (soft 0.1), as the combiner exports them.
    obs = dict(
        utc_ref=t_frame + PROP_S,
        rec_dt=0.001,
        phase=0,
        br=20,
        bits=[[i, 1 if v >= 0 else -1] for i, v in enumerate(air)],
        soft=[10.0 * v if v != 0 else 0.1 for v in air],
    )
    p.ingest(7, obs)
    st = p._p[7]
    assert st.n_str == 30, st.n_str
    e = p.ephemeris(7)
    assert e is not None and e["slot"] == 7 and e["tb"] == tb
    assert abs(e["pos"][0] - 12345678 * 2.0**-11 * 1e3) < 1e-6
    assert abs(e["af0"] - 654321 * 2.0**-30) < 1e-15
    assert abs(e["af1"] + 321 * 2.0**-40) < 1e-20
    print(
        "glonass_nav_predictor self-test: OK (%d strings, eph tb %d)"
        % (st.n_str, e["tb"])
    )


if __name__ == "__main__":
    _selftest()
