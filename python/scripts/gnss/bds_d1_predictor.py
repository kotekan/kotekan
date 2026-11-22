"""BeiDou D1 navigation message decode (B3I and B2I, MEO and IGSO satellites).

Ingests a combiner's bit_export nav_obs: one 20 ms D1 bit per slot, the NH20 already wiped by
the tracker code, each bit's soft value when the combiner exports it. Emits are chained into a
per-PRN bit history (navobs_align.SymbolHistory). Subframes (300 bits, 6 s) are found by their
preamble, confirmed by the next subframe's preamble 300 bits on, and each word is corrected by
BCH(15,11,1). Subframes 1-3 of one frame (SOWs 6 s apart) give the ephemeris, taken when the
next frame repeats it, and built as a gnss_ephemeris record (sys "C"), so sat_pos_clk
propagates it and the BRDC cross-check compares like with like.

Layout per the BDS B1I/B3I ICD D1 message: word 1 is the preamble, 4 reserved bits, then a
BCH(15,11) codeword; words 2-10 interleave two BCH(15,11) codewords bit by bit, stored after
correction as [11 info, 11 info, 4 parity, 4 parity]. The field positions are RTKLIB's
decode_bds_d1_eph (rcvraw.c), on that stored 300-bit subframe, and the BCH syndrome register
PocketSDR's (sdr_nav.py).

The D2 message of the GEOs (500 bps, no NH) is not decoded: no BeiDou GEO rises here.
"""

import math
import time

import numpy as np

import gnss_ephemeris as ge
from navobs_align import HistoryState, SymbolHistory

PREAMBLE = (1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0)
SF_BITS = 300
HIST_BITS = (
    1500  # bits kept per PRN (30 s: a subframe and the next one's preamble, with room)
)
SF_TTL_S = 7200.0  # a cached subframe older than this is stale
# BCH(15,11) is a perfect code: every 15 bits lie one flip from a codeword, so a word always
# "passes" after correction. A subframe is taken only when few of its 19 codewords needed a
# flip and its SOW agrees with the clock.
MAX_CORRECTED = 3
MAX_SOW_ERR_S = 300.0
SC2RAD = 3.1415926535898  # the BDS ICD's pi


def _syndrome(b):
    """BCH(15,11,1) syndrome register over 15 bits, PocketSDR's (sdr_nav.py); 0 = a codeword."""
    r = 0
    for x in b:
        r = (int(x) << 3) ^ ((r & 1) * 0b1100) ^ (r >> 1)
    return r


# syndrome -> the bit a single error sits in: the code is perfect, so all 15 are covered
_ERR_AT = {_syndrome([int(j == k) for j in range(15)]): k for k in range(15)}


def bch_decode(b):
    """15 bits (0/1) -> (corrected 15 bits, whether a bit was flipped)."""
    b = [int(x) for x in b]
    r = _syndrome(b)
    if r:
        b[_ERR_AT[r]] ^= 1
    return b, bool(r)


def bch_encode(info):
    """11 info bits -> the 15-bit systematic codeword, g(x) = x^4 + x + 1 (for the self-test)."""
    reg = [0, 0, 0, 0]
    for x in info:
        fb = int(x) ^ reg[3]
        reg = [fb, reg[0] ^ fb, reg[1], reg[2]]
    return list(info) + reg[::-1]


def decode_subframe(raw):
    """300 received bits (preamble first) -> (stored 300 bits, n corrected), or None if a word
    still fails its BCH check after correction."""
    out = list(raw[:15])
    w, n = bch_decode(raw[15:30])
    out += w
    for i in range(30, SF_BITS, 30):
        s1, e1 = bch_decode(raw[i : i + 30 : 2])
        s2, e2 = bch_decode(raw[i + 1 : i + 31 : 2])
        out += s1[:11] + s2[:11] + s1[11:] + s2[11:]
        n += e1 + e2
    return out, n


def _u(b, p, n):
    v = 0
    for x in b[p : p + n]:
        v = (v << 1) | x
    return v


def _s(b, p, n):
    v = _u(b, p, n)
    return v - (1 << n) if v >> (n - 1) else v


def _u2(b, p1, n1, p2, n2):
    return (_u(b, p1, n1) << n2) | _u(b, p2, n2)


def _s2(b, p1, n1, p2, n2):
    v = _u2(b, p1, n1, p2, n2)
    n = n1 + n2
    return v - (1 << n) if v >> (n - 1) else v


def fra_sow(b):
    return _u(b, 15, 3), _u2(b, 18, 8, 30, 12)


def ephemeris_from(sf1, sf2, sf3, prn):
    """Subframes 1-3 of one frame (stored bits) -> a gnss_ephemeris record, or None."""
    sow1 = fra_sow(sf1)[1]
    toc = _u2(sf1, 73, 9, 90, 8) * 8.0
    toe = ((_u(sf2, 290, 2) << 15) | _u2(sf3, 42, 10, 60, 5)) * 8.0
    if toc != toe:
        return None
    week = _u(sf1, 60, 13)
    if toe - sow1 > 302400.0:
        week -= 1
    elif toe - sow1 < -302400.0:
        week += 1
    toe_gpst = (week + 1356.0) * 604800.0 + toe - ge.BDT_GPST
    sqrta = _u2(sf2, 250, 12, 270, 20) * 2.0**-19
    e = dict(
        sys="C",
        prn=prn,
        health=_u(sf1, 42, 1),
        iodc=_u(sf1, 43, 5),
        accuracy=_u(sf1, 48, 4),
        week=week,
        tgd=_s(sf1, 98, 10) * 0.1e-9,
        tgd2=_s2(sf1, 108, 4, 120, 6) * 0.1e-9,
        af2=_s(sf1, 214, 11) * 2.0**-66,
        af0=_s2(sf1, 225, 7, 240, 17) * 2.0**-33,
        af1=_s2(sf1, 257, 5, 270, 17) * 2.0**-50,
        iode=_u(sf1, 287, 5),
        dn=_s2(sf2, 42, 10, 60, 6) * 2.0**-43 * SC2RAD,
        cuc=_s2(sf2, 66, 16, 90, 2) * 2.0**-31,
        m0=_s2(sf2, 92, 20, 120, 12) * 2.0**-31 * SC2RAD,
        ecc=_u2(sf2, 132, 10, 150, 22) * 2.0**-33,
        cus=_s(sf2, 180, 18) * 2.0**-31,
        crc=_s2(sf2, 198, 4, 210, 14) * 2.0**-6,
        crs=_s2(sf2, 224, 8, 240, 10) * 2.0**-6,
        sqrta=sqrta,
        toe_sow=toe,
        i0=_s2(sf3, 65, 17, 90, 15) * 2.0**-31 * SC2RAD,
        cic=_s2(sf3, 105, 7, 120, 11) * 2.0**-31,
        omegadot=_s2(sf3, 131, 11, 150, 13) * 2.0**-43 * SC2RAD,
        cis=_s2(sf3, 163, 9, 180, 9) * 2.0**-31,
        idot=_s2(sf3, 189, 13, 210, 1) * 2.0**-43 * SC2RAD,
        omega0=_s2(sf3, 211, 21, 240, 11) * 2.0**-31 * SC2RAD,
        omega=_s2(sf3, 251, 11, 270, 21) * 2.0**-31 * SC2RAD,
        toe_gpst=toe_gpst,
        toc_gpst=toe_gpst,
    )
    return e if sqrta > 1000.0 else None


def brdc_xcheck(brdc, sysc, prn, eph, log=None):
    """The decoded ephemeris against BRDC's nearest record, both propagated by sat_pos_clk to
    the decoded toe: a correct decode agrees to well under a metre."""
    try:
        recs = brdc["eph"].get((sysc, prn))
        if not recs:
            return " | BRDC:no-eph"
        be = ge.best_eph(recs, eph["toe_gpst"])
        if be is None:
            return " | BRDC:stale"
        t = eph["toe_gpst"]
        a = ge.sat_pos_clk(be, t)[0]
        b = ge.sat_pos_clk(eph, t)[0]
        return " | BRDC dpos=%.2f m (brdc toe %+.0f s, AODE %d)" % (
            math.dist(a, b),
            be["toe_gpst"] - t,
            eph["iode"],
        )
    except Exception as ex:
        return " | BRDC xcheck err: %s" % ex


class _PrnState(HistoryState):
    def __init__(self):
        super().__init__()
        self.subframes = {}  # (fra, sow) -> (stored bits, t_decoded)
        self.n_sf = 0  # distinct subframes decoded
        self.n_corr = 0  # bits BCH corrected
        self.last_sow = None
        self.cand = None  # the last frame's ephemeris, awaiting the next frame's
        self.eph = None


def bdt_sow_now():
    """BDT seconds of week from the system clock."""
    return (ge.gpst_of_utc(time.time()) + ge.BDT_GPST) % 604800.0


class D1Predictor(SymbolHistory):
    hist_max = HIST_BITS

    def __init__(self):
        super().__init__()
        self.max_sow_err_s = (
            MAX_SOW_ERR_S  # None: no clock check (the self-test's stream)
        )

    def new_state(self):
        return _PrnState()

    def _decode(self, prn, st):
        n = len(st.hist)
        first = st.scan - st.s0
        last = n - (SF_BITS + len(PREAMBLE))  # a start needs the next preamble too
        if last < first:
            return
        pre = 1.0 - 2.0 * np.array(PREAMBLE)
        hard = np.where(st.hist < 0, -1.0, np.where(st.hist > 0, 1.0, 0.0))
        c = np.correlate(hard, pre, "valid")
        now = time.time()
        for i in range(first, last + 1):
            if abs(c[i]) != len(PREAMBLE) or c[i + SF_BITS] != c[i]:
                continue
            pol = 1.0 if c[i] > 0 else -1.0
            raw = [int(x) for x in (pol * st.hist[i : i + SF_BITS] < 0)]
            sf, ncorr = decode_subframe(raw)
            if ncorr > MAX_CORRECTED:
                continue
            fra, sow = fra_sow(sf)
            if not 1 <= fra <= 5 or sow >= 604800 or sow % 6:
                continue
            if self.max_sow_err_s is not None:
                err = (sow - bdt_sow_now() + 302400.0) % 604800.0 - 302400.0
                if abs(err) > self.max_sow_err_s:
                    continue
            if (fra, sow) not in st.subframes:
                st.n_sf += 1
                st.n_corr += ncorr
            st.subframes[(fra, sow)] = (sf, now)
            st.last_sow = sow
            self._assemble(prn, st, fra, sow)
        st.scan = st.s0 + last + 1
        for k in [k for k, v in st.subframes.items() if now - v[1] > SF_TTL_S]:
            del st.subframes[k]

    def _assemble(self, prn, st, fra, sow):
        if fra > 3:
            return
        s1 = sow - 6 * (fra - 1)
        got = [st.subframes.get((f, s1 + 6 * (f - 1))) for f in (1, 2, 3)]
        if all(got):
            e = ephemeris_from(got[0][0], got[1][0], got[2][0], prn)
            # Taken once the next frame repeats it: a two-bit error the BCH "corrects" to the
            # wrong codeword would otherwise land in the orbit.
            if e is not None and e == st.cand:
                st.eph = e
            st.cand = e

    def ephemeris(self, prn):
        st = self._p.get(prn)
        return st.eph if st is not None else None

    def health(self, prn):
        st = self._p.get(prn)
        if st is None:
            return None
        return {
            "pages": st.n_sf,
            "words": st.n_sf,
            "corrected": st.n_corr,
            "sow": st.last_sow,
            "eph": st.eph is not None,
        }


def sf_to_raw(sf):
    """Stored 300 bits -> the on-air order (word 1 as is, words 2-10 re-interleaved): the
    inverse of decode_subframe's layout, so a second decode can confirm every word passes.
    """
    raw = list(sf[:30])
    for i in range(30, SF_BITS, 30):
        w = sf[i : i + 30]
        c1 = w[:11] + w[22:26]
        c2 = w[11:22] + w[26:30]
        for k in range(15):
            raw += [c1[k], c2[k]]
    return raw


def _selftest():
    """Random frames of three subframes (valid preamble, FraID, SOW), BCH-encoded and
    interleaved, with a bit error in word 1 and in one other word, chopped into overlapping
    100-bit emits of random polarity with soft values: every subframe must come back, and the
    BCH must agree with the ICD encoder."""
    rng = np.random.RandomState(3)
    sfs = []
    stream = []
    sow0 = 3600 * 7
    for k in range(9):
        fra = k % 3 + 1
        b = [int(x) for x in rng.randint(0, 2, SF_BITS)]
        b[:11] = PREAMBLE
        b[15:18] = [(fra >> (2 - j)) & 1 for j in range(3)]
        sow = sow0 + 6 * k
        sb = [(sow >> (19 - j)) & 1 for j in range(20)]
        b[18:26] = sb[:8]
        # word 1's codeword, then words 2-10 as [11 info, 11 info, 4 parity, 4 parity]
        b[15:30] = bch_encode(b[15:26])
        b[30:42] = sb[8:]
        for i in range(30, SF_BITS, 30):
            c1 = bch_encode(b[i : i + 11])
            c2 = bch_encode(b[i + 11 : i + 22])
            b[i : i + 30] = c1[:11] + c2[:11] + c1[11:] + c2[11:]
        sfs.append(b)
        raw = sf_to_raw(b)
        raw[
            15 + rng.randint(0, 15)
        ] ^= 1  # an error in word 1 and in one interleaved word
        raw[30 * rng.randint(1, 10) + rng.randint(0, 30)] ^= 1
        stream += raw
    stream += list(PREAMBLE) + [0] * 40
    sym = np.where(np.array(stream) == 0, 1.0, -1.0) * (1.0 + rng.rand(len(stream)))
    pred = D1Predictor()
    pred.max_sow_err_s = None
    prn = 21
    for e in range(0, len(sym) - 100, 40):
        pol = 1 - 2 * rng.randint(0, 2)
        obs = {
            "utc_ref": e * 0.02,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 20,
            "bits": [[i, int(np.sign(sym[e + i]) * pol)] for i in range(100)],
            "soft": [int(round(10 * sym[e + i] * pol)) for i in range(100)],
        }
        pred.ingest(prn, obs)
    st = pred._p[prn]
    got = {k: v[0] for k, v in st.subframes.items()}
    ok = len(got) == len(sfs) and all(
        got.get((k % 3 + 1, sow0 + 6 * k)) == sfs[k] for k in range(9)
    )
    print(
        "subframes %d/%d, corrected %d bits: %s"
        % (len(got), len(sfs), st.n_corr, "PASS" if ok else "FAIL")
    )
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
