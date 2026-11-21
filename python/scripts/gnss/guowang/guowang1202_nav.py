#!/usr/bin/env python3
"""GuoWang 1202.025 MHz low-rate message: subframes from symbols, and the sender's ephemeris.

The low-rate channel is the sign of each 2 ms frame's ranging half, one 4 ms symbol per frame
pair (250 symbols/s). A 3 s subframe on the BDT 3 s grid is 750 symbols: the 24 uncoded symbols
0xE24DE8 (B-CNAV2's preamble), then 363 bits through the rate-1/2 K = 7 code (G1 = 171, G2 = 133
octal, G1 first, +1 for logic 0; the encoder starts each subframe at zero and is not flushed):
[3-bit page 1..4][20-bit BDT SOW of the subframe's start][316-bit payload][CRC-24Q over the 339
before it]. Pages 1-4 cycle every 12 s. The conventions are docs/chord/guowang/PAYLOAD.md's (on
kv/docs-tree) and lib/stages/gnss/guowang1202Nav's.

Each satellite's pages carry its own ephemeris and clock, re-issued at every BDT 5-minute
boundary (PAYLOAD.md, 2026-11-10 and 11-11): a BeiDou-style 16-parameter set plus third-order
harmonic terms, good to ~1 cm from toe - 100 s to toe + 600 s, and a clock bias (2^-36 s) and
drift. Ephemeris assembles one issue from the four pages and evaluates it.

    decode_subframes(symbols) -> [Subframe]          symbols: +-1 (or soft), 4 ms apart
    Ephemeris.add(subframe); Ephemeris.ready(); Ephemeris.position(sow)

Self-test: python3 guowang1202_nav.py (encode, corrupt, decode, and evaluate an ephemeris).
"""

import math
from dataclasses import dataclass, field

import numpy as np

PREAMBLE = 0xE24DE8
PRE_SYMS = 24
SUB_BITS = 363  # page 3 + SOW 20 + payload 316 + CRC 24
CODED_SYMS = 2 * SUB_BITS
SUBFRAME_SYMS = PRE_SYMS + CODED_SYMS  # 750, 3 s
PAYLOAD_BITS = 316
CRC24Q_POLY = 0x1864CFB
G1, G2 = 0o171, 0o133

MU = 3.986004418e14
OMEGA_E = 7.292115e-5  # BeiDou's, used for every system (PAYLOAD.md)
RE = 6378137.0
SHELL = {
    50: (1150e3, 50.0),
    86: (1175e3, 86.5),
}  # nominal height, reference inclination


def crc24q(bits):
    """CRC-24Q of a bit sequence, first bit first, zero initial value (gw1202_crc24q)."""
    c = 0
    for b in bits:
        c ^= (int(b) & 1) << 23
        c = ((c << 1) ^ CRC24Q_POLY) & 0xFFFFFF if c & 0x800000 else (c << 1) & 0xFFFFFF
    return c


def _parity(x):
    return bin(x).count("1") & 1


# Trellis: state = the six previous bits as gw1202_nav_encode keeps them; input b enters at bit 6.
_NEXT = np.zeros((64, 2), dtype=np.int64)
_OUT = np.zeros((64, 2, 2), dtype=np.float64)
for _s in range(64):
    for _b in (0, 1):
        _reg = (_b << 6) | _s
        _NEXT[_s, _b] = _reg >> 1
        _OUT[_s, _b, 0] = 1 - 2 * _parity(_reg & G1)
        _OUT[_s, _b, 1] = 1 - 2 * _parity(_reg & G2)


def conv_encode(bits):
    """Symbols (+-1) of `bits` from the zero state, as gw1202_nav_encode makes them."""
    s, out = 0, []
    for b in bits:
        out += [int(_OUT[s, b, 0]), int(_OUT[s, b, 1])]
        s = int(_NEXT[s, b])
    return np.array(out, dtype=np.int8)


# Each next state ns = (b << 5) | (s >> 1) has two predecessors, s = (ns & 31) << 1 and that
# plus one, both entered with b = ns >> 5.
_NS = np.arange(64)
_B = _NS >> 5
_P0 = (_NS & 31) << 1
_P1 = _P0 | 1


def viterbi(symbols):
    """Bits from symbols (+-1 or soft, +1 = logic 0), the encoder starting at zero, end free."""
    r = np.asarray(symbols, dtype=float).reshape(-1, 2)
    n = len(r)
    pm = np.full(64, -1e18)
    pm[0] = 0.0
    prev = np.zeros((n, 64), dtype=np.int64)
    for i in range(n):
        m0 = pm[_P0] + _OUT[_P0, _B, 0] * r[i, 0] + _OUT[_P0, _B, 1] * r[i, 1]
        m1 = pm[_P1] + _OUT[_P1, _B, 0] * r[i, 0] + _OUT[_P1, _B, 1] * r[i, 1]
        take1 = m1 > m0
        pm = np.where(take1, m1, m0)
        prev[i] = np.where(take1, _P1, _P0)
    s = int(np.argmax(pm))
    out = np.zeros(n, dtype=np.int8)
    for i in range(n - 1, -1, -1):
        out[i] = _B[s]
        s = int(prev[i, s])
    return out


def _preamble_symbols():
    return np.array(
        [1 - 2 * ((PREAMBLE >> (23 - k)) & 1) for k in range(PRE_SYMS)], dtype=float
    )


@dataclass
class Subframe:
    index: int  # the preamble's first symbol in the input
    sign: int  # the input's sign: +1 if it carried the symbols as sent
    page: int
    sow: int
    payload: np.ndarray


def decode_subframes(symbols, max_preamble_errors=3):
    """Every CRC-valid subframe in `symbols` (+-1 or soft, 4 ms apart, in time order)."""
    x = np.sign(np.asarray(symbols, dtype=float))
    pre = _preamble_symbols()
    out = []
    last = -SUBFRAME_SYMS
    for i in range(0, len(x) - SUBFRAME_SYMS + 1):
        if i - last < SUBFRAME_SYMS:
            continue
        c = float(np.dot(pre, x[i : i + PRE_SYMS]))
        if abs(c) < PRE_SYMS - 2 * max_preamble_errors:
            continue
        sign = 1 if c > 0 else -1
        coded = sign * np.asarray(
            symbols[i + PRE_SYMS : i + SUBFRAME_SYMS], dtype=float
        )
        bits = viterbi(coded)
        if crc24q(bits[:339]) != int("".join(str(int(b)) for b in bits[339:363]), 2):
            continue
        page = int("".join(str(int(b)) for b in bits[0:3]), 2)
        sow = int("".join(str(int(b)) for b in bits[3:23]), 2)
        out.append(Subframe(i, sign, page, sow, bits[23:339].copy()))
        last = i
    return out


def _u(p, first, width):
    return int("".join(str(int(b)) for b in p[first : first + width]), 2)


def _s(p, first, width):
    v = _u(p, first, width)
    return v - (1 << width) if v >> (width - 1) else v


# The words of one issue: (page, first payload bit, width, signed, scale), PAYLOAD.md 11-10 and
# 11-11. Angles are in semicircles or circles as the scale says; lengths in metres.
SC, CIRC = math.pi, 2.0 * math.pi
WORDS = {
    "week": (1, 12, 13, False, 1.0),
    "toc_count": (1, 25, 16, False, 1.0),  # SOW / 12 - 1
    "af0": (1, 42, 27, True, 2.0**-36),  # clock bias, s
    "af1": (1, 69, 22, True, 2.0**-50),  # clock drift, s/s
    "issue_tag": (1, 100, 18, False, 1.0),  # boundary SOW / 3 + 1
    "Cus": (1, 140, 25, True, 2.0**-34 * SC),
    "Cuc": (1, 165, 25, True, 2.0**-34 * SC),
    "Cis3": (1, 190, 21, True, 2.0**-34 * SC),
    "Cic3": (1, 211, 21, True, 2.0**-34 * SC),
    "dA": (2, 83, 27, True, 2.0**-10),
    "di": (2, 110, 31, True, 2.0**-35 * SC),
    "Omega_dot": (2, 163, 26, True, 2.0**-45 * SC),
    "Cis": (2, 189, 21, True, 2.0**-34 * SC),
    "dn": (3, 80, 27, True, 2.0**-45 * SC),
    "M0": (3, 130, 36, False, 2.0**-36 * CIRC),
    "Omega0": (3, 166, 36, False, 2.0**-36 * CIRC),
    "e": (3, 202, 30, False, 2.0**-35),
    "omega": (4, 92, 36, False, 2.0**-36 * CIRC),
    "Crs": (4, 128, 26, True, 2.0**-11),
    "Crc": (4, 154, 26, True, 2.0**-11),
    "Crs3": (4, 180, 26, True, 2.0**-11),
    "Crc3": (4, 206, 26, True, 2.0**-11),
}


@dataclass
class Ephemeris:
    """One sender's ephemeris and clock from the four pages of one issue. `shell` is 50 or 86
    (the inclination shell; the satellite's TLE says which). Pages are keyed by the issue, the
    BDT 5-minute boundary their subframe follows (a page is rewritten 12 s after it)."""

    shell: int
    pages: dict = field(default_factory=dict)  # issue toe -> {page: payload}

    @staticmethod
    def issue_of(sow):
        return ((sow - 12) // 300) * 300

    def add(self, sf):
        self.pages.setdefault(self.issue_of(sf.sow), {})[sf.page] = sf.payload

    def words(self, toe):
        """The words of issue `toe`, which must have all four pages."""
        p = self.pages[toe]
        w = {
            k: (_s if sg else _u)(p[pg], first, width) * scale
            for k, (pg, first, width, sg, scale) in WORDS.items()
        }
        w["toe"] = toe
        return w

    def latest(self):
        """(toe, words) of the newest issue with all four pages, or None."""
        for toe in sorted(self.pages, reverse=True):
            if all(k in self.pages[toe] for k in (1, 2, 3, 4)):
                return toe, self.words(toe)
        return None

    def position(self, w, sow):
        """ECEF position (m) at BDT second of week `sow` from words `w` (PAYLOAD.md's user
        algorithm; the receiver adds nothing)."""
        h, i_ref = SHELL[self.shell]
        t = sow - w["toe"]
        A = RE + h + w["dA"]
        n = math.sqrt(MU / A**3) + w["dn"]
        M = w["M0"] + n * t
        e = w["e"]
        E = M
        for _ in range(20):
            E = M + e * math.sin(E)
        nu = math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)
        phi = nu + w["omega"]
        s2, c2, s3, c3 = (
            math.sin(2 * phi),
            math.cos(2 * phi),
            math.sin(3 * phi),
            math.cos(3 * phi),
        )
        u = phi + w["Cus"] * s2 + w["Cuc"] * c2
        r = (
            A * (1 - e * math.cos(E))
            + w["Crs"] * s2
            + w["Crc"] * c2
            + w["Crs3"] * s3
            + w["Crc3"] * c3
        )
        inc = (
            math.radians(i_ref)
            + w["di"]
            + w["Cis"] * s2
            + w["Cis3"] * s3
            + w["Cic3"] * c3
        )
        # Omega0 is the node less GMST at the start of the BDT week, so its Earth-fixed
        # longitude is Omega0 + Omega_dot t - omega_e (toe + t).
        Om = w["Omega0"] + w["Omega_dot"] * t - OMEGA_E * sow
        x, y = r * math.cos(u), r * math.sin(u)
        return np.array(
            [
                x * math.cos(Om) - y * math.cos(inc) * math.sin(Om),
                x * math.sin(Om) + y * math.cos(inc) * math.cos(Om),
                y * math.sin(inc),
            ]
        )

    def clock(self, w, sow):
        """The sender's clock bias (s) at `sow`, from its reference time (toc_count)."""
        toc = (w["toc_count"] + 1) * 12.0
        return w["af0"] + w["af1"] * (sow - toc)


def _selftest():
    rng = np.random.default_rng(1)
    pre = _preamble_symbols()
    stream, sent = [], []
    for k in range(6):
        sow = 300000 + 3 * k
        page = (sow // 3) % 4 + 1
        payload = rng.integers(0, 2, PAYLOAD_BITS)
        head = [int(b) for b in format(page, "03b") + format(sow, "020b")] + list(
            payload
        )
        crc = crc24q(head)
        bits = head + [int(b) for b in format(crc, "024b")]
        stream += list(pre) + list(conv_encode(bits))
        sent.append((page, sow, payload))
    x = -np.array(stream, dtype=float)  # the stream arrives inverted
    x += rng.normal(0, 0.5, len(x))  # 6 dB per symbol
    x = np.concatenate([rng.normal(0, 1, 137), x])
    got = decode_subframes(x)
    assert len(got) == 6, len(got)
    for sf, (page, sow, payload) in zip(got, sent):
        assert (sf.page, sf.sow, sf.sign) == (page, sow, -1)
        assert np.array_equal(sf.payload, payload)
    # An issue built from words: the position at toe sits on the shell.
    eph = Ephemeris(shell=50)
    w = {k: 0.0 for k in WORDS}
    w.update(toe=300000.0, e=1e-4)
    r = np.linalg.norm(eph.position(w, 300000.0))
    assert abs(r - (RE + 1150e3) * (1 - 1e-4)) < 1.0, r
    print(
        "guowang1202_nav self-test OK: 6/6 subframes through noise and an inverted stream"
    )


if __name__ == "__main__":
    _selftest()
