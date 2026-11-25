"""Galileo E6-B C/NAV codec: page symbols to page bits, the HAS page header, and the HAS message:
its recovery from any MS of its pages (Reed-Solomon) and the type 1 parse.

C/NAV uses I/NAV's FEC (galileo_inav's Viterbi and CRC-24Q). A page is 1 s, 1000 symbols at
1000 sps: a 16-symbol sync, then 984 symbols, the rate-1/2 code of 492 bits (486 and the 6-bit
tail) block-interleaved 123x8 as F/NAV's 61x8. Of the 486 bits, 14 are reserved, 24 are the HAS
page header, 424 the page body, and 24 the CRC over the 462 before it. A satellite with no HAS
to send sends the dummy header 0xAF3BC3.

The header: HAS status (2), reserved (2), message type (2), message ID (5), message size less
one (5), page ID (8). A message of MS pages, 53 octets each, is the first MS rows of a 32 x 53
matrix whose columns are each coded RS(255, 32) by a systematic generator; page ID p carries
row p - 1 of the code, so pages 1 to MS are the message itself and any MS distinct pages give it
back.

Refs: Galileo HAS SIS ICD 1.0 (sync, interleaver, page layout, message type 1); the RS code as
gnss-sdr's ReedSolomon("E6B"): GF(2^8) on x^8 + x^4 + x^3 + x^2 + 1, roots alpha^1 to
alpha^223. The sync, interleaver and CRC span were checked by CRC on live pages (11-25), the
generator against gnss-sdr's table and by live pages beyond MS.
"""

import numpy as np

from galileo_inav import conv_encode, crc24q, viterbi_decode, _uint

SYM_S = 0.001
SYNC = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
N_SYNC = len(SYNC)
INTERLEAVE_ROWS = 8
INTERLEAVE_COLS = 123
N_DATA_SYM = INTERLEAVE_ROWS * INTERLEAVE_COLS  # 984
PAGE_SYMS = N_SYNC + N_DATA_SYM  # 1000
PAGE_BITS = N_DATA_SYM // 2 - 6  # 486
CRC_AT = PAGE_BITS - 24  # 462
HDR_AT = 14
BODY_AT = HDR_AT + 24
BODY_BITS = 424
OCTETS = BODY_BITS // 8  # 53
DUMMY = 0xAF3BC3

# HAS SIS ICD 1.0: the validity interval index (s), and the signal index per system.
VALIDITY_S = [5, 10, 15, 20, 30, 60, 90, 120, 180, 240, 300, 600, 900, 1800, 3600]
GNSS = {0: "G", 2: "E"}
SIGNALS = {
    "G": {0: "L1 C/A", 3: "L1C(D)", 4: "L1C(P)", 5: "L1C(D+P)", 6: "L2 CM", 7: "L2 CL",
          8: "L2 CM+CL", 9: "L2 P", 11: "L5 I", 12: "L5 Q", 13: "L5 I+Q"},
    "E": {0: "E1-B", 1: "E1-C", 2: "E1-B+C", 3: "E5a-I", 4: "E5a-Q", 5: "E5a-I+Q", 6: "E5b-I",
          7: "E5b-Q", 8: "E5b-I+Q", 9: "E5-I", 10: "E5-Q", 11: "E5-I+Q", 12: "E6-B",
          13: "E6-C", 14: "E6-B+C"},
}  # fmt: skip


def interleave(sym):
    """Write 984 coded symbols into a 123x8 matrix by rows, read it by columns (F/NAV's)."""
    m = np.asarray(sym, dtype=np.int8).reshape(INTERLEAVE_COLS, INTERLEAVE_ROWS)
    return m.T.reshape(-1)


def deinterleave(sym):
    m = np.asarray(sym, dtype=float).reshape(INTERLEAVE_ROWS, INTERLEAVE_COLS)
    return m.T.reshape(-1)


def encode_page(content_bits):
    """486 page bits -> 1000 page symbols (0/1): FEC, interleave, sync first."""
    if len(content_bits) != PAGE_BITS:
        raise ValueError(
            "a C/NAV page is %d bits, got %d" % (PAGE_BITS, len(content_bits))
        )
    fec = conv_encode([int(b) for b in content_bits], flush=True)
    return np.concatenate([np.array(SYNC, dtype=np.int8), interleave(fec)])


def decode_page(symbols):
    """1000 soft symbols (+ for a 0 bit), sync first -> the 486 page bits, or None if the sync
    does not match."""
    s = np.asarray(symbols, dtype=float)
    if len(s) < PAGE_SYMS or not np.array_equal(
        (s[:N_SYNC] < 0).astype(np.int8), np.array(SYNC, dtype=np.int8)
    ):
        return None
    bits = viterbi_decode(deinterleave(s[N_SYNC:PAGE_SYMS]), known_start=True)
    return [int(b) for b in bits[:PAGE_BITS]]


def check_page(bits):
    """The page's CRC-24Q, over its first 462 bits."""
    return len(bits) == PAGE_BITS and crc24q(bits[:CRC_AT]) == _uint(
        bits[CRC_AT:PAGE_BITS]
    )


def header(bits):
    """The HAS page header {status, type, mid, ms, pid}, or None for a dummy page."""
    h = _uint(bits[HDR_AT:BODY_AT])
    if h == DUMMY:
        return None
    return {
        "status": h >> 22,
        "type": (h >> 18) & 3,
        "mid": (h >> 13) & 31,
        "ms": ((h >> 8) & 31) + 1,
        "pid": h & 255,
    }


def body_octets(bits):
    """The page body as 53 octets."""
    return np.array(
        [_uint(bits[BODY_AT + 8 * k : BODY_AT + 8 * k + 8]) for k in range(OCTETS)],
        dtype=np.int64,
    )


# GF(2^8) on x^8 + x^4 + x^3 + x^2 + 1.
_EXP = np.zeros(512, dtype=np.int64)
_LOG = np.zeros(256, dtype=np.int64)
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
_EXP[255:510] = _EXP[:255]


def _mul(a, b):
    """Elementwise product over GF(2^8)."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return np.where((a == 0) | (b == 0), 0, _EXP[_LOG[a] + _LOG[b]])


def _gen_matrix():
    """The systematic RS(255, 32) generator, 255 x 32: the identity, then each unit message's
    parity, the remainder of m(x) x^223 by g(x) = prod (x - alpha^i), i = 1..223, highest power
    first (the code word's order)."""
    g = [1]
    for i in range(1, 224):
        a = int(_EXP[i])
        g = [c ^ int(_mul(p, a)) for c, p in zip(g + [0], [0] + g)]
    g = np.array(g[1:], dtype=np.int64)
    G = np.zeros((255, 32), dtype=np.int64)
    G[:32] = np.eye(32, dtype=np.int64)
    for j in range(32):
        rem = np.zeros(223, dtype=np.int64)
        for k in range(32):
            fb = (1 if k == j else 0) ^ int(rem[0])
            rem = np.concatenate([rem[1:], [0]])
            if fb:
                rem ^= _mul(g, fb)
        G[32:, j] = rem
    return G


G = _gen_matrix()


def _combine(coef, rows):
    """sum_k coef[k] rows[k] over GF(2^8)."""
    out = np.zeros(rows.shape[1], dtype=np.int64)
    for c, r in zip(coef, rows):
        if c:
            out ^= _mul(r, c)
    return out


def recover(pages, ms):
    """The message's ms x 53 octets from {page ID: 53 octets} holding at least ms distinct page
    IDs, and the number of further pages that agree with it, or (None, 0) when they do not
    determine it or one disagrees. The lowest page IDs are solved for, the rest checked.
    """
    pids = sorted(pages)
    if len(pids) < ms or not all(1 <= p <= 255 for p in pids):
        return None, 0
    a = G[[p - 1 for p in pids[:ms]], :ms].copy()
    r = np.array([pages[p] for p in pids[:ms]], dtype=np.int64)
    for c in range(ms):
        piv = next((k for k in range(c, ms) if a[k, c]), None)
        if piv is None:
            return None, 0
        a[[c, piv]], r[[c, piv]] = a[[piv, c]], r[[piv, c]]
        inv = int(_EXP[255 - _LOG[a[c, c]]])
        a[c], r[c] = _mul(a[c], inv), _mul(r[c], inv)
        for k in range(ms):
            if k != c and a[k, c]:
                f = int(a[k, c])
                a[k] ^= _mul(a[c], f)
                r[k] ^= _mul(r[c], f)
    for p in pids[ms:]:
        if not np.array_equal(_combine(G[p - 1, :ms], r), pages[p]):
            return None, 0
    return r, len(pids) - ms


def message_bits(octets):
    return [
        int(b) for o in np.asarray(octets).reshape(-1) for b in format(int(o), "08b")
    ]


class _Reader:
    def __init__(self, bits):
        self.bits, self.at = bits, 0

    def u(self, n):
        if self.at + n > len(self.bits):
            raise ValueError(
                "message ends at bit %d, %d more wanted" % (len(self.bits), n)
            )
        v = _uint(self.bits[self.at : self.at + n])
        self.at += n
        return v

    def s(self, n):
        v = self.u(n)
        return v - (1 << n) if v >> (n - 1) else v


def _scaled(v, scale, absent):
    return None if v in absent else round(v * scale, 4)


def parse_mt1(bits, masks):
    """A type 1 message: {toh, mask_id, iod_set, flags, ...} with, per satellite ("E04"), the
    blocks it carries: orbit {iod, radial_m, in_track_m, cross_track_m}, clock_m, code_bias_m and
    phase_bias_cyc by signal name. masks ({mask ID: systems}) holds the masks of earlier messages
    and takes this one's; a body whose mask is not held is left unread ("mask": "not held").
    Unavailable values are None. Raises ValueError on a body shorter than its blocks."""
    rd = _Reader(bits)
    out = {"toh": rd.u(12)}
    flags = [rd.u(1) for _ in range(6)]
    names = ("mask", "orbit", "clock", "clock_subset", "code_bias", "phase_bias")
    out["flags"] = [n for n, f in zip(names, flags) if f]
    rd.u(4)
    out["mask_id"], out["iod_set"] = rd.u(5), rd.u(5)
    if out["toh"] >= 3600:
        return out
    if flags[0]:
        systems = []
        for _ in range(rd.u(4)):
            c = GNSS.get(rd.u(4), "?")
            sats, sigs = rd.u(40), rd.u(16)
            prns = [k + 1 for k in range(40) if sats >> (39 - k) & 1]
            sig = [k for k in range(16) if sigs >> (15 - k) & 1]
            cells = None
            if rd.u(1):
                cells = [[rd.u(1) for _ in sig] for _ in prns]
            rd.u(3)
            systems.append({"sys": c, "prns": prns, "sigs": sig, "cells": cells})
        rd.u(6)
        masks[out["mask_id"]] = systems
    systems = masks.get(out["mask_id"])
    if systems is None:
        out["mask"] = "not held"
        return out
    sats = {}
    order = [(s, p) for s in systems for p in s["prns"]]

    def sat(s, p):
        return sats.setdefault("%s%02d" % (s["sys"], p), {})

    if flags[1]:
        out["orbit_validity_s"] = VALIDITY_S[min(rd.u(4), 14)]
        for s, p in order:
            iod = rd.u(8 if s["sys"] == "G" else 10)
            dr, dit, dct = rd.s(13), rd.s(12), rd.s(12)
            sat(s, p)["orbit"] = {
                "iod": iod,
                "radial_m": _scaled(dr, 0.0025, (-4096,)),
                "in_track_m": _scaled(dit, 0.008, (-2048,)),
                "cross_track_m": _scaled(dct, 0.008, (-2048,)),
            }
    if flags[2]:
        out["clock_validity_s"] = VALIDITY_S[min(rd.u(4), 14)]
        mult = {id(s): rd.u(2) + 1 for s in systems}
        for s, p in order:
            sat(s, p)["clock_m"] = _scaled(
                rd.s(13), 0.0025 * mult[id(s)], (-4096, 4095)
            )
    if flags[3]:
        out["clock_subset_validity_s"] = VALIDITY_S[min(rd.u(4), 14)]
        for _ in range(rd.u(4)):
            c, m = GNSS.get(rd.u(4), "?"), rd.u(2) + 1
            s = next((x for x in systems if x["sys"] == c), None)
            if s is None:
                raise ValueError(
                    "clock subset for %s, not in mask %d" % (c, out["mask_id"])
                )
            sub = [p for p in s["prns"] if rd.u(1)]
            for p in sub:
                sat(s, p)["clock_m"] = _scaled(rd.s(13), 0.0025 * m, (-4096, 4095))
    for flag, key, scale, absent in (
        (4, "code_bias_m", 0.02, -1024),
        (5, "phase_bias_cyc", 0.01, -1024),
    ):
        if not flags[flag]:
            continue
        out[key.rsplit("_", 1)[0] + "_validity_s"] = VALIDITY_S[min(rd.u(4), 14)]
        for s in systems:
            for i, p in enumerate(s["prns"]):
                for j, k in enumerate(s["sigs"]):
                    if s["cells"] is not None and not s["cells"][i][j]:
                        continue
                    v = rd.s(11)
                    if flag == 5:
                        rd.u(2)  # the phase discontinuity indicator
                    name = SIGNALS.get(s["sys"], {}).get(k, "signal %d" % k)
                    sat(s, p).setdefault(key, {})[name] = _scaled(v, scale, (absent,))
    out["sats"] = sats
    out["bits_read"] = rd.at
    return out
