"""Candidate ranging-code families for the GuoWang code search (#218), and GF(2) tools.

Codes are int8 arrays of +-1 (ICD logic 0 -> +1, the house convention). The PRN tables come from
the repo's code checks, which stay in kotekan (b1i_b3i_code_check.py reads the B3I G2 table out of
beidouB3ICode.cpp; l3oc_code_check.py holds the L3OC LFSRs); kotekan_src finds that tree from
either repo.

Families:
    b3i      BeiDou B3I PRN 1-63 (10230 chips, 10.23 Mcps, 1 ms)
    b3i_all  every B3I-structured code: the truncated G1 (8190-chip reset) XOR each of the 8191
             phases of the G2 m-sequence, plus G1 alone (8192 codes). A G2 initial state is a G2
             phase, so this is every code the B3I generator can make with any 13-bit G2 state.
    b1i      BeiDou B1I PRN 1-63 (2046 chips, 2.046 Mcps, 1 ms)
    b1i_all  G1 XOR each of the 2047 phases of G2, plus G1 alone (2048 codes, truncated to 2046)
    l3oc     GLONASS L3OC data + pilot: DC1 x each of the 127 phases of the 7-stage LFSR (127)
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import kotekan_src  # noqa: E402

sys.path.insert(0, kotekan_src.src("python", "scripts", "gnss"))
import b1i_b3i_code_check as _b13  # noqa: E402
import l3oc_code_check as _l3  # noqa: E402

B3I_N, B1I_N, L3OC_N = 10230, 2046, 10230
B3I_RC, B1I_RC, L3OC_RC = 10.23e6, 2.046e6, 10.23e6

# Fibonacci LFSR taps in the code checks' convention: chip = reg[0], then reg shifts toward 0 and
# reg[n-1] takes the XOR of the tapped stages, i.e. s[k+n] = XOR_t s[k+t].
B3I_G1_TAPS, B3I_G2_TAPS, B3I_DEG = (0, 9, 10, 12), (0, 1, 3, 4, 6, 7, 8, 12), 13
B1I_G1_TAPS, B1I_G2_TAPS, B1I_DEG = (0, 1, 2, 3, 4, 10), (0, 2, 3, 6, 7, 8, 9, 10), 11


def lfsr_bits(taps, state, n):
    """n output bits (0/1) of the Fibonacci LFSR with reg = state (list, reg[0] = output end)."""
    reg = list(state)
    out = np.empty(n, np.uint8)
    for k in range(n):
        out[k] = reg[0]
        fb = 0
        for t in taps:
            fb ^= reg[t]
        reg = reg[1:] + [fb]
    return out


def mseq_and_states(taps, deg, state):
    """(one period of the m-sequence from `state`, {state tuple: phase})."""
    reg = list(state)
    period = (1 << deg) - 1
    bits = np.empty(period, np.uint8)
    phase = {}
    for k in range(period):
        phase[tuple(reg)] = k
        bits[k] = reg[0]
        fb = 0
        for t in taps:
            fb ^= reg[t]
        reg = reg[1:] + [fb]
    if len(phase) != period or tuple(reg) != tuple(state):
        raise RuntimeError("taps %s are not a maximal-length LFSR" % (taps,))
    return bits, phase


def to_pm(bits):
    """0/1 -> +1/-1 int8."""
    return (1 - 2 * np.asarray(bits, np.int8)).astype(np.int8)


def _b3i_g1():
    """The truncated B3I G1: all-ones start, reset to all-ones after the 8190th chip."""
    g = lfsr_bits(B3I_G1_TAPS, [1] * B3I_DEG, 8190)
    return np.resize(g, B3I_N)


def b3i_structure():
    """(G1 truncated, 10230 bits; G2 m-sequence, 8191 bits; {G2 state: phase})."""
    g1 = _b3i_g1()
    m2, phase = mseq_and_states(B3I_G2_TAPS, B3I_DEG, [1] * B3I_DEG)
    return g1, m2, phase


def gold_codes(g1, m2, phases, n):
    """+-1 codes g1[k] ^ m2[(k + phase) % P] for each phase (phase < 0: g1 alone)."""
    P = len(m2)
    k = np.arange(n)
    out = np.empty((len(phases), n), np.int8)
    for i, ph in enumerate(phases):
        out[i] = to_pm(g1[:n] if ph < 0 else g1[:n] ^ m2[(k + ph) % P])
    return out


def b3i_prn_phases():
    """G2 phase of each B3I PRN's initial state (from beidouB3ICode.cpp via the code check)."""
    _, _, phase = b3i_structure()
    out = []
    for s in _b13._b3i_init():
        reg = [0] * B3I_DEG
        for k in range(B3I_DEG):
            reg[12 - k] = 1 if s[k] == "1" else 0
        out.append(phase[tuple(reg)])
    return out


def b3i_prn(prns=range(1, 64)):
    g1, m2, _ = b3i_structure()
    ph = b3i_prn_phases()
    return gold_codes(g1, m2, [ph[p - 1] for p in prns], B3I_N)


def b3i_all():
    """(codes [8192, 10230], labels): label = G2 phase (-1 = G1 alone), PRN noted where it is one."""
    g1, m2, _ = b3i_structure()
    phases = list(range(len(m2))) + [-1]
    prn_of = {ph: p + 1 for p, ph in enumerate(b3i_prn_phases())}
    labels = [
        ("PRN%d/" % prn_of[ph] if ph in prn_of else "") + "g2ph%d" % ph for ph in phases
    ]
    return gold_codes(g1, m2, phases, B3I_N), labels


def _b1i_init():
    return [1 if j % 2 == 1 else 0 for j in range(B1I_DEG)]


def b1i_structure():
    g1 = lfsr_bits(B1I_G1_TAPS, _b1i_init(), B1I_N)
    m2, phase = mseq_and_states(B1I_G2_TAPS, B1I_DEG, _b1i_init())
    return g1, m2, phase


def b1i_prn(prns=range(1, 64)):
    return np.array([_b13.b1i(p) for p in prns], np.int8)


def b1i_all():
    g1, m2, _ = b1i_structure()
    phases = list(range(len(m2))) + [-1]
    codes = gold_codes(g1, m2, phases, B1I_N)
    # label the PRNs: each is one of these phases (shift-and-add)
    key = {c.tobytes(): i for i, c in enumerate(codes)}
    labels = ["g2ph%d" % ph for ph in phases]
    for p, c in enumerate(b1i_prn(), start=1):
        i = key.get(c.tobytes())
        if i is not None:
            labels[i] = "PRN%d/" % p + labels[i]
    return codes, labels


def l3oc_all():
    """DC1 x the 7-stage LFSR from every non-zero seed: seeds 1-63 = L3OCd PRN, 65-127 = L3OCp."""
    dc1 = np.array(_l3._dc1(), np.int8)
    codes, labels = [], []
    for seed in range(1, 128):
        sh = np.array(
            _l3._lfsr(L3OC_N, seed, _l3.REF["short_tap"], _l3.REF["short_n"]), np.int8
        )
        codes.append(dc1 * sh)
        labels.append(
            "L3OCd%d" % seed
            if seed < 64
            else ("L3OCp%d" % (seed - 64) if seed > 64 else "seed64")
        )
    return np.array(codes, np.int8), labels


FAMILIES = {
    # name: (builder -> (codes, labels), chip rate, code length)
    "b3i": (lambda: (b3i_prn(), ["PRN%d" % p for p in range(1, 64)]), B3I_RC, B3I_N),
    "b3i_all": (b3i_all, B3I_RC, B3I_N),
    "b1i": (lambda: (b1i_prn(), ["PRN%d" % p for p in range(1, 64)]), B1I_RC, B1I_N),
    "b1i_all": (b1i_all, B1I_RC, B1I_N),
    "l3oc": (l3oc_all, L3OC_RC, L3OC_N),
}


def family(name):
    """(codes int8 [n, N], labels, chip rate, N)."""
    build, rc, n = FAMILIES[name]
    codes, labels = build()
    return codes, labels, rc, n


# --------------------------------------------------------------------------- GF(2)


def berlekamp_massey(bits):
    """(linear complexity L, connection polynomial as int: bit i = c_i, c_0 = 1) of a 0/1 sequence,
    s[n] = XOR_{i=1..L} c_i s[n-i]."""
    s = np.asarray(bits, np.uint8)
    n = len(s)
    c = np.zeros(n + 1, np.uint8)
    b = np.zeros(n + 1, np.uint8)
    c[0] = b[0] = 1
    L, m = 0, -1
    for i in range(n):
        d = s[i]
        if L:
            d ^= int(np.bitwise_and(c[1 : L + 1], s[i - L : i][::-1]).sum() & 1)
        if d:
            t = c.copy()
            sh = i - m
            c[sh:] ^= b[: n + 1 - sh]
            if 2 * L <= i:
                L, m, b = i + 1 - L, i, t
    return L, sum(1 << i for i in range(L + 1) if c[i])


def taps_to_connection(taps, deg):
    """Connection polynomial (BM's convention) of s[k+deg] = XOR_t s[k+t]: 1 + sum x^(deg-t)."""
    return (1 << 0) | sum(1 << (deg - t) for t in taps)


def gf2_mod(a, b):
    db = b.bit_length() - 1
    while a and a.bit_length() - 1 >= db:
        a ^= b << (a.bit_length() - 1 - db)
    return a


def poly_str(p):
    terms = [
        "x^%d" % i if i > 1 else ("x" if i == 1 else "1")
        for i in range(p.bit_length())
        if p >> i & 1
    ]
    return " + ".join(reversed(terms))


KNOWN_POLYS = {
    "B3I G1": taps_to_connection(B3I_G1_TAPS, B3I_DEG),
    "B3I G2": taps_to_connection(B3I_G2_TAPS, B3I_DEG),
    "B1I G1": taps_to_connection(B1I_G1_TAPS, B1I_DEG),
    "B1I G2": taps_to_connection(B1I_G2_TAPS, B1I_DEG),
    "L3OC DC1": None,  # filled below
    "L3OC short": None,
}


def _l3_conn(tap, n):
    # _lfsr emits bit 0, shifts right, loads parity(r & tap) into bit n-1: s[k+n] = XOR_{t in tap} s[k+t]
    return taps_to_connection([t for t in range(n) if tap >> t & 1], n)


KNOWN_POLYS["L3OC DC1"] = _l3_conn(_l3.REF["dc1_tap"], _l3.REF["dc1_n"])
KNOWN_POLYS["L3OC short"] = _l3_conn(_l3.REF["short_tap"], _l3.REF["short_n"])


def lfsr_report(bits, windows=(0,), length=600):
    """Berlekamp-Massey over windows of a hard-decision chip sequence: linear complexity, the
    connection polynomial and which known polynomials divide it. A memory (random) code gives
    L ~ length/2; an LFSR / Gold code of degree d gives L <= d on any window that does not cross a
    reset or truncation."""
    bits = np.asarray(bits, np.uint8)
    out = []
    for w in windows:
        seg = np.take(bits, np.arange(w, w + length), mode="wrap")
        # either polarity: a complemented LFSR sequence has L + 1 (the extra factor 1 + x)
        L, c = min(berlekamp_massey(seg), berlekamp_massey(seg ^ 1))
        divs = [
            k for k, p in KNOWN_POLYS.items() if L < length // 2 and gf2_mod(c, p) == 0
        ]
        out.append(dict(start=w, L=L, poly=c, divisors=divs))
    return out
