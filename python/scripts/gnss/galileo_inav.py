"""Galileo E1B I/NAV codec: page-part symbols <-> nav bits, and CRC-24.

The first S5 D-component (2026-07-30). Mirrors gps_cnav.py -- and shares its FEC and CRC
by construction: Galileo I/NAV uses the SAME rate-1/2 K=7 convolutional code (G1=171,
G2=133 oct) and the SAME CRC-24Q (0x1864CFB) as GPS CNAV. What differs, and lives here:
the block interleaver (30x8), the 10-symbol synchronisation pattern, and the even/odd
page-part framing.

E1B is the DATA channel (250 sps = 4 ms symbols); the pilot E1C carries no data. A page
part is 1 s = 250 symbols = 10 sync + 240 data; the 240 data symbols are the rate-1/2 FEC
of 120 bits (114 page-part content + 6 convolutional tail), block-interleaved. A NOMINAL
page is 2 s = an EVEN page part then an ODD page part; the CRC-24 spans the two.

★ TWO CONVENTIONS THIS CODE FIXES BUT THE ICD OWNS, both verifiable only against LIVE
symbols (a roundtrip self-test cannot see them -- encode and decode agree on any choice):
G2 inversion (GPS inverts the second FEC output; Galileo does NOT -- set here to False) and
the interleaver read/write orientation. Both are isolated as module constants so a live
decode that fails CRC can flip them without touching the logic. This file's self-test
proves the codec is SELF-CONSISTENT; ICD-correctness is the live-symbol step.

Refs: Galileo OS SIS ICD (I/NAV page layout, FEC, interleaver, sync). gps_cnav.py is the
structural template; cnav_predictor.py will be the streaming wrapper (next).
"""

import math

import numpy as np

SYM_S = 0.004               # E1B symbol = 4 ms (250 sps)
K = 7
POLY = (0o171, 0o133)       # G1, G2 -- identical to GPS CNAV
G2_INVERT = True            # VERIFIED on live E1B symbols 2026-07-30 (a convention scan
                            # decoded 160 CRC-valid words at True+swap, 0 otherwise). The
                            # roundtrip self-test was blind to this -- encode/decode agree on
                            # any choice -- exactly why it was flagged for the live step.
NSTATES = 1 << (K - 1)      # 64
CRC24Q_POLY = 0x1864CFB     # same CRC-24Q as CNAV

# I/NAV page part: 250 symbols = SYNC (10) + interleaved data (240)
SYNC = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]      # Galileo I/NAV synchronisation pattern
N_SYNC = len(SYNC)                          # 10
INTERLEAVE_ROWS = 8
INTERLEAVE_COLS = 30
N_DATA_SYM = INTERLEAVE_ROWS * INTERLEAVE_COLS   # 240
PAGE_PART_SYMS = N_SYNC + N_DATA_SYM             # 250
PAGE_PART_BITS = N_DATA_SYM // 2 - (K - 1)       # 114 content bits (120 FEC bits - 6 tail)


# --------------------------------------------------------------------------
# rate-1/2 K=7 convolutional code (encoder + soft Viterbi) -- as GPS CNAV
# --------------------------------------------------------------------------
def _tables():
    out = np.zeros((NSTATES, 2, 2), dtype=np.int8)
    nxt = np.zeros((NSTATES, 2), dtype=np.int64)
    for s in range(NSTATES):
        for b in (0, 1):
            reg = (b << (K - 1)) | s
            out[s, b, 0] = bin(reg & POLY[0]).count("1") & 1
            out[s, b, 1] = (bin(reg & POLY[1]).count("1") & 1) ^ (1 if G2_INVERT else 0)
            nxt[s, b] = reg >> 1
    return out, nxt


_OUT, _NXT = _tables()
_EXP = np.where(_OUT == 0, 1.0, -1.0)


def conv_encode(bits, flush=True):
    seq = list(int(b) for b in bits) + ([0] * (K - 1) if flush else [])
    s, out = 0, []
    for b in seq:
        out += [int(_OUT[s, b, 0]), int(_OUT[s, b, 1])]
        s = int(_NXT[s, b])
    return np.array(out, dtype=np.int8)


def viterbi_decode(symbols, known_start=True):
    n = len(symbols) // 2
    r = np.asarray(symbols[:2 * n], dtype=float).reshape(n, 2)
    NEG = -1e18
    pm = np.full(NSTATES, NEG if known_start else 0.0)
    if known_start:
        pm[0] = 0.0
    prev = np.zeros((n, NSTATES), dtype=np.int32)
    pbit = np.zeros((n, NSTATES), dtype=np.int8)
    for i in range(n):
        npm = np.full(NSTATES, NEG)
        r0, r1 = r[i, 0], r[i, 1]
        for s in range(NSTATES):
            if pm[s] <= NEG:
                continue
            base = pm[s]
            for b in (0, 1):
                m = base + _EXP[s, b, 0] * r0 + _EXP[s, b, 1] * r1
                ns = int(_NXT[s, b])
                if m > npm[ns]:
                    npm[ns] = m
                    prev[i, ns] = s
                    pbit[i, ns] = b
        pm = npm
    s = int(np.argmax(pm))
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n - 1, -1, -1):
        bits[i] = pbit[i, s]
        s = int(prev[i, s])
    return bits


# --------------------------------------------------------------------------
# block interleaver (30 cols x 8 rows), CRC-24Q -- I/NAV specific
# --------------------------------------------------------------------------
def interleave(sym):
    """Write the 240 encoded symbols into a 30x8 matrix by ROWS, read by COLUMNS.
    ★ Orientation VERIFIED on live E1B symbols 2026-07-30 (the convention scan: this
    orientation + G2_INVERT True decoded 160 CRC-valid words; the other three combinations
    of {G2, orientation} decoded 0). interleave/deinterleave are exact inverses, so the
    roundtrip self-test could not tell this from the transpose -- the live CRC did."""
    m = np.asarray(sym, dtype=np.int8).reshape(INTERLEAVE_COLS, INTERLEAVE_ROWS)
    return m.T.reshape(-1)


def deinterleave(sym):
    m = np.asarray(sym, dtype=float).reshape(INTERLEAVE_ROWS, INTERLEAVE_COLS)
    return m.T.reshape(-1)


def crc24q(bits):
    crc = 0
    for b in bits:
        crc ^= (int(b) & 1) << 23
        crc <<= 1
        if crc & (1 << 24):
            crc ^= CRC24Q_POLY
    return crc & 0xFFFFFF


def _uint(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (int(b) & 1)
    return v


# --------------------------------------------------------------------------
# page-part encode / decode
# --------------------------------------------------------------------------
def encode_page_part(content_bits):
    """114 content bits -> 250 page-part symbols (0/1): FEC, interleave, prepend sync."""
    content = list(int(b) for b in content_bits)
    if len(content) != PAGE_PART_BITS:
        raise ValueError("page part is %d content bits, got %d"
                         % (PAGE_PART_BITS, len(content)))
    fec = conv_encode(content, flush=True)              # 120 -> 240 symbols
    il = interleave(fec)
    return np.concatenate([np.array(SYNC, dtype=np.int8), il])


def decode_page_part(symbols, want_sync=True):
    """250 soft symbols (bipolar: +~0, -~1) -> (content_bits[114], sync_ok).
    Strips the sync, deinterleaves, Viterbi-decodes. Returns None if too short."""
    s = np.asarray(symbols, dtype=float)
    if len(s) < PAGE_PART_SYMS:
        return None, False
    sync_soft = s[:N_SYNC]
    # sync check: the hard-decided sync symbols match SYNC (0->+, 1->-)
    sync_hard = (sync_soft < 0).astype(np.int8)
    sync_ok = bool(np.array_equal(sync_hard, np.array(SYNC, dtype=np.int8)))
    if want_sync and not sync_ok:
        return None, False
    data = deinterleave(s[N_SYNC:N_SYNC + N_DATA_SYM])
    bits = viterbi_decode(data, known_start=True)       # 240 -> 120 bits
    return bits[:PAGE_PART_BITS], sync_ok


def find_page_parts(symbols):
    """Scan a soft-symbol stream for sync-aligned page parts (both polarities).
    Yields (start_index, polarity, content_bits) for each that decodes with valid sync.
    Polarity: +1 as-is, -1 the whole stream inverted (carrier ambiguity)."""
    s = np.asarray(symbols, dtype=float)
    n = len(s)
    i = 0
    while i + PAGE_PART_SYMS <= n:
        for pol in (1.0, -1.0):
            hard = (pol * s[i:i + N_SYNC] < 0).astype(np.int8)
            if np.array_equal(hard, np.array(SYNC, dtype=np.int8)):
                bits, ok = decode_page_part(pol * s[i:i + PAGE_PART_SYMS], want_sync=True)
                if ok and bits is not None:
                    yield i, int(pol), bits
                    break
        i += 1


# --------------------------------------------------------------------------
# nominal-page word assembly: EVEN + ODD page parts -> 128-bit word + CRC-24
# --------------------------------------------------------------------------
# A nominal page (2 s) is an EVEN page part then an ODD one, each 114 content bits:
#   even: [even/odd=0][page_type=0][data 112]                        (2 + 112 = 114)
#   odd : [even/odd=1][page_type=0][data 16][reserved 64][crc 24][ssp 8]  (2+16+64+24+8=114)
# The 128-bit nav word = even.data(112) + odd.data(16); word type = word[0:6].
# CRC-24Q spans the 196 bits even[0:114] + odd[0:82] (everything before the CRC field).
EVEN_DATA = 112
ODD_DATA = 16
WORD_BITS = EVEN_DATA + ODD_DATA        # 128
CRC_SPAN = 196                          # even[0:114] + odd[0:82]
ODD_CRC_AT = 82                         # crc-24 occupies odd[82:106]


def assemble_word(even_bits, odd_bits):
    """(even 114, odd 114) -> (word_128 bits, crc_ok, word_type) or None if the
    even/odd flags are wrong. CRC gates whether the word is trustworthy."""
    e = [int(b) & 1 for b in even_bits]
    o = [int(b) & 1 for b in odd_bits]
    if len(e) != PAGE_PART_BITS or len(o) != PAGE_PART_BITS:
        return None
    if e[0] != 0 or o[0] != 1:           # even part flag 0, odd part flag 1
        return None
    word = e[2:2 + EVEN_DATA] + o[2:2 + ODD_DATA]
    crc_calc = crc24q(e[0:PAGE_PART_BITS] + o[0:ODD_CRC_AT])
    crc_rx = _uint(o[ODD_CRC_AT:ODD_CRC_AT + 24])
    return word, (crc_calc == crc_rx), _uint(word[0:6])


def build_page(word_128, reserved=None):
    """Inverse of assemble_word for the self-test: a 128-bit word -> (even 114, odd 114)
    with a correct CRC. `reserved` fills odd[18:82] (64 bits); default zeros."""
    w = [int(b) & 1 for b in word_128]
    if len(w) != WORD_BITS:
        raise ValueError("word is %d bits" % WORD_BITS)
    res = list(reserved) if reserved is not None else [0] * (ODD_CRC_AT - 2 - ODD_DATA)
    even = [0, 0] + w[0:EVEN_DATA]
    odd_head = [1, 0] + w[EVEN_DATA:WORD_BITS] + res    # odd[0:82]
    crc = crc24q(even + odd_head)
    crc_bits = [(crc >> (23 - i)) & 1 for i in range(24)]
    odd = odd_head + crc_bits + [0] * 8                 # + 8-bit SSP/tail
    return even, odd


# --------------------------------------------------------------------------
# I/NAV ephemeris: Word Types 1-4 -> Keplerian set (SI), then ECEF
# --------------------------------------------------------------------------
GAL_PI = 3.1415926535898
GAL_MU = 3.986004418e14                  # Galileo GM (WGS-close; GPS uses 3.986005e14)
GAL_OMEGA_E = 7.2921151467e-5
_P = lambda n: 2.0 ** (-n)

# name -> (word type, start bit WITHIN the 128-bit word AFTER the 6-bit type field
# [0-indexed], length, signed, scale). ⚠️ ICD-TRANSCRIBED FROM MEMORY OF THE Galileo
# OS SIS ICD I/NAV layout -- the offsets are the ONE thing a roundtrip cannot check;
# the live BRDC position cross-check (a few metres if right, wildly off if a field is
# misplaced) is the real validator. If that fails, correct THIS table, not the math.
# Layout within each word: [type 6][IODnav 10][fields...]. Start bits below are the
# absolute bit index in the 128-bit word.
INAV_EPH_FIELDS = {
    # Word Type 1: IODnav, t0e, M0, e, sqrtA
    "t0e":     (1, 16, 14, False, 60.0),
    "M0":      (1, 30, 32, True,  _P(31) * GAL_PI),
    "e":       (1, 62, 32, False, _P(33)),
    "sqrtA":   (1, 94, 32, False, _P(19)),
    # Word Type 2: Omega0, i0, omega, iDot
    "OMEGA0":  (2, 16, 32, True,  _P(31) * GAL_PI),
    "i0":      (2, 48, 32, True,  _P(31) * GAL_PI),
    "omega":   (2, 80, 32, True,  _P(31) * GAL_PI),
    "iDot":    (2, 112, 14, True, _P(43) * GAL_PI),
    # Word Type 3: OmegaDot, deltaN, Cuc, Cus, Crc, Crs (SISA ignored here)
    "OMEGA_dot": (3, 16, 24, True, _P(43) * GAL_PI),
    "dn":        (3, 40, 16, True, _P(43) * GAL_PI),
    "Cuc":       (3, 56, 16, True, _P(29)),
    "Cus":       (3, 72, 16, True, _P(29)),
    "Crc":       (3, 88, 16, True, _P(5)),
    "Crs":       (3, 104, 16, True, _P(5)),
    # Word Type 4: Cic, Cis, t0c, af0, af1, af2 (SVID at 16..22 ignored here)
    "Cic":     (4, 22, 16, True, _P(29)),
    "Cis":     (4, 38, 16, True, _P(29)),
    "t0c":     (4, 54, 14, False, 60.0),
    "af0":     (4, 68, 31, True, _P(34)),
    "af1":     (4, 99, 21, True, _P(46)),
    "af2":     (4, 120, 6, True, _P(59)),
}


def _field(word, start, length, signed, scale):
    bits = word[start:start + length]
    v = _uint(bits)
    if signed and bits and bits[0] == 1:
        v -= (1 << length)
    return v * scale


def parse_inav_ephemeris(words_by_type):
    """words_by_type: {word_type: 128-bit word}. Needs types 1-4. Returns the SI
    Keplerian set, or None if a needed word is missing."""
    if not all(t in words_by_type for t in (1, 2, 3, 4)):
        return None
    eph = {}
    for name, (wt, start, length, signed, scale) in INAV_EPH_FIELDS.items():
        eph[name] = _field(words_by_type[wt], start, length, signed, scale)
    # IODnav must agree across the four words (they belong to one ephemeris set)
    iod = [_uint(words_by_type[t][6:16]) for t in (1, 2, 3, 4)]
    eph["IODnav"] = iod[0]
    eph["_iod_consistent"] = len(set(iod)) == 1
    return eph


def sv_position_inav(eph, t):
    """ECEF (x,y,z) m from I/NAV ephemeris. Standard Keplerian propagation (identical
    math to GPS LNAV/CNAV; Galileo transmits sqrtA directly). t in Galileo system time
    of week (seconds)."""
    A = eph["sqrtA"] ** 2
    tk = t - eph["t0e"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    n0 = math.sqrt(GAL_MU / A ** 3)
    n = n0 + eph["dn"]
    M = eph["M0"] + n * tk
    e = eph["e"]
    E = M
    for _ in range(20):
        E = M + e * math.sin(E)
    v = math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)
    phi = v + eph["omega"]
    s2, c2 = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + eph["Cus"] * s2 + eph["Cuc"] * c2
    r = A * (1 - e * math.cos(E)) + eph["Crs"] * s2 + eph["Crc"] * c2
    i = eph["i0"] + eph["iDot"] * tk + eph["Cis"] * s2 + eph["Cic"] * c2
    om = eph["OMEGA0"] + (eph["OMEGA_dot"] - GAL_OMEGA_E) * tk - GAL_OMEGA_E * eph["t0e"]
    xp, yp = r * math.cos(u), r * math.sin(u)
    x = xp * math.cos(om) - yp * math.cos(i) * math.sin(om)
    y = xp * math.sin(om) + yp * math.cos(i) * math.cos(om)
    z = yp * math.sin(i)
    return x, y, z


# --------------------------------------------------------------------------
# self-test: the codec is SELF-CONSISTENT (ICD-correctness is the live step)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    fails = 0

    # 1a. FEC+interleaver roundtrip, CLEAN: content -> symbols -> content, exact.
    # (Tests the decode; sync robustness is 1c. A hard 10/10 sync check under AWGN
    # rejects ~2% of pages -- that is framesync sensitivity, NOT a decode error, so it
    # must not be conflated with codec correctness the way the first cut did.)
    f1a = 0
    for _ in range(200):
        content = rng.integers(0, 2, PAGE_PART_BITS).astype(np.int8)
        soft = np.where(encode_page_part(content) == 0, 1.0, -1.0)
        out, _ = decode_page_part(soft, want_sync=False)
        if out is None or not np.array_equal(out, content):
            f1a += 1
    print("1a. FEC roundtrip, clean: %s" % ("OK" if f1a == 0 else "FAIL %d/200" % f1a))

    # 1b. FEC decode under AWGN (the code's coding gain), sync gate off
    f1b = 0
    for _ in range(200):
        content = rng.integers(0, 2, PAGE_PART_BITS).astype(np.int8)
        soft = np.where(encode_page_part(content) == 0, 1.0, -1.0) \
            + rng.normal(0, 0.35, PAGE_PART_SYMS)
        out, _ = decode_page_part(soft, want_sync=False)
        if out is None or not np.array_equal(out, content):
            f1b += 1
    print("1b. FEC roundtrip @0.35 AWGN: %s" % ("OK" if f1b == 0 else "FAIL %d/200" % f1b))
    fails = f1a + f1b

    # 2. sync + polarity search finds an embedded page part, either polarity
    p2 = 0
    for pol in (1.0, -1.0):
        content = rng.integers(0, 2, PAGE_PART_BITS).astype(np.int8)
        sym01 = encode_page_part(content)
        soft = pol * np.where(sym01 == 0, 1.0, -1.0)
        stream = np.concatenate([rng.normal(0, 1, 37), soft, rng.normal(0, 1, 29)])
        found = list(find_page_parts(stream))
        hit = any(np.array_equal(b, content) and i == 37 for i, _, b in found)
        p2 += 0 if hit else 1
    print("2. sync+polarity search: %s" % ("OK" if p2 == 0 else "FAIL %d/2" % p2))

    # 3. CRC-24Q detects a single-bit flip (over a 196-bit nav word, I/NAV CRC scope)
    word = list(rng.integers(0, 2, 196).astype(int))
    c = crc24q(word)
    bad = word[:]; bad[100] ^= 1
    print("3. CRC-24Q flip-detect: %s" % ("OK" if crc24q(bad) != c else "FAIL"))

    # 4. structural invariants
    ok4 = (PAGE_PART_SYMS == 250 and N_DATA_SYM == 240 and PAGE_PART_BITS == 114
           and WORD_BITS == 128 and CRC_SPAN == 196)
    print("4. page geometry (250 sym = 10+240; 128-bit word; 196-bit CRC span): %s"
          % ("OK" if ok4 else "FAIL"))

    # 5. word assembly roundtrip: word -> even/odd page parts -> word, CRC valid,
    #    and a corrupted odd bit fails the CRC.
    p5 = 0
    for _ in range(100):
        w = list(rng.integers(0, 2, WORD_BITS).astype(int))
        even, odd = build_page(w)
        res = assemble_word(even, odd)
        if res is None or not res[1] or res[0] != w or res[2] != _uint(w[0:6]):
            p5 += 1
    # corruption must be caught
    w = list(rng.integers(0, 2, WORD_BITS).astype(int)); even, odd = build_page(w)
    odd[20] ^= 1
    r = assemble_word(even, odd)
    if r is None or r[1]:      # CRC should now be invalid
        p5 += 1
    print("5. word assembly + CRC roundtrip (even/odd, corruption caught): %s"
          % ("OK" if p5 == 0 else "FAIL %d" % p5))

    # 6. ephemeris field pack/unpack is SELF-CONSISTENT: place known integer codes in
    #    each word by the field table, parse, recover the scaled values. (This checks
    #    the table is internally consistent -- NOT that the offsets match the ICD; only
    #    a live BRDC position check can confirm that.)
    words = {}
    truth = {}
    for name, (wt, start, length, signed, scale) in INAV_EPH_FIELDS.items():
        words.setdefault(wt, [0] * WORD_BITS)
    for wt in (1, 2, 3, 4):
        words[wt][0:6] = [(wt >> (5 - i)) & 1 for i in range(6)]   # word type
        words[wt][6:16] = [0] * 10                                  # IODnav = 0 (consistent)
    for name, (wt, start, length, signed, scale) in INAV_EPH_FIELDS.items():
        code = 5 if not (start + length <= WORD_BITS) else (1 << (length - 1)) - 1  # a value
        # place a distinctive code, guard overlaps by using a small unique per field
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            words[wt][start + k] = (code >> (length - 1 - k)) & 1
        truth[name] = _field(words[wt], start, length, signed, scale)
    eph = parse_inav_ephemeris(words)
    p6 = 0 if (eph is not None and eph.get("_iod_consistent")
               and all(abs(eph[n] - truth[n]) < 1e-12 * (abs(truth[n]) + 1) for n in truth)) else 1
    # overlap check: no two fields in the same word may share a bit
    for wt in (1, 2, 3, 4):
        occ = [0] * WORD_BITS
        for name, (w2, start, length, s, sc) in INAV_EPH_FIELDS.items():
            if w2 != wt:
                continue
            for k in range(start, start + length):
                occ[k] += 1
        if any(x > 1 for x in occ) or any(start + length > WORD_BITS
                                          for n, (w2, start, length, s, sc)
                                          in INAV_EPH_FIELDS.items() if w2 == wt):
            p6 += 1
    print("6. ephemeris field table self-consistent (pack/unpack, no overlaps/overruns): %s"
          % ("OK" if p6 == 0 else "FAIL %d" % p6))

    bad_any = fails or p2 or ok4 is False or crc24q(bad) == c or p5 or p6
    print("\n%s" % ("ALL SELF-CONSISTENT (ICD-correctness pends live E1B symbols)"
                    if not bad_any else "SELF-TEST FAILURES"))
    sys.exit(1 if bad_any else 0)
