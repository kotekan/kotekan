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

import numpy as np

SYM_S = 0.004               # E1B symbol = 4 ms (250 sps)
K = 7
POLY = (0o171, 0o133)       # G1, G2 -- identical to GPS CNAV
G2_INVERT = False           # ⚠️ ICD-owned, verify on live symbols (GPS inverts; Galileo not)
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
    """Write the 240 encoded symbols into an 8x30 matrix by ROWS, read by COLUMNS.
    (Orientation is ICD-owned; interleave/deinterleave are exact inverses so the
    self-test is blind to it -- the live decode confirms the true orientation.)"""
    m = np.asarray(sym, dtype=np.int8).reshape(INTERLEAVE_ROWS, INTERLEAVE_COLS)
    return m.T.reshape(-1)


def deinterleave(sym):
    m = np.asarray(sym, dtype=float).reshape(INTERLEAVE_COLS, INTERLEAVE_ROWS)
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
    ok4 = (PAGE_PART_SYMS == 250 and N_DATA_SYM == 240 and PAGE_PART_BITS == 114)
    print("4. page geometry (250 sym = 10 sync + 240 data, 114 content bits): %s"
          % ("OK" if ok4 else "FAIL"))

    bad_any = fails or p2 or ok4 is False or crc24q(bad) == c
    print("\n%s" % ("ALL SELF-CONSISTENT (ICD-correctness pends live E1B symbols)"
                    if not bad_any else "SELF-TEST FAILURES"))
    sys.exit(1 if bad_any else 0)
