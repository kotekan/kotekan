"""Galileo E5a-I F/NAV codec: page symbols <-> nav bits, and the ephemeris parse.

The second S5 D-component (2026-07-30). Same FEC FAMILY as E1B I/NAV, so it REUSES
galileo_inav's Viterbi (rate-1/2 K=7, G=171/133, G2 inverted -- proven on live E1B) and
CRC-24Q directly. What differs, and lives here:

  * symbol rate 50 sps (20 ms symbols) vs I/NAV's 250 sps (4 ms);
  * a page is 10 s = 500 symbols = 12-symbol SYNC + 488 data, and the 488 data symbols are
    the rate-1/2 FEC of 244 bits (238 content + 6 tail), block-interleaved 61x8;
  * NO even/odd split -- an F/NAV page is a SINGLE 500-symbol unit (unlike I/NAV, where a
    word spans an even + an odd page part); the CRC-24 spans the page's content directly;
  * F/NAV page TYPES 1-4 carry the ephemeris/clock, on a different bit layout from I/NAV.

★ ICD-OWNED CONVENTIONS, isolated as flippable constants and VERIFIED ONLY ON LIVE SYMBOLS
(the E1B lesson: a roundtrip self-test is blind to them -- encode and decode agree on any
choice, and the live CRC is the only judge): the sync pattern, the interleaver orientation,
and the ephemeris field offsets. G2 inversion is inherited from galileo_inav (True, already
live-verified for the shared FEC). If the first real E5a decode finds PAGES (sync) but zero
CRC-valid content, scan these before touching logic.

⚠️ BLOCKED on live validation until the E5a-I spreading-code table exists (galileoE5aCode has
only the E5a-Q pilot; E5a-I needs its own per-PRN LFSR start values from the ICD). This file
is the offline-testable half.

Refs: Galileo OS SIS ICD (F/NAV page layout, FEC, interleaver, sync). gps_cnav.py /
galileo_inav.py are the structural templates.
"""

import math

import numpy as np

# reuse the SHARED FEC + CRC from the I/NAV codec (same convolutional code + CRC-24Q)
from galileo_inav import (conv_encode, viterbi_decode, crc24q, _uint, _field,
                          GAL_PI, GAL_MU, GAL_OMEGA_E, _P)

SYM_S = 0.020               # E5a-I symbol = 20 ms (50 sps)

# F/NAV page: 500 symbols = SYNC (12) + interleaved data (488)
SYNC = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]   # F/NAV sync pattern (ICD; verify on live)
N_SYNC = len(SYNC)                             # 12
INTERLEAVE_ROWS = 8
INTERLEAVE_COLS = 61
N_DATA_SYM = INTERLEAVE_ROWS * INTERLEAVE_COLS # 488
PAGE_SYMS = N_SYNC + N_DATA_SYM                # 500
PAGE_BITS = N_DATA_SYM // 2 - 6                # 238 content bits (244 FEC bits - 6 tail)
CRC_SPAN = PAGE_BITS - 24                      # 214: content before the 24-bit CRC
CRC_AT = CRC_SPAN                               # CRC occupies bits [214:238]


def interleave(sym):
    """Write 488 encoded symbols into a 61x8 matrix by ROWS, read by COLUMNS. Orientation
    ICD-owned -> verify on live symbols (interleave/deinterleave are exact inverses so the
    self-test is blind to it, exactly as for I/NAV)."""
    m = np.asarray(sym, dtype=np.int8).reshape(INTERLEAVE_COLS, INTERLEAVE_ROWS)
    return m.T.reshape(-1)


def deinterleave(sym):
    m = np.asarray(sym, dtype=float).reshape(INTERLEAVE_ROWS, INTERLEAVE_COLS)
    return m.T.reshape(-1)


def encode_page(content_bits):
    """238 content bits -> 500 page symbols (0/1): FEC, interleave, prepend sync."""
    content = list(int(b) for b in content_bits)
    if len(content) != PAGE_BITS:
        raise ValueError("F/NAV page is %d content bits, got %d" % (PAGE_BITS, len(content)))
    fec = conv_encode(content, flush=True)          # 244 -> 488 symbols
    il = interleave(fec)
    return np.concatenate([np.array(SYNC, dtype=np.int8), il])


def decode_page(symbols, want_sync=True):
    """500 soft symbols (bipolar +~0 -~1) -> (content_bits[238], sync_ok). Strips sync,
    deinterleaves, Viterbi-decodes."""
    s = np.asarray(symbols, dtype=float)
    if len(s) < PAGE_SYMS:
        return None, False
    sync_hard = (s[:N_SYNC] < 0).astype(np.int8)
    sync_ok = bool(np.array_equal(sync_hard, np.array(SYNC, dtype=np.int8)))
    if want_sync and not sync_ok:
        return None, False
    data = deinterleave(s[N_SYNC:N_SYNC + N_DATA_SYM])
    bits = viterbi_decode(data, known_start=True)   # 488 -> 244 bits
    return [int(b) for b in bits[:PAGE_BITS]], sync_ok   # list: composes with _field (I/NAV)


def check_page(content_bits):
    """(content 238) -> (crc_ok, page_type). CRC-24 over the first 214 content bits."""
    c = [int(b) & 1 for b in content_bits]
    if len(c) != PAGE_BITS:
        return False, None
    crc_calc = crc24q(c[0:CRC_AT])
    crc_rx = _uint(c[CRC_AT:CRC_AT + 24])
    return (crc_calc == crc_rx), _uint(c[0:6])


def build_page(page_type, payload=None):
    """Inverse of decode/check for the self-test: page_type + 208-bit payload -> 238 content
    bits with a valid CRC. payload defaults to zeros."""
    c = [0] * PAGE_BITS
    c[0:6] = [(int(page_type) >> (5 - i)) & 1 for i in range(6)]
    if payload is not None:
        pl = list(int(b) & 1 for b in payload)
        c[6:6 + len(pl)] = pl[:CRC_AT - 6]
    crc = crc24q(c[0:CRC_AT])
    c[CRC_AT:CRC_AT + 24] = [(crc >> (23 - i)) & 1 for i in range(24)]
    return c


def find_pages(symbols):
    """Scan a soft stream for sync-aligned pages (both polarities). Yields
    (start_index, polarity, content_bits) for each with valid sync."""
    s = np.asarray(symbols, dtype=float)
    n = len(s)
    i = 0
    while i + PAGE_SYMS <= n:
        for pol in (1.0, -1.0):
            hard = (pol * s[i:i + N_SYNC] < 0).astype(np.int8)
            if np.array_equal(hard, np.array(SYNC, dtype=np.int8)):
                bits, ok = decode_page(pol * s[i:i + PAGE_SYMS], want_sync=True)
                if ok and bits is not None:
                    yield i, int(pol), bits
                    break
        i += 1


# --------------------------------------------------------------------------
# F/NAV ephemeris: page types 1-4 -> Keplerian set (SI), then ECEF
# --------------------------------------------------------------------------
# name -> (page type, start bit WITHIN the 238-bit content [0-indexed, the 6-bit page-type
# field is bits 0..5], length, signed, scale). Offsets/pages are the ONE thing a roundtrip
# cannot check (the E1B field-offset lesson) -- LIVE-VERIFIED against the gnss-sdr Galileo_FNAV.h
# tables and confirmed by the BRDC dpos cross-check 2026-07-30. gnss-sdr uses 1-indexed offsets
# from the same page frame, so my_start = gnss_offset - 1 (checked on M0: {17,32} -> start 16 =
# type(6)+IODnav(10)). Scales are IDENTICAL to the I/NAV table (both Galileo Kepler); only the
# page/bit LAYOUT differs -- and it differs a LOT from a naive guess: t0e is on PAGE 3 (not 4),
# OMEGA0 + iDot are on PAGE 2 (not 3), and page 4 carries only Cic/Cis (the rest is GST-UTC).
FNAV_EPH_FIELDS = {
    # Page 1 (clock): IODnav@6, t0c@22, af0@36, af1@67, af2@88
    "t0c":   (1, 22, 14, False, 60.0),
    "af0":   (1, 36, 31, True, _P(34)),
    "af1":   (1, 67, 21, True, _P(46)),
    "af2":   (1, 88, 6, True, _P(59)),
    # Page 2 (orbit A): M0@16, OMEGA_dot@48, e@72, sqrtA@104, OMEGA0@136, iDot@168
    "M0":    (2, 16, 32, True, _P(31) * GAL_PI),
    "OMEGA_dot": (2, 48, 24, True, _P(43) * GAL_PI),
    "e":     (2, 72, 32, False, _P(33)),
    "sqrtA": (2, 104, 32, False, _P(19)),
    "OMEGA0": (2, 136, 32, True, _P(31) * GAL_PI),
    "iDot":  (2, 168, 14, True, _P(43) * GAL_PI),
    # Page 3 (orbit B): i0@16, omega@48, dn@80, Cuc@96, Cus@112, Crc@128, Crs@144, t0e@160
    "i0":    (3, 16, 32, True, _P(31) * GAL_PI),
    "omega": (3, 48, 32, True, _P(31) * GAL_PI),
    "dn":    (3, 80, 16, True, _P(43) * GAL_PI),
    "Cuc":   (3, 96, 16, True, _P(29)),
    "Cus":   (3, 112, 16, True, _P(29)),
    "Crc":   (3, 128, 16, True, _P(5)),
    "Crs":   (3, 144, 16, True, _P(5)),
    "t0e":   (3, 160, 14, False, 60.0),
    # Page 4 (harmonics; rest of page is GST-UTC/GGTO): Cic@16, Cis@32
    "Cic":   (4, 16, 16, True, _P(29)),
    "Cis":   (4, 32, 16, True, _P(29)),
}


def parse_fnav_ephemeris(pages_by_type):
    """pages_by_type: {page_type: 238-bit content}. The ORBIT lives on types 2,3,4 (required
    -- enough for the BRDC position cross-check); type 1 (clock t0c/af*) is folded in when
    present but not required. Returns the SI Keplerian set, or None if an orbit page is
    missing / the IODnav across 2,3,4 disagrees."""
    if not all(t in pages_by_type for t in (2, 3, 4)):
        return None
    iod = [_uint(pages_by_type[t][6:16]) for t in (2, 3, 4)]   # IODnav in the eph pages
    if len(set(iod)) != 1:
        return None
    eph = {}
    for name, (pt, start, length, signed, scale) in FNAV_EPH_FIELDS.items():
        if pt not in pages_by_type:
            continue                                            # clock field, page 1 not in yet
        eph[name] = _field(pages_by_type[pt], start, length, signed, scale)
    eph["IODnav"] = iod[0]
    eph["_iod_consistent"] = True
    return eph


def sv_position_fnav(eph, t):
    """ECEF (x,y,z) m -- identical Kepler math to I/NAV (sv_position_inav). Galileo transmits
    sqrtA directly; F/NAV and I/NAV describe the same orbit, only the framing differs."""
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
# self-test: the codec is SELF-CONSISTENT (ICD-correctness pends live E5a symbols)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    fails = 0

    # 1. page FEC roundtrip clean + under AWGN (sync gate off -> tests decode, not framesync)
    for noise in (0.0, 0.35):
        f = 0
        for _ in range(150):
            content = rng.integers(0, 2, PAGE_BITS).astype(np.int8)
            soft = np.where(encode_page(content) == 0, 1.0, -1.0)
            if noise:
                soft = soft + rng.normal(0, noise, PAGE_SYMS)
            out, _ = decode_page(soft, want_sync=False)
            if out is None or not np.array_equal(out, content):
                f += 1
        print("1. FEC roundtrip @%.2f AWGN: %s" % (noise, "OK" if f == 0 else "FAIL %d/150" % f))
        fails += f

    # 2. CRC page check roundtrip + corruption caught
    p2 = 0
    for _ in range(100):
        c = build_page(rng.integers(1, 6), rng.integers(0, 2, 208))
        ok, pt = check_page(c)
        if not ok:
            p2 += 1
    c = build_page(2, rng.integers(0, 2, 208)); c[50] ^= 1
    if check_page(c)[0]:
        p2 += 1
    print("2. CRC page check + corruption caught: %s" % ("OK" if p2 == 0 else "FAIL %d" % p2))

    # 3. sync + polarity search finds an embedded page, either polarity
    p3 = 0
    for pol in (1.0, -1.0):
        content = build_page(3, rng.integers(0, 2, 208))
        soft = pol * np.where(encode_page(content) == 0, 1.0, -1.0)
        stream = np.concatenate([rng.normal(0, 1, 41), soft, rng.normal(0, 1, 33)])
        found = [(i, b) for i, _, b in find_pages(stream)]
        if not any(np.array_equal(b, content) and i == 41 for i, b in found):
            p3 += 1
    print("3. sync+polarity search: %s" % ("OK" if p3 == 0 else "FAIL %d/2" % p3))

    # 4. geometry + ephemeris field-table self-consistency (no overlaps/overruns, pack/unpack)
    ok4 = (PAGE_SYMS == 500 and N_DATA_SYM == 488 and PAGE_BITS == 238 and CRC_SPAN == 214)
    pages = {}
    for pt in (1, 2, 3, 4):
        pages[pt] = [0] * PAGE_BITS
        pages[pt][0:6] = [(pt >> (5 - i)) & 1 for i in range(6)]
        pages[pt][6:16] = [0] * 10          # IODnav = 0
    truth = {}
    for name, (pt, start, length, signed, scale) in FNAV_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            pages[pt][start + k] = (code >> (length - 1 - k)) & 1
        truth[name] = _field(pages[pt], start, length, signed, scale)
    eph = parse_fnav_ephemeris(pages)
    p4 = 0 if (ok4 and eph is not None
               and all(abs(eph[n] - truth[n]) < 1e-9 * (abs(truth[n]) + 1) for n in truth)) else 1
    for pt in (1, 2, 3, 4):
        occ = [0] * PAGE_BITS
        for n, (p, st, ln, s_, sc) in FNAV_EPH_FIELDS.items():
            if p != pt:
                continue
            if st + ln > PAGE_BITS:
                p4 += 1
            for k in range(st, min(st + ln, PAGE_BITS)):
                occ[k] += 1
        if any(x > 1 for x in occ):
            p4 += 1
    print("4. geometry + ephemeris field table self-consistent: %s" % ("OK" if p4 == 0 else "FAIL %d" % p4))

    bad = fails or p2 or p3 or p4
    print("\n%s" % ("ALL SELF-CONSISTENT (ICD-correctness pends live E5a symbols + the code table)"
                    if not bad else "SELF-TEST FAILURES"))
    sys.exit(1 if bad else 0)
