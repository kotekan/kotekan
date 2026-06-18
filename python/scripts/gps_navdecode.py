#!/usr/bin/env python3
"""GPS LNAV navigation-message decoder for GpsReplicaCorrelator records.

Reads the per-PRN soft prompt symbols out of the correlator's gps_*.raw record
stream (peak_re/peak_im = the carrier-aligned complex prompt, one per 1 ms
integration with track_phase on) and runs the GPS L1 C/A legacy nav-message
(LNAV) demod chain:

    soft symbols -> carrier de-rotation -> bit sync (20 ms) -> frame sync
    (TLM preamble + parity) -> word parity check -> field parse (TOW, subframe
    id, week number, ...).

This is the "physical-layer demod" half of the split design: the correlator
does the DSP and emits symbols; this decodes them. It is content-agnostic --
it matches on the encoding (the LNAV parity/framing), not the meaning -- so the
unknown nav bits are recovered by sync + parity, never assumed.

We also expose the LNAV *encoder* (parity, framing, trimming) so a synthetic
known message can be generated and round-tripped in tests without live sky data.

Extending to ephemeris/clock is mechanical (more bit-field entries); extending
to other constellations means a new framing/parser behind the same symbol-in
interface.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gps_beamtrack import read_records  # noqa: E402  (shared record reader)

PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]   # LNAV TLM preamble (as transmitted)
WORD_BITS = 30
DATA_BITS = 24                         # source bits per 30-bit word (rest = parity)
WORDS_PER_SUBFRAME = 10
SUBFRAME_BITS = WORD_BITS * WORDS_PER_SUBFRAME   # 300
SYMS_PER_BIT = 20                      # 50 bps nav data over 1 ms code periods

# IS-GPS-200 parity: source-bit index sets (1-indexed into d1..d24) for D25..D30,
# and which previous-word bit (D29* or D30*) each parity bit XORs in.
_PARITY_SETS = [
    [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23],     # D25
    [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24],     # D26
    [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22],      # D27
    [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23],      # D28
    [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24],  # D29
    [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24],         # D30
]
_PARITY_PREV = [29, 30, 29, 30, 30, 29]  # D25<-D29*, D26<-D30*, ... D30<-D29*


# --------------------------------------------------------------------------
# LNAV parity / word coding (encoder + decoder share this)
# --------------------------------------------------------------------------
def compute_parity(d, d29s, d30s):
    """6 parity bits (D25..D30) from the 24 source data bits d (0-indexed) and
    the previous word's last two transmitted bits d29s, d30s."""
    prev = {29: d29s, 30: d30s}
    out = []
    for k in range(6):
        x = prev[_PARITY_PREV[k]]
        for idx in _PARITY_SETS[k]:
            x ^= int(d[idx - 1])
        out.append(x)
    return out


def encode_word(source24, d29s, d30s):
    """Build a 30-bit transmitted word: D[1..24] = source XOR D30*, then parity.
    Returns (word30, new_d29s, new_d30s)."""
    tx = [int(source24[i]) ^ d30s for i in range(DATA_BITS)]
    par = compute_parity(source24, d29s, d30s)
    word = tx + par
    return word, par[4], par[5]


def check_word(word30, d29s, d30s):
    """Parity-check a received 30-bit word. Returns (ok, recovered_source24,
    new_d29s, new_d30s). Recovery d = D XOR D30* is transparent to a global bit
    inversion (the LNAV polarity ambiguity), so the source comes out right
    regardless of stream polarity."""
    tx = [int(b) for b in word30[:DATA_BITS]]
    d = [tx[i] ^ d30s for i in range(DATA_BITS)]
    par = compute_parity(d, d29s, d30s)
    ok = par == [int(b) for b in word30[DATA_BITS:WORD_BITS]]
    return ok, d, int(word30[28]), int(word30[29])


def _trim_word(source24, d29s, d30s):
    """Set bits 23-24 (the two non-information bits of words 2 and 10) so the
    word's D29=D30=0, leaving the next word's previous-bits clean."""
    s = list(source24)
    for d24v in (0, 1):
        for d23v in (0, 1):
            s[22], s[23] = d23v, d24v
            par = compute_parity(s, d29s, d30s)
            if par[4] == 0 and par[5] == 0:
                return s
    raise RuntimeError("trim solve failed")  # unreachable for valid inputs


def encode_subframe(sfid, tow_count, words_data, week=None):
    """Encode one 300-bit subframe (list of ints, transmitted order).

    words_data: optional list of 10 lists of 24 data bits for words 1..10; the
    TLM preamble, HOW (tow/sfid), and trims are overwritten here. week (10 bits)
    is written to SF1 word 3 bits 1-10 when sfid==1.
    """
    words = [list(w) if w is not None else [0] * DATA_BITS
             for w in (words_data or [None] * WORDS_PER_SUBFRAME)]
    words[0][0:8] = PREAMBLE                       # TLM preamble
    how = [0] * DATA_BITS
    for i in range(17):                            # TOW count, 17 bits MSB-first
        how[i] = (tow_count >> (16 - i)) & 1
    for i in range(3):                             # subframe id, bits 20-22
        how[19 + i] = (sfid >> (2 - i)) & 1
    words[1] = how
    if sfid == 1 and week is not None:
        for i in range(10):                        # WN, word 3 bits 1-10
            words[2][i] = (week >> (9 - i)) & 1

    bits, d29s, d30s = [], 0, 0                     # subframe starts clean
    for w in range(WORDS_PER_SUBFRAME):
        src = words[w]
        if w in (1, 9):                            # HOW and word 10 are trimmed
            src = _trim_word(src, d29s, d30s)
        word, d29s, d30s = encode_word(src, d29s, d30s)
        bits.extend(word)
    return bits


# --------------------------------------------------------------------------
# symbols -> bits
# --------------------------------------------------------------------------
def derotate(symbols):
    """Remove the BPSK carrier (mod pi) by squaring: angle(mean(s^2))/2 is the
    carrier phase regardless of the nav-bit sign. Returns the real projection
    A*D (+ noise), with the +-1 nav data on the sign."""
    s = np.asarray(symbols, dtype=complex)
    phi = 0.5 * np.angle(np.mean(s * s)) if np.any(s) else 0.0
    return np.real(s * np.exp(-1j * phi))


def bit_sync(real_syms):
    """Find the 0..19 symbol offset of the 20 ms bit boundaries: the phase whose
    per-bit coherent sums are largest (a wrong phase straddles transitions and
    cancels)."""
    best_ph, best = 0, -1.0
    for ph in range(SYMS_PER_BIT):
        n = (len(real_syms) - ph) // SYMS_PER_BIT
        blk = real_syms[ph:ph + n * SYMS_PER_BIT].reshape(n, SYMS_PER_BIT)
        score = np.sum(np.abs(blk.sum(axis=1)))
        if score > best:
            best, best_ph = score, ph
    return best_ph


def symbols_to_bits(symbols):
    """Soft symbols -> hard bits (0/1) at 50 bps, plus the bit-sync offset.
    Global polarity is left unresolved here; LNAV parity recovery is transparent
    to it."""
    real = derotate(symbols)
    ph = bit_sync(real)
    n = (len(real) - ph) // SYMS_PER_BIT
    blk = real[ph:ph + n * SYMS_PER_BIT].reshape(n, SYMS_PER_BIT)
    sums = blk.sum(axis=1)
    bits = (sums < 0).astype(int)   # arbitrary sign convention; polarity-blind
    return bits, ph


# --------------------------------------------------------------------------
# frame sync + parse
# --------------------------------------------------------------------------
def _matches_preamble(bits8):
    b = [int(x) for x in bits8]
    inv = [1 - x for x in PREAMBLE]
    return b == PREAMBLE or b == inv


def _decode_subframe_at(bits, i):
    """Try to decode a 300-bit subframe at index i. Returns a dict or None."""
    if i + SUBFRAME_BITS > len(bits):
        return None
    window = [int(b) for b in bits[i:i + SUBFRAME_BITS]]
    if window[0:8] == [1 - x for x in PREAMBLE]:
        window = [1 - x for x in window]   # de-invert: resolve LNAV polarity so
        #                                    the subframe again starts D29*=D30*=0
    elif window[0:8] != PREAMBLE:
        return None
    d29s, d30s, words = 0, 0, []
    for w in range(WORDS_PER_SUBFRAME):
        word = window[w * WORD_BITS:(w + 1) * WORD_BITS]
        ok, d, d29s, d30s = check_word(word, d29s, d30s)
        if not ok:
            return None
        words.append(d)
    if words[0][0:8] != PREAMBLE:        # recovered TLM preamble must be clean
        return None
    tow_count = _bits_to_uint(words[1][0:17])
    sfid = _bits_to_uint(words[1][19:22])
    if not 1 <= sfid <= 5:
        return None
    out = {"index": i, "sfid": sfid, "tow_count": tow_count,
           "tow_s": tow_count * 6, "words": words}
    if sfid == 1:
        out["week"] = _bits_to_uint(words[2][0:10])
    return out


def find_subframes(bits):
    """All parity-valid subframes, chained at 300-bit spacing once locked."""
    subs, i, n = [], 0, len(bits)
    while i + SUBFRAME_BITS <= n:
        if _matches_preamble(bits[i:i + 8]):
            sf = _decode_subframe_at(bits, i)
            if sf is not None:
                subs.append(sf)
                i += SUBFRAME_BITS
                continue
        i += 1
    return subs


def _bits_to_uint(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def decode_symbols(symbols):
    """Full chain: complex prompt symbols -> list of decoded subframes."""
    bits, _ph = symbols_to_bits(symbols)
    return find_subframes(bits)


def read_symbols(paths, prn, n_prn):
    """Time-ordered complex prompt symbols for one PRN from record files."""
    recs = read_records(paths, n_prn=n_prn)
    sel = recs[recs["prn"].astype(int) == int(prn)]
    return sel["peak_re"].astype(float) + 1j * sel["peak_im"].astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="record file, directory, or glob")
    ap.add_argument("--n-prn", type=int, required=True, help="PRNs per record frame")
    ap.add_argument("--prn", type=int, action="append", required=True,
                    help="PRN(s) to decode (repeatable)")
    args = ap.parse_args(argv)

    paths = (sorted(glob.glob(os.path.join(args.path, "*.raw")))
             if os.path.isdir(args.path) else sorted(glob.glob(args.path)))
    if not paths:
        ap.error("no record files matched: %s" % args.path)

    for prn in args.prn:
        syms = read_symbols(paths, prn, args.n_prn)
        subs = decode_symbols(syms)
        print("PRN %2d: %d symbols, %d valid subframes" % (prn, len(syms), len(subs)))
        for sf in subs:
            extra = "  week=%d" % sf["week"] if "week" in sf else ""
            print("   sfid=%d  TOW=%ds%s" % (sf["sfid"], sf["tow_s"], extra))


if __name__ == "__main__":
    main()
