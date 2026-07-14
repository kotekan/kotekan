#!/usr/bin/env python3
"""GPS L1 C/A navigation-message decoder (IS-GPS-200): parity, bit/frame sync, TLM/HOW.

The 50 bps data (20 ms/bit) is organized as 30-bit words (24 data + 6 parity),
10 words per 300-bit / 6 s subframe, 5 subframes per frame. This module:

  * parity_check / parity_encode -- the IS-GPS-200 (1+2) Hamming parity, including
    the D30* data inversion and D29*/D30* chaining between words.
  * frame_sync -- locate subframe starts by the TLM preamble (0x8B) confirmed by
    parity over the first two words, and resolve the stream POLARITY (the data may
    arrive inverted; parity on the recovered bits is polarity-independent, and the
    preamble on the recovered bits fixes the absolute sign).
  * decode_how -- TOW-count and subframe ID from the HOW word.

The point for the calibrator: a validated, polarity-locked bit sequence to WIPE off
a satellite's data modulation so coherent integration runs past the 20 ms nav bit.
Bits are decoded from a STRONG reference (main beam / bright sat) and reused on the
weak sidelobe measurement -- the message is identical for every beam.

Decoder works on hard bits (0/1). Upstream forms them from the per-20 ms despread
sign; this module validates/syncs/polarity-locks them.
"""
import numpy as np

PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)  # TLM preamble, 0x8B

# IS-GPS-200 parity: each D25..D30 = (prev D29* or D30*) XOR (a fixed subset of d1..d24),
# where d_i = D_i XOR D30* are the data bits corrected for the previous word's bit 30.
# Subsets below are 1-based data-bit indices (Table 20-XIV).
_PAR = [
    (29, [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23]),
    (30, [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24]),
    (29, [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22]),
    (30, [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23]),
    (30, [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24]),
    (29, [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24]),
]


def _parity_bits(d24, D29s, D30s):
    """Six parity bits for data bits d24 (already D30*-corrected), given prev D29*/D30*."""
    out = []
    for src, idx in _PAR:
        v = D29s if src == 29 else D30s
        for i in idx:
            v ^= int(d24[i - 1])
        out.append(v & 1)
    return out


def parity_encode(data24, D29s, D30s):
    """Encode 24 data bits into a 30-bit word with correct parity (for tests/prediction).
    data24 are the source bits BEFORE the D30* inversion the receiver sees."""
    d = [(int(b) ^ D30s) & 1 for b in data24]          # transmitted bits = source XOR D30*
    par = _parity_bits([b ^ D30s for b in d], D29s, D30s)  # parity over recovered (=source) bits
    return np.array(d + par, dtype=np.int8)


def parity_check(word30, D29s, D30s):
    """Validate a 30-bit word against IS-GPS-200 parity given the previous word's bits
    29,30. Returns (recovered_data_24, ok). Recovered data is polarity-independent."""
    D = np.asarray(word30, dtype=np.int8) & 1
    d = (D[:24] ^ D30s) & 1                              # recover data bits (undo D30*)
    exp = _parity_bits(d, D29s, D30s)
    ok = bool(np.all(np.array(exp, dtype=np.int8) == D[24:30]))
    return d, ok


def frame_sync(bits):
    """Find subframe starts in a 0/1 bit stream. Returns a list of (start_index, tow,
    sfid): a subframe is accepted when the TLM preamble matches on the recovered bits
    and words 1-2 (TLM, HOW) both parity-check. GPS parity is polarity-independent (the
    recovered data is the same if the whole stream inverts), so the global sign of the
    despread symbols need not be resolved -- decode the bits as given."""
    bits = np.asarray(bits, dtype=np.int8) & 1
    hits = []
    n = len(bits)
    i = 0
    while i + 60 <= n:
        D29s = int(bits[i - 2]) if i >= 2 else 0
        D30s = int(bits[i - 1]) if i >= 1 else 0
        d1, ok1 = parity_check(bits[i:i + 30], D29s, D30s)
        if ok1 and np.array_equal(d1[:8], PREAMBLE):
            d2, ok2 = parity_check(bits[i + 30:i + 60], int(bits[i + 28]), int(bits[i + 29]))
            if ok2:
                tow, sfid = decode_how(d2)
                hits.append((i, tow, sfid))
                i += 299                                 # skip ~a subframe past a good hit
                continue
        i += 1
    return hits


def decode_how(how_data24):
    """HOW word (word 2): bits 1-17 = TOW count (x6 s = next subframe start), 20-22 = ID."""
    d = np.asarray(how_data24, dtype=np.int8) & 1
    tow = int("".join(str(int(x)) for x in d[:17]), 2)
    sfid = int("".join(str(int(x)) for x in d[19:22]), 2)
    return tow, sfid


# ---- ephemeris-predicted bits ------------------------------------------------
# Decode the message once from a STRONG reference, then PREDICT the satellite's
# future transmitted bits to wipe a beam too weak to self-decode. The message is
# deterministic given the decoded state: subframes 1-3 (ephemeris/clock) hold the
# same data words for ~2 h, the HOW TOW counter increments +1 per 6 s subframe (+5
# per 30 s frame for a given subframe ID), and parity is recomputed. Subframes 4-5
# (almanac) cycle 25 pages over 12.5 min, so they are predicted from the matching
# page 25 frames back -- here we predict the whole frame assuming held data, exact
# for 1-3; the caller supplies the right page words for 4-5.
def decode_subframe(bits, i, D29s, D30s):
    """Decode the 10 words of a subframe at bits[i:i+300]. Returns (data[10][24],
    ok, tow, sfid, D29s, D30s) -- the recovered data words, all-parity-ok, the HOW
    TOW/subframe-ID, and the chaining bits at the subframe's end."""
    bits = np.asarray(bits, dtype=np.int8) & 1
    data, ok = [], True
    for w in range(10):
        word = bits[i + 30 * w:i + 30 * w + 30]
        d, wok = parity_check(word, D29s, D30s)
        ok = ok and wok
        data.append(d)
        D29s, D30s = int(word[28]), int(word[29])
    tow, sfid = decode_how(data[1])
    return data, ok, tow, sfid, D29s, D30s


def encode_subframe(data, D29s, D30s):
    """Re-encode 10 data words (each 24 bits) into the 300 transmitted bits, chaining
    parity. Returns (bits300, D29s, D30s). encode(decode(sf)) round-trips a subframe."""
    out = []
    for d in data:
        word = parity_encode(d, D29s, D30s)
        out.extend(int(x) for x in word)
        D29s, D30s = int(word[28]), int(word[29])
    return np.array(out, dtype=np.int8), D29s, D30s


def set_how_tow(how_data, tow):
    """Return a copy of a HOW data word with the 17-bit TOW count (bits 1-17) set."""
    d = np.array(how_data, dtype=np.int8).copy()
    d[:17] = [int(x) for x in format(tow & 0x1FFFF, "017b")]
    return d


def predict_next_frame(frame_data, frame_tows, D29s, D30s):
    """Predict the transmitted bits of the NEXT frame from a decoded one: re-encode each
    subframe with its HOW TOW advanced +5 (one frame later) and the data words held
    (exact for subframes 1-3). frame_data[k]/frame_tows[k] = subframe k+1's decoded data
    and TOW; (D29s, D30s) = the chaining at this frame's end. Returns the 1500-bit stream."""
    out = []
    for k in range(5):
        data = [np.array(d, dtype=np.int8).copy() for d in frame_data[k]]
        data[1] = set_how_tow(data[1], frame_tows[k] + 5)
        bits, D29s, D30s = encode_subframe(data, D29s, D30s)
        out.extend(int(x) for x in bits)
    return np.array(out, dtype=np.int8)


if __name__ == "__main__":
    # Self-test: build a parity-correct multi-subframe stream, then decode it back.
    rng = np.random.RandomState(0)
    stream = []
    D29s, D30s = 0, 0
    sfid_seq = []
    for sf in range(3):
        for w in range(10):
            data = rng.randint(0, 2, 24).astype(np.int8)
            if w == 0:                                   # word 1: TLM preamble in bits 1-8
                data[:8] = PREAMBLE
            if w == 1:                                   # word 2: HOW -- stuff a TOW + ID
                tow = 1000 + sf
                data[:17] = [int(x) for x in format(tow, "017b")]
                data[19:22] = [int(x) for x in format(sf + 1, "03b")]
                sfid_seq.append(sf + 1)
            word = parity_encode(data, D29s, D30s)
            stream.extend(int(x) for x in word)
            D29s, D30s = int(word[28]), int(word[29])
    # Prepend the two virtual history bits (encoder started D29*=D30*=0) so the first
    # subframe has real history -- mirrors a continuous stream (subframes never at idx 0).
    stream = np.concatenate([[0, 0], np.array(stream, dtype=np.int8)]).astype(np.int8)
    hits = frame_sync(stream)
    print("self-test: %d subframes encoded, %d frame-synced" % (3, len(hits)))
    for (idx, tow, sfid) in hits:
        print("  start %4d  TOW %d  subframe %d" % (idx, tow, sfid))
    assert [h[2] for h in hits] == sfid_seq, "subframe IDs mismatch"
    assert [h[1] for h in hits] == [1000, 1001, 1002], "TOW mismatch"
    # Polarity independence: a globally-inverted stream recovers the SAME data.
    inv = frame_sync(stream ^ 1)
    assert [(h[1], h[2]) for h in inv] == [(h[1], h[2]) for h in hits], "polarity-dependent!"
    # Single-bit error must break parity (the error detection we rely on for clean bits).
    bad = stream.copy()
    bad[2 + 35] ^= 1
    assert len(frame_sync(bad)) < len(hits), "parity missed a bit error"
    print("OK: parity + frame sync + HOW + polarity-independence + error-detect all pass")

    # --- ephemeris-predicted bits: decode one frame, predict the next, must match ---
    # Build 2 frames: held data (the ~2 h-constant ephemeris), TOW +1 per subframe.
    eph = {sf: [rng.randint(0, 2, 24).astype(np.int8) for _ in range(10)] for sf in range(1, 6)}
    for sf in range(1, 6):
        eph[sf][0][:8] = PREAMBLE

    def subframe(sf, tow):
        d = [w.copy() for w in eph[sf]]
        d[1] = set_how_tow(d[1], tow)
        d[1][19:22] = [int(x) for x in format(sf, "03b")]
        return d

    s2, D29s, D30s, f0_tows = [0, 0], 0, 0, []
    for frame in range(2):
        for k in range(5):
            tow = 1000 + frame * 5 + k
            bits, D29s, D30s = encode_subframe(subframe(k + 1, tow), D29s, D30s)
            s2.extend(int(x) for x in bits)
            if frame == 0:
                f0_tows.append(tow)
    s2 = np.array(s2, dtype=np.int8)
    # Decode frame 0 (the strong-reference capture): its data words + the chaining at end.
    i, c29, c30, fdata = 2, 0, 0, []
    for k in range(5):
        data, ok, tow, sfid, c29, c30 = decode_subframe(s2, i + 300 * k, c29, c30)
        assert ok, "subframe %d parity failed" % (k + 1)
        fdata.append(data)
    # Predict frame 1 from the decoded frame 0 and compare to what was actually transmitted.
    pred = predict_next_frame(fdata, f0_tows, c29, c30)
    actual = s2[2 + 1500:2 + 3000]
    assert np.array_equal(pred, actual), "predicted frame != actual (%d/1500 bits differ)" \
        % int(np.sum(pred != actual))
    print("OK: ephemeris-predicted bits reproduce the next frame exactly (%d bits)" % len(pred))
