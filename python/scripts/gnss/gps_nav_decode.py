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
import math

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


# ---- ephemeris extraction (subframes 1-3, IS-GPS-200 Table 20-III) -----------
# The decoder already yields per-subframe DATA words (10 x 24 bits, parity stripped and
# D30*-corrected -- decode_subframe's `data`). This turns subframes 1/2/3 into the Keplerian
# clock+orbit set and propagates it, so an L1-only GPS satellite feeds the decoded-eph BRDC
# fallback exactly as the CNAV (L2C/L5) decoders do. GPS LNAV is the LEGACY orbit model
# (sqrtA direct, no A_dot/dn0_dot) -- the same math as gnss_ephemeris.sat_pos_clk / Galileo
# sv_position_inav -- so this reuses that propagation, only fed seconds-of-week.
#
# Fields are addressed as (word_index 0-9, start-bit 0-indexed within that word's 24 DATA bits,
# length); multi-word quantities (M0/e/sqrtA/OMEGA0/i0/omega, IODC) list their MSB then LSB
# segments, concatenated MSB-first. Angular fields are in semicircles -> * pi for radians.
# ⚠️ These bit offsets are the ONE thing a pack/unpack round-trip cannot check; the LIVE BRDC
# position cross-check (_lnav_brdc_xcheck in the broker; near-zero since BRDC *is* LNAV) is the
# real validator. If it disagrees, correct THIS table, not the propagation.
GPS_PI = 3.1415926535898
GPS_MU = 3.986005e14
GPS_OMEGA_E = 7.2921151467e-5
_LP = lambda n: 2.0 ** (-n)

LNAV_EPH_FIELDS = {
    # subframe 1: clock + health (word 3 = index 2 ... word 10 = index 9)
    "WN":     (1, [(2, 0, 10)],           False, 1.0),
    "health": (1, [(2, 16, 6)],           False, 1.0),
    "IODC":   (1, [(2, 22, 2), (7, 0, 8)], False, 1.0),
    "TGD":    (1, [(6, 16, 8)],           True,  _LP(31)),
    "toc":    (1, [(7, 8, 16)],           False, 16.0),
    "af2":    (1, [(8, 0, 8)],            True,  _LP(55)),
    "af1":    (1, [(8, 8, 16)],           True,  _LP(43)),
    "af0":    (1, [(9, 0, 22)],           True,  _LP(31)),
    # subframe 2: ephemeris part 1
    "IODE2":  (2, [(2, 0, 8)],            False, 1.0),
    "Crs":    (2, [(2, 8, 16)],           True,  _LP(5)),
    "dn":     (2, [(3, 0, 16)],           True,  _LP(43) * GPS_PI),
    "M0":     (2, [(3, 16, 8), (4, 0, 24)], True, _LP(31) * GPS_PI),
    "Cuc":    (2, [(5, 0, 16)],           True,  _LP(29)),
    "e":      (2, [(5, 16, 8), (6, 0, 24)], False, _LP(33)),
    "Cus":    (2, [(7, 0, 16)],           True,  _LP(29)),
    "sqrtA":  (2, [(7, 16, 8), (8, 0, 24)], False, _LP(19)),
    "toe":    (2, [(9, 0, 16)],           False, 16.0),
    # subframe 3: ephemeris part 2
    "Cic":    (3, [(2, 0, 16)],           True,  _LP(29)),
    "OMEGA0": (3, [(2, 16, 8), (3, 0, 24)], True, _LP(31) * GPS_PI),
    "Cis":    (3, [(4, 0, 16)],           True,  _LP(29)),
    "i0":     (3, [(4, 16, 8), (5, 0, 24)], True, _LP(31) * GPS_PI),
    "Crc":    (3, [(6, 0, 16)],           True,  _LP(5)),
    "omega":  (3, [(6, 16, 8), (7, 0, 24)], True, _LP(31) * GPS_PI),
    "OMEGA_dot": (3, [(8, 0, 24)],        True,  _LP(43) * GPS_PI),
    "IODE3":  (3, [(9, 0, 8)],            False, 1.0),
    "idot":   (3, [(9, 8, 14)],           True,  _LP(43) * GPS_PI),
}


def _lnav_field(sf_data, segments, signed, scale):
    """Concatenate the bit segments (MSB-first) of one field from a subframe's 10x24 data
    words and scale. `sf_data` is {sfid: [word0..word9]}, each word a length-24 0/1 array."""
    bits = []
    for widx, start, length in segments:
        w = sf_data[widx]
        bits.extend(int(w[start + k]) for k in range(length))
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    if signed and bits and bits[0] == 1:
        v -= (1 << len(bits))
    return v * scale


def parse_lnav_ephemeris(sf1, sf2, sf3):
    """Decode the LNAV clock+ephemeris (SI) from subframes 1, 2, 3 (each a list of 10 data
    words of 24 bits). Returns the eph dict, or None if the two IODEs / IODC LSBs disagree
    (a mixed upload set). Angular elements are radians; toe/toc/WN in their native units."""
    src = {1: sf1, 2: sf2, 3: sf3}
    if any(v is None or len(v) < 10 for v in src.values()):
        return None
    eph = {}
    for name, (sf, segs, signed, scale) in LNAV_EPH_FIELDS.items():
        eph[name] = _lnav_field({0: src[sf][0], 1: src[sf][1], 2: src[sf][2], 3: src[sf][3],
                                 4: src[sf][4], 5: src[sf][5], 6: src[sf][6], 7: src[sf][7],
                                 8: src[sf][8], 9: src[sf][9]}, segs, signed, scale)
    # Issue-of-data consistency: the two IODEs and IODC's 8 LSBs must match for one orbit set.
    iode2, iode3, iodc = int(eph["IODE2"]), int(eph["IODE3"]), int(eph["IODC"])
    eph["IODE"] = iode2
    eph["_iod_consistent"] = (iode2 == iode3 == (iodc & 0xFF))
    return eph


def sv_position_lnav(eph, t):
    """ECEF (x,y,z) m from LNAV ephemeris (IS-GPS-200 Table 20-IV). Legacy Keplerian --
    identical math to gnss_ephemeris.sat_pos_clk / galileo sv_position_inav, GPS constants;
    `t` in GPS seconds-of-week (the propagator week-wraps tk = t - toe)."""
    A = eph["sqrtA"] ** 2
    tk = t - eph["toe"]
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800
    n = math.sqrt(GPS_MU / A ** 3) + eph["dn"]
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
    i = eph["i0"] + eph["idot"] * tk + eph["Cis"] * s2 + eph["Cic"] * c2
    om = eph["OMEGA0"] + (eph["OMEGA_dot"] - GPS_OMEGA_E) * tk - GPS_OMEGA_E * eph["toe"]
    xp, yp = r * math.cos(u), r * math.sin(u)
    x = xp * math.cos(om) - yp * math.cos(i) * math.sin(om)
    y = xp * math.sin(om) + yp * math.cos(i) * math.cos(om)
    z = yp * math.sin(i)
    return x, y, z


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

    # --- ephemeris field table: SELF-CONSISTENT pack/unpack + no overlaps/overruns ---
    # (This proves the table is internally consistent, NOT that the offsets match the ICD --
    # only the live BRDC position cross-check confirms that, see the field-table note.)
    sfw = {sf: [np.zeros(24, dtype=np.int8) for _ in range(10)] for sf in (1, 2, 3)}
    truth = {}
    for name, (sf, segs, signed, scale) in LNAV_EPH_FIELDS.items():
        nbits = sum(L for _, _, L in segs)
        code = (hash(name) % ((1 << (nbits - 1)) - 1 if nbits > 1 else 1)) + 1
        cb = [(code >> (nbits - 1 - k)) & 1 for k in range(nbits)]
        off = 0
        for (widx, start, L) in segs:
            for k in range(L):
                sfw[sf][widx][start + k] = cb[off + k]
            off += L
    for name, (sf, segs, signed, scale) in LNAV_EPH_FIELDS.items():
        truth[name] = _lnav_field({i: sfw[sf][i] for i in range(10)}, segs, signed, scale)
    eph_t = parse_lnav_ephemeris(sfw[1], sfw[2], sfw[3])
    assert eph_t is not None and all(
        abs(eph_t[n] - truth[n]) < 1e-12 * (abs(truth[n]) + 1) for n in truth), \
        "ephemeris field pack/unpack mismatch"
    # overlap / overrun guard, per subframe
    for sf in (1, 2, 3):
        occ = [0] * (10 * 24)
        for name, (s, segs, signed, scale) in LNAV_EPH_FIELDS.items():
            if s != sf:
                continue
            for (widx, start, L) in segs:
                assert start + L <= 24, "%s overruns word %d" % (name, widx)
                for k in range(start, start + L):
                    occ[widx * 24 + k] += 1
        assert max(occ) <= 1, "field overlap in subframe %d" % sf
    print("OK: LNAV ephemeris field table self-consistent (pack/unpack, no overlap/overrun)")

    # --- propagator agreement: sv_position_lnav == the trusted legacy sat_pos_clk ---
    # Feed one synthetic orbit to BOTH the new LNAV propagator and gnss_ephemeris.sat_pos_clk
    # (the RINEX/BRDC propagator the dpos cross-check already validates) at several epochs; they
    # implement the same legacy Keplerian math, so they must agree to ~mm.
    import gnss_ephemeris as _ge
    orbit = dict(sqrtA=5153.7, e=0.008, M0=0.3, omega=-1.1, i0=0.96, OMEGA0=-0.7,
                 OMEGA_dot=-8.0e-9, idot=1.0e-10, dn=4.5e-9, toe=345600.0,
                 Cuc=1.0e-6, Cus=8.0e-6, Crc=200.0, Crs=-30.0, Cic=-1.0e-7, Cis=2.0e-7)
    week = 2200
    e_leg = dict(sys="G", sqrta=orbit["sqrtA"], ecc=orbit["e"], m0=orbit["M0"],
                 omega=orbit["omega"], i0=orbit["i0"], omega0=orbit["OMEGA0"],
                 omegadot=orbit["OMEGA_dot"], idot=orbit["idot"], dn=orbit["dn"],
                 cuc=orbit["Cuc"], cus=orbit["Cus"], crc=orbit["Crc"], crs=orbit["Crs"],
                 cic=orbit["Cic"], cis=orbit["Cis"], toe_sow=orbit["toe"],
                 toe_gpst=week * 604800.0 + orbit["toe"], af0=0.0, af1=0.0, af2=0.0,
                 toc_gpst=week * 604800.0 + orbit["toe"])
    worst = 0.0
    for dtk in (-7200.0, -1800.0, 0.0, 1800.0, 7200.0):
        xl = sv_position_lnav(orbit, orbit["toe"] + dtk)
        xg = _ge.sat_pos_clk(e_leg, week * 604800.0 + orbit["toe"] + dtk)[0]
        worst = max(worst, math.sqrt(sum((a - b) ** 2 for a, b in zip(xl, xg))))
    assert worst < 1e-3, "sv_position_lnav disagrees with sat_pos_clk by %.3e m" % worst
    print("OK: sv_position_lnav agrees with the legacy sat_pos_clk (worst %.2e m over +-2 h)"
          % worst)
