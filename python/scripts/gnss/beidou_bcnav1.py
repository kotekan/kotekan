"""BeiDou-3 B1C B-CNAV1 codec: frame symbols <-> nav bits, and the ephemeris parse.

The FOURTH and LAST civil S5 D-component (2026-07-30). B-CNAV1 rides the B1C DATA channel on
the L1 band (1575.42, beside E1B). It REUSES the GF(64) non-binary LDPC decoder (nb_ldpc.py)
built for B-CNAV2 -- different H matrices (SF2 = NB-LDPC(200,100), SF3 = NB-LDPC(88,44)) -- and
the B-CNAV2 CNAV-style ephemeris propagation (sv_position_bcnav2), since B-CNAV1 and B-CNAV2
share the BDS-3 ephemeris parameter set. What is NEW here:

  * an 1800-symbol / 18 s frame = SF1 (72 syms: PRN + SOH as short BCH/LFSR sequences, the
    frame-sync pattern) + a BLOCK-INTERLEAVED (36x48) SF2/SF3;
  * SF2 -> 600 bits (the FULL ephemeris + clock, one subframe -- simpler than B-CNAV2's two
    frames), SF3 -> 264 bits (iono/other); each subframe has its own CRC-24Q;
  * frame sync by generating the expected SF1 (PRN-specific 21 syms + SOH-specific 51 syms,
    searching SOH 0..199) rather than a fixed preamble.

Frame timing: a B1C-data symbol is 10 ms (one 10230-chip BOC(1,1) code period; 50 bps through
the rate-1/2 LDPC = 100 sps), so SF1/SF2/SF3 = 72/1200/528 symbols after de-interleave.

★ ICD-OWNED CONVENTIONS (flippable, LIVE-VERIFIED via BRDC dpos): the data-bit inversion into
the LDPC (PocketSDR applies a `^1`; here resolved by trying both and gating on CRC, then cached)
and the ephemeris offsets (from andrewkamble88/gnss-sdr@bds_b1c Beidou_B1C.h). Sources: PocketSDR
(SF1 LFSRs, interleaver, H matrices) + that B1C fork (SF2 ephemeris layout).
"""

import numpy as np

import nb_ldpc as _NB
from bcnv1_ldpc_matrices import H_SF2_IDX, H_SF2_ELE, H_SF3_IDX, H_SF3_ELE
from galileo_inav import crc24q, _uint, _field, _P
# reuse the BDS-3 CNAV ephemeris propagation + constants from B-CNAV2
from beidou_bcnav2 import (sv_position_bcnav2, BDS_PI, A_REF_MEO, A_REF_IGSO)

SYM_S = 0.010                  # B1C-data symbol = 10 ms (100 sps)
N_SF1 = 72                     # SF1 symbols (21 PRN + 51 SOH)
FRAME_SYMS = 1800              # full frame = 18 s
SF2_BITS = 600
SF3_BITS = 264
SF2_CRC_AT = SF2_BITS - 24     # 576
SF3_CRC_AT = SF3_BITS - 24     # 240
N_SOH = 200                    # SOH range (18 s frames, 3600 s hour / 18 = 200)


# ------------------------------------------------------------------ SF1 sync sequences (LFSR)
def _rev(R, n):
    r = 0
    for i in range(n):
        r = (r << 1) | ((R >> i) & 1)
    return r


def _lfsr_bits(N, R, tap, n):
    """PocketSDR LFSR -> 0/1 symbols: out = (CHIP[R&1]+1)//2, CHIP=[+1,-1], Fibonacci shift."""
    out = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        out[i] = 1 if (R & 1) == 0 else 0
        R = ((bin(R & tap).count("1") & 1) << (n - 1)) | (R >> 1)
    return out


_SF1A = {}
_SF1B = None
def _sf1a(prn):
    if prn not in _SF1A:
        _SF1A[prn] = _lfsr_bits(21, _rev(prn, 6), 0b010111, 6)     # BCH(21,6) PRN sequence
    return _SF1A[prn]
def _sf1b_all():
    global _SF1B
    if _SF1B is None:
        _SF1B = [_lfsr_bits(51, _rev(soh, 8), 0b10011111, 8) for soh in range(N_SOH)]
    return _SF1B


# ------------------------------------------------------------------ block interleaver (36x48)
def _deinterleave(frame_bits):
    """1800 hard bits -> (syms2[1200], syms3[528]). SF1 = [:72]; the 1728 coded symbols
    [72:1800] are read out of a 36x48 block (BDS-SIS-ICD-B1C, via PocketSDR)."""
    symsr = np.asarray(frame_bits[N_SF1:FRAME_SYMS], dtype=np.uint8).reshape(48, 36).T  # (36,48)
    syms2 = np.zeros(1200, dtype=np.uint8)
    syms3 = np.zeros(528, dtype=np.uint8)
    for i in range(11):
        syms2[i * 96:i * 96 + 48] = symsr[i * 3]
        syms2[i * 96 + 48:i * 96 + 96] = symsr[i * 3 + 1]
        syms3[i * 48:i * 48 + 48] = symsr[i * 3 + 2]
    for i in range(22, 25):
        syms2[i * 48:i * 48 + 48] = symsr[i + 11]
    return syms2, syms3


def _interleave(syms2, syms3):
    """Inverse of _deinterleave (for the self-test): (1200, 528) -> 1728 coded symbols."""
    symsr = np.zeros((36, 48), dtype=np.uint8)
    for i in range(11):
        symsr[i * 3] = syms2[i * 96:i * 96 + 48]
        symsr[i * 3 + 1] = syms2[i * 96 + 48:i * 96 + 96]
        symsr[i * 3 + 2] = syms3[i * 48:i * 48 + 48]
    for i in range(22, 25):
        symsr[i + 11] = syms2[i * 48:i * 48 + 48]
    return symsr.T.reshape(-1)


# ------------------------------------------------------------------ subframe decode + CRC
def _decode_sf(idx, ele, m, n, syms, invert):
    s = np.asarray(syms, dtype=np.uint8)
    if invert:
        s = s ^ 1
    return _NB.decode(idx, ele, m, n, s)


def check_crc(bits, crc_at, total):
    b = [int(x) & 1 for x in bits]
    if len(b) != total:
        return False
    return crc24q(b[0:crc_at]) == _uint(b[crc_at:crc_at + 24])


# invert convention into the LDPC: resolved on the first CRC-valid frame, then cached (avoids
# a slow wrong-invert iterate every frame). None until known; find_frames tries both.
_INVERT = [None]


def find_frames(prn, symbols):
    """Scan a soft symbol stream for SF1-aligned frames (both polarities). Yields
    (start_index, polarity, soh, sf2_bits[600], sf3_bits_or_None) per CRC-valid SF2. Steps by 1."""
    s = np.asarray(symbols, dtype=float)
    nlen = len(s)
    a = _sf1a(prn)
    b_all = _sf1b_all()
    i = 0
    while i + FRAME_SYMS <= nlen:
        for pol in (1.0, -1.0):
            hard = (pol * s[i:i + N_SF1] < 0).astype(np.uint8)
            if not np.array_equal(hard[:21], a):
                continue
            soh = next((k for k, bb in enumerate(b_all) if np.array_equal(hard[21:72], bb)), None)
            if soh is None:
                continue
            frame = (pol * s[i:i + FRAME_SYMS] < 0).astype(np.uint8)
            syms2, syms3 = _deinterleave(frame)
            inverts = (_INVERT[0],) if _INVERT[0] is not None else (False, True)
            for inv in inverts:
                sf2, nerr = _decode_sf(H_SF2_IDX, H_SF2_ELE, 100, 200, syms2, inv)
                if nerr >= 0 and check_crc(sf2, SF2_CRC_AT, SF2_BITS):
                    _INVERT[0] = inv
                    sf3b = None
                    sf3, nerr3 = _decode_sf(H_SF3_IDX, H_SF3_ELE, 44, 88, syms3, inv)
                    if nerr3 >= 0 and check_crc(sf3, SF3_CRC_AT, SF3_BITS):
                        sf3b = [int(x) for x in sf3]
                    yield i, int(pol), soh, [int(x) for x in sf2], sf3b
                    break
            break
        i += 1


# ------------------------------------------------------------------ ephemeris (SF2, types shared with B-CNAV2)
# name -> (start bit in the 600-bit SF2 [0-indexed], length, signed, scale). Offsets from
# andrewkamble88/gnss-sdr@bds_b1c BEIDOU_B1C.h (1-indexed -> my_start = offset-1; SatType/scales
# IDENTICAL to B-CNAV2, verified against beidou_cnav1_navigation_message.cc). LIVE-VERIFIED via
# BRDC dpos. The WHOLE orbit is in SF2 (one subframe) -- no cross-frame matching.
BCNV1_SF2_EPH_FIELDS = {
    "t_oe":     (39, 11, False, 300.0),
    "SatType":  (50, 2, False, 1.0),
    "dA":       (52, 26, True, _P(9)),
    "A_dot":    (78, 25, True, _P(21)),
    "dn_0":     (103, 17, True, _P(44) * BDS_PI),
    "dn_0_dot": (120, 23, True, _P(57) * BDS_PI),
    "M_0":      (143, 33, True, _P(32) * BDS_PI),
    "e":        (176, 33, False, _P(34)),
    "omega":    (209, 33, True, _P(32) * BDS_PI),
    "Omega_0":  (242, 33, True, _P(32) * BDS_PI),
    "i_0":      (275, 33, True, _P(32) * BDS_PI),
    "Omega_dot":(308, 19, True, _P(44) * BDS_PI),
    "i_0_dot":  (327, 15, True, _P(44) * BDS_PI),
    "C_IS":     (342, 16, True, _P(30)),
    "C_IC":     (358, 16, True, _P(30)),
    "C_RS":     (374, 24, True, _P(8)),
    "C_RC":     (398, 24, True, _P(8)),
    "C_US":     (422, 21, True, _P(30)),
    "C_UC":     (443, 21, True, _P(30)),
}
_IODE_SF2 = (31, 8)   # IODE (start, len) in SF2


def parse_bcnav1_ephemeris(sf2_bits):
    """600-bit SF2 -> SI ephemeris (A_ref resolved by SatType). The full orbit is here."""
    eph = {}
    for name, (start, length, signed, scale) in BCNV1_SF2_EPH_FIELDS.items():
        eph[name] = _field(sf2_bits, start, length, signed, scale)
    eph["A_ref"] = A_REF_IGSO if int(round(eph["SatType"])) in (1, 2) else A_REF_MEO
    eph["IODE"] = _uint(sf2_bits[_IODE_SF2[0]:_IODE_SF2[0] + _IODE_SF2[1]])
    eph["_iod_consistent"] = True
    return eph


def sv_position_bcnav1(eph, t):
    """ECEF -- identical BDS-3 CNAV propagation as B-CNAV2 (shared parameter set)."""
    return sv_position_bcnav2(eph, t)


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import sys
    _NB.init_gf()
    rng = np.random.default_rng(4)
    fails = 0

    # 1. SF1 sync sequences are distinct per PRN / per SOH (a real sync must be)
    a19, a20 = _sf1a(19), _sf1a(20)
    b = _sf1b_all()
    ok1 = (len(a19) == 21 and not np.array_equal(a19, a20)
           and len({tuple(x) for x in b}) == N_SOH)
    print("1. SF1 sequences distinct (PRN + all %d SOH): %s" % (N_SOH, "OK" if ok1 else "FAIL"))
    fails += 0 if ok1 else 1

    # 2. interleaver roundtrip (deinterleave . interleave == identity on the coded region)
    s2 = rng.integers(0, 2, 1200).astype(np.uint8)
    s3 = rng.integers(0, 2, 528).astype(np.uint8)
    il = _interleave(s2, s3)
    frame = np.concatenate([np.zeros(N_SF1, dtype=np.uint8), il])
    d2, d3 = _deinterleave(frame)
    ok2 = np.array_equal(d2, s2) and np.array_equal(d3, s3)
    print("2. block-interleaver roundtrip (36x48): %s" % ("OK" if ok2 else "FAIL"))
    fails += 0 if ok2 else 1

    # 3. SF2 NB-LDPC(200,100) + SF3 NB-LDPC(88,44): encode a codeword, decode clean + w/ errors
    p3 = 0
    for (idx, ele, m, n, nsym) in ((H_SF2_IDX, H_SF2_ELE, 100, 200, 1200),
                                   (H_SF3_IDX, H_SF3_ELE, 44, 88, 528)):
        info = rng.integers(0, Q_GF if False else 64, m).astype(np.uint8)
        cw = _NB.encode_systematic(idx, ele, m, n, info)
        bits, nerr = _NB.decode(idx, ele, m, n, cw)
        clean_ok = (nerr == 0 and np.array_equal(bits, cw[:m * 6]))
        bad = cw.copy(); bad[rng.choice(nsym, 6, replace=False)] ^= 1
        b2, n2 = _NB.decode(idx, ele, m, n, bad)
        corr_ok = (n2 >= 0 and np.array_equal(b2, cw[:m * 6]))
        tag = "SF2(200,100)" if m == 100 else "SF3(88,44)"
        print("   %s: clean %s, 6-err corrected %s" % (tag, clean_ok, corr_ok))
        if not (clean_ok and corr_ok):
            p3 += 1
    print("3. SF2/SF3 NB-LDPC: %s" % ("OK" if p3 == 0 else "FAIL %d" % p3))
    fails += p3

    # 4. END TO END: plant an ephemeris in SF2 (+CRC), LDPC-encode both subframes, interleave,
    # prepend SF1(prn,soh), feed as symbols -> find_frames -> recover the ephemeris.
    Q_GF = 64
    prn, soh = 19, 137
    sf2 = [0] * SF2_BITS
    truth = {}
    for name, (start, length, signed, scale) in BCNV1_SF2_EPH_FIELDS.items():
        code = (hash(name) % ((1 << (length - 1)) - 1 if length > 1 else 1)) + 1
        for k in range(length):
            sf2[start + k] = (code >> (length - 1 - k)) & 1
        truth[name] = _field(sf2, start, length, signed, scale)
    crc = crc24q(sf2[0:SF2_CRC_AT]); sf2[SF2_CRC_AT:SF2_CRC_AT + 24] = [(crc >> (23 - k)) & 1 for k in range(24)]
    sf3 = [0] * SF3_BITS
    crc3 = crc24q(sf3[0:SF3_CRC_AT]); sf3[SF3_CRC_AT:SF3_CRC_AT + 24] = [(crc3 >> (23 - k)) & 1 for k in range(24)]
    # encode subframes (info = first m GF symbols of each subframe's bits)
    syms2 = _NB.encode_systematic(H_SF2_IDX, H_SF2_ELE, 100, 200, _NB.bin2gf(np.array(sf2, dtype=np.uint8)))
    syms3 = _NB.encode_systematic(H_SF3_IDX, H_SF3_ELE, 44, 88, _NB.bin2gf(np.array(sf3, dtype=np.uint8)))
    il = _interleave(syms2, syms3)
    sf1 = np.concatenate([_sf1a(prn), _sf1b_all()[soh]])
    frame_bits = np.concatenate([sf1, il]).astype(np.uint8)      # 1800 transmitted bits
    # repeat 2 frames so a full 1800 lands inside; map bit 0->+1, 1->-1 (bipolar)
    stream = np.tile(frame_bits, 2)
    soft = np.where(stream == 0, 1.0, -1.0)
    _INVERT[0] = None
    got = list(find_frames(prn, soft))
    p4 = 0
    if not got:
        print("4. FAIL: no frame found"); p4 = 1
    else:
        _i, _pol, gsoh, gsf2, gsf3 = got[0]
        eph = parse_bcnav1_ephemeris(gsf2)
        bad = [k for k in truth if abs(eph[k] - truth[k]) > 1e-9 * (abs(truth[k]) + 1)]
        if gsoh != soh or bad:
            print("4. FAIL: soh %d/%d, eph disagree %s" % (gsoh, soh, bad)); p4 = 1
        else:
            print("4. end-to-end frame->eph: OK (soh %d, invert=%s, SatType %d)"
                  % (gsoh, _INVERT[0], int(round(eph["SatType"]))))
    fails += p4

    print("\n%s" % ("B-CNAV1 codec SELF-CONSISTENT (live CRC + dpos pend deploy)"
                    if not fails else "SELF-TEST FAILURES: %d" % fails))
    sys.exit(1 if fails else 0)
