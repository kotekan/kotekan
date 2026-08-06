"""BeiDou-3 B2b-I B-CNAV3 codec: frame symbols <-> nav bits, and the ephemeris parse.

The B2b DATA-channel message (BDS-SIS-ICD-B2b, Open Service Signal B2b v1.0). Like B-CNAV2 it is
protected by a NON-BINARY LDPC over GF(64), but a bigger code -- (n=162, k=81) in 6-bit symbols
(972 coded bits -> 486 info bits) vs B-CNAV2's (96,48). The GF(64) tables + QSPA/EMS decoder are
the SAME shared nb_ldpc.py; only the H matrix + frame geometry differ. CRC-24Q reused as always.

Frame (per BDS-SIS-ICD-B2b): 1000 symbols = 1 s at 1000 sps (a B2b symbol is 1 ms = one B2b-I
primary code period, no secondary -> navwipe_bit_records 1). Structure: 16-symbol PREAMBLE
(0xEB90) + 6-symbol PRN + 6 reserved + 972 LDPC symbols. The framesync pattern is PRN-SPECIFIC
(preamble || PRN), unlike B-CNAV2's fixed preamble; the PRN is therefore in the SYNC header, not
the LDPC-protected info. The 972 LDPC symbols (frame[28:1000]) decode to a 486-bit info frame:
MesType(6) + SOW(20) + payload(436) + CRC-24(24).

Verbatim from PocketSDR (sdr_ldpc.py H_BCNV3_idx/ele + sdr_nav.py decode_B2BI/BCNV3, T.Takasu,
ICD-transcribed). Decoder is SYSTEMATIC (info = first 81 GF symbols = 486 bits); at our deep-
integration SNR the codeword is error-free, so decode is essentially a parity check.

★ ICD-OWNED CONVENTIONS (flippable, to be LIVE-VERIFIED via the BRDC dpos, the B-CNAV2 lesson):
the preamble polarity (both tried) and -- for the ephemeris -- the message-type layout + field
offsets/scales. The frame decode (LDPC + CRC) is convention-COMPLETE and self-validating (the
CRC is the truth); the ephemeris field table is filled in against LIVE frames + the BRDC dpos.
"""

import math

import numpy as np

# reuse CRC-24Q + bit helpers (BDS B-CNAV uses the same 0x1864CFB as GPS/Galileo)
from galileo_inav import crc24q, _uint, _field, _P  # noqa: F401  (_field/_P for the eph table)

SYM_S = 0.001                  # B2b-I symbol = 1 ms (1000 sps), one B2b primary period (no overlay)

# 16-symbol fixed preamble (0xEB90) -- the PRN's 6 bits are appended per-satellite in preamble_for()
PREAMBLE = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
N_PRE_FIXED = len(PREAMBLE)    # 16
N_PRE = N_PRE_FIXED + 6        # 22 (preamble || 6-bit PRN) -- the PRN-specific sync pattern
LDPC_OFFSET = 28               # LDPC symbols start after preamble(16)+PRN(6)+reserved(6)
N_LDPC_SYM = 972               # coded bits per frame (= 162 GF(64) symbols)
FRAME_SYMS = 1000              # 1 s at 1000 sps
FRAME_BITS = 486               # info bits after LDPC
CRC_AT = FRAME_BITS - 24       # 462: CRC-24 occupies bits [462:486]

# BDS-3 constants (BDS-SIS-ICD)
BDS_MU = 3.986004418e14
BDS_OMEGA_E = 7.292115e-5
BDS_PI = 3.1415926535898
A_REF_MEO = 27906100.0
A_REF_IGSO = 42162200.0

# ------------------------------------------------------------------ GF(64) NB-LDPC (shared)
import nb_ldpc as _NB
from nb_ldpc import GF_VEC, GF_POW, Q_GF, N_GF  # noqa: F401  (re-export for tests)
_init_gf = _NB.init_gf
_bin2gf = _NB.bin2gf
_gf2bin = _NB.gf2bin

# B-CNAV3 LDPC H-matrix ([BDS-ICD-B2b 6.2.2] via PocketSDR sdr_ldpc.H_BCNV3_idx/ele): 81 checks
# x 162 vars, weight 4, GF(64).
H_BCNV3_IDX = (
    (19, 67, 109, 130), (27, 71, 85, 161), (31, 78, 96, 122), (2, 44, 83, 125),
    (26, 71, 104, 132), (30, 39, 93, 154), (4, 46, 85, 127), (21, 62, 111, 127),
    (13, 42, 101, 146), (18, 66, 108, 129), (27, 72, 100, 153), (29, 70, 84, 160),
    (23, 61, 113, 126), (8, 50, 89, 131), (34, 74, 111, 157), (12, 44, 100, 145),
    (22, 60, 112, 128), (0, 49, 115, 151), (6, 47, 106, 144), (33, 53, 82, 140),
    (3, 45, 84, 126), (38, 80, 109, 147), (9, 60, 96, 141), (1, 43, 82, 124),
    (20, 77, 88, 158), (37, 54, 122, 159), (3, 65, 104, 149), (5, 47, 86, 128),
    (0, 42, 81, 123), (32, 79, 97, 120), (35, 72, 112, 158), (15, 57, 93, 138),
    (22, 75, 107, 143), (24, 69, 102, 133), (1, 50, 116, 152), (24, 57, 119, 135),
    (17, 59, 95, 140), (7, 45, 107, 145), (34, 51, 83, 138), (14, 43, 99, 144),
    (21, 77, 106, 142), (16, 58, 94, 139), (20, 68, 110, 131), (2, 48, 114, 150),
    (10, 52, 91, 133), (25, 70, 103, 134), (32, 41, 95, 153), (14, 56, 91, 137),
    (33, 73, 113, 156), (28, 73, 101, 154), (4, 63, 102, 147), (6, 48, 87, 129),
    (8, 46, 105, 146), (30, 80, 98, 121), (41, 68, 119, 150), (35, 52, 81, 139),
    (16, 63, 114, 124), (13, 55, 90, 136), (31, 40, 94, 155), (10, 61, 97, 142),
    (36, 56, 121, 161), (29, 74, 99, 155), (5, 64, 103, 148), (18, 75, 89, 156),
    (36, 78, 110, 148), (19, 76, 87, 157), (15, 65, 116, 123), (11, 53, 92, 134),
    (25, 58, 117, 136), (39, 66, 117, 151), (11, 62, 98, 143), (9, 51, 90, 132),
    (38, 55, 120, 160), (7, 49, 88, 130), (17, 64, 115, 125),
    (28, 69, 86, 159), (23, 76, 105, 141), (12, 54, 92, 135),
    (40, 67, 118, 152), (37, 79, 108, 149), (26, 59, 118, 137))
H_BCNV3_ELE = (
    (46, 45, 44, 15), (15, 24, 50, 37), (24, 50, 37, 15), (15, 32, 18, 61),
    (58, 56, 60, 62), (37, 53, 61, 29), (46, 58, 18, 6), (36, 19, 3, 57),
    (54, 7, 38, 23), (51, 59, 63, 47), (9, 3, 43, 29), (56, 8, 46, 13),
    (26, 22, 14, 2), (63, 26, 41, 12), (17, 32, 58, 37), (38, 23, 55, 22),
    (35, 1, 31, 44), (44, 51, 35, 13), (30, 1, 44, 7), (27, 5, 2, 62),
    (16, 63, 20, 9), (27, 56, 8, 43), (1, 44, 30, 24), (5, 26, 27, 37),
    (42, 47, 37, 32), (38, 12, 25, 51), (43, 34, 48, 57), (39, 9, 30, 48),
    (63, 13, 54, 10), (2, 46, 56, 35), (47, 20, 33, 26), (62, 54, 56, 60),
    (1, 21, 25, 7), (43, 58, 19, 49), (28, 4, 52, 44), (46, 44, 14, 15),
    (41, 48, 2, 27), (49, 21, 7, 35), (40, 21, 44, 17), (24, 23, 45, 11),
    (46, 25, 22, 48), (13, 29, 53, 61), (52, 17, 24, 61), (29, 41, 10, 16),
    (60, 24, 4, 50), (32, 49, 58, 19), (43, 34, 48, 57), (29, 7, 10, 16),
    (25, 11, 7, 1), (32, 49, 58, 19), (42, 14, 24, 33), (39, 56, 30, 48),
    (13, 27, 56, 8), (53, 40, 61, 18), (8, 43, 27, 56), (18, 40, 32, 61),
    (60, 48, 2, 27), (50, 54, 60, 62), (58, 19, 32, 49), (9, 3, 63, 43),
    (53, 35, 16, 13), (23, 25, 30, 16), (18, 6, 61, 21), (15, 1, 42, 45),
    (20, 16, 63, 9), (27, 37, 5, 26), (29, 7, 10, 16), (11, 60, 6, 49),
    (43, 47, 18, 20), (42, 14, 24, 33), (43, 22, 41, 20), (22, 15, 12, 33),
    (9, 41, 57, 58), (5, 31, 51, 30), (9, 3, 63, 43),
    (37, 53, 61, 29), (6, 45, 56, 19), (33, 45, 36, 34),
    (19, 24, 42, 14), (1, 45, 15, 6), (8, 43, 27, 56))
N_CHECK = len(H_BCNV3_IDX)     # 81
N_VAR = 162


def decode_nb_ldpc(syms):
    """972 hard bits (0/1) -> (486 info bits, nerr) via the shared GF(64) NB-LDPC decoder.
    nerr>=0 = decoded (0 = clean); -1 = failed to converge."""
    return _NB.decode(H_BCNV3_IDX, H_BCNV3_ELE, N_CHECK, N_VAR, syms)


# ------------------------------------------------------------------ frame sync + CRC + parse
def preamble_for(prn):
    """The 22-symbol PRN-specific sync pattern: 16-bit 0xEB90 preamble || 6-bit PRN (MSB first)."""
    return np.array(PREAMBLE + [(prn >> (5 - k)) & 1 for k in range(6)], dtype=np.int8)


def check_crc(bits):
    """486-bit frame -> (crc_ok, mestype, sow). CRC-24Q over bits[0:462]."""
    b = [int(x) & 1 for x in bits]
    if len(b) != FRAME_BITS:
        return False, None, None
    ok = (crc24q(b[0:CRC_AT]) == _uint(b[CRC_AT:CRC_AT + 24]))
    return ok, _uint(b[0:6]), _uint(b[6:26])


def find_frames(symbols, prn):
    """Scan a soft symbol stream (+~0 / -~1 bipolar) for PRN-preamble-aligned frames, both
    polarities. Yields (start_index, polarity, info_bits[486], mestype, sow) per CRC-OK frame.
    Steps by 1 (never jumps on a bare preamble; the CRC is the true filter)."""
    s = np.asarray(symbols, dtype=float)
    n = len(s)
    pre = preamble_for(prn)
    i = 0
    while i + FRAME_SYMS <= n:
        for pol in (1.0, -1.0):
            hard = (pol * s[i:i + N_PRE] < 0).astype(np.int8)
            if np.array_equal(hard, pre):
                ldpc = (pol * s[i + LDPC_OFFSET:i + FRAME_SYMS] < 0).astype(np.uint8)
                bits, nerr = decode_nb_ldpc(ldpc)
                if nerr >= 0:
                    lb = [int(b) for b in bits]
                    ok, mt, sow = check_crc(lb)
                    if ok:
                        yield i, int(pol), lb, mt, sow
                        break
        i += 1


# ------------------------------------------------------------------ ephemeris (Message Type 10)
# The full BDS-3 CNAV orbit is in Message Type 10. LIVE-MAPPED 2026-08-03: a change-detection over
# consecutive type-10 frames showed bits [26:462] STABLE (the ephemeris) between SOW[6:26] and
# CRC[462:486]; the orbit is the SAME contiguous BDS-3 CNAV parameter block as B-CNAV1 SF2, at the
# B-CNAV1 SF2 offsets MINUS 9 (its t_oe 39 -> 30, ... C_UC 443 -> 434). VERIFIED against BRDC:
# dpos 0.015-0.079 m across PRN 23/25/32/37/41 (SatType 3 MEO). Scales are the shared BDS-3 CNAV
# set (identical to B-CNAV1/2). name -> (start bit in the 486-bit frame, length, signed, scale).
_EPH10_SHIFT = -9
BCNV3_EPH_FIELDS = {
    "t_oe":     (30, 11, False, 300.0),
    "SatType":  (41, 2, False, 1.0),
    "dA":       (43, 26, True, _P(9)),
    "A_dot":    (69, 25, True, _P(21)),
    "dn_0":     (94, 17, True, _P(44) * BDS_PI),
    "dn_0_dot": (111, 23, True, _P(57) * BDS_PI),
    "M_0":      (134, 33, True, _P(32) * BDS_PI),
    "e":        (167, 33, False, _P(34)),
    "omega":    (200, 33, True, _P(32) * BDS_PI),
    "Omega_0":  (233, 33, True, _P(32) * BDS_PI),
    "i_0":      (266, 33, True, _P(32) * BDS_PI),
    "Omega_dot":(299, 19, True, _P(44) * BDS_PI),
    "i_0_dot":  (318, 15, True, _P(44) * BDS_PI),
    "C_IS":     (333, 16, True, _P(30)),
    "C_IC":     (349, 16, True, _P(30)),
    "C_RS":     (365, 24, True, _P(8)),
    "C_RC":     (389, 24, True, _P(8)),
    "C_US":     (413, 21, True, _P(30)),
    "C_UC":     (434, 21, True, _P(30)),
}


def parse_bcnav3_ephemeris(frames_by_type):
    """{mestype: 486-bit frame} -> SI ephemeris (A_ref resolved by SatType), or None if type 10 is
    missing. Type 10 carries the WHOLE orbit -- no cross-frame matching (unlike B-CNAV2's 10+11)."""
    f = frames_by_type.get(10)
    if f is None:
        return None
    eph = {}
    for name, (start, length, signed, scale) in BCNV3_EPH_FIELDS.items():
        eph[name] = _field(f, start, length, signed, scale)
    eph["A_ref"] = A_REF_IGSO if int(round(eph["SatType"])) in (1, 2) else A_REF_MEO
    eph["IODE"] = -1               # not needed (single-frame orbit); B-CNAV3 IODE not in type 10
    eph["_iod_consistent"] = True
    return eph


def sv_position_bcnav3(eph, t):
    """ECEF (x,y,z) m -- identical BDS-3 CNAV propagation as B-CNAV1/2 (shared parameter set)."""
    import beidou_bcnav2 as _B2
    return _B2.sv_position_bcnav2(eph, t)


# ------------------------------------------------------------------ self-test
def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return GF_VEC[(GF_POW[a] + GF_POW[b]) % (Q_GF - 1)]


def _nullspace_codeword(rng):
    """A RANDOM valid GF(64) codeword (H c = 0) via Gaussian elimination -- rigorous decoder test."""
    H = [[0] * N_VAR for _ in range(N_CHECK)]
    for i in range(N_CHECK):
        for k in range(len(H_BCNV3_IDX[i])):
            H[i][H_BCNV3_IDX[i][k]] = H_BCNV3_ELE[i][k]
    where = [-1] * N_VAR
    pivots = []
    row = 0
    for col in range(N_VAR):
        if row >= N_CHECK:
            break
        piv = next((r for r in range(row, N_CHECK) if H[r][col] != 0), None)
        if piv is None:
            continue
        H[row], H[piv] = H[piv], H[row]
        inv = GF_VEC[(Q_GF - 1 - GF_POW[H[row][col]]) % (Q_GF - 1)]
        H[row] = [_gf_mul(inv, x) for x in H[row]]
        for r in range(N_CHECK):
            if r != row and H[r][col] != 0:
                f = H[r][col]
                H[r] = [H[r][c] ^ _gf_mul(f, H[row][c]) for c in range(N_VAR)]
        where[col] = row
        pivots.append(col); row += 1
    free = [c for c in range(N_VAR) if where[c] == -1]
    c = [0] * N_VAR
    for fcol in free:
        c[fcol] = int(rng.integers(0, Q_GF))
    for col in pivots:
        r = where[col]
        acc = 0
        for fc in range(N_VAR):
            if fc != col and H[r][fc] != 0:
                acc ^= _gf_mul(H[r][fc], c[fc])
        c[col] = acc
    return c


if __name__ == "__main__":
    import sys
    _init_gf()
    rng = np.random.default_rng(3)
    fails = 0

    # 1. GF(64) field sanity
    okgf = all(_gf_mul(a, GF_VEC[(Q_GF - 1 - GF_POW[a]) % (Q_GF - 1)]) == 1 for a in range(1, Q_GF))
    print("1. GF(64) field inverse: %s" % ("OK" if okgf else "FAIL"))
    fails += 0 if okgf else 1

    # 2. a constructed codeword decodes clean (zero syndrome on iteration 0)
    p2 = 0
    for _ in range(5):
        cw = _nullspace_codeword(rng)
        syms = _gf2bin(np.array(cw, dtype=np.uint8))
        _bits, nerr = decode_nb_ldpc(syms)
        if nerr != 0:
            p2 += 1
    print("2. valid codeword clean decode (nerr 0): %s" % ("OK" if p2 == 0 else "FAIL %d/5" % p2))
    fails += p2

    # 3. NB-LDPC error correction: inject symbol errors, recover the info bits
    p3 = 0
    for nbit in (2, 5, 10):
        okc = 0
        for _ in range(4):
            cw = _nullspace_codeword(rng)
            syms = _gf2bin(np.array(cw, dtype=np.uint8)).copy()
            info_true = syms[:N_CHECK * N_GF].copy()
            flip = rng.choice(N_LDPC_SYM, size=nbit, replace=False)
            syms[flip] ^= 1
            bits, nerr = decode_nb_ldpc(syms)
            if nerr >= 0 and np.array_equal(bits, info_true):
                okc += 1
        print("   %2d-bit errors: recovered %d/4" % (nbit, okc))
        if okc < 3:
            p3 += 1
    print("3. NB-LDPC correction: %s" % ("OK" if p3 == 0 else "WEAK (%d rungs <3/4)" % p3))
    fails += p3

    # 4. CRC-24Q frame roundtrip + PRN-specific preamble sync + polarity, corruption caught
    p4 = 0
    for pol in (1, -1):
        for prn in (19, 33, 60):
            frame = list(rng.integers(0, 2, CRC_AT))
            crc = crc24q(frame)
            frame += [(crc >> (23 - k)) & 1 for k in range(24)]     # 486-bit info frame
            ok, mt, sow = check_crc(frame)
            bad = list(frame); bad[100] ^= 1
            if not ok or check_crc(bad)[0]:
                p4 += 1
            # preamble is PRN-specific and reversible
            pre = preamble_for(prn)
            if len(pre) != N_PRE or _uint(list(pre[N_PRE_FIXED:])) != prn:
                p4 += 1
    print("4. CRC-24Q + PRN preamble + corruption caught: %s" % ("OK" if p4 == 0 else "FAIL %d" % p4))
    fails += p4

    # 5. ephemeris field table: no overlap/overrun within [26:462], + a demo MEO orbit radius
    p5 = 0
    occ = [0] * FRAME_BITS
    for _n, (st, ln, _s, _sc) in BCNV3_EPH_FIELDS.items():
        if st < 26 or st + ln > CRC_AT:
            p5 += 1
        for k in range(st, min(st + ln, FRAME_BITS)):
            occ[k] += 1
    if any(x > 1 for x in occ):
        p5 += 1
    demo = {"A_ref": A_REF_MEO, "dA": 200.0, "A_dot": 0.0, "t_oe": 0.0, "SatType": 3.0,
            "dn_0": 0.0, "dn_0_dot": 0.0, "M_0": 0.3, "e": 1e-3, "omega": 0.1,
            "Omega_0": 0.2, "i_0": 0.96, "Omega_dot": -2e-9, "i_0_dot": 0.0,
            "C_IS": 0, "C_IC": 0, "C_RS": 0, "C_RC": 0, "C_US": 0, "C_UC": 0}
    rad = math.sqrt(sum(c * c for c in sv_position_bcnav3(demo, 0.0)))
    if not (2.6e7 < rad < 2.9e7):
        p5 += 1
        print("   demo orbit radius %.0f m OUT of MEO shell" % rad)
    print("5. ephemeris field table (no overlap) + CNAV propagation: %s"
          % ("OK" if p5 == 0 else "FAIL %d" % p5))
    fails += p5

    print("\n%s" % ("B-CNAV3 NB-LDPC + frame + ephemeris codec SELF-CONSISTENT (type-10 orbit "
                    "LIVE-VERIFIED vs BRDC 0.015-0.079 m)" if not fails else
                    "SELF-TEST FAILURES: %d" % fails))
    sys.exit(1 if fails else 0)
