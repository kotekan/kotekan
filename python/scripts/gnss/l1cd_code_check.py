#!/usr/bin/env python3
"""GPS L1C-D (data) spreading-code bit-exactness check (IS-GPS-800).

Mirrors lib/stages/gnss/gpsL1CCode.cpp::generate_l1cd_code (the shared L1C Weil construction --
Legendre L(t)*L(t+w) + a 7-chip insertion at p -- with the L1C-D per-PRN indices) and asserts it
against self-contained fingerprints, so the C++ transcription can be regression-checked WITHOUT
PocketSDR. If PocketSDR's sdr_code is importable, it additionally cross-checks bit-exact against
gen_code_L1CPD for all 32 PRNs. The RAW ranging code (pre-BOC) is what the C++ returns; the
BOC(1,1) modulation is applied downstream in the replica bank.

    python3 l1cd_code_check.py     # exits non-zero on any mismatch
"""
import sys
import zlib

N = 10230
N_L = 10223

# IS-GPS-800 L1C-D Weil + insertion indices (== gpsL1CCode.cpp L1CD_WEIL / L1CD_INS), PRN 1..32.
L1CD_WEIL = (5097, 5110, 5079, 4403, 4121, 5043, 5042, 5104, 4940, 5035, 4372, 5064,
             5084, 5048, 4950, 5019, 5076, 3736, 4993, 5060, 5061, 5096, 4983, 4783,
             4991, 4815, 4443, 4769, 4879, 4894, 4985, 5056)
L1CD_INS = (181, 359, 72, 1110, 1480, 5034, 4622, 1, 4547, 826, 6284, 4195,
            368, 1, 4796, 523, 151, 713, 9850, 5734, 34, 6142, 190, 644,
            467, 5384, 801, 594, 4450, 9437, 4307, 5906)
_INS7 = (1, -1, -1, 1, -1, 1, 1)  # "0110100" bipolar

# PocketSDR-verified fingerprints: (crc32 of the +/-1 code as a "01" bitstring, popcount of +1).
_FINGERPRINT = {
    1: (29149742, 5115), 7: (1054620445, 5115), 19: (4184466511, 5115), 32: (3058363443, 5115),
}


def _legendre():
    L = [1] * N_L
    for i in range(1, N_L):
        L[(i * i) % N_L] = -1
    return L


_L = _legendre()


def generate_l1cd_code(prn):
    """L1C-D raw ranging code (PRN 1..32) as a list of +-1 chips -- mirrors gpsL1CCode.cpp."""
    if prn < 1 or prn > 32:
        raise ValueError("GPS L1C-D PRN must be 1..32")
    w, p = L1CD_WEIL[prn - 1], L1CD_INS[prn - 1]
    code = [0] * N
    for t in range(0, p - 1):
        code[t] = _L[t] * _L[(t + w) % N_L]
    for t in range(p - 1, p + 6):
        code[t] = _INS7[t - p + 1]
    for t in range(p + 6, N):
        code[t] = _L[t - 7] * _L[(t - 7 + w) % N_L]
    return code


def _fp(code):
    bits = "".join("1" if x > 0 else "0" for x in code)
    return (zlib.crc32(bits.encode()) & 0xFFFFFFFF, sum(1 for x in code if x > 0))


def main():
    fails = 0
    try:
        import sdr_code  # PocketSDR, optional
        pocket = True
    except Exception:
        pocket = False
    for prn in range(1, 33):
        code = generate_l1cd_code(prn)
        assert len(code) == N
        line = "PRN %2d:" % prn
        checked = False
        if prn in _FINGERPRINT:
            got = _fp(code)
            ok = got == _FINGERPRINT[prn]
            fails += 0 if ok else 1
            line += " crc=0x%08x pop=%d vs stored: %s" % (got[0], got[1], "OK" if ok else "MISMATCH")
            checked = True
        if pocket:
            ref = list(sdr_code.gen_code_L1CPD(N, L1CD_WEIL[prn - 1], L1CD_INS[prn - 1]))
            bx = ref == code
            fails += 0 if bx else 1
            line += "  | PocketSDR: %s" % ("BIT-EXACT" if bx else "MISMATCH")
            checked = True
        if checked and (prn in _FINGERPRINT or prn % 8 == 0):
            print(line)
    print("\n%s" % ("ALL OK" if fails == 0 else "FAIL (%d mismatches)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
