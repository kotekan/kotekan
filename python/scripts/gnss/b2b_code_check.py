#!/usr/bin/env python3
"""BeiDou-3 B2b-I spreading-code bit-exactness check (BDS-SIS-ICD-B2b / PPP-B2b).

Mirrors lib/stages/gnss/beidouB2bCode.cpp exactly (same 13-stage Fibonacci LFSRs, taps and
per-PRN G2 start table) and asserts it against PocketSDR-verified fingerprints, so the C++
transcription can be regression-checked WITHOUT PocketSDR present. If PocketSDR's sdr_code is
importable (e.g. on PATH), it additionally cross-checks bit-exact against gen_code('B2BI', prn).

Same discipline as the B2a/B1C tables: the code itself is an ICD transcription (via PocketSDR
T.Takasu), so its correctness is asserted here and finally proven by live acquisition off the
retuned 1207.14 MHz dongle.

    python3 b2b_code_check.py     # exits non-zero on any mismatch
"""
import sys
import zlib

CODE_LENGTH = 10230

# Per-PRN G2 register start values (== beidouB2bCode.cpp B2BI_G2_START == PocketSDR B2BI_G2_init).
B2BI_G2_START = (
    0b1000000100101, 0b1000000110100, 0b1000010101101, 0b1000101001111,
    0b1000101010101, 0b1000110101110, 0b1000111101110, 0b1000111111011,
    0b1001100101001, 0b1001111011010, 0b1010000110101, 0b1010001000100,
    0b1010001010101, 0b1010001011011, 0b1010001011100, 0b1010010100011,
    0b1010011110111, 0b1010100000001, 0b1010100111110, 0b1010110101011,
    0b1010110110001, 0b1011001010011, 0b1011001100010, 0b1011010011000,
    0b1011010110110, 0b1011011110010, 0b1011011111111, 0b1011100010010,
    0b1011100111100, 0b1011110100001, 0b1011111001000, 0b1011111010100,
    0b1011111101011, 0b1011111110011, 0b1100001010001, 0b1100010010100,
    0b1100010110111, 0b1100100010001, 0b1100100011001, 0b1100110101011,
    0b1100110110001, 0b1100111010010, 0b1101001010101, 0b1101001110100,
    0b1101011001011, 0b1101101010111, 0b1110000110100, 0b1110010000011,
    0b1110010001011, 0b1110010100011, 0b1110010101000, 0b1110100111011,
    0b1110110010111, 0b1111001001000, 0b1111010010100, 0b1111010011001,
    0b1111011011010, 0b1111011111000, 0b1111011111111, 0b1111110110101,
    0b1111110111101, 0b0101110000101, 0b0101100111011,
)

# PocketSDR-verified fingerprints: (crc32 of the +/-1 code as a "01" bitstring, popcount of +1).
_FINGERPRINT = {
    1: (0x141e0447, 5161), 19: (0x2b36a621, 5212), 20: (0x7470642c, 5101),
    30: (0x1e423fbd, 5092), 45: (0x64b214fc, 5054), 63: (0xfc033b9d, 5026),
}


def _lfsr13(count, reg, tap):
    out = [0] * count
    for i in range(count):
        out[i] = -1 if (reg & 1) else 1
        fb = bin(reg & tap).count("1") & 1
        reg = (fb << 12) | (reg >> 1)
    return out


def _g1():
    base = _lfsr13(8190, 0b1111111111111, 0b1000000011001)
    return base + base[: CODE_LENGTH - 8190]


_G1 = _g1()


def generate_b2bi_code(prn):
    """B2b-I primary code (PRN 1..63) as a list of +-1 chips -- mirrors beidouB2bCode.cpp."""
    if prn < 1 or prn > 63:
        raise ValueError("BeiDou B2b PRN must be 1..63")
    g2 = _lfsr13(CODE_LENGTH, B2BI_G2_START[prn - 1], 0b0011010010011)
    return [a * b for a, b in zip(_G1, g2)]


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
    for prn in range(1, 64):
        code = generate_b2bi_code(prn)
        assert len(code) == CODE_LENGTH
        if prn in _FINGERPRINT:
            got = _fp(code)
            ok = got == _FINGERPRINT[prn]
            fails += 0 if ok else 1
            line = "PRN %2d: crc=0x%08x pop=%d  vs stored: %s" % (
                prn, got[0], got[1], "OK" if ok else "MISMATCH")
            if pocket:
                ref = list(sdr_code.gen_code("B2BI", prn))
                bx = ref == code
                fails += 0 if bx else 1
                line += "  | PocketSDR: %s" % ("BIT-EXACT" if bx else "MISMATCH")
            print(line)
    print("\n%s" % ("ALL OK" if fails == 0 else "FAIL (%d mismatches)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
