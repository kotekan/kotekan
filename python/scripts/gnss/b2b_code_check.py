#!/usr/bin/env python3
"""BeiDou-3 B2b-I spreading-code bit-exactness check (BDS-SIS-ICD-B2b / PPP-B2b).

Mirrors lib/stages/gnss/beidouB2bCode.cpp (same 13-stage Fibonacci LFSRs, taps and per-PRN G2
start table) and asserts the mirror against PocketSDR-verified fingerprints, so it runs without
PocketSDR present. If a C++ compiler is available, the module is compiled with a throwaway main()
and all 63 of its codes must equal the mirror chip for chip, so a typo in the C++ table fails
here; a compile failure fails too. If PocketSDR's sdr_code is importable (on PYTHONPATH), the
fingerprinted PRNs are also cross-checked bit-exact against gen_code('B2BI', prn).

As with the B2a and B1C tables, the code is an ICD transcription (via PocketSDR, T. Takasu)
asserted here; live acquisition is the final proof.

    python3 b2b_code_check.py     # exits non-zero on any mismatch
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import kotekan_src

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = kotekan_src.gnss_cpp()
CPP = os.path.join(CPP_DIR, "beidouB2bCode.cpp")

CODE_LENGTH = 10230

# Per-PRN G2 register start values (== beidouB2bCode.cpp B2BI_G2_START == PocketSDR B2BI_G2_init).
B2BI_G2_START = (
    0b1000000100101,
    0b1000000110100,
    0b1000010101101,
    0b1000101001111,
    0b1000101010101,
    0b1000110101110,
    0b1000111101110,
    0b1000111111011,
    0b1001100101001,
    0b1001111011010,
    0b1010000110101,
    0b1010001000100,
    0b1010001010101,
    0b1010001011011,
    0b1010001011100,
    0b1010010100011,
    0b1010011110111,
    0b1010100000001,
    0b1010100111110,
    0b1010110101011,
    0b1010110110001,
    0b1011001010011,
    0b1011001100010,
    0b1011010011000,
    0b1011010110110,
    0b1011011110010,
    0b1011011111111,
    0b1011100010010,
    0b1011100111100,
    0b1011110100001,
    0b1011111001000,
    0b1011111010100,
    0b1011111101011,
    0b1011111110011,
    0b1100001010001,
    0b1100010010100,
    0b1100010110111,
    0b1100100010001,
    0b1100100011001,
    0b1100110101011,
    0b1100110110001,
    0b1100111010010,
    0b1101001010101,
    0b1101001110100,
    0b1101011001011,
    0b1101101010111,
    0b1110000110100,
    0b1110010000011,
    0b1110010001011,
    0b1110010100011,
    0b1110010101000,
    0b1110100111011,
    0b1110110010111,
    0b1111001001000,
    0b1111010010100,
    0b1111010011001,
    0b1111011011010,
    0b1111011111000,
    0b1111011111111,
    0b1111110110101,
    0b1111110111101,
    0b0101110000101,
    0b0101100111011,
)

# PocketSDR-verified fingerprints: (crc32 of the +/-1 code as a "01" bitstring, popcount of +1).
_FINGERPRINT = {
    1: (0x141E0447, 5161),
    19: (0x2B36A621, 5212),
    20: (0x7470642C, 5101),
    30: (0x1E423FBD, 5092),
    45: (0x64B214FC, 5054),
    63: (0xFC033B9D, 5026),
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


def _cpp_codes():
    """Compile the module with a throwaway main() and read back every code. Returns
    {prn: [chips]}, None if no compiler is available, or a string saying why the compile or
    the dump failed."""
    cxx = next((c for c in ("g++-12", "g++", "clang++") if shutil.which(c)), None)
    if cxx is None:
        return None
    main = r"""
#include "beidouB2bCode.hpp"
#include <cstdio>
int main() {
    for (int prn = 1; prn <= 63; ++prn) {
        std::printf("%d ", prn);
        for (int8_t v : beidou::generate_b2bi_code(prn)) std::putchar(v > 0 ? '1' : '0');
        std::putchar('\n');
    }
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src, exe = os.path.join(td, "d.cpp"), os.path.join(td, "d")
        with open(src, "w") as f:
            f.write(main)
        r = subprocess.run(
            [cxx, "-std=c++17", "-O1", "-I", CPP_DIR, src, CPP, "-o", exe],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return "compile failed\n%s" % r.stderr[:800]
        r = subprocess.run([exe], capture_output=True, text=True)
        if r.returncode != 0:
            return "dump exited %d\n%s" % (r.returncode, r.stderr[:800])
    codes = {}
    for line in r.stdout.splitlines():
        prn, bits = line.split()
        codes[int(prn)] = [1 if b == "1" else -1 for b in bits]
    return codes


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
                prn,
                got[0],
                got[1],
                "OK" if ok else "MISMATCH",
            )
            if pocket:
                ref = list(sdr_code.gen_code("B2BI", prn))
                bx = ref == code
                fails += 0 if bx else 1
                line += "  | PocketSDR: %s" % ("BIT-EXACT" if bx else "MISMATCH")
            print(line)
    cpp = _cpp_codes()
    # GNSS_REQUIRE_CXX (the test runner sets it): this host must compile the C++, so no
    # compiler fails rather than skips.
    if cpp is None and os.environ.get("GNSS_REQUIRE_CXX"):
        print("C++: no compiler, and GNSS_REQUIRE_CXX is set")
        fails += 1
    elif cpp is None:
        print("C++: no compiler, bit-exact check SKIPPED")
    elif isinstance(cpp, str):
        print("C++: %s" % cpp)
        fails += 1
    else:
        bad = [p for p in range(1, 64) if cpp.get(p) != generate_b2bi_code(p)]
        fails += len(bad)
        print(
            "C++: %s"
            % ("PRNs %s DIFFER from the mirror" % bad if bad else "all 63 BIT-EXACT")
        )
    print("\n%s" % ("ALL OK" if fails == 0 else "FAIL (%d mismatches)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
