#!/usr/bin/env python3
"""Galileo E6-B / E6-C spreading-code bit-exactness check (EU E6-B/C Codes Technical Note).

Regression-checks lib/stages/gnss/galileoE6Code.cpp WITHOUT needing gnss-sdr or PocketSDR on
the box: it re-decodes the hex tables straight out of the .cpp with an INDEPENDENT reimpl of
the bit unpacking, then asserts structural invariants + stored fingerprints.

★ THE INVARIANT THAT PINS THE PADDING (the only subtle thing about E6): the hex is 1279 chars
= 5116 bits for a 5115-chip code, so exactly one bit is padding. Chars 0..1277 give all four
bits (5112) and char 1278 gives only its TOP THREE -- gnss-sdr's reader is the authority
("last bit is filled up with a zero"). The check: EVERY one of the 100 primary codes is
balanced at exactly 2558 ones / 2557 zeros (sum +1 over an odd length). Take 5116 bits and
every code reads +2; take 5114 and the balance scatters. Uniform +1 across all 100 is only
consistent with the correct extraction -- which is why this is a real test, not a tautology.

    python3 e6_code_check.py     # exits non-zero on any mismatch
"""
import os
import re
import sys
import zlib

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "lib", "stages", "gnss", "galileoE6Code.cpp")

N_PRN = 50
CHIPS = 5115
SEC_CHIPS = 100

# (crc32 of the +/-1 code as a "01" string, popcount of +1) -- from the transcribed tables.
_FP = {
    "E6B_HEX": {1: (205759734, 2558), 7: (2199043268, 2558),
                19: (3558976337, 2558), 36: (2264841301, 2558)},
    "E6C_HEX": {1: (578628357, 2558), 7: (1079021178, 2558),
                19: (4085486462, 2558), 36: (1506237582, 2558)},
    "E6C_SEC_HEX": {1: (4688809, 46), 7: (2603738921, 48),
                    19: (429929171, 54), 36: (2647661557, 53)},
}


def _tables(text, name, per):
    i = text.index("const char* const %s[%d]" % (name, N_PRN))
    j = text.index("};", i)
    out, cur = [], ""
    for s in re.findall(r'"([0-9A-Fa-f]+)"', text[i:j]):
        cur += s
        if len(cur) >= per:
            out.append(cur)
            cur = ""
    return out


def _decode(hexs, n):
    """Independent reimpl of galileoE6Code.cpp::decode -- MSB-first, bit 0 -> +1, 1 -> -1."""
    out = []
    for i in range(n):
        v = int(hexs[i // 4], 16)
        out.append(-1 if (v >> (3 - (i % 4))) & 1 else 1)
    return out


def main():
    text = open(os.path.normpath(CPP)).read()
    ok = True
    for name, n, per in (("E6B_HEX", CHIPS, 1279), ("E6C_HEX", CHIPS, 1279),
                         ("E6C_SEC_HEX", SEC_CHIPS, 25)):
        tab = _tables(text, name, per)
        if len(tab) != N_PRN or any(len(x) != per for x in tab):
            print("%s: BAD table shape (%d entries)" % (name, len(tab)))
            ok = False
            continue
        codes = [_decode(x, n) for x in tab]
        if len({tuple(c) for c in codes}) != N_PRN:
            print("%s: codes NOT distinct" % name)
            ok = False
        if n == CHIPS:
            bal = {sum(c) for c in codes}
            if bal != {1}:
                print("%s: balance %s, expected {1} -- PADDING CONVENTION WRONG" % (name, bal))
                ok = False
            else:
                print("%s: 50 codes, all distinct, balance +1 (2558/2557) -- padding OK" % name)
        else:
            print("%s: 50 codes, all distinct, %d chips" % (name, n))
        for prn, (crc, pop) in sorted(_FP[name].items()):
            c = codes[prn - 1]
            got = (zlib.crc32("".join("1" if x > 0 else "0" for x in c).encode()),
                   sum(1 for x in c if x > 0))
            status = "OK" if got == (crc, pop) else "MISMATCH (stored %s)" % ((crc, pop),)
            if got != (crc, pop):
                ok = False
            print("   PRN %2d: crc=0x%08x pop=%d %s" % (prn, got[0], got[1], status))
    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
