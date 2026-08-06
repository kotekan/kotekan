#!/usr/bin/env python3
"""BeiDou B1I / B3I ranging-code check (BDS-SIS-ICD-B1I v3.0 / -B3I v1.0).

Regression-checks lib/stages/gnss/beidouB1ICode.cpp and beidouB3ICode.cpp by re-running the
ICD LFSR constructions here, INDEPENDENTLY of the C++, and asserting structural invariants plus
stored fingerprints. Needs neither gnss-sdr nor PocketSDR (neither is installed on this box).

★ Both signals are LEGACY BeiDou, ALGORITHMIC (not memory codes like B1C/B2a/B2b), 1 ms period,
DATA-ONLY (no pilot), and both carry the SAME 20-chip D1 NH secondary 00000100110101001110.

⚠️ THE TWO TRAPS THIS GUARDS, one per signal:
  * B1I -- the G2 output tap is TWO taps XORed for PRN 1..37 and **THREE** for PRN 38..63. Most
    published tables stop at 37, so a transcription that stops there produces garbage for
    exactly the BDS-3 satellites (the ones actually overhead). Checked by asserting PRN 38..63
    are distinct from each other AND balanced -- a dropped third tap breaks both.
  * B3I -- G1 is NOT free-running: its natural period exceeds the 10230-chip code, so the ICD
    RESETS it to all-ones on reaching 0011111111111. Omit the reset and the first ~8190 chips
    are still correct while the tail diverges, giving a code that correlates WEAKLY instead of
    not at all -- the hardest failure to attribute. Checked by the fingerprints, which cover
    the whole 10230 and so cannot pass without the reset.

⚠️ These are STRUCTURAL checks, not a bit-exact comparison against an external reference. The
real validation is a sky lock: coherent tracking requires the code, its rate and its secondary
to all be right at once (that is what settled Galileo E6 -- see galileoE6Code.hpp).

    python3 b1i_b3i_code_check.py     # exits non-zero on any mismatch
"""
import os
import re
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "lib", "stages", "gnss"))

B1I_N, B3I_N, N_PRN = 2046, 10230, 63

# (crc32 of the code as a "01" string with +1 -> "1", popcount of +1)
_FP_B1I = {1: (2885436505, 1023), 7: (1143448711, 1022), 19: (3041313231, 1023),
           38: (4062613400, 1023), 63: (1363507217, 1023)}
_FP_B3I = {1: (2623635003, 5105), 7: (2160214820, 5105), 19: (858640360, 5113),
           38: (4140033058, 5173), 63: (1481222794, 5095)}

P1 = [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6,
      6, 8, 8, 8, 9, 9, 10, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 8, 9, 9, 3, 5, 7, 4, 4, 5, 5,
      5, 5, 6]
P2 = [3, 4, 5, 6, 8, 9, 10, 11, 7, 4, 5, 6, 8, 9, 10, 11, 5, 6, 8, 9, 10, 11, 6, 8, 9, 10, 11, 8,
      9, 10, 11, 9, 10, 11, 10, 11, 11, 7, 4, 6, 8, 10, 11, 5, 9, 6, 8, 10, 11, 9, 9, 10, 11, 7,
      7, 9, 5, 9, 6, 8, 10, 11, 9]
P3 = [0] * 37 + [1] * 16 + [2] * 3 + [3] * 7


def b1i(prn):
    """B1I code as +-1, mirroring beidouB1ICode.cpp (0 -> +1)."""
    i = prn - 1
    g1 = [0] * 11
    g2 = [0] * 11
    for k in range(11):                       # "01010101010" -> reg[10-k] = s[k]
        b = 1 if (10 - k) % 2 == 1 else 0
        g1[10 - k] = b
        g2[10 - k] = b
    out = []
    for _ in range(B1I_N):
        go = g2[11 - P1[i]] ^ g2[11 - P2[i]]
        if P3[i]:
            go ^= g2[11 - P3[i]]              # the PRN 38..63 third tap
        out.append(-1 if (g1[0] ^ go) else 1)
        f1 = g1[0] ^ g1[1] ^ g1[2] ^ g1[3] ^ g1[4] ^ g1[10]
        f2 = g2[0] ^ g2[2] ^ g2[3] ^ g2[6] ^ g2[7] ^ g2[8] ^ g2[9] ^ g2[10]
        g1[:10], g2[:10] = g1[1:], g2[1:]
        g1[10], g2[10] = f1, f2
    return out


def _b3i_init():
    txt = open(os.path.join(CPP, "beidouB3ICode.cpp")).read()
    i = txt.index("B3I_G2_INIT[63]")
    j = txt.index("};", i)
    return re.findall(r'"([01]{13})"', txt[i:j])


def b3i(prn, init=None):
    """B3I code as +-1, mirroring beidouB3ICode.cpp (0 -> +1), G1 reset included."""
    s = (init or _b3i_init())[prn - 1]
    g1 = [1] * 13
    g2 = [0] * 13
    for k in range(13):
        g2[12 - k] = 1 if s[k] == "1" else 0
    out = []
    for _ in range(B3I_N):
        out.append(-1 if (g1[0] ^ g2[0]) else 1)
        f1 = g1[0] ^ g1[9] ^ g1[10] ^ g1[12]
        f2 = g2[0] ^ g2[1] ^ g2[3] ^ g2[4] ^ g2[6] ^ g2[7] ^ g2[8] ^ g2[12]
        g1[:12], g2[:12] = g1[1:], g2[1:]
        g1[12], g2[12] = f1, f2
        if g1[0] == 0 and g1[1] == 0 and all(g1[k] == 1 for k in range(2, 13)):
            g1 = [1] * 13                     # ★ the ICD reset -- see the header
    return out


def _fp(code):
    return (zlib.crc32("".join("1" if x > 0 else "0" for x in code).encode()),
            sum(1 for x in code if x > 0))


def main():
    ok = True
    init = _b3i_init()
    if len(init) != N_PRN:
        print("B3I: expected %d G2 initial states, found %d" % (N_PRN, len(init)))
        return 1
    for name, fn, n, fps, bal_lim in (("B1I", b1i, B1I_N, _FP_B1I, 4),
                                      ("B3I", lambda p: b3i(p, init), B3I_N, _FP_B3I, 400)):
        codes = [fn(p) for p in range(1, N_PRN + 1)]
        if len({tuple(c) for c in codes}) != N_PRN:
            print("%s: codes NOT distinct" % name)
            ok = False
        bal = [sum(c) for c in codes]
        if max(abs(b) for b in bal) > bal_lim:
            print("%s: balance out of range: %d..%d" % (name, min(bal), max(bal)))
            ok = False
        # the BDS-3 half specifically (B1I's three-tap rows / B3I's high PRNs)
        hi = codes[37:]
        if len({tuple(c) for c in hi}) != len(hi):
            print("%s: PRN 38..63 not distinct -- suspect the extended tap/init table" % name)
            ok = False
        print("%s: %d codes, all distinct (incl. PRN 38..63), balance %d..%d, len %d"
              % (name, N_PRN, min(bal), max(bal), n))
        for prn, want in sorted(fps.items()):
            got = _fp(codes[prn - 1])
            if got != want:
                print("   PRN %2d: crc=0x%08x pop=%d MISMATCH (stored %s)" % (prn, got[0], got[1], want))
                ok = False
            else:
                print("   PRN %2d: crc=0x%08x pop=%d OK" % (prn, got[0], got[1]))
    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
