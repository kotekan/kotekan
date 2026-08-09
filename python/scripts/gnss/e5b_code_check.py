#!/usr/bin/env python3
"""Galileo E5b-I / E5b-Q spreading-code bit-exactness check (Galileo OS SIS ICD).

Mirrors lib/stages/gnss/galileoE5bCode.cpp exactly (same shared 14-stage Fibonacci lfsr14,
all-ones X1, per-PRN X2 start tables, and the pre-resolved X1/X2 taps) and asserts it against
PocketSDR-verified fingerprints, so the C++ transcription can be regression-checked WITHOUT
PocketSDR present. If PocketSDR's sdr_code is importable (e.g. on PATH), it additionally
cross-checks bit-exact against gen_code('E5BI'/'E5BQ', prn) for all 50 PRNs.

E5b shares 1207.14 MHz with BeiDou B2b, so one mid-band capture validates both. Same discipline
as the E5a/B2b tables: an ICD transcription (via PocketSDR T.Takasu) asserted here and finally
proven by live acquisition off the retuned 1207.14 MHz dongle (mid_band_decode.py --sig e5b).

    python3 e5b_code_check.py     # exits non-zero on any mismatch
"""
import sys
import zlib

CODE_LENGTH = 10230

# Pre-resolved LFSR taps (bit-reversed ICD polynomials), == galileoE5bCode.cpp.
X1_TAP = 0x040B    # shared X1 tap (ICD 0o64021)
E5BI_X2_TAP = 0x1265  # E5b-I X2 tap (ICD 0o51445)
E5BQ_X2_TAP = 0x2331  # E5b-Q X2 tap (ICD 0o43143)

# Per-PRN X2 START values (bit-reversed ICD Annex C), == galileoE5bCode.cpp tables.
E5BI_X2_START = (
    0x025C, 0x390D, 0x1540, 0x1B9E, 0x2386, 0x01A8, 0x3EB0, 0x320D,
    0x2AF0, 0x3E61, 0x0BC8, 0x26FD, 0x0163, 0x0F87, 0x0B2C, 0x1F7C,
    0x264B, 0x35C2, 0x2DE0, 0x3F0D, 0x0972, 0x3EB2, 0x2C97, 0x1521,
    0x2967, 0x0B75, 0x3031, 0x354B, 0x3EC8, 0x3FE3, 0x129D, 0x3FC9,
    0x2F8F, 0x2F1B, 0x259B, 0x0E9A, 0x363C, 0x1C42, 0x1B37, 0x36A8,
    0x25D4, 0x2283, 0x0406, 0x1FAB, 0x1AC3, 0x363F, 0x120D, 0x07F7,
    0x076C, 0x3C0A,
)
E5BQ_X2_START = (
    0x26D8, 0x318C, 0x12D5, 0x27D9, 0x3420, 0x2E4F, 0x05FE, 0x29E4,
    0x281A, 0x019F, 0x2D83, 0x3E41, 0x2338, 0x1373, 0x199D, 0x39B4,
    0x1D28, 0x3403, 0x12C0, 0x23E2, 0x3BED, 0x080F, 0x34E3, 0x137C,
    0x0471, 0x0E81, 0x1352, 0x18E2, 0x3187, 0x2512, 0x0335, 0x2B7E,
    0x0DCB, 0x0651, 0x1981, 0x13D2, 0x195F, 0x09CE, 0x0E16, 0x2735,
    0x1975, 0x3D89, 0x1EC4, 0x1DCE, 0x2F04, 0x1A68, 0x18F2, 0x241F,
    0x211D, 0x0A13,
)

# PocketSDR-verified fingerprints: (crc32 of the +/-1 code as a "01" bitstring, popcount of +1).
_FP_E5BI = {
    1: (858306990, 5157), 19: (3597122107, 5143), 25: (1377709193, 5093), 50: (2948600139, 5155),
}
_FP_E5BQ = {
    1: (3178291707, 5141), 19: (19817131, 5131), 25: (194131014, 5129), 50: (4058147432, 5089),
}


def _lfsr14(count, reg, tap):
    out = [0] * count
    for i in range(count):
        out[i] = -1 if (reg & 1) else 1
        fb = bin(reg & tap).count("1") & 1
        reg = (fb << 13) | (reg >> 1)
    return out


_X1 = _lfsr14(CODE_LENGTH, 0x3FFF, X1_TAP)  # all-ones start, shared by both components


def _gen(prn, tap, start):
    if prn < 1 or prn > 50:
        raise ValueError("Galileo E5b PRN must be 1..50")
    x2 = _lfsr14(CODE_LENGTH, start[prn - 1], tap)
    return [a * b for a, b in zip(_X1, x2)]


def generate_e5bi_code(prn):
    """E5b-I (data) primary code (PRN 1..50) as a list of +-1 chips -- mirrors galileoE5bCode.cpp."""
    return _gen(prn, E5BI_X2_TAP, E5BI_X2_START)


def generate_e5bq_code(prn):
    """E5b-Q (pilot) primary code (PRN 1..50) as a list of +-1 chips -- mirrors galileoE5bCode.cpp."""
    return _gen(prn, E5BQ_X2_TAP, E5BQ_X2_START)


def _fp(code):
    bits = "".join("1" if x > 0 else "0" for x in code)
    return (zlib.crc32(bits.encode()) & 0xFFFFFFFF, sum(1 for x in code if x > 0))


# -- E5b-Q CS100 SECONDARY -----------------------------------------------------------------
#
# ⚠️ WHAT THIS CAN AND CANNOT PROVE. The per-PRN CS100 overlay in galileoE5bCode.cpp was, until
# 2026-08-09, asserted by NOTHING: the primaries above have this bit-exactness gate for all 50
# PRNs, while CS100_E5BQ_HEX appeared in exactly one place -- its own definition -- with a
# comment claiming provenance (PocketSDR CS100[50:100]) and no check behind it. That is the
# failure mode that hides longest, because a wrong 100-chip overlay leaves the 1 ms primary
# correlation intact and only destroys integration BEYOND one code period.
#
# The fingerprints below are computed FROM THE CURRENT TABLE, so they are a REGRESSION gate --
# they catch an edit, a re-paste or a half-applied patch. They are NOT a correctness proof, and
# calling them one would be exactly the "gate that cannot fail" this project keeps re-learning.
#
# THE CORRECTNESS EVIDENCE IS THE SKY, and it is per-PRN. The deep window is ~1.05 s = 1000
# primary periods = 10 full CS100 cycles, so a wrong secondary cannot produce a coherent
# detection: it would cancel itself. Every PRN in SKY_CONFIRMED below reached deep_snr > 20 at
# coh_frac > 0.90 on the 2026-08-09 soak, i.e. its overlay is right. PRN 29 peaked at deep 670
# with coh 1.00. That is stronger evidence than any table comparison, and it only became
# available once the second band could track at all (task #34).
CS100_TABLE_CRC32 = 0xc19c50ab       # crc32 of all 50 hex strings concatenated
CS100_FP = {                          # prn: (crc32 of the 100-bit string, popcount of 1s)
    1: (0x88582830, 54), 7: (0x6976bba6, 54), 13: (0x9c50c1b6, 53), 19: (0xbf89a421, 53),
    25: (0xcdc2e950, 54), 29: (0xb65e472d, 54), 33: (0xf83bfb17, 54), 50: (0x1dd70a1b, 48),
}
SKY_CONFIRMED = [2, 7, 10, 11, 12, 19, 26, 27, 28, 29, 30, 33]
_CPP = "lib/stages/gnss/galileoE5bCode.cpp"
_CPP_A = "lib/stages/gnss/galileoE5aCode.cpp"


def _cs100_table(path, name):
    """Parse a CS100 hex table straight out of the C++ -- never transcribe it a second time."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    src = open(os.path.join(root, path)).read()
    i = src.index(name)
    j = src.index("};", i)
    import re as _re
    return _re.findall(r'"([0-9A-Fa-f]+)"', src[i:j])


def check_cs100():
    fails = 0
    try:
        b = _cs100_table(_CPP, "CS100_E5BQ_HEX[50]")
        a = _cs100_table(_CPP_A, "CS100_HEX[50]")
    except Exception as e:                       # noqa: BLE001
        print("E5bQ CS100: cannot read the C++ table (%s)" % e)
        return 1

    def bits(h):
        return bin(int(h, 16))[2:].zfill(100)

    def say(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print("  %-46s %s%s" % (name, "OK" if ok else "FAIL", "  " + detail if detail else ""))

    print("")
    print("E5b-Q CS100 secondary (per-PRN, 100 chips):")
    say("50 entries, 25 hex chars each",
        len(b) == 50 and {len(x) for x in b} == {25}, "%d entries" % len(b))
    say("all 50 distinct", len(set(b)) == 50)
    say("disjoint from the E5a-Q table", not (set(a) & set(b)),
        "shared: %s" % sorted(set(a) & set(b))[:3])
    pops = [bits(x).count("1") for x in b]
    say("balanced (popcount within 45..55)", min(pops) >= 45 and max(pops) <= 55,
        "range %d..%d" % (min(pops), max(pops)))
    # Near-orthogonality: a duplicated or shifted-in entry survives every check above but
    # shows here. For random +-1 length-100 sequences sd = 10, so over 1225 pairs an extreme
    # of ~3.5 sd is EXPECTED -- the bar is set where a genuine collision (|corr| -> 100) lands.
    worst = 0
    for i in range(50):
        bi = bits(b[i])
        for j in range(i + 1, 50):
            bj = bits(b[j])
            c = abs(sum(1 if bi[k] == bj[k] else -1 for k in range(100)))
            worst = max(worst, c)
    say("mutually near-orthogonal (no collision)", worst <= 50, "worst |corr| %d/100" % worst)
    import zlib as _z
    say("whole-table crc32 unchanged (regression gate)",
        (_z.crc32("".join(b).encode()) & 0xFFFFFFFF) == CS100_TABLE_CRC32,
        "0x%08x" % (_z.crc32("".join(b).encode()) & 0xFFFFFFFF))
    for prn, (crc, pop) in sorted(CS100_FP.items()):
        st = bits(b[prn - 1])
        got = (_z.crc32(st.encode()) & 0xFFFFFFFF, st.count("1"))
        say("PRN %2d fingerprint" % prn, got == (crc, pop),
            "0x%08x pop=%d" % got)
    print("  sky-confirmed overlays (deep>20 at coh>0.90, 2026-08-09 soak): %s"
          % SKY_CONFIRMED)
    print("  -> %d/50 PRNs proven ON SKY; the rest are regression-gated only."
          % len(SKY_CONFIRMED))
    return fails


def main():
    fails = 0
    try:
        import sdr_code  # PocketSDR, optional
        pocket = True
    except Exception:
        pocket = False
    for comp, gen, table, sig in (("E5bI", generate_e5bi_code, _FP_E5BI, "E5BI"),
                                  ("E5bQ", generate_e5bq_code, _FP_E5BQ, "E5BQ")):
        for prn in range(1, 51):
            code = gen(prn)
            assert len(code) == CODE_LENGTH
            checked = False
            line = "%s PRN %2d:" % (comp, prn)
            if prn in table:
                got = _fp(code)
                ok = got == table[prn]
                fails += 0 if ok else 1
                line += " crc=0x%08x pop=%d  vs stored: %s" % (
                    got[0], got[1], "OK" if ok else "MISMATCH")
                checked = True
            if pocket:
                ref = list(sdr_code.gen_code(sig, prn))
                bx = ref == code
                fails += 0 if bx else 1
                line += "  | PocketSDR: %s" % ("BIT-EXACT" if bx else "MISMATCH")
                checked = True
            if checked and (prn in table or not (prn % 10)):
                print(line)
    fails += check_cs100()
    print("\n%s" % ("ALL OK" if fails == 0 else "FAIL (%d mismatches)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
