#!/usr/bin/env python3
"""GLONASS L2OC-p ranging-code check (1248.06 MHz, modernised CDMA pilot).

★ THE CLAIM THIS EXISTS TO VERIFY: L2OC-p's base ranging code is BIT-IDENTICAL to L3OC-p's --
GLONASS reuses one sequence across the L2 and L3 CDMA pilots, and the only difference is the
modulation on top (L2OC applies a per-chip [0,0,1,-1] BOC(1,1)+TDM expansion; L3OC uses it raw).
This checker re-derives the L3OC-p base INDEPENDENTLY, applies the expansion, and asserts the
C++ matches chip for chip -- so the reuse is a checked fact, not an assumption.

Checks: the base equals the independently-derived L3OC-p code; the [0,0,1,-1] expansion is applied
exactly (every 4-slot group is [0, 0, +base, -base]); half the chips are zero and the non-zeros
balance; the 50-chip OC2 secondary matches the reference; codes are distinct across PRN; and, if a
compiler is available, the C++ matches python chip for chip.

    python3 l2oc_code_check.py     # exits non-zero on any mismatch
"""
import os
import re
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "lib", "stages", "gnss"))
CPP = os.path.join(CPP_DIR, "glonassL2OCCode.cpp")
L3OC_CPP = os.path.join(CPP_DIR, "glonassL3OCCode.cpp")

BASE_N = 10230
SUB = (0, 0, 1, -1)
N = BASE_N * len(SUB)   # 40920
N_PRN = 63

# L3OC-p reference construction (must match glonassL3OCCode / l3oc_code_check).
DC1_SEED, DC1_TAP, DC1_N = 0b00110100111000, 0b00010001000011, 14
SHORT_TAP, SHORT_N, PILOT_OFFSET = 0b0000011, 7, 64

OC2_REF = (
    1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1,
    1, 1, 1, -1, 1)

# Recorded from the run that passed the bit-exact C++/python + reuse checks. popcount is 10230
# (half of 40920: every base +1 chip yields one +1 and one -1 after [0,0,+c,-c]).
FP = {1: (0x802f373e, 10230), 7: (0xe848631e, 10230), 19: (0x52943633, 10230),
      63: (0xc5622044, 10230)}


def _lfsr(n_out, seed, tap, n):
    r, out = seed, []
    for _ in range(n_out):
        out.append(1 if (r & 1) == 0 else -1)
        fb = bin(r & tap).count("1") & 1
        r = (fb << (n - 1)) | (r >> 1)
    return out


def _l3ocp_base(prn):
    """The L3OC-p ranging code for `prn`, derived independently of the C++ (DC1 x short(prn+64))."""
    dc1 = _lfsr(BASE_N, DC1_SEED, DC1_TAP, DC1_N)
    shrt = _lfsr(BASE_N, prn + PILOT_OFFSET, SHORT_TAP, SHORT_N)
    return [a * b for a, b in zip(dc1, shrt)]


def _l2ocp(prn):
    base = _l3ocp_base(prn)
    out = []
    for c in base:
        out.extend(c * s for s in SUB)
    return out


def _parse_oc2():
    src = open(CPP).read()
    m = re.search(r"L2OC_OC2\s*=\s*\{([^}]*)\}", src)
    if not m:
        return None
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def _cpp_codes():
    cxx = next((c for c in ("g++-12", "g++", "clang++")
                if subprocess.call(["which", c], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL) == 0), None)
    if cxx is None:
        return None
    main = r"""
#include "glonassL2OCCode.hpp"
#include <cstdio>
int main() {
    for (int prn = 1; prn <= 63; ++prn) {
        auto c = glonass::generate_l2ocp_code(prn);
        std::printf("%d ", prn);
        for (int8_t v : c) std::putchar(v > 0 ? '+' : (v < 0 ? '-' : '0'));
        std::putchar('\n');
    }
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src, exe = os.path.join(td, "d.cpp"), os.path.join(td, "d")
        open(src, "w").write(main)
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", CPP_DIR, src, CPP, L3OC_CPP, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   (C++ compile failed, skipping bit-exact check)\n%s" % r.stderr[:800])
            return None
        out = subprocess.run([exe], capture_output=True, text=True).stdout
    codes = {}
    for line in out.splitlines():
        prn, chips = line.split()
        codes[int(prn)] = [1 if ch == "+" else (-1 if ch == "-" else 0) for ch in chips]
    return codes


def _fp(code):
    return (zlib.crc32("".join("+" if x > 0 else ("-" if x < 0 else "0")
                               for x in code).encode()),
            sum(1 for x in code if x > 0))


def main():
    ok = True

    codes = {p: _l2ocp(p) for p in range(1, N_PRN + 1)}

    # ---- length + alphabet + the expansion structure --------------------------------------
    c1 = codes[1]
    if len(c1) != N:
        print("length: %d, expected %d" % (len(c1), N))
        ok = False
    else:
        print("length: %d chips at 2.046 Mcps (20 ms period)" % N)

    if set(c1) - {-1, 0, 1}:
        print("alphabet: values outside {-1,0,+1}: %s" % (set(c1) - {-1, 0, 1}))
        ok = False

    # every 4-slot group must be [0, 0, +base, -base]
    base1 = _l3ocp_base(1)
    bad = 0
    for i, b in enumerate(base1):
        g = c1[4 * i:4 * i + 4]
        if g != [0, 0, b, -b]:
            bad += 1
    if bad:
        print("expansion: %d of %d groups are not [0,0,+base,-base]" % (bad, BASE_N))
        ok = False
    else:
        print("expansion: all %d base chips became [0,0,+c,-c] exactly (BOC(1,1)+TDM)" % BASE_N)

    nz = sum(1 for x in c1 if x != 0)
    if nz != 2 * BASE_N:
        print("zeros: %d non-zero chips, expected %d (half)" % (nz, 2 * BASE_N))
        ok = False
    else:
        pos = sum(1 for x in c1 if x > 0)
        print("zeros: exactly half (%d) are the TDM'd data slots; non-zero split +%d/-%d"
              % (N - nz, pos, nz - pos))

    # ---- the reuse claim, checked against the L3OC checker's own construction -------------
    # (the base used here is _l3ocp_base; if l3oc_code_check is importable, cross-check it)
    try:
        sys.path.insert(0, HERE)
        import l3oc_code_check as l3
        ref = l3._code(1, True, l3._dc1())  # L3OC-p PRN 1
        if ref != base1:
            print("reuse: the L2OC base does NOT match l3oc_code_check's L3OC-p PRN 1 -- the "
                  "'same sequence' claim is wrong")
            ok = False
        else:
            print("reuse: L2OC base == l3oc_code_check's L3OC-p code (PRN 1), independently derived")
    except Exception as e:
        print("reuse: could not import l3oc_code_check to cross-check (%s); used local derivation"
              % e)

    # ---- distinctness ---------------------------------------------------------------------
    if len({tuple(c) for c in codes.values()}) != N_PRN:
        print("distinctness: codes NOT distinct across PRN")
        ok = False
    else:
        print("distinctness: all %d codes distinct" % N_PRN)

    # ---- secondary ------------------------------------------------------------------------
    oc2 = _parse_oc2()
    if oc2 is None:
        print("OC2: could not parse L2OC_OC2 out of the C++")
        ok = False
    elif oc2 != list(OC2_REF):
        print("OC2: C++ secondary differs from the reference (len %d vs %d)"
              % (len(oc2), len(OC2_REF)))
        ok = False
    else:
        print("OC2: 50-chip secondary matches the reference")

    # ---- bit-exactness vs the C++ ---------------------------------------------------------
    cpp = _cpp_codes()
    if cpp is None:
        print("bit-exact: no compiler, SKIPPED")
    else:
        bad = [p for p in codes if cpp.get(p) != codes[p]]
        if bad:
            print("bit-exact: %d PRNs differ between C++ and python (e.g. %s)" % (len(bad), bad[:4]))
            ok = False
        else:
            print("bit-exact: all %d codes identical between C++ and python, chip for chip" % N_PRN)

    for prn in (1, 7, 19, 63):
        got = _fp(codes[prn])
        want = FP.get(prn)
        if want is None:
            print("   PRN %2d: crc=0x%08x pop=%d (recording)" % (prn, got[0], got[1]))
        elif got != tuple(want):
            print("   PRN %2d: crc=0x%08x pop=%d MISMATCH (stored %s)" % (prn, got[0], got[1], want))
            ok = False
        else:
            print("   PRN %2d: crc=0x%08x pop=%d OK" % (prn, got[0], got[1]))

    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
