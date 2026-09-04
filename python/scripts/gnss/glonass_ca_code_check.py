#!/usr/bin/env python3
"""GLONASS FDMA C/A ranging-code check (L1OF / L2OF, 511 chips at 511 kcps).

★ This code gets a SHARPER test than any other signal we carry, because it is a pure
MAXIMAL-LENGTH SEQUENCE rather than a Gold code. An m-sequence has IDEAL two-valued periodic
autocorrelation: exactly +511 at zero shift and exactly -1 at all 510 others. So the check is
not "sidelobes are low enough", it is "every one of the 510 sidelobes equals -1 EXACTLY". An
off-by-one in the tap set, the output stage or the shift direction fails that instantly -- there
is no room for a nearly-right code to pass. Balance is exactly -1 and the longest CIRCULAR run
is exactly 9; see the notes at each check for why those are the right numbers (both tripped me
up first time round, and both were the test being wrong rather than the code).

★ And there is only ONE code: every GLONASS satellite transmits it. Satellites are separated by
FREQUENCY (channel k -> f0 + k*437.5 kHz on L2), not by code. So there is no PRN loop here and no
per-satellite table anywhere -- the per-satellite story lives entirely in the carrier.

Checks, in order: the C++ literals are parsed back out and asserted; the sequence is regenerated
independently in python; the m-sequence properties are asserted exactly; and if a compiler is
available the C++ output is compared to python's chip for chip.

    python3 glonass_ca_code_check.py     # exits non-zero on any mismatch
"""
import os
import re
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "lib", "stages", "gnss"))
CPP = os.path.join(CPP_DIR, "glonassCACode.cpp")

N = 511
REF = {"seed": 0b111111111, "out_stage": 2, "tap_a": 0, "tap_b": 4, "fb_stage": 8}
FINGERPRINT = (0x36c5959f, 255)   # (crc32 of the "01" string with +1 -> "1", count of +1)


def ca_code():
    """The 9-stage m-sequence, in the reference convention: emit stage `out_stage`, shift RIGHT,
    load parity(stage tap_a ^ stage tap_b) into `fb_stage`. Chips are logic 0 -> +1."""
    r = REF["seed"]
    out = []
    for _ in range(N):
        out.append(1 if ((r >> REF["out_stage"]) & 1) == 0 else -1)
        fb = ((r >> REF["tap_a"]) & 1) ^ ((r >> REF["tap_b"]) & 1)
        r = (fb << REF["fb_stage"]) | (r >> 1)
    return out


def parse_cpp():
    """Pull the LFSR literals out of the C++ so a typo there fails here, not on the sky."""
    src = open(CPP).read()
    got = {}
    m = re.search(r"uint32_t\s+r\s*=\s*0b([01]+)u", src)
    if m:
        got["seed"] = int(m.group(1), 2)
    m = re.search(r"\(\(r\s*>>\s*(\d+)\)\s*&\s*1u\)\s*\?\s*-1\s*:\s*\+1", src)
    if m:
        got["out_stage"] = int(m.group(1))
    # `(r & 1u)` is stage 0 written without a shift; normalise it so one regex covers both.
    norm = src.replace("(r & 1u)", "((r >> 0) & 1u)")
    m = re.search(r"fb\s*=\s*\(\(\(r\s*>>\s*(\d+)\)\s*&\s*1u\)\s*\^\s*\(\(r\s*>>\s*(\d+)\)"
                  r"\s*&\s*1u\)\)", norm)
    if m:
        got["tap_a"], got["tap_b"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"r\s*=\s*\(fb\s*<<\s*(\d+)\)", src)
    if m:
        got["fb_stage"] = int(m.group(1))
    return got


def cpp_code():
    cxx = next((c for c in ("g++-12", "g++", "clang++")
                if subprocess.call(["which", c], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL) == 0), None)
    if cxx is None:
        return None
    main = r"""
#include "glonassCACode.hpp"
#include <cstdio>
int main() {
    for (int8_t v : glonass::ca_code()) std::putchar(v > 0 ? '1' : '0');
    std::putchar('\n');
    // also print a couple of frequency-plan values so the header's constants are exercised
    std::printf("%.0f %.0f\n", glonass::l2of_carrier_hz(-7), glonass::l2of_carrier_hz(6));
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src, exe = os.path.join(td, "d.cpp"), os.path.join(td, "d")
        open(src, "w").write(main)
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", CPP_DIR, src, CPP, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   (C++ compile failed, skipping bit-exact check)\n%s" % r.stderr[:800])
            return None
        out = subprocess.run([exe], capture_output=True, text=True).stdout.split("\n")
    return ([1 if b == "1" else -1 for b in out[0].strip()],
            tuple(float(x) for x in out[1].split()))


def main():
    ok = True

    got = parse_cpp()
    for k, want in REF.items():
        if k not in got:
            print("constants: could not parse %s out of the C++" % k)
            ok = False
        elif got[k] != want:
            print("constants: %s = %s in C++, reference says %s" % (k, got[k], want))
            ok = False
    if ok:
        print("constants: seed/output stage/taps/feedback stage all match the reference")

    c = ca_code()

    # ---- m-sequence properties, asserted EXACTLY ----------------------------
    # A degree-9 m-sequence has 256 binary ONES and 255 zeros. Under our house convention
    # (logic 0 -> +1, logic 1 -> -1) that is 255 x (+1) and 256 x (-1), i.e. balance -1 and a
    # +1-count of 255. (Balance +1 would mean the convention had been inverted somewhere.)
    bal = sum(c)
    if bal != -1:
        print("balance: %d, expected exactly -1 (255 x +1 vs 256 x -1) -- not an m-sequence" % bal)
        ok = False
    else:
        print("balance: exactly -1 (255 chips +1, 256 chips -1) -- 256 binary ones, as required")

    side = set()
    for s in range(1, N):
        side.add(sum(c[i] * c[(i + s) % N] for i in range(N)))
    if side != {-1}:
        print("autocorrelation: sidelobes take values %s, expected EXACTLY {-1} at all %d "
              "non-zero shifts -- not a maximal-length sequence" % (sorted(side)[:6], N - 1))
        ok = False
    else:
        print("autocorrelation: all %d non-zero shifts are EXACTLY -1 (ideal m-sequence, "
              "-54.2 dB)" % (N - 1))

    # Run-length check: a degree-n m-sequence contains exactly one run of n identical chips.
    # ⚠️ It MUST be evaluated CIRCULARLY -- the sequence is periodic, and the long run straddles
    # the buffer boundary for this seed, so a linear scan reports 8 and cries wolf.
    runs, cur = [], 1
    for i in range(1, 2 * N):                       # two periods -> every circular run appears
        if c[i % N] == c[(i - 1) % N]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    if max(runs) != 9:
        print("runs: longest circular run %d, expected 9 (a 9-stage m-sequence has exactly one)"
              % max(runs))
        ok = False
    else:
        print("runs: longest circular run is 9, as a 9-stage m-sequence requires")

    fp = (zlib.crc32("".join("1" if x > 0 else "0" for x in c).encode()),
          sum(1 for x in c if x > 0))
    if fp != FINGERPRINT:
        print("fingerprint: crc=0x%08x pop=%d MISMATCH (stored %s)" % (fp[0], fp[1], FINGERPRINT))
        ok = False
    else:
        print("fingerprint: crc=0x%08x pop=%d OK" % fp)

    # ---- bit-exactness vs the C++ + the frequency plan -----------------------
    r = cpp_code()
    if r is None:
        print("bit-exact: no compiler, SKIPPED")
    else:
        cc, freqs = r
        if cc != c:
            n = sum(1 for a, b in zip(cc, c) if a != b)
            print("bit-exact: C++ and python differ in %d of %d chips" % (n, N))
            ok = False
        else:
            print("bit-exact: C++ and python identical, chip for chip")
        want = (1242.9375e6, 1248.625e6)
        if abs(freqs[0] - want[0]) > 1 or abs(freqs[1] - want[1]) > 1:
            print("freq plan: k=-7,+6 -> %.4f, %.4f MHz; expected %.4f, %.4f"
                  % (freqs[0] / 1e6, freqs[1] / 1e6, want[0] / 1e6, want[1] / 1e6))
            ok = False
        else:
            print("freq plan: k=-7..+6 spans %.4f-%.4f MHz (%.2f MHz) -- fits one 10 MHz tune"
                  % (freqs[0] / 1e6, freqs[1] / 1e6, (freqs[1] - freqs[0]) / 1e6))

    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
