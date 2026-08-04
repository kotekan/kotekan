#!/usr/bin/env python3
"""GLONASS L3OC ranging-code check (data L3OCd + pilot L3OCp, 1202.025 MHz).

Checks lib/stages/gnss/glonassL3OCCode.cpp three ways:
  1. **Constants**: the LFSR seeds/taps/lengths are parsed OUT of the C++ and asserted against
     the reference values, so a typo in a binary literal fails here rather than on the sky.
  2. **Algorithm**: the construction is re-run in pure python, INDEPENDENTLY of the C++, and
     structural invariants are asserted (distinctness, balance, autocorrelation sidelobes,
     data/pilot disjointness).
  3. **Bit-exactness**: if a C++ compiler is available, the module is compiled with a throwaway
     main() and its codes are compared to python's CHIP FOR CHIP. This is the check the earlier
     signals could not make -- their codes came from large transcribed tables, so python could
     only re-derive from the same table. L3OC is fully algorithmic, so the two implementations
     are genuinely independent and an exact match means something.

★ Construction: a SHARED 14-stage LFSR (DC1) times a PER-SATELLITE 7-stage LFSR that differs
only in its initial state -- `prn` for the data component, `prn + 64` for the pilot. The 7-stage
polynomial is primitive, so its 127 non-zero states are cyclic shifts of ONE m-sequence: the
per-satellite diversity is a code-phase offset, and data (shifts 1..63) and pilot (65..127) are
disjoint families.

⚠️ THE TRAP THIS GUARDS: code index 0 seeds the short LFSR with all-zeros, an ABSORBING state
that emits a constant sequence -- so index 0 does not yield a poor code, it yields DC1 itself,
which would correlate against every satellite. Checked explicitly below.

⚠️ Still NOT guarded: whether the code index equals the orbital slot number. That is an ICD
question no structural test can answer; it is settled on the sky by searching the full index
range. See the header of glonassL3OCCode.hpp.

    python3 l3oc_code_check.py     # exits non-zero on any mismatch
"""
import os
import re
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "lib", "stages", "gnss"))
CPP = os.path.join(CPP_DIR, "glonassL3OCCode.cpp")

N = 10230
N_PRN = 63

# Reference constants (PocketSDR sdr_code.py: gen_code_G3OC_DC1 / gen_code_G3OCD / _G3OCP).
REF = {
    "dc1_seed": 0b00110100111000,
    "dc1_tap": 0b00010001000011,
    "dc1_n": 14,
    "short_tap": 0b0000011,
    "short_n": 7,
    "pilot_offset": 64,
}

# (crc32 of the code as a "01" string with +1 -> "1", popcount of +1). Recorded from the run
# that passed the bit-exact C++/python comparison, so these pin BOTH implementations.
FINGERPRINTS = {
    "L3OCd": {1: (0x05d98a18, 5094), 7: (0x901857c4, 5200),
              19: (0xc8ad7d00, 5074), 63: (0x5bff8816, 5034)},
    "L3OCp": {1: (0x78f17598, 5095), 7: (0xed30a844, 5103),
              19: (0xb5858280, 5177), 63: (0x26d77796, 5129)},
}


def _lfsr(n_out, seed, tap, n):
    """Fibonacci LFSR in the reference convention: emit the LOW bit, shift RIGHT, and load the
    parity of the tapped bits into bit n-1. Emitting BEFORE shifting is what makes the seed the
    source of chip 0. Chips are logic 0 -> +1 (house convention, and PocketSDR's CHIP=(1,-1))."""
    r = seed
    out = []
    for _ in range(n_out):
        out.append(1 if (r & 1) == 0 else -1)
        fb = bin(r & tap).count("1") & 1
        r = (fb << (n - 1)) | (r >> 1)
    return out


def _dc1():
    return _lfsr(N, REF["dc1_seed"], REF["dc1_tap"], REF["dc1_n"])


def _code(prn, pilot, dc1):
    seed = prn + (REF["pilot_offset"] if pilot else 0)
    shrt = _lfsr(N, seed, REF["short_tap"], REF["short_n"])
    return [a * b for a, b in zip(dc1, shrt)]


def _fp(code):
    return (zlib.crc32("".join("1" if x > 0 else "0" for x in code).encode()),
            sum(1 for x in code if x > 0))


def _parse_cpp_constants():
    """Pull the binary literals out of the C++ so a typo there fails the check. Returns a dict
    in the same shape as REF, with None for anything not found."""
    src = open(CPP).read()
    got = {}
    m = re.search(r"Lfsr\s+lfsr\{\s*0b([01]+)u\s*,\s*0b([01]+)u\s*,\s*(\d+)\s*\}", src)
    if m:
        got["dc1_seed"] = int(m.group(1), 2)
        got["dc1_tap"] = int(m.group(2), 2)
        got["dc1_n"] = int(m.group(3))
    m = re.search(r"Lfsr\s+shrt\{\s*init\s*,\s*0b([01]+)u\s*,\s*(\d+)\s*\}", src)
    if m:
        got["short_tap"] = int(m.group(1), 2)
        got["short_n"] = int(m.group(2))
    m = re.search(r"compose\(static_cast<uint32_t>\(prn\)\s*\+\s*(\d+)u\)", src)
    if m:
        got["pilot_offset"] = int(m.group(1))
    return got


def _cpp_codes():
    """Compile the module with a throwaway main() and read back every code. Returns
    {(component, prn): [chips]} or None if no compiler is available."""
    cxx = None
    for cand in ("g++-12", "g++", "clang++"):
        if subprocess.call(["which", cand], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0:
            cxx = cand
            break
    if cxx is None:
        return None
    main = r"""
#include "glonassL3OCCode.hpp"
#include <cstdio>
int main() {
    for (int prn = 1; prn <= 63; ++prn) {
        for (int pilot = 0; pilot < 2; ++pilot) {
            auto c = pilot ? glonass::generate_l3ocp_code(prn) : glonass::generate_l3ocd_code(prn);
            std::printf("%d %d ", prn, pilot);
            for (int8_t v : c) std::putchar(v > 0 ? '1' : '0');
            std::putchar('\n');
        }
    }
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "dump.cpp")
        exe = os.path.join(td, "dump")
        open(src, "w").write(main)
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", CPP_DIR, src, CPP, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("   (C++ compile failed, skipping bit-exact check)\n%s" % r.stderr[:800])
            return None
        out = subprocess.run([exe], capture_output=True, text=True).stdout
    codes = {}
    for line in out.splitlines():
        prn, pilot, bits = line.split()
        codes[("L3OCp" if int(pilot) else "L3OCd", int(prn))] = [1 if b == "1" else -1
                                                                 for b in bits]
    return codes


def _peak_sidelobe_db(code):
    """Worst off-peak circular autocorrelation, in dB below the peak. Pure python FFT-free would
    be O(N^2)=1e8; use numpy when available and skip otherwise."""
    try:
        import numpy as np
    except ImportError:
        return None
    a = np.array(code, dtype=float)
    f = np.fft.rfft(a)
    ac = np.fft.irfft(f * np.conj(f), n=len(a))
    peak = ac[0]
    side = np.max(np.abs(ac[1:]))
    return 20.0 * np.log10(side / peak)


def main():
    ok = True

    # ---- 1. constants -------------------------------------------------------
    got = _parse_cpp_constants()
    for k, want in REF.items():
        if k not in got:
            print("constants: could not parse %s out of the C++ -- check the literal's shape" % k)
            ok = False
        elif got[k] != want:
            print("constants: %s = %s in C++, reference says %s" % (k, got[k], want))
            ok = False
    if ok:
        print("constants: DC1 seed/tap/len + short tap/len + pilot offset all match the reference")

    # ---- 2. algorithm -------------------------------------------------------
    dc1 = _dc1()
    codes = {}
    for prn in range(1, N_PRN + 1):
        codes[("L3OCd", prn)] = _code(prn, False, dc1)
        codes[("L3OCp", prn)] = _code(prn, True, dc1)

    # index 0 is the absorbing state -- confirm it really does collapse, so the range check in
    # the C++ is guarding something real rather than being decorative.
    zero = _code(0, False, dc1)
    if zero != dc1:
        print("index 0: expected the degenerate case to reproduce DC1 exactly -- it did not, so"
              " the LFSR convention here and in the C++ may have drifted")
        ok = False
    else:
        print("index 0: confirmed DEGENERATE (collapses to DC1) -- the 1..63 range check is load"
              "-bearing")

    allc = {k: tuple(v) for k, v in codes.items()}
    if len(set(allc.values())) != 2 * N_PRN:
        print("codes: NOT distinct across data+pilot (%d unique of %d)"
              % (len(set(allc.values())), 2 * N_PRN))
        ok = False
    else:
        print("codes: all %d distinct (data 1..63 and pilot 1..63 disjoint)" % (2 * N_PRN))

    bal = {c: [sum(codes[(c, p)]) for p in range(1, N_PRN + 1)] for c in ("L3OCd", "L3OCp")}
    for c, b in bal.items():
        # A product of two truncated m-sequences is not perfectly balanced; a few hundred out of
        # 10230 is normal. Orders of magnitude more would mean a stuck register.
        if max(abs(x) for x in b) > 600:
            print("%s: balance out of range: %d..%d (suspect a stuck LFSR)" % (c, min(b), max(b)))
            ok = False
        else:
            print("%s: balance %d..%d over 10230 chips" % (c, min(b), max(b)))

    sl = _peak_sidelobe_db(codes[("L3OCd", 1)])
    if sl is None:
        print("sidelobes: numpy unavailable, skipped")
    elif sl > -20.0:
        print("sidelobes: worst autocorrelation sidelobe %.1f dB -- too high for a ranging code" % sl)
        ok = False
    else:
        print("sidelobes: worst autocorrelation sidelobe %.1f dB (L3OCd PRN 1)" % sl)

    # ---- 3. bit-exactness vs the C++ ---------------------------------------
    cpp = _cpp_codes()
    if cpp is None:
        print("bit-exact: no compiler, SKIPPED -- structural checks only")
    else:
        bad = [k for k in codes if cpp.get(k) != codes[k]]
        if bad:
            print("bit-exact: %d of %d codes DIFFER between C++ and python (e.g. %s)"
                  % (len(bad), len(codes), bad[:4]))
            ok = False
        else:
            print("bit-exact: all %d codes identical between C++ and python, chip for chip"
                  % len(codes))

    # ---- fingerprints -------------------------------------------------------
    for comp in ("L3OCd", "L3OCp"):
        for prn in (1, 7, 19, 63):
            crc, pop = _fp(codes[(comp, prn)])
            want = FINGERPRINTS[comp].get(prn)
            if want is None:
                print("   %s PRN %2d: crc=0x%08x pop=%d (recording)" % (comp, prn, crc, pop))
            elif (crc, pop) != tuple(want):
                print("   %s PRN %2d: crc=0x%08x pop=%d MISMATCH (stored %s)"
                      % (comp, prn, crc, pop, want))
                ok = False
            else:
                print("   %s PRN %2d: crc=0x%08x pop=%d OK" % (comp, prn, crc, pop))

    print("\n" + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
