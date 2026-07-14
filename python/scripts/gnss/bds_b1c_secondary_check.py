#!/usr/bin/env python3
"""GROUND TRUTH for the B1C pilot SECONDARY (overlay) code table.

The 13.7 h beam map showed C35/C37/C38/C39 sitting ~10 dB low in the COHERENT C/N0 while
sitting exactly on the mean in the INCOHERENT one. The incoherent estimator never touches the
secondary code; the coherent one depends on it completely. That points at our per-PRN
secondary table -- but the table has no duplicates and nothing out of range, so it LOOKS fine.
A wrong-but-legal Weil parameter produces a perfectly well-behaved sequence with good
autocorrelation: no self-consistency check can catch it. Only the sky can.

So ask the sky. Despread the pilot PRIMARY code (which demonstrably works -- the search
detects these satellites 100% of the time), and the per-record complex amplitude is then
A_k = +-|A|, its sign being the secondary chip actually being transmitted. Recover that sign
sequence from the raw voltage and correlate it against what our table generates, over every
cyclic shift. A correct table correlates ~1.0. A wrong one correlates ~0.

*** STATUS: NOT YET TRUSTWORTHY -- DO NOT USE ITS VERDICTS. *** It reproduces C25 exactly
(corr = 1.000, all 600 signs), so the idea is sound, but it still fails its own CONTROLS:
C23 comes back "wrong" when we know it is right (the pipeline deep-integrates C23 at +1.9 dB
vs the constellation mean over 19k emits). Something in the code-drift / carrier tracking is
still fragile for all but one satellite. A tool that condemns a satellite we can prove is
healthy cannot be used to condemn one we suspect. Fix the controls first; the script REFUSES
to print verdicts until they pass.

Suspected remaining fault: the code-phase acquisition latches onto a BOC(1,1) sidelobe
(half a chip either side of the true peak) at some epochs, so the measured drift is wrong and
the code walks off over the 6 s correlation. Median-slope filtering was not enough. Next:
scan the drift rate directly, maximising the sign-free incoherent sum, instead of measuring it
from FFT peaks.

Usage: python3 bds_b1c_secondary_check.py <raw.bin> [PRN ...]
"""
CONTROLS = {23: "deep-integrates at +1.9 dB vs mean (19k emits)",
            25: "deep-integrates at +1.8 dB vs mean (13k emits)"}
import os
import re
import sys

import numpy as np

FS = 20e6
IF = 5e6
T_SKIP = 2.0
T_P = 10e-3            # B1C primary period (10230 chips @ 1.023 Mcps)
SEC_LEN = 1800         # pilot secondary length
CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "../../../lib/stages/gnss/beidouB1CCode.cpp")


def _table(src, name):
    m = re.search(name + r"\[63\]\s*=\s*\{(.*?)\};", src, re.S)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def legendre(n):
    L = np.ones(n, dtype=np.int8)
    i = np.arange(1, n, dtype=np.int64)
    L[(i * i) % n] = -1
    return L


def b1c_codes(prn):
    """(primary 10230, secondary 1800) exactly as lib/stages/gnss/beidouB1CCode.cpp builds them."""
    src = open(CPP).read()
    Lp, Ls = legendre(10243), legendre(3607)
    w, p = _table(src, "B1CP_PH_DIFF")[prn - 1], _table(src, "B1CP_TRUNC")[prn - 1]
    i = np.arange(10230)
    j = (i + p - 1) % 10243
    prim = (Lp[j] * Lp[(j + w) % 10243]).astype(np.int8)
    w, p = _table(src, "B1CS_PH_DIFF")[prn - 1], _table(src, "B1CS_TRUNC")[prn - 1]
    i = np.arange(SEC_LEN)
    j = (i + p - 1) % 3607
    sec = (Ls[j] * Ls[(j + w) % 3607]).astype(np.int8)
    return prim, sec


def replica(code, n_samp, chip_rate=1.023e6):
    """BOC(1,1): each chip becomes +1,-1 at twice the rate."""
    code = np.repeat(code, 2) * np.tile([1, -1], len(code))
    idx = (np.arange(n_samp) * (2 * chip_rate) / FS).astype(np.int64) % len(code)
    return code[idx].astype(np.float32)


def acquire(raw, prim, n_samp, rec=0, dops=None):
    """Doppler + code phase from ONE 10 ms record at record index `rec` (FFT correlation).

    The Doppler grid must be FINE (default 10 Hz): a coarse grid leaves a residual carrier
    that aliases the per-record sign estimate downstream. At 10 ms records, a residual of
    more than ~12 Hz advances the bit-robust (squared) phase by >1/4 cycle per record and the
    sign sequence scrambles -- which is exactly how the first version of this test managed to
    declare the KNOWN-GOOD satellites broken."""
    REPF = np.conj(np.fft.fft(replica(prim, n_samp)))
    off = int(T_SKIP * FS) + rec * n_samp
    tt = np.arange(n_samp) / FS
    best = (-1, 0, 0)
    for dop in (np.arange(-4000, 4001, 100.0) if dops is None else dops):
        x = raw[off:off + n_samp].astype(np.float32)
        xb = x * np.exp(-2j * np.pi * (IF + dop) * (tt + off / FS))
        c = np.abs(np.fft.ifft(np.fft.fft(xb.astype(np.complex64)) * REPF)) ** 2
        pk = int(np.argmax(c))
        m = np.ones(n_samp, bool)
        m[max(0, pk - 40):pk + 40] = False
        snr = (c[pk] - c[m].mean()) / c[m].mean()
        if snr > best[0]:
            best = (snr, dop, pk)
    return best


def main():
    raw = np.memmap(sys.argv[1], dtype=np.int16, mode="r")
    prns = [int(a) for a in sys.argv[2:]] or [23, 25, 35, 37, 38, 39]
    n_samp = int(round(FS * T_P))
    n_rec = min(600, int(len(raw) / n_samp) - int(T_SKIP * FS / n_samp) - 2)
    print("raw: %.1f s @ 20 MSPS -> %d records of 10 ms usable\n" % (len(raw) / FS, n_rec))
    print("  PRN   acq_snr   recovered-vs-OUR-secondary")
    results = {}
    for prn in prns:
        prim, sec = b1c_codes(prn)
        snr, dop, pk = acquire(raw, prim, n_samp)
        if snr < 3:
            print("  C%-3d  %7.1f   (not up in this capture -- skipped)" % (prn, snr))
            continue
        # refine Doppler on a 10 Hz grid (see acquire(): a coarse residual aliases the signs)
        snr, dop, pk = acquire(raw, prim, n_samp, 0,
                               np.arange(dop - 120, dop + 121, 10.0))
        # CODE DRIFT: the chips march at chip_rate*(dop/f_carrier + receiver clock offset) --
        # ~2 chips/s here, i.e. tens of SAMPLES over the correlation. Measure it directly
        # (code phase now vs code phase K records later) rather than modelling it; despreading
        # a drifting code at a fixed phase is what destroyed the first attempt.
        # Robust slope: a BOC(1,1) autocorrelation has SIDELOBES half a chip either side of
        # the true peak (~10 samples at 20 MSPS), so any single code-phase measurement can
        # latch onto the wrong lobe. Two points then give a wrong drift, the code walks off
        # over the correlation, and the satellite looks broken -- which is exactly what this
        # test did to C21 (SNR 441!) while C25 came out at 1.000. Sample the code phase at
        # several epochs and take the MEDIAN successive slope: one bad lobe cannot move it.
        ks = [k for k in range(0, n_rec, max(1, n_rec // 8))][:8]
        pks = [acquire(raw, prim, n_samp, k, np.array([dop]))[2] for k in ks]
        slopes = []
        for i in range(1, len(ks)):
            d = pks[i] - pks[i - 1]
            d -= n_samp * round(d / n_samp)       # unwrap across the record boundary
            slopes.append(d / (ks[i] - ks[i - 1]))
        rate = float(np.median(slopes))            # samples per record
        rep0 = replica(prim, n_samp)
        off0 = int(T_SKIP * FS)
        tt = np.arange(n_samp) / FS
        A = np.empty(n_rec, complex)
        for k in range(n_rec):
            o = off0 + k * n_samp
            x = raw[o:o + n_samp].astype(np.float32)
            xb = x * np.exp(-2j * np.pi * (IF + dop) * (tt + o / FS))
            A[k] = np.sum(xb * np.roll(rep0, int(round(pk + rate * k))))
        # DEROTATE THE CARRIER PER RECORD, sign-free. A_k^2 cancels the overlay's +-1 exactly
        # (that is the whole point of squaring), leaving 2*theta_k -- the carrier phase, which
        # we can unwrap and halve. Fitting a straight LINE to the phase is not enough: the
        # Doppler RATE (~0.5 Hz/s) contributes tens of radians of QUADRATIC phase over a
        # 6 s correlation, which scrambles the signs on its own. Following the phase record by
        # record handles any drift shape, as long as it moves slowly (it does: <12 Hz residual
        # = <1/4 cycle per 10 ms record).
        theta = np.unwrap(np.angle(A * A)) / 2.0
        s = np.sign((A * np.exp(-1j * theta)).real).astype(np.int8)
        # correlate the RECOVERED sign sequence against OUR table, over every cyclic shift
        ours = sec[:n_rec].astype(float)
        best = max(abs(float(np.dot(s, np.roll(sec.astype(float), -sh)[:n_rec])) / n_rec)
                   for sh in range(SEC_LEN))
        results[prn] = best
        print("  C%-3d  %7.1f   corr = %.3f" % (prn, snr, best))

    # The tool does not get to render a verdict until it can reproduce the satellites we
    # already KNOW are healthy. Otherwise it is just laundering its own bugs into a finding.
    ctrl = {p: results[p] for p in CONTROLS if p in results}
    if not ctrl:
        print("\n  NO VERDICT: no control PRNs (%s) in this capture."
              % ", ".join("C%d" % p for p in CONTROLS))
        return
    failed = [p for p, v in ctrl.items() if v < 0.6]
    if failed:
        print("\n  *** NO VERDICT: the test FAILED ITS OWN CONTROLS ***")
        for p in failed:
            print("      C%-2d corr=%.3f but it is KNOWN GOOD (%s)" % (p, ctrl[p], CONTROLS[p]))
        print("      The tool is broken, not the table. Fix it before believing any row above.")
        return
    print("\n  controls pass (%s) -> verdicts are meaningful:"
          % ", ".join("C%d=%.2f" % (p, v) for p, v in ctrl.items()))
    for prn, v in results.items():
        if prn in CONTROLS:
            continue
        print("      C%-2d  %s" % (prn, "MATCHES our table" if v > 0.6
                                   else "*** OUR SECONDARY IS WRONG ***"))


if __name__ == "__main__":
    main()
