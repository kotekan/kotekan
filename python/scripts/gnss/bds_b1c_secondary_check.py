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
# Controls = sats whose OVERLAY (not just primary!) is certified by the 13.7 h map:
# coherent-vs-incoherent residual gap ~0 over thousands of emits. C21 does NOT qualify --
# a strong acquisition certifies only the primary code, which is how it snuck into an
# earlier control list and confused a night's debugging.
CONTROLS = {20: "map coh/inc gap +1.6 dB (10k emits)",
            25: "map coh/inc gap -1.0 dB (13k emits)",
            27: "map coh/inc gap -1.2 dB (16k emits)",
            29: "map coh/inc gap +2.9 dB (13k emits)",
            30: "map coh/inc gap +0.8 dB (17k emits)"}
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


def acquire(raw, prim, n_samp, rec=0, dops=None, n_win=8):
    """Doppler + code phase from ONE 10 ms record at record index `rec` (FFT correlation).

    The Doppler grid must be FINE (default 10 Hz): a coarse grid leaves a residual carrier
    that aliases the per-record sign estimate downstream. At 10 ms records, a residual of
    more than ~12 Hz advances the bit-robust (squared) phase by >1/4 cycle per record and the
    sign sequence scrambles -- which is exactly how the first version of this test managed to
    declare the KNOWN-GOOD satellites broken."""
    REPF = np.conj(np.fft.fft(replica(prim, n_samp)))
    tt = np.arange(n_samp) / FS
    best = (-1, 0, 0)
    for dop in (np.arange(-4000, 4001, 100.0) if dops is None else dops):
        # INCOHERENT SUM over several windows. A CW spur concentrates its power better
        # than a spread signal in any SINGLE window (it captured "C21" at -1830 Hz with
        # snr 441 while the real satellite sat at +1875), but its correlation peak lands
        # somewhere new every window, so averaging kills it; the true peak repeats.
        c = np.zeros(n_samp)
        for w in range(n_win):
            off = int(T_SKIP * FS) + (rec + w) * n_samp
            x = raw[off:off + n_samp].astype(np.float32)
            xb = x * np.exp(-2j * np.pi * (IF + dop) * (tt + off / FS))
            c += np.abs(np.fft.ifft(np.fft.fft(xb.astype(np.complex64)) * REPF)) ** 2
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
        if snr < 8:
            print("  C%-3d  %7.1f   (not meaningfully present in this capture -- skipped)"
                  % (prn, snr))
            continue
        # refine Doppler on a 10 Hz grid (see acquire(): a coarse residual aliases the signs)
        snr, dop, pk = acquire(raw, prim, n_samp, 0,
                               np.arange(dop - 120, dop + 121, 10.0))
        # ---- PERIOD-ALIGNED WINDOWS. The FFT peak at pk means the primary-code period
        # boundary falls at sample pk of the search window -- so START each despread window
        # THERE. A window on an arbitrary grid straddles TWO secondary chips, and the sign
        # it recovers is the mixture's; worse, as the code drifts the mixture fraction
        # sweeps through 1/2 where the sign is pure noise.
        # ---- PHYSICS-DERIVED DRIFT. Never measure the drift from FFT peak positions: a
        # BOC(1,1) correlation has -6 dB sidelobes half a chip (10 samples) either side of
        # the peak, and one sidelobe-captured epoch poisons the slope; the walked-off code
        # then reads as garbage signs at full acquisition SNR -- this is what condemned C21
        # at SNR 441 while C25 sailed through at 1.000 (its slope got lucky). The code rate
        # is DETERMINED by the refined Doppler up to the small receiver clock term:
        # chips/s = chip_rate*(sign*dop/f_carrier + (l-a)); scan the two signs and a small
        # (l-a) residual grid, maximizing total incoherent despread power -- the sign-free,
        # sidelobe-free estimator -- then hand the winner to the full pass.
        spc = FS / 1.023e6                     # samples per chip
        base = 1.023e6 * dop / 1575.42e6 * spc * T_P   # samples per record from Doppler
        off0 = int(T_SKIP * FS)
        tt = np.arange(n_samp) / FS
        rep0 = replica(prim, n_samp)

        def despread(rate_spr, ks):
            out = np.empty(len(ks), complex)
            for i, k in enumerate(ks):
                # the RECORD STRIDE k*n_samp is the whole point -- dropping it made every
                # 'record' re-despread the same 10 ms of data (the r=0 self-fulfilling max)
                o = off0 + pk + k * n_samp + int(round(rate_spr * k))
                x = raw[o:o + n_samp].astype(np.float32)
                xb = x * np.exp(-2j * np.pi * (IF + dop) * (tt + o / FS))
                out[i] = np.sum(xb * rep0)
            return out

        ks_probe = np.arange(0, n_rec, 10)
        cands = [sgn * base + d * spc * T_P
                 for sgn in (1.0, -1.0) for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]  # (l-a) in chips/s
        pwr = [float(np.sum(np.abs(despread(c, ks_probe)) ** 2)) for c in cands]
        rate = cands[int(np.argmax(pwr))]
        # fine refine around the winner
        cands2 = [rate + d for d in np.linspace(-0.03, 0.03, 7)]
        pwr2 = [float(np.sum(np.abs(despread(c, ks_probe)) ** 2)) for c in cands2]
        rate = cands2[int(np.argmax(pwr2))]
        A = despread(rate, np.arange(n_rec))
        # SELF-REFINE THE CARRIER from the despread series itself: the acquisition's
        # Doppler can settle tens of Hz off when RFI shares the refine window (the -1830
        # spur pulled C21 to -1900, 25 Hz from its mirror truth -- and 25 Hz is exactly
        # pi per 10 ms record, the unwrap ambiguity, which scrambles every sign while the
        # amplitude sails through untouched). The bit-robust squared product estimates the
        # residual unambiguously to +-1/(4*T_P) = +-25 Hz, from the data, spur-free.
        for _ in range(2):
            prod2 = A[1:] * np.conj(A[:-1])
            f_res = float(np.angle(np.sum(prod2 * prod2))) / (4.0 * np.pi * T_P)
            if abs(f_res) < 0.5:
                break
            dop += f_res
            A = despread(rate, np.arange(n_rec))
        # DEROTATE THE CARRIER PER RECORD, sign-free: A^2 cancels the overlay's +-1 exactly,
        # leaving 2*theta -- unwrap and halve. (A LINE fit is not enough: the Doppler RATE
        # puts tens of radians of quadratic phase across 6 s.)
        theta = np.unwrap(np.angle(A * A)) / 2.0
        s_rec = np.sign((A * np.exp(-1j * theta)).real).astype(np.int8)
        print("        [pk=%d (frac %.2f of window), rate=%+.3f samp/rec, mean|A| ratio %.2f]"
              % (pk, pk / n_samp, rate,
                 float(np.mean(np.abs(A))) / (np.mean(np.abs(A[:20])) + 1e-9)))
        # correlate the RECOVERED sign sequence against OUR table, over every cyclic shift
        best = max(abs(float(np.dot(s_rec, np.roll(sec.astype(float), -sh)[:n_rec])) / n_rec)
                   for sh in range(SEC_LEN))
        results[prn] = best
        print("  C%-3d  %7.1f   corr = %.3f   (dop %+.0f)" % (prn, snr, best, dop))
        if best < 0.6:
            # The recovered sign sequence is DATA -- if it doesn't match this PRN's row,
            # ask which sequence it IS. First every other row (a table row-swap), then
            # every possible Weil phase-difference w (a wrong parameter): the truth is in
            # the sky, and 600 clean chips identify it uniquely.
            src = open(CPP).read()
            Ls = legendre(3607)
            phs, trs = _table(src, "B1CS_PH_DIFF"), _table(src, "B1CS_TRUNC")
            row_hits = []
            for q in range(1, 64):
                i = np.arange(SEC_LEN)
                j = (i + trs[q - 1] - 1) % 3607
                sq = (Ls[j] * Ls[(j + phs[q - 1]) % 3607]).astype(float)
                c = max(abs(float(np.dot(s_rec, np.roll(sq, -sh)[:len(s_rec)])) / len(s_rec))
                        for sh in range(SEC_LEN))
                if c > 0.6:
                    row_hits.append((q, c))
            if row_hits:
                print("          -> recovered signs MATCH table row(s): %s  (ROW SWAP!)"
                      % ", ".join("C%d (%.3f)" % rc for rc in row_hits))
            else:
                # full Weil search: W_w[i] = Ls[i]*Ls[(i+w)%3607]; FFT-correlate the
                # recovered snippet against each w over all 3607 lags
                n = len(s_rec)
                bestw = (0.0, 0, 0)
                for w in range(1, 3607):
                    Ww = (Ls * np.roll(Ls, -w)).astype(float)
                    W2 = np.concatenate([Ww, Ww[:n - 1]])
                    c = np.correlate(W2, s_rec.astype(float))
                    i = int(np.argmax(np.abs(c)))
                    v = abs(float(c[i])) / n
                    if v > bestw[0]:
                        bestw = (v, w, i)
                print("          -> best Weil phase-difference from the SKY: w=%d "
                      "(corr %.3f at lag %d); our table says w=%d"
                      % (bestw[1], bestw[0], bestw[2], phs[prn - 1]))

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
