#!/usr/bin/env python3
"""On-sky validation of the GPS L2C CL pilot code table (lib/stages/gnss/gpsL2CCode.cpp).

CM is sky-proven (live L2C tracking); CL's 767250-chip table has never been checked against a
real signal. CL's 1.5 s period is un-searchable blind, but CM and CL epochs are LOCKED:
cp_CL = cp_CM + k*10230 with k in [0,75). So: acquire CM (Doppler + cp + code drift), then test
the 75 CL alignments with a coherent sum over ~1 s -- the right k pops if and only if the CL
table, the LFSR, and the epoch relation are all correct. Then, at the winning k, integrate
coherently over growing spans: the CL coherence curve (dataless -- no wipe needed), directly
measuring the depth the GPSDO buys on the pilot.

Usage:  python3 l2_cl_check.py [l2_5msps_YYYY-MM-DD.bin] [PRN ...]   (default: newest, PRNs 20 15 18)
"""
import glob
import os
import sys

import numpy as np

FS = 5.0e6          # real sample rate
FOFF = 1.25e6       # IF after the (-1)^n flip = Fs/4
CHIP = 511.5e3      # L2C effective chip rate (CM and CL each, time-multiplexed)
FCARR = 1227.6e6    # L2 carrier
NW = 100000         # samples per 20 ms CM period at 5 MSPS
CMLEN = 10230       # CM chips / period
CLLEN = 767250      # CL chips / period = 75 CM segments

# --- L2C codes: 27-stage Galois LFSR, constants transcribed from lib/stages/gnss/gpsL2CCode.cpp ---
FEEDBACK = 0o445112474
CM_INIT = [0o742417664, 0o756014035, 0o002747144, 0o066265724, 0o601403471, 0o703232733,
           0o124510070, 0o617316361, 0o047541621, 0o733031046, 0o713512145, 0o024437606,
           0o021264003, 0o230655351, 0o001314400, 0o222021506, 0o540264026, 0o205521705,
           0o064022144, 0o120161274, 0o044023533, 0o724744327, 0o045743577, 0o741201660,
           0o700274134, 0o010247261, 0o713433445, 0o737324162, 0o311627434, 0o710452007,
           0o722462133, 0o050172213]
CL_INIT = [0o624145772, 0o506610362, 0o220360016, 0o710406104, 0o001143345, 0o053023326,
           0o652521276, 0o206124777, 0o015563374, 0o561522076, 0o023163525, 0o117776450,
           0o606516355, 0o003037343, 0o046515565, 0o671511621, 0o605402220, 0o002576207,
           0o525163451, 0o266527765, 0o006760703, 0o501474556, 0o743747443, 0o615534726,
           0o763621420, 0o720727474, 0o700521043, 0o222567263, 0o132765304, 0o746332245,
           0o102300466, 0o255231716]


def l2c_code(init, length):
    """Galois LFSR (matches gpsL2CCode.cpp generate()): LSB 0 -> +1, 1 -> -1."""
    reg = init
    out = np.empty(length, np.int8)
    for i in range(length):
        out[i] = -1 if (reg & 1) else 1
        reg = (reg >> 1) ^ (FEEDBACK * (reg & 1))
    return out.astype(np.float64)


def load(path, skip_s=1.0):
    n_tot = os.path.getsize(path) // 2
    off = int(skip_s * FS)
    raw = np.memmap(path, dtype=np.int16, mode="r")
    return raw, off, (n_tot - off) / FS


def window(raw, n0, dop):
    """One 20 ms window, flipped + carrier-wiped at absolute-sample phase (coherent across calls)."""
    x = raw[n0:n0 + NW].astype(np.float64)
    x *= 1 - 2 * ((n0 + np.arange(NW)) & 1)                       # (-1)^n flip
    return x * np.exp(-2j * np.pi * (FOFF + dop) * (np.arange(n0, n0 + NW)) / FS)


def cm_acquire(raw, off, prn, nwin=20):
    """Coarse+fine Doppler and cp (CM chips) via incoherent FFT correlation over nwin windows."""
    cm = l2c_code(CM_INIT[prn - 1], CMLEN)
    rep = cm[(np.arange(NW) * CHIP / FS % CMLEN).astype(np.int64) % CMLEN]
    crep = np.conj(np.fft.fft(rep))

    def acc(dop, n_start, n):
        p = np.zeros(NW)
        for i in range(n):
            p += np.abs(np.fft.ifft(np.fft.fft(window(raw, n_start + i * NW, dop)) * crep)) ** 2
        return p

    d0 = max(np.arange(-4500, 4501, 100), key=lambda d: acc(d, off, nwin).max())
    dop = max(np.arange(d0 - 100, d0 + 101, 10), key=lambda d: acc(d, off, nwin).max())
    p = acc(dop, off, nwin)
    snr = p.max() / np.median(p)
    cp = p.argmax() * CHIP / FS % CMLEN                            # peak sample -> CM chips
    return dop, cp, snr


def cp_at(raw, off, prn, dop, n_start, nwin=20):
    """cp (CM chips) measured at a later block -- for the linear code-drift fit."""
    cm = l2c_code(CM_INIT[prn - 1], CMLEN)
    rep = cm[(np.arange(NW) * CHIP / FS % CMLEN).astype(np.int64) % CMLEN]
    crep = np.conj(np.fft.fft(rep))
    p = np.zeros(NW)
    for i in range(nwin):
        p += np.abs(np.fft.ifft(np.fft.fft(window(raw, n_start + i * NW, dop)) * crep)) ** 2
    return p.argmax() * CHIP / FS % CMLEN


def main():
    args = [a for a in sys.argv[1:]]
    files = [a for a in args if a.endswith(".bin")]
    prns = [int(a) for a in args if not a.endswith(".bin")] or [20, 15, 18]
    here = os.path.dirname(os.path.abspath(__file__))
    path = files[0] if files else max(glob.glob(os.path.join(here, "l2_5msps_*.bin")),
                                      key=os.path.getmtime)
    raw, off, dur = load(path)
    print("capture: %s  (%.1f s usable)" % (os.path.basename(path), dur))

    chipidx = np.arange(NW) * CHIP / FS                            # chip advance within a window

    for prn in prns:
        dop, cp0, snr = cm_acquire(raw, off, prn)
        if snr < 8:
            print("\nPRN %d: CM not found (snr %.1f) -- skipping" % (prn, snr))
            continue
        # Linear code drift from a second CM fix near the end (the l-a code-rate term).
        t2 = int((dur - 3.0) * FS / NW) * NW
        cp2 = cp_at(raw, off, prn, dop, off + t2)
        drift = ((cp2 - cp0 + CMLEN / 2) % CMLEN - CMLEN / 2) / (t2 / FS)   # chips/s, wrapped
        print("\nPRN %d: CM dop %+０.0f Hz  cp %.1f  snr %.1f  drift %+.3f chips/s"
              .replace("０", "") % (prn, dop, cp0, snr, drift))

        # Refine the carrier to <~0.1 Hz from the CM despread phase history (the acquisition
        # grid is only ~5 Hz -- useless for coherent sums over seconds). CNAV flips the sign
        # per 20 ms window, so use the bit-robust squared-phase discriminator.
        cm = l2c_code(CM_INIT[prn - 1], CMLEN)
        for _ in range(2):
            nref = 100
            c = np.empty(nref, np.complex128)
            for i in range(nref):
                cp_i = cp0 + drift * (i * NW / FS)
                idx = ((chipidx - cp_i) % CMLEN).astype(np.int64) % CMLEN
                c[i] = np.dot(window(raw, off + i * NW, dop), cm[idx])
            dphi = np.angle((c[1:] * np.conj(c[:-1])) ** 2) / 2.0
            dop += np.median(dphi) / (2 * np.pi * (NW / FS))
        ctrl = np.empty(nref, np.complex128)
        for i in range(nref):
            cp_i = cp0 + drift * (i * NW / FS)
            idc = ((chipidx - cp_i - 500.0) % CMLEN).astype(np.int64) % CMLEN   # 500 chips off-peak
            ctrl[i] = np.dot(window(raw, off + i * NW, dop), cm[idc])
        print("  carrier refined: dop %+.2f Hz   CM direct-despread check: on/off-peak = %.1f"
              % (dop, np.abs(c).mean() / np.abs(ctrl).mean()))

        cl = l2c_code(CL_INIT[prn - 1], CLLEN)
        # 75-hypothesis test, INCOHERENT power sum over 50 windows (1 s) -- alignment does not
        # need carrier coherence, and per-window CL SNR matches CM's (same power split), so the
        # right k separates exactly like CM acquisition does. Window i uses CL segment (k+i)%75.
        nwin = 50
        w = [window(raw, off + i * NW, dop) for i in range(nwin)]
        P = np.zeros(75)
        for i in range(nwin):
            cp_i = cp0 + drift * (i * NW / FS)                     # received-code phase, CM chips
            for k in range(75):
                idx = ((chipidx - cp_i + (k + i) * CMLEN) % CLLEN).astype(np.int64) % CLLEN
                P[k] += np.abs(np.dot(w[i], cl[idx])) ** 2
        k_best = int(P.argmax())
        others = np.delete(P, k_best)
        snr_k = (P[k_best] - np.median(others)) / others.std()
        print("  CL alignment: k=%d  power %.3g vs others %.3g+-%.2g  -> %.1f sigma  %s"
              % (k_best, P[k_best], np.median(others), others.std(), snr_k,
                 "*** CL CODE VALIDATED ***" if snr_k > 8 else "(weak)"))
        if snr_k <= 8:
            continue

        # CL coherence curve at the winning k: coherent |mean| over growing spans, with the
        # SAME sums at a wrong alignment (k+37) as the noise reference. Dataless pilot -- the
        # coherent build needs no wipe; where it stops growing is the clock's coherence knee.
        print("  CL coherent build (dataless -- no wipe):")
        print("    span(s)   |<A>|/rec   SNR_vs_wrong-k")
        max_s = min(12.0, dur - 5.0)
        nmax = int(max_s * FS / NW)
        c_all = np.empty(nmax, np.complex128)
        c_bad = np.empty(nmax, np.complex128)
        for i in range(nmax):
            t_i = i * NW / FS
            cp_i = cp0 + drift * t_i
            wi = window(raw, off + i * NW, dop)
            idx = ((chipidx - cp_i + ((k_best + i) % 75) * CMLEN) % CLLEN).astype(np.int64) % CLLEN
            idb = ((chipidx - cp_i + ((k_best + 37 + i) % 75) * CMLEN) % CLLEN).astype(np.int64) % CLLEN
            c_all[i] = np.dot(wi, cl[idx])
            c_bad[i] = np.dot(wi, cl[idb])
        scale = np.abs(c_all).mean()                               # per-window amplitude scale
        for span in (0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0):
            n = int(span * FS / NW)
            if n < 2 or n > nmax:
                continue
            sums = np.array([np.abs(c_all[j:j + n].mean()) for j in range(0, nmax - n + 1, n)])
            bad = np.array([np.abs(c_bad[j:j + n].mean()) for j in range(0, nmax - n + 1, n)])
            print("    %6.1f    %.4f      %6.1f" % (span, sums.mean() / scale,
                                                    sums.mean() / bad.mean()))


if __name__ == "__main__":
    main()
