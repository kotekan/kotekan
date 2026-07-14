#!/usr/bin/env python3
"""L5 Q5 ground truth + the search-vs-tracker cp-convention test (2026-07-04).

Live L5: the (FFT-acquire) search sees snr 30-43 but the trackers despread NOTHING at the
seeded cp (|A| at the noise floor). Everything else is ruled out (fits fire, slopes sane,
carrier sign right, coverage full, FLL-divergence refuted by a frozen-FLL run). Prime
suspect: a constant OFFSET between the cp convention the search REPORTS and the chip-index
model the tracker's replica DESPREADS at -- invisible to fits (search-internal) and to the
DLL (E/L both in noise). This script settles it on a raw capture:

  acquire mode (default):  python3 l5_q5_check.py [capture.bin] [PRN ...]
      FFT circular correlation per 1 ms window, incoherent over many windows (NH20-immune).
      Confirms the capture has signal and how strong. (Own cp convention -- no claim.)

  tracker-model scan:      python3 l5_q5_check.py capture.bin PRN --at-cp CP0 --dop D [--sign S]
      Despread with the EXACT C++ replica chip model (gnssChannelizedReplica.cpp):
      chip(n) = code[floor(cp0 + n * chip_per_sample) % 10230], n = ABSOLUTE capture sample,
      chip_per_sample = CHIP/FS * (1 + sign*dop/carrier). Scans cp0 + delta over +-scan chips
      and prints |A|(delta): the peak's delta IS the search->tracker convention offset
      (0 => convention fine, tracker should work; nonzero constant => the bug, read it off).
"""
import argparse
import glob
import os
import sys

import numpy as np

FS = 20.0e6         # real sample rate (l5_20msps capture)
FOFF = 5.0e6        # IF after the (-1)^n flip = Fs/4
CHIP = 10.23e6      # L5 chip rate
FCARR = 1176.45e6   # L5 carrier
NW = 20000          # samples per 1 ms code period at 20 MSPS
CODELEN = 10230

# --- L5 codes, transcribed from lib/stages/gnss/gpsL5Code.cpp (PocketSDR convention) ---
L5Q_XB_ADV = [1701, 323, 5292, 2020, 5429, 7136, 1041, 5947, 4315, 148, 535, 1939, 5206,
              5910, 3595, 5135, 6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961, 7106, 5299,
              4660, 276, 4389, 3783, 1591, 1601]
NH20 = np.array([1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1],
                np.float64)


def _lfsr(count, reg, tap, n=13):
    out = np.empty(count, np.int8)
    for i in range(count):
        out[i] = -1 if (reg & 1) else 1
        fb = bin(reg & tap).count("1") & 1
        reg = (fb << (n - 1)) | (reg >> 1)
    return out


def l5q_code(prn):
    """Q5 code: XA (8190-cycle, short-cycled to 10230) * XB (full 10230) advanced per PRN."""
    base = _lfsr(8190, 0x1FFF, 0b0000000011011)
    xa = np.concatenate([base, base[:CODELEN - 8190]])
    xb = _lfsr(CODELEN, 0x1FFF, 0b1011011100011)
    sh = L5Q_XB_ADV[prn - 1]
    return (xa * np.roll(xb, -sh)).astype(np.float64)   # xb[(i+sh) % L] == roll(xb, -sh)[i]


def load(path, skip_s=1.0):
    n_tot = os.path.getsize(path) // 2
    off = int(skip_s * FS)
    raw = np.memmap(path, dtype=np.int16, mode="r")
    return raw, off, (n_tot - off) / FS


def window(raw, n0, dop):
    x = raw[n0:n0 + NW].astype(np.float64)
    x *= 1 - 2 * ((n0 + np.arange(NW)) & 1)                       # (-1)^n flip
    return x * np.exp(-2j * np.pi * (FOFF + dop) * np.arange(n0, n0 + NW) / FS)


def acquire(raw, off, prn, nwin=100):
    """Coarse+fine Doppler and cp via incoherent FFT correlation (NH20-immune)."""
    q = l5q_code(prn)
    rep = q[(np.arange(NW) * CHIP / FS % CODELEN).astype(np.int64)]
    crep = np.conj(np.fft.fft(rep))

    def acc(dop, n):
        p = np.zeros(NW)
        for i in range(n):
            p += np.abs(np.fft.ifft(np.fft.fft(window(raw, off + i * NW, dop)) * crep)) ** 2
        return p

    d0 = max(np.arange(-4000, 4001, 250), key=lambda d: acc(d, 40).max())
    dop = max(np.arange(d0 - 250, d0 + 251, 50), key=lambda d: acc(d, 40).max())
    p = acc(dop, nwin)
    snr = p.max() / np.median(p)
    cp = p.argmax() * CHIP / FS % CODELEN
    return dop, cp, snr


def tracker_scan(raw, prn, cp0, dop, sign, nwin, scan, step):
    """Despread with the C++ replica chip model at cp0+delta; return |A|(delta) profile.

    Incoherent |A| over nwin 1 ms windows starting at file sample 0 timeline (matching the
    pipeline's absolute anchoring: the replay stream's sample 0 == this file's sample 0).
    NH20 flips whole windows only -> power sum immune.
    """
    q = l5q_code(prn)
    cps = CHIP / FS * (1.0 + sign * dop / FCARR)   # chip_per_sample, as the C++ replica
    deltas = np.arange(-scan, scan + step / 2, step)
    prof = np.zeros(len(deltas))
    n0s = [int(1.0 * FS) + i * NW for i in range(nwin)]   # 1 s in, contiguous windows
    wins = [window(raw, n0, dop) for n0 in n0s]
    for k, dlt in enumerate(deltas):
        acc = 0.0
        for n0, w in zip(n0s, wins):
            idx = ((cp0 + dlt + (n0 + np.arange(NW)) * cps) % CODELEN).astype(np.int64)
            acc += np.abs(np.dot(w, q[idx])) ** 2
        prof[k] = acc
    return deltas, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="*")
    ap.add_argument("--at-cp", type=float, default=None, help="tracker-model scan around this cp0")
    ap.add_argument("--dop", type=float, default=None)
    ap.add_argument("--sign", type=float, default=1.0, help="code_doppler_sign (config: 1.0)")
    ap.add_argument("--scan", type=float, default=20.0, help="scan half-width, chips")
    ap.add_argument("--step", type=float, default=0.25, help="scan step, chips")
    ap.add_argument("--nwin", type=int, default=50)
    a = ap.parse_args()
    files = [x for x in a.args if x.endswith(".bin")]
    prns = [int(x) for x in a.args if not x.endswith(".bin")] or [28, 32]
    here = os.path.dirname(os.path.abspath(__file__))
    path = files[0] if files else max(glob.glob(os.path.join(here, "l5_20msps_*.bin")),
                                      key=os.path.getmtime)
    raw, off, dur = load(path)
    print("capture: %s (%.1f s usable)" % (os.path.basename(path), dur))

    if a.at_cp is not None:
        prn = prns[0]
        deltas, prof = tracker_scan(raw, prn, a.at_cp, a.dop, a.sign, a.nwin, a.scan, a.step)
        med = np.median(prof)
        k = int(prof.argmax())
        print("PRN %d tracker-model scan around cp0=%.2f (dop %+.0f, sign %+g):"
              % (prn, a.at_cp, a.dop, a.sign))
        print("  peak at delta = %+.2f chips, power %.3g = %.1f x median"
              % (deltas[k], prof[k], prof[k] / med))
        for kk in range(len(deltas)):
            if prof[kk] > 3 * med:
                print("    delta %+7.2f -> %5.1f x med" % (deltas[kk], prof[kk] / med))
        return

    for prn in prns:
        dop, cp, snr = acquire(raw, off, prn)
        print("PRN %d: dop %+6.0f Hz  cp %8.2f chips (own convention)  snr %.1f %s"
              % (prn, dop, cp, snr, "" if snr > 8 else "(weak)"))


if __name__ == "__main__":
    main()
