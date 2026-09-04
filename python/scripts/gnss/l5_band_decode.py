#!/usr/bin/env python3
"""L5-band (1176.45) code-generator bring-up bench against a grab_l5.sh capture.

Stage 0 (tonight): GPS L5 Q5 via the sky-proven gpsL5Code.cpp algorithm ported here --
validates the CAPTURE + IF/flip conventions before any NEW generator is judged.
Stage 1 (next): Galileo E5a-Q + BeiDou B2a-pilot generators validated by UNSEEDED
acquisition on this same capture (the only proof a code table is right).

Usage: python3 l5_band_decode.py captures/l5_20msps_2026-07-15.bin --prn 28,32,10
"""
import argparse
import numpy as np

FS = 20e6
IF = 5e6          # Fs/4 after the (-1)^n flip (real capture, high-side by convention)
T_SKIP = 0.5
L = 10230

def lfsr(count, reg, tap, n):
    out = np.empty(count, np.int8)
    for i in range(count):
        out[i] = -1 if (reg & 1) else 1
        fb = bin(reg & tap).count("1") & 1
        reg = (fb << (n - 1)) | (reg >> 1)
    return out

def _xa():
    base = lfsr(8190, 0x1FFF, 0b0000000011011, 13)  # XA: restarts at 8190 (gpsL5Code.cpp)
    return np.concatenate([base, base[:L - 8190]])

def _xb():
    return lfsr(L, 0x1FFF, 0b1011011100011, 13)     # XB: free-runs the full 10230

L5Q_XB_ADV = [1701, 323, 5292, 2020, 5429, 7136, 1041, 5947, 4315, 148, 535, 1939,
              5206, 5910, 3595, 5135, 6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961,
              7106, 5299, 4660, 276, 4389, 3783, 1591, 1601]

def l5q_code(prn):
    xa, xb = _xa(), _xb()
    s = L5Q_XB_ADV[prn - 1]
    return xa * np.roll(xb, -s)   # XB advanced (left-rolled) by s, like the C++

def measure(raw, name, code, chip_rate, T_p, dops, n_blocks, flip):
    n_samp = int(round(FS * T_p))
    idx = (np.arange(n_samp) * chip_rate / FS).astype(np.int64) % len(code)
    rep = code[idx].astype(np.float32)
    REPF = np.conj(np.fft.fft(rep))
    off0 = int(T_SKIP * FS)
    tt = np.arange(n_samp) / FS
    guard = int(2 * FS / chip_rate)
    best = (-1.0, 0.0, 0)
    for dop in dops:
        f0 = IF + dop
        xs = []
        pks = []
        for b in range(n_blocks):
            o = off0 + b * n_samp
            x = raw[o:o + n_samp].astype(np.float32)
            if flip:
                x = x * np.where(np.arange(o, o + n_samp) % 2, -1, 1).astype(np.float32)
            xb_ = x * np.exp(-2j * np.pi * f0 * (tt + o / FS)).astype(np.complex64)
            c = np.abs(np.fft.ifft(np.fft.fft(xb_) * REPF)) ** 2
            pk = int(np.argmax(c))
            m = np.ones(n_samp, bool)
            m[max(0, pk - guard):pk + guard] = False
            xs.append((c[pk] - c[m].mean()) / c[m].mean())
            pks.append(pk)
        mx = float(np.mean(xs))
        if mx > best[0]:
            best = (mx, dop, int(np.median(pks)))
    x, dop, pk = best
    cn0 = 10 * np.log10(max(x, 1e-9) / T_p)
    cp = pk * chip_rate / FS % L
    print("%-14s best dop %+7.1f Hz  x=%8.2f  C/N0=%5.1f dB-Hz  cp=%7.1f chips"
          % (name, dop, x, cn0, cp))
    return x, dop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--prn", default="28,32,10")
    ap.add_argument("--flip", type=int, default=-1, help="-1 = calibrate on first PRN")
    args = ap.parse_args()
    raw = np.memmap(args.raw, dtype=np.int16, mode="r")
    print("raw: %.1f s @ 20 MSPS" % (len(raw) / FS))
    prns = [int(p) for p in args.prn.split(",")]
    dops = np.arange(-4000, 4001, 250.0)
    flip = args.flip
    if flip < 0:
        for f in (False, True):
            x, _ = measure(raw, "G%d flip=%d" % (prns[0], f), l5q_code(prns[0]),
                           10.23e6, 1e-3, dops, 10, f)
            if x > 3:
                flip = f
                break
        else:
            raise SystemExit("calibration failed: no detection either flip")
        print("-> flip convention: %s" % flip)
    for prn in prns:
        measure(raw, "G%d L5Q" % prn, l5q_code(prn), 10.23e6, 1e-3, dops, 10, bool(flip))
