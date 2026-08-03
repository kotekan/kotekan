#!/usr/bin/env python3
"""Mid-band (1207.14 MHz) code-generator bring-up bench, off a capture_b2b_raw.yaml grab.

The 1207.14 MHz centre carries BeiDou B2b AND Galileo E5b at once, so ONE capture validates
both new generators by UNSEEDED acquisition (the only proof a code table is right on-sky) --
the same discipline that brought up B2a/E5a on the l5 band (l5_band_decode.py).

The kotekan rawFileWrite grab is one file per frame: [4-byte hdr][49920 int16 real samples].
This concatenates them (once) into a single int16 stream, then FFT-acquires the requested
signal's PRNs. IF = Fs/4 = 5 MHz after the airspyInput (-1)^n high-side flip, Fs = 20 MSPS.

    python3 mid_band_decode.py /tmp/gpsin_b2b --sig b2b --prn 19,20,29,30
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FS = 20e6
IF = 5e6           # Fs/4 (real capture, high-side by convention)
T_SKIP = 0.5
HDR = 4            # rawFileWrite per-frame header bytes (size word)
FRAME_SAMP = 49920


def load_stream(path):
    """Return an int16 memmap of the capture. `path` is either a concatenated .bin or a
    rawFileWrite directory (frames stitched into <dir>/mid_1207_20msps.bin on first use)."""
    if os.path.isfile(path):
        return np.memmap(path, dtype=np.int16, mode="r")
    files = sorted(glob.glob(os.path.join(path, "*.raw")))
    if not files:
        raise SystemExit("no .raw frames in %s" % path)
    out = os.path.join(path, "mid_1207_20msps.bin")
    if not os.path.isfile(out) or os.path.getsize(out) != len(files) * FRAME_SAMP * 2:
        print("stitching %d frames -> %s ..." % (len(files), out))
        buf = np.empty(len(files) * FRAME_SAMP, dtype=np.int16)
        for i, f in enumerate(files):
            buf[i * FRAME_SAMP:(i + 1) * FRAME_SAMP] = np.fromfile(
                f, dtype=np.int16, count=FRAME_SAMP, offset=HDR)
        buf.tofile(out)
    return np.memmap(out, dtype=np.int16, mode="r")


def code_fn(sig):
    if sig == "b2b":
        from b2b_code_check import generate_b2bi_code
        return lambda prn: np.array(generate_b2bi_code(prn), dtype=np.int8), 10.23e6, 10230
    raise SystemExit("unknown signal %s (b2b; e5b to come)" % sig)


# Detection statistic: RATIO of the correlation peak to the 2nd-highest peak outside a guard
# window, on the INCOHERENTLY summed |corr|^2 across blocks. Unlike (peak-mean)/mean -- whose
# noise floor is the max-of-N-bins ~ln(N) ~ 11, which made a "x>3" gate call pure noise a hit --
# a noise peak/2nd-peak sits at ~1.1-1.3, so >~1.6 is a genuine detection regardless of N.
def measure(raw, name, code, chip_rate, L, dops, n_blocks, flip, thresh=1.6):
    T_p = L / chip_rate
    n_samp = int(round(FS * T_p))
    idx = (np.arange(n_samp) * chip_rate / FS).astype(np.int64) % len(code)
    REPF = np.conj(np.fft.fft(code[idx].astype(np.float32)))
    off0 = int(T_SKIP * FS)
    tt = np.arange(n_samp) / FS
    guard = int(2 * FS / chip_rate)
    best = (0.0, 0.0, 0, 0.0)  # ratio, dop, peak-bin, peak-power
    for dop in dops:
        f0 = IF + dop
        acc = np.zeros(n_samp, np.float64)
        for b in range(n_blocks):
            o = off0 + b * n_samp
            x = raw[o:o + n_samp].astype(np.float32)
            if flip:
                x = x * np.where(np.arange(o, o + n_samp) % 2, -1, 1).astype(np.float32)
            xb = x * np.exp(-2j * np.pi * f0 * (tt + o / FS)).astype(np.complex64)
            acc += np.abs(np.fft.ifft(np.fft.fft(xb) * REPF)) ** 2
        pk = int(np.argmax(acc))
        m = np.ones(n_samp, bool)
        m[max(0, pk - guard):pk + guard] = False
        ratio = float(acc[pk] / acc[m].max())
        if ratio > best[0]:
            best = (ratio, dop, pk, float(acc[pk] / acc[m].mean()))
    ratio, dop, pk, pom = best
    cn0 = 10 * np.log10(max(pom, 1e-9) / T_p)  # rough C/N0 off the peak/mean (floor-biased)
    cp = pk * chip_rate / FS % L
    hit = "  <== ACQUIRED" if ratio > thresh else ""
    print("%-16s best dop %+7.1f Hz  pk/2nd=%6.2f  ~C/N0=%5.1f  cp=%7.1f%s"
          % (name, dop, ratio, cn0, cp, hit))
    return ratio, dop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="capture .bin or rawFileWrite directory")
    ap.add_argument("--sig", default="b2b")
    ap.add_argument("--prn", default=",".join(str(p) for p in range(19, 47)))
    ap.add_argument("--flip", type=int, default=-1, help="-1 = auto-calibrate on the first hit")
    ap.add_argument("--nblk", type=int, default=10)
    args = ap.parse_args()
    raw = load_stream(args.raw)
    print("raw: %.1f s @ 20 MSPS" % (len(raw) / FS))
    gen, chip, L = code_fn(args.sig)
    prns = [int(p) for p in args.prn.split(",")]
    dops = np.arange(-4500, 4501, 250.0)
    flip = args.flip
    if flip < 0:  # calibrate the (-1)^n flip on whichever PRN detects first
        for prn in prns:
            for f in (False, True):
                x, _ = measure(raw, "%s%d flip=%d" % (args.sig.upper(), prn, f),
                               gen(prn), chip, L, dops, args.nblk, f)
                if x > 1.6:
                    flip = int(f); break
            if flip >= 0:
                break
        if flip < 0:
            raise SystemExit("calibration failed: no detection on any PRN either flip")
        print("-> flip convention: %s\n" % bool(flip))
    for prn in prns:
        measure(raw, "%s C%02d" % (args.sig.upper(), prn), gen(prn), chip, L, dops,
                args.nblk, bool(flip))


if __name__ == "__main__":
    main()
