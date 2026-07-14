#!/usr/bin/env python3
"""Watch the BEAM CUBE live: per-PRN, per-frequency-channel deep-integrated |A| -- the antenna
beam pattern resolved across the signal band (the GnssBeamCube product). The combiner gives one
full-band |A| per PRN (frequency collapsed); this shows |A| vs frequency channel, so as a sat
transits you read beam(time, frequency) -- essential for the wide signals (L5 spans ~10 MHz).

Record layout (GnssBeamCube), per PRN: [0]=PRN [1]=dop [2]=cp [3]=channel0 then n_chan |A_c|
(sqrt<|A_c|^2>, one per covering channel) then a float64 UTC at slot (4 + n_chan). Frames are
written by rawFileWrite as 4-byte meta length (0) + n_prn records.

  python3 python/scripts/gps_beam_watch.py [base_dir] [n_prn] [n_chan] [--fs HZ --n N]
  # L5 default: /tmp/gpsl5 11 10 --fs 20e6 --n 10  (channels 1 MHz wide, -5..+4 MHz from carrier)
  # L2C:        /tmp/gpsl2c 11 7  --fs 5e6  --n 10
"""
import argparse
import glob
import os
import struct
import time

import numpy as np

OUT_HEADER = 4  # PRN, dop, cp, channel0


def latest_file(d):
    fs = glob.glob(os.path.join(d, "beam*.raw"))
    return max(fs, key=os.path.getmtime) if fs else None


def read_latest_frame(path, n_prn, recflo):
    rec_bytes = recflo * 4
    stride = 4 + n_prn * rec_bytes
    buf = open(path, "rb").read()
    if len(buf) < stride:
        return None
    off = (len(buf) // stride - 1) * stride  # last complete frame
    meta = struct.unpack_from("<I", buf, off)[0]
    base = off + 4 + meta
    rows = []
    for p in range(n_prn):
        ro = base + p * rec_bytes
        rows.append(np.frombuffer(buf, dtype="<f4", count=recflo, offset=ro))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="?", default="/tmp/gpsl5")
    ap.add_argument("n_prn", nargs="?", type=int, default=11)
    ap.add_argument("n_chan", nargs="?", type=int, default=10)
    ap.add_argument("--fs", type=float, default=20e6, help="sample rate Hz (for freq axis)")
    ap.add_argument("--n", type=int, default=10, help="PFB spectrum_length N (for freq axis)")
    ap.add_argument("--floor", type=float, default=0.0, help="only show PRNs with max |A| above this")
    args = ap.parse_args()

    recflo = OUT_HEADER + args.n_chan + 2  # + UTC float64
    bin_w = args.fs / (2 * args.n)         # channel width, Hz
    print("watching %s -- beam cube (n_prn=%d, n_chan=%d, %.0f kHz bins, carrier = bin N/2=%d)"
          % (args.base_dir, args.n_prn, args.n_chan, bin_w / 1e3, args.n // 2))
    while True:
        time.sleep(2)
        f = latest_file(args.base_dir)
        if not f:
            print("  (no beam records yet)")
            continue
        rows = read_latest_frame(f, args.n_prn, recflo)
        if not rows:
            continue
        # rank PRNs by peak per-channel |A|
        def amps(r):
            return np.asarray(r[OUT_HEADER:OUT_HEADER + args.n_chan], dtype=float)
        ranked = sorted(rows, key=lambda r: -amps(r).max())
        shown = 0
        for r in ranked:
            a = amps(r)
            if a.max() <= args.floor or r[0] <= 0:
                continue
            ch0 = int(round(r[3]))
            pk = int(np.argmax(a))
            pk_off = (ch0 + pk - args.n // 2) * bin_w / 1e6  # MHz from carrier
            amax = a.max() or 1.0
            bar = "".join(" .:-=+*#%@"[min(9, int(9 * v / amax))] for v in a)
            lo = (ch0 - args.n // 2) * bin_w / 1e6
            hi = (ch0 + args.n_chan - 1 - args.n // 2) * bin_w / 1e6
            print("PRN%-2d dop%+5.0f |%s|  peak %.2f @ %+.2f MHz   (%+.1f..%+.1f MHz from carrier)"
                  % (int(r[0]), r[1], bar, a.max(), pk_off, lo, hi))
            shown += 1
            if shown >= 6:
                break
        if shown == 0:
            print("  (nothing above floor %.2f)" % args.floor)


if __name__ == "__main__":
    main()
