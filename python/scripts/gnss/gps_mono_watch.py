#!/usr/bin/env python3
"""Watch the MONOLITHIC (GpsReplicaCorrelator) records live: per-PRN SNR + amplitude.

The monolithic correlator writes one [n_prn x RECORD_FLOATS] record block per emit; the SNR
is right there in the record (rec[6]), so this is the ground-truth "what should we be getting"
readout, independent of the channelized pipeline. Record layout (GpsReplicaCorrelator):
  [0]=prn [1]=doppler_hz [2]=code_phase_chips [3]=peak_amp [4..5]=cpx [6]=SNR
  [7]=nav_sign [8]=phase_cont, then a float64 UTC at slot 9 (RECORD_UTC_SLOT).
Frames are written by rawFileWrite as: 4-byte meta length (0) + n_prn records.

  python3 python/scripts/gps_mono_watch.py [base_dir] [n_prn]   # default /tmp/gpsmono 11
"""
import sys, os, glob, struct, time
import numpy as np

RECORD_FLOATS = 11
RECORD_BYTES = RECORD_FLOATS * 4
UTC_SLOT = 9


def latest_file(d):
    fs = glob.glob(os.path.join(d, "mono*.raw"))
    return max(fs, key=os.path.getmtime) if fs else None


def read_latest_frame(path, n_prn):
    stride = 4 + n_prn * RECORD_BYTES
    buf = open(path, "rb").read()
    if len(buf) < stride:
        return None
    off = (len(buf) // stride - 1) * stride  # last complete frame
    meta = struct.unpack_from("<I", buf, off)[0]
    base = off + 4 + meta
    rows = []
    for p in range(n_prn):
        ro = base + p * RECORD_BYTES
        rows.append(np.frombuffer(buf, dtype="<f4", count=9, offset=ro))
    return rows


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gpsmono"
    n_prn = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    print("watching %s (n_prn=%d) -- monolithic per-PRN SNR (ground truth)" % (d, n_prn))
    while True:
        time.sleep(2)
        f = latest_file(d)
        if not f:
            print("  (no records yet -- correlator still filling its first integration)")
            continue
        rows = read_latest_frame(f, n_prn)
        if not rows:
            continue
        dets = sorted(rows, key=lambda r: -r[6])  # by SNR
        shown = ["PRN%d snr%.1f amp%.2g dop%+.0f" % (r[0], r[6], r[3], r[1])
                 for r in dets if r[6] > 0][:6]
        print("[mono] " + ("  ".join(shown) if shown else "nothing above SNR 0 yet"))


if __name__ == "__main__":
    main()
