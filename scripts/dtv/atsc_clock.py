#!/usr/bin/env python3
"""Residual symbol-clock error from the timing drift of the known field syncs: least-squares channel on the
training symbols, then per field a complex gain + timing shift fitted inside the PN511; the slope of the
timing over the fields is the clock correction to pass back to atsc_rx.py --delta.

    atsc_clock.py --rx rxa24_k00.npz
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import SEG, FIELD, cubic_at  # noqa: E402
from atsc_eq import known_symbols, ls_cir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx", required=True)
    a = ap.parse_args()
    d = np.load(a.rx)
    y2, fslot, rsign, delta = (
        d["y2"].astype(np.complex128),
        int(d["fslot"]),
        complex(d["rsign"]),
        float(d["delta"]),
    )
    L = 2 * SEG
    nseg = len(y2) // L
    y2 = y2[: nseg * L]
    lg, h = ls_cir(y2, known_symbols(nseg, fslot, rsign, "all"), 108)
    n = len(y2)
    hf = np.zeros(n, complex)
    hf[lg % n] = h
    pred = sfft.ifft(sfft.fft(known_symbols(nseg, fslot, rsign, "all")) * sfft.fft(hf))
    taus, fi = [], []
    for i, s in enumerate(range(fslot, nseg, FIELD)):
        a0, a1 = 2 * (s * SEG + 60), 2 * (s * SEG + 460)
        yy = y2[a0:a1]
        best = (1e18, 0.0)
        for tau in np.arange(-2.0, 2.001, 0.02):
            ps = cubic_at(pred, np.arange(a0, a1) - tau)
            al = np.vdot(ps, yy) / np.vdot(ps, ps)
            rr = np.sum(np.abs(yy - al * ps) ** 2)
            if rr < best[0]:
                best = (rr, tau)
        taus.append(best[1])
        fi.append(i)
    taus, fi = np.array(taus), np.array(fi)
    p = np.polyfit(fi, taus, 1)
    corr = p[0] / (2 * SEG * FIELD)
    print(
        "timing per field (samples at 2/symbol): %s"
        % " ".join("%+.2f" % v for v in taus)
    )
    print(
        "drift %.4f samples/field -> clock correction %+.4f ppm: use --delta %.4f (was %.4f); fit rms %.3f samples"
        % (
            p[0],
            corr * 1e6,
            (delta + corr) * 1e6,
            delta * 1e6,
            np.std(taus - np.polyval(p, fi)),
        )
    )


if __name__ == "__main__":
    main()
