#!/usr/bin/env python3
"""Decision-directed fine phase track from the decoded symbols: the dense least-squares model of the beam
(h * c, every symbol known) and a complex gain per window of --win samples (2 per symbol); the phase is
added to atsc_eq2's segment-sync track. Iterates --iters times (h re-fitted on the re-derotated beam).
Writes an eq-like npz (phi per symbol) for atsc_sub.py --eq.

    atsc_phase2.py --rx rxa30d_k00.npz --eq eq2_30_k00_w8.npz --tcm tcm30_k00.npz --win 256 --out eq3_30_k00.npz
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_eq import ls_cir  # noqa: E402

W = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx", required=True)
    ap.add_argument("--eq", required=True)
    ap.add_argument("--tcm", required=True)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--smooth", type=int, default=3, help="windows in the moving average of the complex gains")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--span", type=int, default=220)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r, e, t = np.load(a.rx), np.load(a.eq), np.load(a.tcm)
    y2, rsign = r["y2"].astype(np.complex128), complex(r["rsign"])
    sym = t["sym"].astype(float)
    nsym = len(sym)
    k = np.arange(nsym)
    phi = e["phi"].astype(float)[:nsym].copy()
    c = np.zeros(2 * nsym, np.complex128)
    c[0::2] = sym * rsign ** (k % 4)
    C = sfft.fft(c, workers=W)
    for it in range(a.iters + 1):
        y = y2[: 2 * nsym] * np.repeat(np.exp(-1j * phi), 2)
        lg, h = ls_cir(y, c, a.span)
        n = len(y)
        hf = np.zeros(n, np.complex128)
        hf[lg % n] = h
        pred = sfft.ifft(C * sfft.fft(hf, workers=W), workers=W)
        ratio = np.sum(np.abs(y - pred) ** 2) / np.sum(np.abs(pred) ** 2)
        print("iter %d: beam residual / replica %.4f (%.2f dB)" % (it, ratio, 10 * np.log10(ratio)), flush=True)
        if it == a.iters:
            break
        m = n // a.win * a.win
        g = np.sum(np.conj(pred[:m]).reshape(-1, a.win) * y[:m].reshape(-1, a.win), axis=1) / np.sum(np.abs(pred[:m].reshape(-1, a.win)) ** 2, axis=1)
        if a.smooth > 1:
            g = np.convolve(g, np.ones(a.smooth) / a.smooth, mode="same")
        dph = np.unwrap(np.angle(g))
        tw = (np.arange(len(g)) + 0.5) * a.win / 2.0  # symbol index of each window centre
        phi = phi + np.interp(k, tw, dph)
        print("   phase update rms %.2f deg (window %d samples = %.1f us)" % (np.degrees(np.std(dph - dph.mean())), a.win, a.win / 2 / 10.762238), flush=True)
    np.savez(a.out, phi=phi.astype(np.float32), fslot=e["fslot"])
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
