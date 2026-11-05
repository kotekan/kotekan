#!/usr/bin/env python3
"""Iterative equaliser with a carrier phase tracker for the 8-VSB stream of atsc_rx.py.

Each pass: derotate the 2-per-symbol stream by the current phase track; least-squares channel on the
training symbols (every segment sync + the even fields' PN511; with --dd after the first passes, the
symbol decisions too); MMSE equaliser to the ideal raised cosine; the phase of every segment sync
(4 known symbols per 77 us), smoothed over --win segments, updates the phase track. MER is scored on
the odd fields' PN511 (never used for training or tracking).

    atsc_eq2.py --rx /var/tmp/bb1030/atsc/rxa30_k00.npz --iters 4 --win 8 --out /var/tmp/bb1030/atsc/eq2_30_k00.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import SEG, FIELD  # noqa: E402
from atsc_eq import pn511, rc_2sps, known_symbols, ls_cir, LEVELS  # noqa: E402

W = 8
SS = np.array([5, -5, -5, 5], float)


def equalise(y, lg, h, eps):
    n = len(y)
    hf = np.zeros(n, np.complex128)
    hf[lg % n] = h
    g = rc_2sps()
    gf = np.zeros(n)
    gl = np.arange(-(len(g) // 2), len(g) // 2 + 1)
    gf[gl % n] = g
    H = sfft.fft(hf, workers=W)
    E = sfft.fft(gf, workers=W) * np.conj(H) / (np.abs(H) ** 2 + eps * np.max(np.abs(H) ** 2))
    return sfft.ifft(sfft.fft(y, workers=W) * E, workers=W)


def smooth_complex(v, win):
    if win <= 1:
        return v
    k = np.ones(win) / win
    return np.convolve(v, k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--win", type=int, default=8, help="segments per phase estimate (77 us each)")
    ap.add_argument("--span", type=int, default=220)
    ap.add_argument("--eps", type=float, default=3e-3)
    ap.add_argument("--dd", type=int, default=0, help="from this pass on (>0), train on the decisions as well")
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.rx)
    y2, fslot, rsign = d["y2"].astype(np.complex128), int(d["fslot"]), complex(d["rsign"])
    L = 2 * SEG
    nseg = len(y2) // L
    y2 = y2[: nseg * L]
    nsym = nseg * SEG
    k = np.arange(nsym)
    rot = np.conj(rsign) ** (k % 4)
    c_train = known_symbols(nseg, fslot, rsign, "even")
    pn = pn511()
    odd = np.arange(fslot, nseg, FIELD)[1::2]
    data = np.ones((nseg, SEG), bool)
    data[:, :4] = False
    data[fslot::FIELD] = False
    phi = np.zeros(nsym)
    dec = None
    for it in range(a.iters):
        y = y2 * np.repeat(np.exp(-1j * phi), 2)
        if a.dd and it >= a.dd and dec is not None:
            c = np.zeros(2 * nsym, np.complex128)
            c[0::2] = dec * rsign ** (k % 4)
            kn = c_train != 0
            c[kn] = c_train[kn]
            lg, h = ls_cir(y, c, a.span)
            how = "decisions + training"
        else:
            lg, h = ls_cir(y, c_train, a.span)
            how = "training only"
        ye = equalise(y, lg, h, a.eps)
        u = ye[0::2] * rot
        U = u.reshape(nseg, SEG)
        beta = (U[:, :4] * SS).sum() / (nseg * np.sum(SS ** 2))
        u = u / beta
        U = u.reshape(nseg, SEG)
        # phase track from the segment syncs
        v = (U[:, :4] * SS).sum(axis=1)
        vs = smooth_complex(v, a.win)
        ph_seg = np.unwrap(np.angle(vs))
        dphi = np.interp(k, np.arange(nseg) * SEG + 1.5, ph_seg)
        uc = u * np.exp(-1j * dphi)
        ar = np.real(uc)
        Ar = ar.reshape(nseg, SEG)
        dc = Ar[data].mean()
        ar = ar - dc
        Ar = ar.reshape(nseg, SEG)
        dec = LEVELS[np.argmin(np.abs(ar[:, None] - LEVELS[None, :]), axis=1)]
        held = Ar[odd, 4:515]
        mer_h = 10 * np.log10(25.0 / np.mean((held - pn) ** 2))
        ser_h = np.mean(LEVELS[np.argmin(np.abs(held.reshape(-1)[:, None] - LEVELS[None, :]), axis=1)] != np.tile(pn, len(odd)))
        sign_h = np.mean(np.sign(held) != pn / 5)
        Dd = dec.reshape(nseg, SEG)
        mer_dd = 10 * np.log10(np.mean(Dd[data] ** 2) / np.mean((Ar[data] - Dd[data]) ** 2))
        ssok = np.mean(np.all(Dd[:, :4] == SS, axis=1))
        print("pass %d (%s): phase-track step rms %.1f deg; held-out PN511 MER %.1f dB, symbol errors %.4f, sign errors %.4f; "
              "seg syncs exact %.3f; decision MER %.1f dB" % (it, how, np.degrees(np.std(dphi - dphi.mean())), mer_h, ser_h, sign_h, ssok, mer_dd), flush=True)
        phi = phi + dphi + np.angle(beta)
    np.savez(a.out, a=ar.astype(np.float32), dec=dec.astype(np.int8), phi=phi.astype(np.float32), lg=lg, h=h, fslot=fslot, mer_held=mer_h)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
