#!/usr/bin/env python3
"""Fine-frequency array beam on one ATSC channel for the 8-VSB decode test.

Per input: the alias-cancelling stitch (atsc_ipfb) of the PFB channels into one wideband stream, then
the FFT of the whole record (1.2 Hz bins). Per block of B bins (~600 Hz, far inside the channel's
coherence bandwidth, so one transmitter is ONE spatial mode there): the per-input-whitened covariance
of the 128 inputs and its top eigenvector u_b; the beam is u_b^H x per bin, the eigenvector's phase
chained block to block (max Re u_{b-1}^H u_b) so the combined response is smooth across the band.
Unlike the per-195-kHz-channel beam (atsc_beam.py), no weight has to serve a band wider than the
multipath's coherence bandwidth, so the response has no notches. Output is atsc_beam's format (z, fs, fc).

    atsc_arr.py --chunk 0 --freq 2889-2937 --band 565.9e6,572.1e6 --out /var/tmp/bb1030/atsc/arr30_k00.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_beam import load_channel, CS  # noqa: E402
from atsc_ipfb import response  # noqa: E402

W = 8


def ipfb_inputs(X, eps, nfft=4096):
    """X: list over channels of [t, i] -> wideband per input [i, n_t * n_chan] (atsc_ipfb's inverse per fine bin)."""
    n = len(X)
    u = (np.arange(nfft) - nfft // 2) / nfft
    A0, Ap, Am = response(u), response(u + 1), response(u - 1)
    M = np.zeros((nfft, n, n))
    idx = np.arange(n)
    M[:, idx, idx] = A0[:, None]
    M[:, idx[:-1], idx[1:]] = Ap[:, None]
    M[:, idx[1:], idx[:-1]] = Am[:, None]
    MtM = np.einsum("jkl,jkm->jlm", M, M)
    P = np.linalg.solve(MtM + eps * np.max(A0 ** 2) * np.eye(n)[None], np.transpose(M, (0, 2, 1))).astype(np.complex64)
    Xa = np.stack(X)  # [k, t, i]
    ni = Xa.shape[2]
    n_t = Xa.shape[1] // nfft * nfft
    out = np.empty((ni, n_t * n), np.complex64)
    for b in range(0, n_t, nfft):
        F = sfft.fftshift(sfft.fft(Xa[:, b: b + nfft, :], axis=1, workers=W), axes=1)  # [l, j, i]
        Bj = np.matmul(P, np.ascontiguousarray(F.transpose(1, 0, 2)))  # [j, k, i]
        Wd = np.ascontiguousarray(Bj.transpose(2, 1, 0)).reshape(ni, n * nfft)  # [i, k*nfft + j]
        out[:, b * n: (b + nfft) * n] = sfft.ifft(sfft.ifftshift(Wd, axes=1), axis=1, workers=W) * n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--freq", default="2889-2937")
    ap.add_argument("--band", default="565.9e6,572.1e6", help="absolute band to beam (outside: zero)")
    ap.add_argument("--block", type=int, default=512, help="bins per covariance block")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--ref", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resdir", help="read channels from atsc_resid.py residuals (res_<fid>.npy) instead of the dump")
    ap.add_argument("--mode", type=int, default=0, help="beam on this eigenvector (0 = top)")
    a = ap.parse_args()
    t0 = time.time()
    lo, hi = (int(v) for v in a.freq.split("-"))
    fids = list(range(lo, hi + 1))
    if a.resdir:
        X = [np.load(os.path.join(a.resdir, "res_%d.npy" % fid)) for fid in fids]
    else:
        X = [load_channel(a.dir, a.chunk, fid) for fid in fids]
    xw = ipfb_inputs(X, a.eps)
    del X
    ni, n = xw.shape
    fs = len(fids) * CS
    fc = fids[0] * CS + (len(fids) - 1) * CS / 2.0
    print("per-input wideband: %d inputs x %d samples at %.4f MHz (%.0f s)" % (ni, n, fs / 1e6, time.time() - t0), flush=True)
    Xf = sfft.fft(xw, axis=1, workers=W)
    del xw
    f = sfft.fftfreq(n, 1 / fs) + fc
    b0, b1 = (float(v) for v in a.band.split(","))
    sel = np.nonzero((f >= b0) & (f < b1))[0]
    sel = sel[np.argsort(f[sel])]
    nb = len(sel) // a.block
    sel = sel[: nb * a.block]
    p = np.mean(np.abs(Xf[:, sel]) ** 2, axis=1)
    wt = (1 / np.sqrt(np.maximum(p, 1e-12))).astype(np.complex64)
    Xb = (Xf[:, sel] * wt[:, None]).reshape(ni, nb, a.block).transpose(1, 0, 2)  # [block, i, bin]
    R = np.matmul(Xb, np.conj(Xb.transpose(0, 2, 1))) / a.block  # [block, i, i]
    print("covariances: %d blocks of %d bins (%.0f Hz) (%.0f s)" % (nb, a.block, a.block * fs / n, time.time() - t0), flush=True)
    ev, U = np.linalg.eigh(R)
    u = U[:, :, -1 - a.mode]  # [block, i]
    noise = np.median(ev[:, 8:100], axis=1)
    lam = ev[:, -1] / noise
    lam2 = ev[:, -2] / noise
    lam3 = ev[:, -3] / noise
    print("eigenvalue / noise per block, percentiles 5/50/95: 1st %s; 2nd %s; 3rd %s" % tuple(
        " / ".join("%.1f" % v for v in np.percentile(q, [5, 50, 95])) for q in (lam, lam2, lam3)))
    # phase chain: align each block's eigenvector to the previous one
    ph = np.ones(nb, complex)
    for b in range(1, nb):
        c = np.vdot(u[b - 1] * ph[b - 1], u[b])
        ph[b] = np.conj(c) / max(abs(c), 1e-12)
    coh = np.abs(np.einsum("bi,bi->b", np.conj(u[:-1]), u[1:]))
    u = u * ph[:, None]
    # beam per bin: y = u_b^H (whitened x), scaled by sqrt(lam_b * noise) so the combined amplitude is the
    # transmitter's array-summed amplitude (smooth), not normalised away
    yb = np.einsum("bi,bij->bj", np.conj(u), Xb)  # [block, bin]
    Y = np.zeros(n, np.complex64)
    Y[sel] = yb.reshape(-1)
    z = sfft.ifft(Y, workers=W).astype(np.complex64)
    print("top eigenvalue / noise per block: percentiles 5/50/95 %.1f / %.1f / %.1f; second %.1f / %.1f / %.1f; "
          "block-to-block |u'u| median %.3f, 1st pct %.3f" % (
              *np.percentile(lam, [5, 50, 95]), *np.percentile(lam2, [5, 50, 95]), np.median(coh), np.percentile(coh, 1)))
    np.savez(a.out, z=z, fs=fs, fc=fc, fids=np.array(fids), lam=lam, lam2=lam2, coh=coh, band=np.array([b0, b1]),
             block=a.block)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
