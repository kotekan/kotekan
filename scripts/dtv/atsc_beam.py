#!/usr/bin/env python3
"""Wideband array beam on one ATSC channel of the 2026-10-30 baseband dump, for an 8-VSB decode test.

Per 195 kHz channel: the own-power-whitened covariance of all 128 inputs, its top eigenvector v
(the strongest transmitter), the beam y = sum_i conj(w_i) x_i with w = D^-1/2 v scaled so the beam is
the transmitter as seen at ONE reference input (the same input in every channel, so the stitched
response is that input's, continuous across channels). The channels are stitched into one wideband
stream (guowang_search.channels_to_baseband, conj=False) and the spectrum is searched for the pilot.

    atsc_beam.py --dir /var/tmp/bb1030/dtv --chunk 0 --freq 2897-2929 --out /var/tmp/bb1030/atsc/beam30_k00.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/python/scripts/gnss/guowang"))
import guowang_search as gs  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NT = 163840
CS = 195312.5
LUT = np.empty(256, dtype=np.complex64)
for b in range(256):
    LUT[b] = complex((b >> 4) - 8, (b & 0xF) - 8)


def load_channel(dirp, chunk, fid):
    r = fid % 16
    fr = np.load("%s/k%02d_r%02d_freq.npy" % (dirp, chunk, r))
    col = int(np.nonzero(fr == fid)[0][0])
    a = np.load("%s/k%02d_r%02d.npy" % (dirp, chunk, r), mmap_mode="r")
    return LUT[np.asarray(a[:, col]).reshape(NT, 128)]


def ipfb_beam(X, V, D, ref, freq_hz, eps, nfft=4096):
    """Per fine bin: undo the PFB aliasing across channels on every input (atsc_ipfb's tridiagonal
    Wiener inverse), then beam each channel with its own weights, then stitch."""
    from atsc_ipfb import response
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
    # weights per channel, in the reference input's units: [i, k]
    Wk = np.array([V[c] / np.sqrt(np.maximum(D[c], 1e-3)) * (np.sqrt(D[c][ref]) * np.conj(V[c][ref])) for c in range(n)]).T.astype(np.complex64)
    Xa = np.stack(X)  # [k, t, i]
    n_t = Xa.shape[1] // nfft * nfft
    out = np.empty(n_t * n, np.complex64)
    for b in range(0, n_t, nfft):
        F = sfft.fftshift(sfft.fft(Xa[:, b: b + nfft, :], axis=1, workers=8), axes=1)  # [l, j, i]
        G = (F.reshape(n * nfft, -1) @ Wk).reshape(n, nfft, n)  # G[l, j, k] = channel l's data beamed with channel k's weights
        B = np.einsum("jkl,ljk->kj", P, G)
        out[b * n: (b + nfft) * n] = sfft.ifft(sfft.ifftshift(B.reshape(-1)), workers=8) * n
    fs = n * CS
    fc = freq_hz.min() + (n - 1) * CS / 2.0
    return out, fs, fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--freq", default="2897-2929")
    ap.add_argument("--ref", type=int, default=-1, help="reference input (default: best median |v|)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ipfb", action="store_true", help="alias-cancelling stitch per input before beamforming (atsc_ipfb.py)")
    ap.add_argument("--eps", type=float, default=1e-3)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.freq.split("-"))
    fids = list(range(lo, hi + 1))
    t0 = time.time()
    V, D, LAM, X = [], [], [], []
    for fid in fids:
        x = load_channel(a.dir, a.chunk, fid)
        p = (np.abs(x) ** 2).mean(axis=0).astype(np.float64)
        xw = x / np.sqrt(np.maximum(p, 1e-3)).astype(np.float32)
        R = (xw.T @ xw.conj()).astype(np.complex128) / NT  # R_ik = <x_i conj(x_k)>
        ev, U = np.linalg.eigh(R)
        noise = np.mean(ev[8:108])
        V.append(U[:, -1].conj())  # with R_ik = <x_i x_k*>, the beam yw @ conj(u) maximises; keep conj(u) as the weight
        D.append(p)
        LAM.append(ev[::-1][:6] / noise)
        X.append(x)
        print("freq_id %d (%.3f MHz): lam/noise %s" % (fid, fid * CS / 1e6, " ".join("%.1f" % v for v in LAM[-1])), flush=True)
    V, D, LAM = np.array(V), np.array(D), np.array(LAM)
    ref = a.ref if a.ref >= 0 else int(np.argmax(np.median(np.abs(V), axis=0)))
    print("reference input %d (median |v| %.3f, max %.3f)" % (ref, np.median(np.abs(V[:, ref])), np.median(np.abs(V), axis=0).max()))
    beams = np.empty((len(fids), NT), np.complex64)
    for c, x in enumerate(X):
        # beam b = sum_i (x_i / sqrt(D_i)) * wv_i with wv = conj(u): b = c_tx * sum|u|^2 = c_tx for x = c_tx sqrt(D) u.
        # The transmitter at the reference input is c_tx * sqrt(D_ref) * u_ref, so scale by sqrt(D_ref) u_ref = sqrt(D_ref) conj(wv_ref).
        wv = V[c] / np.sqrt(np.maximum(D[c], 1e-3))
        b = x @ wv.astype(np.complex64)
        beams[c] = b * (np.sqrt(D[c][ref]) * np.conj(V[c][ref]))
    freq_hz = np.array(fids) * CS
    if a.ipfb:
        z, fs, fc = ipfb_beam(X, V, D, ref, freq_hz, a.eps)
    else:
        z, fs, fc = gs.channels_to_baseband(beams, freq_hz, CS, conj=False, nfft=4096)
    del X
    # pilot search: the strongest narrow line in the stitched spectrum
    n = len(z)
    Z = sfft.fftshift(sfft.fft(z.astype(np.complex128), workers=8))
    f = sfft.fftshift(sfft.fftfreq(n, 1 / fs)) + fc
    P = np.abs(Z) ** 2
    k = int(np.argmax(P))
    med = np.median(P)
    print("stitched: %d samples at %.4f MHz, centre %.4f MHz; strongest line %.6f MHz at %.1f dB over the median bin"
          % (n, fs / 1e6, fc / 1e6, f[k] / 1e6, 10 * np.log10(P[k] / med)))
    order = np.argsort(P)[::-1]
    shown = []
    for j in order:
        if all(abs(f[j] - s) > 5e3 for s in shown):
            shown.append(f[j])
            print("  line %.6f MHz  %.1f dB" % (f[j] / 1e6, 10 * np.log10(P[j] / med)))
        if len(shown) >= 6:
            break
    # coarse spectrum in 100 kHz bins
    eb = np.arange(f[0], f[-1], 100e3)
    idx = np.digitize(f, eb)
    prof = np.array([P[idx == i].mean() if np.any(idx == i) else 0 for i in range(1, len(eb))])
    print("spectrum (100 kHz bins, dB rel median) from %.2f MHz:" % (eb[0] / 1e6))
    print(" ".join("%.0f" % (10 * np.log10(max(v, 1e-30) / med)) for v in prof))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, z=z, fs=fs, fc=fc, fids=np.array(fids), lam=LAM, v=V, d=D, ref=ref, beams=beams)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
