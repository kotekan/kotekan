#!/usr/bin/env python3
"""Alias-cancelling stitch of contiguous critically sampled PFB channels into one wideband stream.

A critically sampled PFB channel k, read at fine frequency u (in channel units, -1/2..1/2), holds
its own content and the edge content of its neighbours folded in:
    Y_k(u) = H(u) X_k(u) + H(u + 1) X_{k+1}(u) + H(u - 1) X_{k-1}(u)
with H the window's response (one function for every channel). Per fine bin this is a tridiagonal
system across the channels; solving it (Wiener-regularised) undoes the aliasing that
guowang_search.channels_to_baseband leaves at every channel edge. Channels outside the set are
taken as empty, so keep a margin channel on each side of the band of interest.

    python3 atsc_ipfb.py sim [--window hamming] [--nch 33]      # fidelity on white noise vs the plain stitch
    python3 atsc_ipfb.py coh --dir D --chunk 0 --freq 2899-2927  # measured adjacent-channel coherence vs models
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

CS = 195312.5
W = 8


def window(name, L):
    if name == "hamming":
        return np.hamming(L)
    if name == "hann":
        return np.hanning(L)
    if name == "rect":
        return np.ones(L)
    raise SystemExit("window %s?" % name)


def response(u, win="hamming", taps=4, nsub=512):
    """A(u): the PFB channel response at offset u channels, with the common linear phase removed
    (real for a symmetric window), normalised to A(0) = 1."""
    L = taps * nsub
    m = np.arange(L)
    w = np.sinc((m - L / 2 + 0.5) / nsub) * window(win, L)
    t = (m - (L - 1) / 2) / nsub  # centred, in channel periods
    u = np.atleast_1d(np.asarray(u, float))
    A = np.array([np.sum(w * np.cos(2 * np.pi * uu * t)) for uu in u])  # symmetric window: real
    return A / np.sum(w)


def stitch_ipfb(chan, freq_hz, fs_chan=CS, win="hamming", sign=1.0, nfft=4096, eps=1e-3):
    """chan [n_chan, n_t] (contiguous grid channels), returns (x, fs, fc) like channels_to_baseband."""
    chan = np.asarray(chan)
    f = np.asarray(freq_hz, float)
    slot = np.round((f - f.min()) / fs_chan).astype(int)
    n_grid = int(slot.max()) + 1
    if len(slot) != n_grid:
        raise SystemExit("stitch_ipfb needs contiguous channels")
    n_t = chan.shape[1] // nfft * nfft
    fs = n_grid * fs_chan
    fc = f.min() + (n_grid - 1) * fs_chan / 2.0
    u = (np.arange(nfft) - nfft // 2) / nfft
    A0, Ap, Am = response(u, win), sign * response(u + 1, win), sign * response(u - 1, win)
    # per fine bin j: M_j = diag(A0) + super-diagonal Ap (X_{k+1} into Y_k) + sub-diagonal Am (X_{k-1} into Y_k)
    n = n_grid
    M = np.zeros((nfft, n, n))
    idx = np.arange(n)
    M[:, idx, idx] = A0[:, None]
    M[:, idx[:-1], idx[1:]] = Ap[:, None]
    M[:, idx[1:], idx[:-1]] = Am[:, None]
    MtM = np.einsum("jkl,jkm->jlm", M, M)
    lam = eps * np.max(A0 ** 2)
    P = np.linalg.solve(MtM + lam * np.eye(n)[None], np.transpose(M, (0, 2, 1)))  # [nfft, n, n]
    out = np.empty(n_t * n_grid, np.complex64)
    for b in range(0, n_t, nfft):
        F = sfft.fftshift(sfft.fft(chan[:, b: b + nfft], axis=1, workers=W), axes=1)  # [n, nfft]
        X = np.einsum("jkl,lj->kj", P, F)
        out[b * n_grid: (b + nfft) * n_grid] = sfft.ifft(sfft.ifftshift(X.reshape(-1)), workers=W) * n_grid
    return out, fs, fc


def pfb_sim(x, nch, win="hamming", taps=4):
    """dg_stitchcheck.py's complex PFB: frames of nch, window taps*nch sinc*win, fftshifted channels [nt, nch]."""
    L = taps * nch
    m = np.arange(L)
    w = np.sinc((m - L / 2 + 0.5) / nch) * window(win, L)
    nt = len(x) // nch
    xp = np.concatenate([np.zeros(L - nch, x.dtype), x])
    idx = np.arange(nt)[:, None] * nch + m[None, :]
    seg = (xp[idx] * w).reshape(nt, taps, nch).sum(axis=1)
    return sfft.fftshift(sfft.fft(seg, axis=1, workers=W), axes=1) / np.sqrt(nch)


def best_corr(x, y):
    nf = sfft.next_fast_len(2 * len(y))
    r = sfft.ifft(sfft.fft(x[: len(y)], nf) * np.conj(sfft.fft(y, nf)))
    j = int(np.argmax(np.abs(r)))
    lag = j if j < nf // 2 else j - nf
    xs = np.roll(x[: len(y)], -lag)
    g = np.vdot(y, xs) / np.vdot(y, y)
    res = xs - g * y
    sl = slice(4096, len(y) - 4096)  # drop the block edges of the circular shift
    return np.abs(r[j]) / (np.linalg.norm(x[: len(y)]) * np.linalg.norm(y)), lag, np.sum(np.abs(res[sl]) ** 2) / np.sum(np.abs(xs[sl]) ** 2)


def cmd_sim(a):
    sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/python/scripts/gnss/guowang"))
    import guowang_search as gs
    rng = np.random.default_rng(0)
    nch, nt = a.nch, 16384
    x = (rng.standard_normal(nch * nt) + 1j * rng.standard_normal(nch * nt))
    C = pfb_sim(x, nch, a.window)
    freq_hz = (np.arange(nch) - nch // 2) * CS
    # keep the inner channels only, as with real data (the outer ones see un-modelled neighbours)
    keep = slice(2, nch - 2)
    xb = x  # the reference: the input band-limited to the kept channels
    X = sfft.fft(xb)
    fr = sfft.fftfreq(len(xb), 1 / (nch * CS))
    lo, hi = freq_hz[keep][0] - CS / 2, freq_hz[keep][-1] + CS / 2
    X[(fr < lo) | (fr >= hi)] = 0
    xb = sfft.ifft(X)
    for name, fn in (("plain", lambda: gs.channels_to_baseband(C.T[keep].astype(np.complex64), freq_hz[keep], CS, conj=False, nfft=4096)),
                     ("ipfb", lambda: stitch_ipfb(C.T[keep], freq_hz[keep], CS, win=a.window, nfft=4096, eps=a.eps))):
        y, fs, fc = fn()
        # bring the reference to the stitched grid: shift to fc, decimate in frequency to the stitched band
        n_grid = len(freq_hz[keep])
        t = np.arange(len(xb)) / (nch * CS)
        xr = xb * np.exp(-2j * np.pi * fc * t)
        XR = sfft.fftshift(sfft.fft(xr))
        nyb = len(y)
        mid = len(XR) // 2
        xr2 = sfft.ifft(sfft.ifftshift(XR[mid - nyb // 2: mid + nyb // 2])) * nyb / len(XR)
        c, lag, rr = best_corr(xr2, y)
        xs = np.roll(xr2[: len(y)], -lag); g = np.vdot(y, xs) / np.vdot(y, y); res = xs - g * y
        Rr = np.abs(sfft.fftshift(sfft.fft(res))) ** 2; Ss = np.abs(sfft.fftshift(sfft.fft(xs))) ** 2
        ff = (np.arange(len(y)) - len(y) // 2) / len(y) * n_grid + n_grid / 2  # position in channel units from the band's lower edge
        ch = np.clip(ff.astype(int), 0, n_grid - 1); pos = ((ff % 1) * 16).astype(int)
        print("   residual/signal by channel:", " ".join("%.3f" % (Rr[ch == k].sum() / Ss[ch == k].sum()) for k in range(n_grid)))
        inner = (ch >= 2) & (ch < n_grid - 2)
        print("   inner channels by position within the channel (16 sub-bins):", " ".join("%.3f" % (Rr[inner & (pos == i)].sum() / Ss[inner & (pos == i)].sum()) for i in range(16)))
        print("   inner channels total: %.4f (%.1f dB)" % (Rr[inner].sum() / Ss[inner].sum(), 10 * np.log10(Rr[inner].sum() / Ss[inner].sum())))
        print("%s stitch (%s window, %d kept channels): peak correlation %.4f at lag %d, residual/signal %.4f (%.1f dB)"
              % (name, a.window, n_grid, c, lag, rr, 10 * np.log10(rr)))


def cmd_coh(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from atsc_beam import load_channel
    lo, hi = (int(v) for v in a.freq.split("-"))
    nfft = 1024
    Y = []
    for fid in range(lo, hi + 1):
        x = load_channel(a.dir, a.chunk, fid)[:, : a.inputs]  # [NT, ni]
        n_b = x.shape[0] // nfft
        F = sfft.fftshift(sfft.fft(x[: n_b * nfft].reshape(n_b, nfft, -1), axis=1, workers=W), axes=1)
        Y.append(F)
    Y = np.array(Y)  # [k, block, nu, input]
    cross = np.sum(Y[:-1] * np.conj(Y[1:]), axis=(0, 1, 3))
    p0 = np.sum(np.abs(Y[:-1]) ** 2, axis=(0, 1, 3))
    p1 = np.sum(np.abs(Y[1:]) ** 2, axis=(0, 1, 3))
    coh = cross / np.sqrt(p0 * p1)
    u = (np.arange(nfft) - nfft // 2) / nfft
    sel = np.r_[0:24:2, 24:64:8, 64:nfft - 64:64, nfft - 64:nfft - 24:8, nfft - 24:nfft:2]
    models = {}
    for win in ("hamming", "hann", "rect"):
        A0, Ap, Am = response(u, win), response(u + 1, win), response(u - 1, win)
        # flat spectrum: coherence of Y_k, Y_{k+1} = (A0 Am + Ap A0) / (A0^2 + Ap^2 + Am^2)
        models[win] = (A0 * Am + Ap * A0) / (A0 ** 2 + Ap ** 2 + Am ** 2)
    print("u      |coh| meas  phase(deg)   model hamming / hann / rect")
    for j in sel:
        print("%+.4f  %.3f   %+7.1f      %.3f / %.3f / %.3f" % (u[j], abs(coh[j]), np.degrees(np.angle(coh[j])),
                                                               models["hamming"][j], models["hann"][j], models["rect"][j]))
    for win, m in models.items():
        print("rms |coh| error vs %s: %.4f" % (win, np.sqrt(np.mean((np.abs(coh) - np.abs(m)) ** 2))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sim")
    p.add_argument("--window", default="hamming")
    p.add_argument("--nch", type=int, default=37)
    p.add_argument("--eps", type=float, default=1e-3)
    p.set_defaults(fn=cmd_sim)
    p = sub.add_parser("coh")
    p.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--freq", default="2899-2927")
    p.add_argument("--inputs", type=int, default=128)
    p.set_defaults(fn=cmd_coh)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
