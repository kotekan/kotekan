#!/usr/bin/env python3
"""GuoWang code search (#218): find the spreading code of a GNSS-like signal in complex baseband.

Input is one complex stream (a single element, or a beam formed toward the satellite) with its
sample rate and centre frequency. The lobe of interest (b3: 1268.52 MHz / 10.23 Mcps; l3: 1202.025
MHz / ~2.046 Mcps) is mixed to 0 Hz and resampled to 2 samples per nominal chip. Then:

  1. blind estimators, which need no code:
     - chip rate: the line at the chip rate in x(t) x*(t - d) (the chip transitions);
     - code period: the lag of the strongest peak of the segmented delay-and-multiply
       sum_s |sum_{t in s} x(t) x*(t + lag)|^2, which data bits cannot cancel (segments are shorter
       than a bit), with the full-record (coherent) autocorrelation beside it: equal = no data on
       the code, i.e. spectral lines at 1/period;
     - carrier: the line at order x Doppler in x^order (order 2 for BPSK, 4 for QPSK), searched
       semi-coherently over Doppler and Doppler rate, then refined coherently over the record;
  2. blind code recovery: wipe the carrier, fold the record at the code period (with the code
     Doppler and its rate), align the periods' data signs by power iteration, read the chips; then
     identify them against the code families and run Berlekamp-Massey on them;
  3. parallel-code-phase acquisition (FFT circular correlation per code period, non-coherent sum
     over periods) against the code families over Doppler and Doppler rate, with the code Doppler
     applied as a fractional delay per period;
  4. verification of the best code: the code delay tracked over the record gives the received chip
     rate, the code period and the code Doppler, to compare with the carrier Doppler.

    python guowang_search.py run capture.npz --lobe b3 --families b3i,b3i_all
    python guowang_search.py run capture.npy --fs 25e6 --fc 1268.52e6 --lobe b3

Inputs: .npy (complex, needs --fs/--fc), .npz with x, fs, fc, or a channelised .npz with chan
[n_chan, n_t], freq_hz [n_chan], fs_chan (stitched by channels_to_baseband; --conj for the CHORD
F-engine's conjugated output), or raw interleaved ci8/ci16 (--format, --fs, --fc).
FFTs use scipy.fft with GUOWANG_FFT_WORKERS threads (default 4). CPU only.
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
import scipy.fft as sfft
import scipy.signal as ssig
import scipy.special as sspec
import scipy.stats as sstats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import guowang_codes as gcodes  # noqa: E402

WORKERS = int(os.environ.get("GUOWANG_FFT_WORKERS", "4"))
LOBES = {"b3": (1268.52e6, 10.23e6), "l3": (1202.025e6, 2.046e6)}
NOMINAL_CHIPS = {
    "b3": 10230,
    "l3": 2046,
}  # 1 ms, B3I / B1I-like: the fold's guess with no period peak
GAP_DB = (
    6.0  # a code is identified when its excess is this far above its family's runner-up
)
DEFAULT_FAMILIES = {"b3": "b3i,b3i_all,l3oc", "l3": "b1i,b1i_all"}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- input


def channels_to_baseband(chan, freq_hz, fs_chan, conj=False, nfft=4096):
    """Stitch channelised data [n_chan, n_t] (channel centres freq_hz on a grid of fs_chan) into one
    wideband stream at n_grid * fs_chan: per block of nfft samples, FFT each channel, place its fine
    spectrum at its centre, inverse FFT. Missing grid channels are zero. PFB ripple and the
    aliasing at channel edges are ignored; the PFB's group delay (taps / 2 FFT lengths) stays in.
    Returns (x, fs, fc)."""
    chan = np.asarray(chan)
    if conj:
        chan = np.conj(chan)
    f = np.asarray(freq_hz, float)
    slot = np.round((f - f.min()) / fs_chan).astype(int)
    n_grid = int(slot.max()) + 1
    n_t = chan.shape[1] // nfft * nfft
    fs = n_grid * fs_chan
    fc = f.min() + (n_grid - 1) * fs_chan / 2.0
    out = np.empty(n_t * n_grid, np.complex64)
    for b in range(0, n_t, nfft):
        F = sfft.fftshift(
            sfft.fft(chan[:, b : b + nfft], axis=1, workers=WORKERS), axes=1
        )
        W = np.zeros((n_grid, nfft), np.complex64)
        W[slot] = F
        # the grid's fine bins run from fc - fs/2 upward; wideband bin order is fftshifted
        out[b * n_grid : (b + nfft) * n_grid] = (
            sfft.ifft(sfft.ifftshift(W.reshape(-1)), workers=WORKERS) * n_grid
        )
    return out, fs, fc


def load(path, fs=None, fc=None, fmt=None, conj=False):
    """(x complex64, fs, fc) from .npy/.npz/raw (see the module docstring)."""
    if path.endswith(".npz"):
        d = np.load(path)
        if "chan" in d:
            return channels_to_baseband(
                d["chan"], d["freq_hz"], float(d["fs_chan"]), conj=conj
            )
        x = d["x"].astype(np.complex64)
        return (
            x,
            float(d["fs"]) if fs is None else fs,
            float(d["fc"]) if fc is None else fc,
        )
    if path.endswith(".npy"):
        x = np.load(path).astype(np.complex64)
    else:
        dt = {"ci8": np.int8, "ci16": np.int16}[fmt]
        r = np.fromfile(path, dt).astype(np.float32)
        x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
    if fs is None or fc is None:
        raise SystemExit("%s: --fs and --fc are required" % path)
    return (np.conj(x) if conj else x), fs, fc


def extract_lobe(x, fs, fc, f_car, chip_rate, n_spc=2, n_max=None):
    """Mix f_car to 0 Hz and resample to n_spc * chip_rate (polyphase, scipy.signal.resample_poly).
    Returns (y complex64, fs_out)."""
    x = x[:n_max]
    fs_out = n_spc * chip_rate
    ratio = (
        Fraction(repr(float(fs_out))) / Fraction(repr(float(fs)))
    ).limit_denominator(20000)
    if abs(float(ratio) - fs_out / fs) > 1e-12 * fs_out / fs:
        raise ValueError(
            "fs_out/fs = %r has no small rational form; resample first" % (fs_out / fs)
        )
    df = f_car - fc
    y = np.empty(len(x), np.complex64)
    step = 1 << 22
    for a in range(0, len(x), step):
        n = np.arange(a, min(a + step, len(x)), dtype=np.float64)
        ph = 2 * np.pi * ((df / fs * n) % 1.0)
        y[a : a + len(n)] = x[a : a + len(n)] * np.exp(-1j * ph).astype(np.complex64)
    if ratio != 1:
        y = ssig.resample_poly(y, ratio.numerator, ratio.denominator).astype(
            np.complex64
        )
    return y, fs_out


def wipe(y, fs, f_c, rate, t_c, phase=0.0):
    """y * exp(-j (2 pi (f_c (t - t_c) + rate (t - t_c)^2 / 2) + phase)), t in s from the first sample."""
    out = np.empty(len(y), np.complex64)
    step = 1 << 22
    for a in range(0, len(y), step):
        t = np.arange(a, min(a + step, len(y))) / fs - t_c
        out[a : a + len(t)] = y[a : a + len(t)] * np.exp(
            -1j * (2 * np.pi * (f_c * t + 0.5 * rate * t ** 2) + phase)
        )
    return out


def boxcar_decimate(y, q):
    n = len(y) // q * q
    return y[:n].reshape(-1, q).mean(axis=1) if q > 1 else y


def gamma_threshold(K, n_trials, pfa=1e-3):
    """Level (in units of the noise mean) that a sum of K exponentials exceeds with probability
    pfa over n_trials independent cells."""
    return float(sstats.gamma.isf(pfa / max(n_trials, 1), K, scale=1.0 / K))


def noise_mean(a, K):
    """Mean of a sum-of-K-exponentials population from its median (robust to a few signal cells)."""
    return float(np.median(a)) * K / max(K - 1.0 / 3.0, 0.5)


def _parabola(a, b, c):
    den = a - 2 * b + c
    return 0.5 * (a - c) / den if den != 0 else 0.0


# --------------------------------------------------------------------------- blind estimators


def blind_chip_rate(y, fs, rc_lo, rc_hi, t_use=0.5, lag=1, pfa=1e-3, pad=4):
    """Chip-rate line in z = y(t) y*(t - lag): E[z] is periodic at the chip period whatever the code,
    data or carrier. Needs rc_hi < fs/2. Returns the received chip rate and its significance."""
    n = min(len(y), int(t_use * fs))
    z = y[lag:n] * np.conj(y[: n - lag])
    z = z - z.mean()
    nfft = sfft.next_fast_len(pad * len(z))
    Z = np.abs(sfft.fft(z, nfft, workers=WORKERS)) ** 2
    f = np.arange(nfft) * fs / nfft
    sel = np.where((f >= rc_lo) & (f <= rc_hi))[0]
    p = Z[sel] + Z[(nfft - sel) % nfft]  # the +Rc and -Rc lines
    mu = noise_mean(p, 2)
    k = int(np.argmax(p))
    frac = _parabola(p[k - 1], p[k], p[k + 1]) if 0 < k < len(p) - 1 else 0.0
    thr = gamma_threshold(2, len(sel) / pad, pfa)
    return dict(
        chip_rate=float(f[sel[k]] + frac * fs / nfft),
        snr=float(p[k] / mu),
        thr=thr,
        detected=bool(p[k] / mu > thr),
        t_use=n / fs,
    )


def blind_period(
    y,
    fs,
    tau_min=0.05e-3,
    tau_max=2.2e-3,
    t_seg=0.25e-3,
    t_use=0.5,
    decim=1,
    pfa=1e-3,
    top=6,
):
    """Code period from the segmented delay-and-multiply (see the module docstring). Returns the
    significant lags (strongest first), each with its coherent (full-record) |R|/R(0)."""
    yd = boxcar_decimate(y[: int(t_use * fs) + int(tau_max * fs) + 1], decim)
    fsd = fs / decim
    L = max(8, int(round(t_seg * fsd)))
    m0, m1 = int(np.floor(tau_min * fsd)), int(np.ceil(tau_max * fsd))
    W = m1 - m0 + 1
    nseg = (len(yd) - m1 - L) // L
    if nseg < 4:
        raise ValueError("record too short for tau_max %.3g s" % tau_max)
    nfft = sfft.next_fast_len(L + W + L)
    P = np.zeros(W)
    for s0 in range(0, nseg, 64):
        ss = np.arange(s0, min(s0 + 64, nseg)) * L
        A = np.stack([yd[s : s + L] for s in ss])
        B = np.stack([yd[s + m0 : s + m0 + W + L] for s in ss])
        R = sfft.ifft(
            np.conj(sfft.fft(A, nfft, axis=1, workers=WORKERS))
            * sfft.fft(B, nfft, axis=1, workers=WORKERS),
            axis=1,
            workers=WORKERS,
        )[:, :W]
        P += (np.abs(R) ** 2).sum(axis=0)
    S = P / noise_mean(P, nseg)
    thr = gamma_threshold(nseg, W, pfa)
    # coherent full-record autocorrelation at the same lags
    n = nseg * L
    yc = yd[: n + m1]
    nf = sfft.next_fast_len(2 * len(yc))
    Yf = sfft.fft(yc, nf, workers=WORKERS)
    Rc = sfft.ifft(np.abs(Yf) ** 2, workers=WORKERS)
    Rc = np.abs(Rc[m0 : m1 + 1]) / np.abs(Rc[0])
    peaks = []
    S2 = S.copy()
    for _ in range(top):
        k = int(np.argmax(S2))
        if S2[k] <= thr:
            break
        frac = _parabola(S[k - 1], S[k], S[k + 1]) if 0 < k < W - 1 else 0.0
        peaks.append(
            dict(period=(m0 + k + frac) / fsd, snr=float(S[k]), coherent=float(Rc[k]))
        )
        S2[max(0, k - 3) : k + 4] = 0
    return dict(peaks=peaks, thr=thr, n_seg=nseg, fs=fsd)


def fundamental(peaks, frac=0.5, standout=3.0):
    """The code period among the period peaks, or None: the shortest lag harmonically related to the
    strongest and at least frac of it (its multiples are as strong, and data / secondary codes can
    tip the order). A strong code also lifts lags that are not its period a little (its own
    correlation structure, not noise), so the strongest must stand standout x above the unrelated
    peaks' median excess."""
    if not peaks:
        return None
    top = max(peaks, key=lambda p: p["snr"])

    def related(p):
        r = max(p["period"], top["period"]) / min(p["period"], top["period"])
        return abs(r - round(r)) < 0.01

    others = [p["snr"] - 1.0 for p in peaks if not related(p)]
    if others and top["snr"] - 1.0 < standout * float(np.median(others)):
        return None
    return min(
        p["period"] for p in peaks if related(p) and p["snr"] >= frac * top["snr"]
    )


def blind_carrier(
    y,
    fs,
    order=2,
    f_lo=-35e3,
    f_hi=35e3,
    r_lo=-200.0,
    r_hi=200.0,
    t_seg=0.25,
    t_use=1.0,
    pfa=1e-3,
    top=4,
    t_ref=None,
):
    """Doppler and Doppler rate from the line at order x f in y^order, semi-coherent over segments
    of t_seg after a global dechirp per trial rate, then refined coherently over t_use. Times are s
    from the first sample; f_c is the Doppler at t_c (the centre of the span used)."""
    n = min(len(y), int(t_use * fs))
    span = max(abs(f_lo), abs(f_hi)) * order
    q = max(1, int(fs / (2.5 * span)))
    z = boxcar_decimate(y[:n] ** order, q)
    fsd = fs / q
    T = len(z) / fsd
    t = np.arange(len(z)) / fsd
    t_c = T / 2.0 if t_ref is None else t_ref
    tt = t - t_c
    L = int(round(t_seg * fsd))
    nseg = len(z) // L
    nfft = sfft.next_fast_len(2 * L)
    dr = min(1.0 / (order * t_seg ** 2), 1.0 / (order * T * t_seg))
    rates = np.arange(r_lo, r_hi + dr / 2, dr) if r_hi > r_lo else np.array([r_lo])
    fb = sfft.fftfreq(nfft, 1.0 / fsd)
    sel = np.where((fb >= order * f_lo) & (fb <= order * f_hi))[0]
    S = np.empty((len(rates), len(sel)), np.float32)
    win = np.hanning(L).astype(
        np.float32
    )  # sidelobes that fall fast: a burst line is ~1e6 x noise
    for i, r in enumerate(rates):
        w = z * np.exp(-1j * np.pi * order * r * tt ** 2)
        seg = w[: nseg * L].reshape(nseg, L) * win
        S[i] = (np.abs(sfft.fft(seg, nfft, axis=1, workers=WORKERS)[:, sel]) ** 2).sum(
            axis=0
        )
    mu = noise_mean(S, nseg)
    S /= mu
    thr = gamma_threshold(nseg, S.size / 2.0, pfa)
    lines = []
    S2 = S.copy()
    fsel = fb[sel] / order
    for _ in range(top):
        i, k = np.unravel_index(int(np.argmax(S2)), S2.shape)
        if S2[i, k] <= thr and lines:
            break
        ref = _refine_line(z, tt, order, fsel[k], rates[i], fsd, T)
        ref.update(snr_semi=float(S[i, k]), detected=bool(S[i, k] > thr), t_c=t_c)
        lines.append(ref)
        # blank the line and its leakage: dechirped at a wrong rate r it spreads over
        # |r - r_line| T / 2 around its frequency, and the window's sidelobes (~n^-6 at n bins)
        # stay above the threshold out to ~(peak / thr)^(1/6) bins
        hw = (
            np.abs(rates - ref["rate"]) * T / 2
            + (2.0 + (S[i, k] / thr) ** (1 / 6.0)) / t_seg
        )
        S2[np.abs(fsel[None, :] - ref["f_c"]) < hw[:, None]] = 0
    return dict(
        lines=lines, thr=thr, n_seg=nseg, rate_step=dr, n_rates=len(rates), order=order
    )


def _refine_line(z, tt, order, f0, r0, fsd, T):
    """Coherent zoom on |sum z exp(-j(2 pi order f t + pi order r t^2))| around (f0, r0)."""
    df, dr = 1.0 / (order * T), 4.0 / (order * T ** 2)
    best = (f0, r0)
    for _ in range(4):
        fs_ = best[0] + df * np.linspace(-2, 2, 9)
        rs_ = best[1] + dr * np.linspace(-2, 2, 9)
        V = np.empty((len(rs_), len(fs_)))
        for i, r in enumerate(rs_):
            w = z * np.exp(-1j * np.pi * order * r * tt ** 2)
            V[i] = np.abs(np.exp(-2j * np.pi * order * np.outer(fs_, tt)) @ w)
        i, k = np.unravel_index(int(np.argmax(V)), V.shape)
        best = (fs_[k], rs_[i])
        df /= 4.0
        dr /= 4.0
    f, r = best
    s = np.sum(
        z * np.exp(-1j * (2 * np.pi * order * f * tt + np.pi * order * r * tt ** 2))
    )
    noise = np.sqrt(np.sum(np.abs(z - np.mean(z)) ** 2))
    return dict(
        f_c=float(f),
        rate=float(r),
        phase=float(np.angle(s) / order),
        coh=float(np.abs(s) / noise),
    )


# --------------------------------------------------------------------------- blind code recovery


def _n_sub(nb, want=16):
    """A divisor of nb near want (sub-blocks per period for the fold's sign fit)."""
    divs = [d for d in range(4, 65) if nb % d == 0]
    return min(divs, key=lambda d: abs(d - want)) if divs else 1


def blind_fold(
    y,
    fs,
    period,
    f_c,
    rate,
    t_c,
    phase,
    f_rf,
    n_chips,
    bins_per_chip=2,
    t_use=0.5,
    n_iter=30,
):
    """Recover the chips: wipe the carrier, fold at the received period (with the code Doppler rate),
    fit a template and a unit-modulus weight per sub-block of each period, read the chips.

    The weights carry the data / secondary-code signs, which change at the code epoch, not at the
    fold origin, so each period is cut into sub-blocks with their own weights. That leaves each
    sub-block of the template with its own phase: the phases are chained across the sub-block
    boundaries where the weight ratio is constant, and the one boundary where it is random holds
    the epoch, which is then placed to the bin. Chip 0 of the result is the epoch when the signs
    change (else the fold origin). Also returns the chip SNR, the energy of the first two
    components (BPSK vs QPSK), and the sign of each code period."""
    n = min(len(y), int(t_use * fs))
    t = np.arange(n) / fs - t_c
    yw = y[:n] * np.exp(
        -1j * (2 * np.pi * (f_c * t + 0.5 * rate * t ** 2) + phase)
    ).astype(np.complex64)
    T0 = period * (1.0 + f_c / f_rf)
    chi = t / period + 0.5 * rate * t ** 2 / (f_rf * T0)
    chi -= np.floor(chi[0])
    p = np.floor(chi).astype(np.int64)
    NB = int(n_chips * bins_per_chip)
    b = np.minimum((chi - p) * NB, NB - 1).astype(np.int64)
    n_p = int(p[-1]) + 1
    idx = p * NB + b
    V = (
        np.bincount(idx, yw.real, n_p * NB) + 1j * np.bincount(idx, yw.imag, n_p * NB)
    ).reshape(n_p, NB)
    cnt = np.bincount(idx, minlength=n_p * NB).reshape(n_p, NB)
    V, cnt = V[1:-1], cnt[1:-1]  # partial first/last periods
    n_p = V.shape[0]
    S = _n_sub(NB)
    L = NB // S
    # [S, n_p, L]: sub-block-major so each sub-block's fit is a batched matrix-vector product
    Vt = np.ascontiguousarray(V.reshape(n_p, S, L).transpose(1, 0, 2)).astype(
        np.complex64
    )
    Ct = np.ascontiguousarray(cnt.reshape(n_p, S, L).transpose(1, 0, 2)).astype(
        np.float32
    )
    del V, cnt

    def components(Vt):
        """Rank-1 fit Vt[s, p, l] ~ a[s, p] u[s, l] Ct[s, p, l] by alternating least squares (power
        iteration); a projected to unit modulus (data signs) in the second half."""
        u = (np.random.default_rng(0).standard_normal((S, L)) + 0j).astype(np.complex64)
        for it in range(n_iter):
            a = (Vt @ np.conj(u)[:, :, None])[:, :, 0] / np.maximum(
                (Ct @ (np.abs(u) ** 2)[:, :, None])[:, :, 0], 1e-30
            )
            if it >= n_iter // 2:
                a = np.mean(np.abs(a)) * a / np.maximum(np.abs(a), 1e-30)
            u = (np.conj(a)[:, None, :] @ Vt)[:, 0, :] / np.maximum(
                ((np.abs(a) ** 2)[:, None, :] @ Ct)[:, 0, :], 1e-30
            )
        return u.astype(np.complex64), a.astype(np.complex64)

    u1, a1 = components(Vt)
    M1 = a1[:, :, None] * u1[:, None, :] * Ct
    u2, a2 = components(Vt - M1)
    M2 = a2[:, :, None] * u2[:, None, :] * Ct
    Cn = np.maximum(Ct, 1)
    e1 = float(np.sum(np.abs(M1) ** 2 / Cn))
    e2 = float(np.sum(np.abs(M2) ** 2 / Cn))
    sig2 = float(np.sum(np.abs(Vt - M1 - M2) ** 2) / Ct.sum())
    del M1, M2
    # a fit to noise alone takes ~sig2 (sqrt(n_p S) + sqrt(NB))^2
    e_noise = sig2 * (np.sqrt(n_p * S) + np.sqrt(NB)) ** 2

    # weights in time order: flattened (period, sub-block); boundary k joins sub-block k to k+1
    w = (a1 / np.maximum(np.abs(a1), 1e-30)).T.reshape(-1)  # [n_p * S]
    ratio = np.array(
        [
            np.mean(w[k + 1 :: S][: n_p - 1] * np.conj(w[k::S][: n_p - 1]))
            for k in range(S)
        ]
    )
    D = 1.0 - np.abs(ratio)  # ~0 where the ratio is constant, ~1 at the epoch boundary
    kb = int(np.argmax(D))
    flips = float(D[kb])
    # chain the sub-block phases from the sub-block after the epoch boundary round to it
    rot = np.ones(S, np.complex64)
    for step in range(1, S):
        k = (kb + step) % S  # boundary between sub-block k and k+1 is consistent
        prev = k
        nxt = (k + 1) % S
        rot[nxt] = rot[prev] * ratio[k] / max(abs(ratio[k]), 1e-30)
    # a[s] u[s] = (a[s] conj(rot[s])) (u[s] rot[s])
    u1 = u1 * rot[:, None]
    a1 = a1 * np.conj(rot)[:, None]
    u = u1.reshape(-1)
    epoch_bin = 0
    if flips > 0.1:
        # the epoch lies in sub-block kb (late) or kb+1 (early). In time order (period-major
        # flattened sub-blocks j), split the pair (j, j+1), j = p S + kb, between the weights of j-1
        # and j+2; the pair runs into the next period when kb = S-1.
        Vf = Vt.transpose(1, 0, 2).reshape(n_p * S, L)
        Cf = Ct.transpose(1, 0, 2).reshape(n_p * S, L)
        wf = (a1 / np.maximum(np.abs(a1), 1e-30)).T.reshape(-1)
        ja = np.arange(n_p) * S + kb
        ja = ja[(ja >= 1) & (ja + 2 < n_p * S)]
        wl, wr = wf[ja - 1], wf[ja + 2]
        Vab = np.concatenate([Vf[ja], Vf[ja + 1]], axis=1)  # [n, 2L]
        Cab = np.concatenate([Cf[ja], Cf[ja + 1]], axis=1)
        SL = np.abs(np.conj(wl) @ Vab) ** 2
        SR = np.abs(np.conj(wr) @ Vab) ** 2
        score = np.concatenate([[0.0], np.cumsum(SL)]) + np.concatenate(
            [np.cumsum(SR[::-1])[::-1], [0.0]]
        )
        e = int(np.argmax(score))
        # wl / wr are chained weights, so these are already in the template's frame
        ul = (np.conj(wl) @ Vab) / np.maximum(Cab.sum(axis=0), 1)
        ur = (np.conj(wr) @ Vab) / np.maximum(Cab.sum(axis=0), 1)
        seg = np.concatenate([ul[:e], ur[e:]])
        sa, sb = kb, (kb + 1) % S
        u[sa * L : (sa + 1) * L], u[sb * L : (sb + 1) * L] = seg[:L], seg[L:]
        epoch_bin = (sa * L + e) % NB
    u = np.roll(u, -epoch_bin)
    # chips: the bins_per_chip grouping with the most energy marks the chip boundaries
    best = None
    for o in range(bins_per_chip):
        g = np.roll(u, -o).reshape(n_chips, bins_per_chip).sum(axis=1)
        en = float(np.sum(np.abs(g) ** 2))
        if best is None or en > best[0]:
            best = (en, o, g)
    _, off, g = best
    if off > bins_per_chip // 2:
        g = np.roll(g, 1)  # the epoch bin sat late in chip 0: keep chip 0 first
    ph = np.exp(-0.5j * np.angle(np.sum(g ** 2)))
    g = g * ph
    soft = g.real
    mu = np.mean(np.abs(soft))
    sd = np.std(np.abs(soft))
    snr = float(mu ** 2 / max(sd ** 2, 1e-30))
    # per code period sign: two sub-blocks after the epoch boundary (clear of the split), in the
    # template phase
    signs = np.sign(np.real(a1[(kb + 2) % S] * np.conj(ph))).astype(np.int8)
    return dict(
        soft=soft,
        chips=np.where(soft >= 0, 1, -1).astype(np.int8),
        quad=float(np.sum(g.imag ** 2) / np.sum(g.real ** 2)),
        chip_snr=snr,
        chip_err=float(0.5 * sspec.erfc(np.sqrt(snr / 2.0))),
        e1=e1 / e_noise,
        e2=e2 / e_noise,
        n_periods=int(n_p),
        flips=flips,
        epoch_bin=int(epoch_bin),
        signs=signs,
    )


def nh_report(signs, nh=None):
    """Does the per-period sign sequence carry the B3I/B1I NH20 secondary code? Returns the best
    offset's |mean agreement| over 20-period windows (1 = NH20 exactly, with data bits at 50 bps)."""
    nh = (
        np.array([1 - 2 * int(c) for c in "00000100110101001110"]) if nh is None else nh
    )
    s = np.asarray(signs, float)
    nw = len(s) // 20
    if nw < 2:
        return dict(
            nh20=float("nan"),
            offset=-1,
            transitions=float(np.mean(s[1:] != s[:-1])) if len(s) > 1 else float("nan"),
        )
    best = (0.0, -1)
    for o in range(20):
        ss = s[o : o + (len(s) - o) // 20 * 20].reshape(-1, 20)
        m = float(np.mean(np.abs(ss @ nh) / 20.0))
        best = max(best, (m, o))
    return dict(
        nh20=best[0], offset=best[1], transitions=float(np.mean(s[1:] != s[:-1]))
    )


def fold_period(period_lag, cr, f_c, f_rf, rc_nom):
    """Received period for the fold. The lag estimate is good to a few ns, but the fold needs ~0.01 ns
    over 0.5 s, so the period is taken from an integer number of chips at the transmitted chip rate,
    compressed by the carrier Doppler (code and carrier from one clock, as in GNSS):
    T_rx = N / rc_tx / (1 + f_c / f_rf). rc_tx is the measured chip rate with the Doppler taken out,
    snapped to the nominal when within 2 Hz (the chip-rate line's precision over 0.5 s is ~0.2 Hz);
    a chip clock further off keeps its measured rate."""
    rc_tx = cr["chip_rate"] / (1.0 + f_c / f_rf) if cr["detected"] else rc_nom
    if abs(rc_tx - rc_nom) < 2.0:
        rc_tx = rc_nom
    n_chips = int(round(period_lag * (1.0 + f_c / f_rf) * rc_tx))
    return n_chips / rc_tx / (1.0 + f_c / f_rf), n_chips, rc_tx


def period_scan(y, fs, period, car, f_rf, n_chips, t_use=0.05, span=10e-9):
    """Fallback when the Doppler-compressed period does not fold (a chip clock not locked to the
    carrier): the period (+-span) with the most rank-1 energy over a short fold (drift <= 0.25 chip
    over t_use per step). Moves only to a clear fold (> 10 x noise and 2 x the given period's), so a
    weak signal, whose short folds are all noise, keeps the given period."""
    step = 0.25 * period / n_chips / (t_use / period)
    trial = period + np.arange(-span, span + step / 2, step)
    trial = np.concatenate([[period], trial[np.abs(trial - period) > step / 4]])
    e = []
    for p in trial:
        e1 = blind_fold(
            y,
            fs,
            p,
            car["f_c"],
            car["rate"],
            car["t_c"],
            car["phase"],
            f_rf,
            n_chips,
            t_use=t_use,
            n_iter=16,
        )["e1"]
        e.append(e1)
    k = int(np.argmax(e))
    return trial[k] if k > 0 and e[k] > 10.0 and e[k] > 2.0 * e[0] else period


def identify(soft, families, top=3):
    """Circular correlation of recovered soft chips against every code of each family of the same
    length. rho = |peak| / (|soft| sqrt(N)): ~1 - 2 x chip error rate for the right code, ~sqrt(2
    ln N / N) for any other."""
    soft = np.asarray(soft, np.float64)
    N = len(soft)
    S = sfft.fft(soft / np.linalg.norm(soft), workers=WORKERS)
    out = {}
    for name in families:
        codes, labels, rc, n = gcodes.family(name)
        if n != N:
            out[name] = dict(skipped="code length %d != recovered %d" % (n, N))
            continue
        best = []
        for a in range(0, len(codes), 512):
            C = sfft.fft(codes[a : a + 512].astype(np.float32), axis=1, workers=WORKERS)
            r = np.abs(
                sfft.ifft(S[None, :] * np.conj(C), axis=1, workers=WORKERS)
            ) / np.sqrt(N)
            k = np.argmax(r, axis=1)
            for i in range(len(k)):
                best.append((float(r[i, k[i]]), labels[a + i], int(k[i])))
        best.sort(reverse=True)
        out[name] = dict(
            top=best[:top], noise_rho=float(np.sqrt(2 * np.log(N * len(codes)) / N))
        )
    return out


# --------------------------------------------------------------------------- acquisition


def replica_fft(codes, chip_rate, fs, spb):
    """conj FFT of each code sampled at fs over one block of spb samples (the code repeats)."""
    n = np.arange(spb)
    idx = np.floor(n * (chip_rate / fs) + 1e-9).astype(np.int64) % codes.shape[1]
    return np.conj(
        sfft.fft(codes[:, idx].astype(np.complex64), axis=1, workers=WORKERS)
    )


def pcps(
    y,
    fs,
    codes,
    chip_rate,
    f_rf,
    dopplers,
    rates=(0.0,),
    n_blocks=None,
    start=0,
    labels=None,
    pfa=1e-3,
    mem=768e6,
    progress=False,
    drift=(0.0, 0.0),
):
    """Parallel-code-phase acquisition: per code period, the circular correlation with each code at
    each (Doppler, rate) hypothesis (half-bin Doppler grid by FFT shift; the code Doppler applied as a
    fractional delay so the periods add in place), summed non-coherently over n_blocks periods.
    dopplers / rates: Doppler at the first block's start (Hz) and its rate (Hz/s). drift: (Doppler,
    rate) added to each hypothesis for the code drift only, for a stream whose carrier is already
    wiped. Returns one dict per code, best first."""
    N = codes.shape[1]
    T = N / chip_rate
    spb = int(round(fs * T))
    if abs(fs * T - spb) > 1e-6:
        raise ValueError("fs * code period = %.6f samples is not an integer" % (fs * T))
    nb = (
        (len(y) - start) // spb
        if n_blocks is None
        else min(n_blocks, (len(y) - start) // spb)
    )
    hf = np.repeat(np.asarray(dopplers, float)[None, :], len(rates), 0).ravel()
    hr = np.repeat(np.asarray(rates, float)[:, None], len(dopplers), 1).ravel()
    nH = len(hf)
    n = np.arange(spb)
    half = np.exp(-1j * np.pi * n / spb).astype(np.complex64)
    ksig = (sfft.fftfreq(spb) * spb).astype(np.float64)
    Xsub = np.empty((nb, 2, spb), np.complex64)
    for b in range(nb):
        xb = y[start + b * spb : start + (b + 1) * spb]
        Xsub[b, 0] = sfft.fft(xb, workers=WORKERS)
        Xsub[b, 1] = sfft.fft(xb * half, workers=WORKERS)
    nc = max(1, int(mem / (nH * spb * 16)))
    res = []
    t_start = time.time()
    for a in range(0, len(codes), nc):
        C = replica_fft(codes[a : a + nc], chip_rate, fs, spb)
        acc = np.zeros((nH, len(C), spb), np.float32)
        for b in range(nb):
            tb = b * T
            fb = hf + hr * (tb + T / 2.0)
            h = np.round(2.0 * fb * T).astype(np.int64)
            kint, sub = h // 2, h % 2
            Xs = Xsub[b][sub[:, None], (n[None, :] + kint[:, None]) % spb]
            d = -fs * ((hf + drift[0]) * tb + 0.5 * (hr + drift[1]) * tb ** 2) / f_rf
            ramp = np.exp(2j * np.pi * np.outer(d, ksig) / spb).astype(np.complex64)
            Z = (Xs * ramp)[:, None, :] * C[None, :, :]
            acc += np.abs(sfft.ifft(Z, axis=-1, workers=WORKERS)) ** 2
        for i in range(len(C)):
            A = acc[:, i, :]
            mu = noise_mean(A, nb)
            j, k = np.unravel_index(int(np.argmax(A)), A.shape)
            pk = float(A[j, k])
            frac = (
                _parabola(A[j, k - 1], A[j, k], A[j, (k + 1) % spb]) if k > 0 else 0.0
            )
            fr = 0.0
            if 0 < j % len(dopplers) < len(dopplers) - 1:
                fr = _parabola(A[j - 1, k], A[j, k], A[j + 1, k])
            res.append(
                dict(
                    code=a + i,
                    label=labels[a + i] if labels is not None else str(a + i),
                    pmr=pk / mu,
                    z=(pk / mu - 1.0) * np.sqrt(nb),
                    doppler=float(
                        hf[j]
                        + fr * (dopplers[1] - dopplers[0] if len(dopplers) > 1 else 0.0)
                    ),
                    rate=float(hr[j]),
                    delay=float(k + frac),
                    chip0=float((-(k + frac) * chip_rate / fs) % N),
                )
            )
        if progress:
            log(
                "   pcps: %d/%d codes, %.0f s"
                % (min(a + nc, len(codes)), len(codes), time.time() - t_start)
            )
    thr = gamma_threshold(nb, nH * spb * len(codes) * chip_rate / fs, pfa)
    for r in res:
        r["thr"] = thr
        r["margin_db"] = float(10 * np.log10(max(r["pmr"] - 1.0, 1e-9) / (thr - 1.0)))
        r["detected"] = bool(r["pmr"] > thr)
        r["cn0_dbhz"] = float(10 * np.log10(max(r["pmr"] - 1.0, 1e-9) / T))
    res.sort(key=lambda r: -r["pmr"])
    # a strong signal of another code also passes thr on every code (its cross-correlation is
    # deterministic, so the periods do not average it down): the right code must stand out of its
    # family, so identification is the excess over the runner-up
    gap = (
        10 * np.log10(max(res[0]["pmr"] - 1.0, 1e-9) / max(res[1]["pmr"] - 1.0, 1e-9))
        if len(res) > 1
        else float("inf")
    )
    return dict(
        results=res,
        n_blocks=nb,
        n_hyp=nH,
        thr=thr,
        block_s=T,
        gap_db=float(gap),
        identified=bool(res[0]["detected"] and gap > GAP_DB),
    )


def doppler_grid(f_lo, f_hi, block_s, step=None):
    """Half-bin grid (the FFT shift resolution) unless a step is given."""
    step = step or 0.5 / block_s
    return np.arange(np.floor(f_lo / step) * step, f_hi + step / 2, step)


def rate_grid(r_lo, r_hi, span_s, fs, f_rf):
    """Rates whose code-drift error stays under a quarter sample over span_s (the carrier tolerates
    far more): half the spacing x span^2 / 2 x fs / f_rf <= 0.25."""
    step = f_rf / (fs * span_s ** 2)
    if step >= r_hi - r_lo:
        return np.array([0.5 * (r_lo + r_hi)])
    n = int(np.ceil((r_hi - r_lo) / step))
    return np.linspace(r_lo, r_hi, n + 1)


def blind_doppler_step(block_s, span_s, fs, f_rf):
    """Doppler step for a search without a carrier estimate: the half-bin FFT shift, or finer so the
    code drift of half a step over span_s stays under a quarter sample."""
    return min(0.5 / block_s, 0.5 * f_rf / (fs * span_s))


# --------------------------------------------------------------------------- verification


def verify(
    y, fs, code, chip_rate, f_rf, f_c, rate, t_c, n_win=8, t_win=0.05, carrier=True
):
    """Track the code delay over the record (PCPS per window, fractional peak), fit
    delay(t) = a + b t + c t^2, and return the received chip rate, code period and code Doppler
    (-f_rf d(delay)/dt) next to the carrier Doppler; then despread and re-measure the carrier."""
    N = len(code)
    T = N / chip_rate
    spb = int(round(fs * T))
    nbw = max(1, int(round(t_win / T)))
    nblk = len(y) // spb
    starts = np.linspace(0, nblk - nbw, n_win).astype(int) * spb
    yw = wipe(y, fs, f_c, rate, t_c)
    tw, dl, zz = [], [], []
    for s in starts:
        t0 = s / fs
        fd = f_c + rate * (t0 - t_c)
        r = pcps(
            yw,
            fs,
            code[None, :],
            chip_rate,
            f_rf,
            [0.0],
            [0.0],
            nbw,
            start=s,
            drift=(fd, rate),
        )
        best = r["results"][0]
        tw.append(t0)
        dl.append(best["delay"])
        zz.append(best["pmr"])
    tw, dl = np.array(tw), np.array(dl)
    dl = np.unwrap(dl * 2 * np.pi / spb) * spb / (2 * np.pi)
    A = np.vstack([np.ones_like(tw), tw - t_c, (tw - t_c) ** 2]).T
    coef, *_ = np.linalg.lstsq(A, dl / fs, rcond=None)
    resid = dl / fs - A @ coef
    tau_dot = coef[1]
    out = dict(
        t=tw.tolist(),
        delay_samples=dl.tolist(),
        pmr=[float(v) for v in zz],
        code_doppler=float(-f_rf * tau_dot),
        code_doppler_rate=float(-f_rf * 2 * coef[2]),
        chip_rate_rx=float(chip_rate * (1.0 - tau_dot)),
        period_rx=float(N / (chip_rate * (1.0 - tau_dot))),
        delay_rms_chips=float(np.std(resid) * chip_rate),
        carrier_doppler=float(f_c),
        carrier_rate=float(rate),
    )
    if carrier:
        # despread with the fitted code phase, then the carrier from the squared narrowband signal
        n = len(y)
        t = np.arange(n) / fs
        tau = coef[0] + coef[1] * (t - t_c) + coef[2] * (t - t_c) ** 2
        idx = np.floor((t - tau) * chip_rate + 1e-9).astype(np.int64) % N
        yd = y * code[idx]
        bc = blind_carrier(
            yd,
            fs,
            2,
            f_c - 500,
            f_c + 500,
            rate - 5,
            rate + 5,
            t_seg=0.1,
            t_use=n / fs,
            top=1,
            t_ref=t_c,
        )
        ln = bc["lines"][0]
        out.update(
            despread_doppler=ln["f_c"], despread_rate=ln["rate"], despread_coh=ln["coh"]
        )
    return out


# --------------------------------------------------------------------------- pipeline


def run(
    x,
    fs,
    fc,
    lobe="b3",
    families=None,
    t_blind=0.5,
    n_blocks=200,
    dmax=30e3,
    rmax=150.0,
    full_family_blind=False,
    out=None,
    n_blocks_big=20,
    period_max=2.2e-3,
):
    """The whole search on one stream; returns a JSON-able report (and prints a summary)."""
    f_car, rc = LOBES[lobe]
    families = (families or DEFAULT_FAMILIES[lobe]).split(",")
    rep = dict(
        lobe=lobe, f_car=f_car, fs_in=fs, fc_in=fc, n_in=len(x), t_in=len(x) / fs
    )
    T0 = time.time()
    y, fsy = extract_lobe(x, fs, fc, f_car, rc, 2)
    y4, fs4 = extract_lobe(x, fs, fc, f_car, rc, 4, n_max=int((t_blind + 0.01) * fs))
    rep["t_extract"] = time.time() - T0
    log(
        "lobe %s: %.3f MHz, %d samples at %.3f MSPS (%.2f s); extract %.1f s"
        % (lobe, f_car / 1e6, len(y), fsy / 1e6, len(y) / fsy, rep["t_extract"])
    )

    t = time.time()
    # near the nominal first (Doppler moves it < 300 Hz; channelised data carry chip-rate images
    # one channel away), then +-10 %
    cr = blind_chip_rate(y4, fs4, rc - 1000.0, rc + 1000.0, t_use=t_blind)
    if not cr["detected"]:
        cr = blind_chip_rate(y4, fs4, 0.9 * rc, 1.1 * rc, t_use=t_blind)
    cr["code_doppler"] = float(f_car * (cr["chip_rate"] / rc - 1.0))
    rep["chip_rate"] = cr
    log(
        "chip rate: %.3f Hz (x%.1f, thr %.1f, %s) -> code Doppler %+.0f Hz vs nominal %.0f; %.1f s"
        % (
            cr["chip_rate"],
            cr["snr"],
            cr["thr"],
            "DETECTED" if cr["detected"] else "no",
            cr["code_doppler"],
            rc,
            time.time() - t,
        )
    )

    t = time.time()
    # beyond ~2 ms the lag search runs on 8x boxcar-decimated data (lag resolution ~4 chips)
    pr = blind_period(
        y,
        fsy,
        tau_max=period_max,
        t_use=t_blind,
        decim=1 if period_max <= 2.2e-3 else 8,
    )
    rep["period"] = pr
    for p in pr["peaks"]:
        log(
            "period: %.6f ms (%.1f chips at the measured chip rate) x%.1f (thr %.1f), coherent %.3f"
            % (
                p["period"] * 1e3,
                p["period"] * cr["chip_rate"],
                p["snr"],
                pr["thr"],
                p["coherent"],
            )
        )
    if not pr["peaks"]:
        log(
            "period: nothing above x%.1f in 0.05-%.1f ms"
            % (pr["thr"], period_max * 1e3)
        )
    log("   (%.1f s)" % (time.time() - t))

    t = time.time()
    bc = blind_carrier(
        y, fsy, 2, -dmax, dmax, -rmax - 50, rmax + 50, t_use=min(1.0, len(y) / fsy)
    )
    if not bc["lines"][0]["detected"]:
        bc4 = blind_carrier(
            y, fsy, 4, -dmax, dmax, -rmax - 50, rmax + 50, t_use=min(1.0, len(y) / fsy)
        )
        if bc4["lines"][0]["detected"]:
            bc = bc4
    rep["carrier"] = bc
    for ln in bc["lines"]:
        log(
            "carrier (order %d): f %+.1f Hz at t %.3f s, rate %+.2f Hz/s, x%.1f (thr %.1f) coh %.0f %s"
            % (
                bc["order"],
                ln["f_c"],
                ln["t_c"],
                ln["rate"],
                ln["snr_semi"],
                bc["thr"],
                ln["coh"],
                "DETECTED" if ln["detected"] else "-",
            )
        )
    log(
        "   (%d rates x %d segments, %.1f s)"
        % (bc["n_rates"], bc["n_seg"], time.time() - t)
    )
    car = bc["lines"][0] if bc["lines"][0]["detected"] else None

    if car is not None:
        t = time.time()
        lag = fundamental(pr["peaks"])
        if lag is None:
            lag = NOMINAL_CHIPS[lobe] / rc
            log(
                "fold: no code period stands out: folding at the nominal %.1f ms (raise --period-max for longer codes)"
                % (lag * 1e3)
            )
        per, n_chips, rc_tx = fold_period(lag, cr, car["f_c"], f_car, rc)
        rep["fold_period"] = dict(period=per, n_chips=n_chips, chip_rate_tx=rc_tx)
        fo = blind_fold(
            y,
            fsy,
            per,
            car["f_c"],
            car["rate"],
            car["t_c"],
            car["phase"],
            f_car,
            n_chips,
            t_use=t_blind,
        )
        if fo["e1"] < 2.0:
            per2 = period_scan(y, fsy, per, car, f_car, n_chips)
            if per2 != per:
                log("fold: period scan moved %.6f -> %.6f ms" % (per * 1e3, per2 * 1e3))
                per = per2
                fo = blind_fold(
                    y,
                    fsy,
                    per,
                    car["f_c"],
                    car["rate"],
                    car["t_c"],
                    car["phase"],
                    f_car,
                    n_chips,
                    t_use=t_blind,
                )
        idn = identify(fo["soft"], families)
        lf = gcodes.lfsr_report(
            (fo["chips"] < 0).astype(np.uint8),
            windows=(0, n_chips // 3, 2 * n_chips // 3),
            length=min(600, n_chips // 3),
        )
        rep["fold"] = {
            k: v for k, v in fo.items() if k not in ("soft", "chips", "signs")
        }
        rep["_chips"] = fo["soft"]
        rep["identify"] = idn
        # an LFSR code of unknown polynomials still has a small L: the connection polynomial
        # (the product of its generators' polynomials) is then the answer
        rep["lfsr"] = [
            dict(
                start=r["start"],
                L=r["L"],
                divisors=r["divisors"],
                poly=gcodes.poly_str(r["poly"]) if r["L"] < 100 else "",
            )
            for r in lf
        ]
        nh = nh_report(fo["signs"])
        rep["fold"]["nh"] = nh
        log(
            "fold: %d chips, %d periods, chip SNR %.1f (err ~%.1e), components %.0f / %.1f x noise, quad/real %.3f; %.1f s"
            % (
                n_chips,
                fo["n_periods"],
                fo["chip_snr"],
                fo["chip_err"],
                fo["e1"],
                fo["e2"],
                fo["quad"],
                time.time() - t,
            )
        )
        log(
            "   sign changes at the code epoch in %.0f %% of periods (chip 0 = fold bin %d); NH20 agreement %.2f at offset %d"
            % (100 * nh["transitions"], fo["epoch_bin"], nh["nh20"], nh["offset"])
        )
        for name, v in idn.items():
            if "skipped" in v:
                log("   identify %s: %s" % (name, v["skipped"]))
            else:
                log(
                    "   identify %s: %s (noise ~%.3f)"
                    % (
                        name,
                        ", ".join("%s rho %.3f" % (l, r) for r, l, _ in v["top"]),
                        v["noise_rho"],
                    )
                )
        for r in rep["lfsr"]:
            log(
                "   Berlekamp-Massey from chip %d: L = %d%s%s"
                % (
                    r["start"],
                    r["L"],
                    (", divisible by " + " and ".join(r["divisors"]))
                    if r["divisors"]
                    else "",
                    "" if r["divisors"] or not r["poly"] else ", " + r["poly"],
                )
            )
        if out:
            np.save(out + "_chips.npy", fo["chips"])

    # acquisition per family; with a carrier estimate the stream is wiped once and searched at 0 Hz
    rep["pcps"] = {}
    best_all = None
    yw = wipe(y, fsy, car["f_c"], car["rate"], car["t_c"]) if car is not None else None
    for name in families:
        codes, labels, rcf, N = gcodes.family(name)
        if abs(rcf - rc) > 1e-6:
            log(
                "pcps %s: chip rate %.3f != lobe %.3f Mcps, skipped"
                % (name, rcf / 1e6, rc / 1e6)
            )
            continue
        Tb = N / rcf
        nb = min(n_blocks, len(y) // int(round(fsy * Tb)))
        if car is not None:
            fd0 = car["f_c"] - car["rate"] * car["t_c"]
            dop, rates, drift, ys = [0.0], [0.0], (fd0, car["rate"]), yw
            if len(codes) > 200:
                nb = min(nb, n_blocks_big)
        else:
            drift, ys = (0.0, 0.0), y
            if len(codes) > 200 and not full_family_blind:
                log(
                    "pcps %s: %d codes with no carrier estimate: skipped (--full-family-blind runs it)"
                    % (name, len(codes))
                )
                continue
            dop = doppler_grid(
                -dmax, dmax, Tb, blind_doppler_step(Tb, nb * Tb, fsy, f_car)
            )
            rates = rate_grid(-rmax, rmax, nb * Tb, fsy, f_car)
        t = time.time()
        r = pcps(
            ys,
            fsy,
            codes,
            rcf,
            f_car,
            dop,
            rates,
            nb,
            labels=labels,
            progress=len(codes) > 200,
            drift=drift,
        )
        dtm = time.time() - t
        top = r["results"][:3]
        rep["pcps"][name] = dict(
            top=top,
            gap_db=r["gap_db"],
            identified=r["identified"],
            n_blocks=r["n_blocks"],
            n_hyp=r["n_hyp"],
            thr=r["thr"],
            seconds=dtm,
            n_codes=len(codes),
        )
        log(
            "pcps %s: %d codes x %d hyp x %d periods, %.1f s; thr x%.3f; %s (top %.1f dB over the next)"
            % (
                name,
                len(codes),
                r["n_hyp"],
                r["n_blocks"],
                dtm,
                r["thr"],
                "IDENTIFIED %s" % top[0]["label"]
                if r["identified"]
                else "no code stands out",
                r["gap_db"],
            )
        )
        for b in top:
            log(
                "   %-16s pmr %.3f margin %+.1f dB %s  dop %+.1f Hz rate %+.1f chip0 %.2f  C/N0 ~%.1f dB-Hz"
                % (
                    b["label"],
                    b["pmr"],
                    b["margin_db"],
                    "DET" if b["detected"] else "-  ",
                    b["doppler"] + drift[0],
                    b["rate"] + drift[1],
                    b["chip0"],
                    b["cn0_dbhz"],
                )
            )
        if r["identified"] and (best_all is None or top[0]["pmr"] > best_all[1]["pmr"]):
            best_all = (name, top[0], codes[top[0]["code"]], rcf)

    if best_all is not None:
        name, b, code, rcf = best_all
        t = time.time()
        if car is not None:
            f_c, rate, t_c = car["f_c"], car["rate"], car["t_c"]
        else:
            t_c = len(y) / fsy / 2
            f_c, rate = b["doppler"] + b["rate"] * t_c, b["rate"]
        v = verify(y, fsy, code, rcf, f_car, f_c, rate, t_c)
        rep["verify"] = dict(family=name, label=b["label"], **v)
        log(
            "verify %s %s: chip rate rx %.3f Hz, period rx %.9f ms, code Doppler %+.1f Hz (rate %+.1f) vs carrier %+.1f Hz (%+.1f); despread carrier %+.2f Hz %+.2f Hz/s (coh %.0f); delay fit rms %.3f chip; %.1f s"
            % (
                name,
                b["label"],
                v["chip_rate_rx"],
                v["period_rx"] * 1e3,
                v["code_doppler"],
                v["code_doppler_rate"],
                v["carrier_doppler"],
                v["carrier_rate"],
                v["despread_doppler"],
                v["despread_rate"],
                v["despread_coh"],
                v["delay_rms_chips"],
                time.time() - t,
            )
        )
    rep["t_total"] = time.time() - T0
    log("total %.1f s" % rep["t_total"])
    if out:
        with open(out + ".json", "w") as f:
            json.dump(
                {k: v for k, v in rep.items() if not k.startswith("_")},
                f,
                indent=1,
                default=_jsonable,
            )
    return rep


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("input")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--fc", type=float)
    ap.add_argument("--format", choices=["ci8", "ci16"])
    ap.add_argument(
        "--conj",
        action="store_true",
        help="conjugate the input (CHORD F-engine output)",
    )
    ap.add_argument("--t0", type=float, default=0.0, help="s into the record to start")
    ap.add_argument("--t-len", type=float, default=0.0, help="s to use (0 = all)")
    ap.add_argument("--lobe", choices=sorted(LOBES), default="b3")
    ap.add_argument(
        "--families",
        help="comma list of %s (default per lobe: %s)"
        % (",".join(gcodes.FAMILIES), DEFAULT_FAMILIES),
    )
    ap.add_argument(
        "--t-blind",
        type=float,
        default=0.5,
        help="s used by the blind estimators and the fold",
    )
    ap.add_argument(
        "--n-blocks",
        type=int,
        default=200,
        help="code periods summed by the acquisition",
    )
    ap.add_argument(
        "--period-max", type=float, default=2.2, help="ms: longest code period searched"
    )
    ap.add_argument(
        "--n-blocks-big",
        type=int,
        default=20,
        help="the same for families of > 200 codes",
    )
    ap.add_argument("--doppler-max", type=float, default=30e3)
    ap.add_argument("--rate-max", type=float, default=150.0)
    ap.add_argument(
        "--full-family-blind",
        action="store_true",
        help="search the large families over the full Doppler grid",
    )
    ap.add_argument("--out", help="prefix for <out>.json and <out>_chips.npy")
    a = ap.parse_args(argv)
    x, fs, fc = load(a.input, a.fs, a.fc, a.format, a.conj)
    i0 = int(a.t0 * fs)
    x = x[i0 : i0 + int(a.t_len * fs)] if a.t_len else x[i0:]
    run(
        x,
        fs,
        fc,
        a.lobe,
        a.families,
        a.t_blind,
        a.n_blocks,
        a.doppler_max,
        a.rate_max,
        a.full_family_blind,
        a.out,
        a.n_blocks_big,
        a.period_max * 1e-3,
    )


if __name__ == "__main__":
    main()
