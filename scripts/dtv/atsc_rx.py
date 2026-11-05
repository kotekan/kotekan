#!/usr/bin/env python3
"""8-VSB receiver front end on a stitched ATSC beam (atsc_beam.py output): pilot lock, band to centre,
matched filter, symbol clock from the segment syncs, field-sync search, PN511 check and the channel
impulse response from the field sync.

Model: with the pilot at f_p and the band centre f_p + Rs/4 shifted to 0 Hz, the complex baseband is
z_c(t) = sum_k a_k (-j)^k p(t - kT), a_k in {+-1,+-3,+-5,+-7} (+1.25 pilot), T = 1/Rs, p a root raised
cosine with Nyquist frequency Rs/4 (roll-off 0.1152). After the matched filter, Re{y(kT) j^k e^-jtheta}
is a_k scaled by the main path.

    atsc_rx.py --beam /var/tmp/bb1030/atsc/beam30i_k00.npz --pilot 566.3092e6 --out /var/tmp/bb1030/atsc/rx30_k00.npz
"""
import argparse
import os
import time
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft
import scipy.signal as ss

RS = 4.5e6 * 684 / 286  # 10.762237762 MHz
ALPHA = 0.1152
SEG = 832
FIELD = 313
W = 8


def rrc(ts, beta, sps, span):
    """Root raised cosine, symbol period ts samples*sps... returns taps for symbol period = sps samples."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(x) - 1 / (4 * beta)) < 1e-9:
            h[i] = (
                beta
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            h[i] = (
                np.sin(np.pi * x * (1 - beta))
                + 4 * beta * x * np.cos(np.pi * x * (1 + beta))
            ) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def cubic_at(x, pos):
    """Cubic (Catmull-Rom) interpolation of x at fractional sample positions pos."""
    i = np.floor(pos).astype(np.int64)
    f = (pos - i).astype(np.float32)
    i = np.clip(i, 1, len(x) - 3)
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    return x0 + 0.5 * f * (
        x1 - xm1 + f * (2 * xm1 - 5 * x0 + 4 * x1 - x2 + f * (3 * (x0 - x1) + x2 - xm1))
    )


def lfsr(taps, nbits, init, n):
    """Fibonacci LFSR: state bits s[0..nbits-1]; output s[0]; new bit = xor of s[t] for t in taps."""
    s = list(init)
    out = []
    for _ in range(n):
        out.append(s[0])
        nb = 0
        for t in taps:
            nb ^= s[t]
        s = s[1:] + [nb]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", required=True)
    ap.add_argument(
        "--pilot", type=float, required=True, help="approximate pilot frequency, Hz"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sps", type=int, default=4, help="oversampling of the resampled stream"
    )
    ap.add_argument(
        "--delta",
        type=float,
        default=None,
        help="symbol clock offset (ppm) to use instead of the field-autocorrelation estimate",
    )
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.beam)
    z, fs, fc = d["z"].astype(np.complex64), float(d["fs"]), float(d["fc"])
    n = len(z)
    t = np.arange(n) / fs
    # 1. pilot: refine the frequency (peak + parabola on |Z|), then a per-block complex amplitude
    Z = sfft.fft(z.astype(np.complex128), workers=W)
    fr = sfft.fftfreq(n, 1 / fs)
    k0 = int(np.round((a.pilot - fc) * n / fs)) % n
    win = (np.arange(k0 - 50, k0 + 51)) % n
    k = win[np.argmax(np.abs(Z[win]))]
    ym1, y0, yp1 = (np.log(np.abs(Z[(k + o) % n])) for o in (-1, 0, 1))
    dk = 0.5 * (ym1 - yp1) / (ym1 - 2 * y0 + yp1)
    fp = fr[k] + dk * fs / n
    blk = int(fs * 0.005)  # 5 ms blocks
    nb = n // blk
    rot = np.exp(-2j * np.pi * fp * t[: nb * blk]).astype(np.complex64)
    amp = (z[: nb * blk] * rot).reshape(nb, blk).mean(axis=1)
    ph = np.unwrap(np.angle(amp))
    print(
        "pilot %.3f Hz (absolute %.6f MHz), |amp| %.3f (beam rms %.3f); phase drift over the record %.1f deg (fit slope %.3f Hz)"
        % (
            fp,
            (fp + fc) / 1e6,
            np.abs(amp).mean(),
            np.sqrt(np.mean(np.abs(z) ** 2)),
            np.degrees(ph[-1] - ph[0]),
            np.polyfit(np.arange(nb) * blk / fs, ph, 1)[0] / (2 * np.pi),
        )
    )
    # remove the pilot (re-added as a +1.25 symbol offset later) and derotate by its smoothed phase
    tb = (np.arange(nb) + 0.5) * blk / fs
    php = np.interp(t, tb, ph)
    ampi = np.interp(t, tb, np.abs(amp))
    pil = (ampi * np.exp(1j * (2 * np.pi * fp * t + php))).astype(np.complex64)
    zc = (z - pil) * np.exp(-1j * (2 * np.pi * fp * t + php)).astype(
        np.complex64
    )  # pilot -> 0 Hz, phase-locked
    zc = zc * np.exp(-2j * np.pi * (RS / 4) * t).astype(
        np.complex64
    )  # band centre -> 0 Hz
    pilot_amp = float(np.abs(amp).mean())
    del Z, rot, pil
    # 2. resample to ~sps * Rs (rational approximation; record the exact rate)
    ratio = Fraction(a.sps * RS / fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    fo = fs * up / down
    zr = ss.resample_poly(zc, up, down).astype(np.complex64)
    print(
        "resampled x%d/%d to %.6f MHz = %.6f samples per symbol"
        % (up, down, fo / 1e6, fo / RS)
    )
    # 3. matched filter: RRC with Nyquist Rs/4 (symbol period 2T)
    sps2 = 2 * fo / RS  # samples per 2T
    tt = np.arange(-int(24 * sps2), int(24 * sps2) + 1) / sps2
    hh = np.empty_like(tt)
    beta = ALPHA
    for i, x in enumerate(tt):
        if abs(x) < 1e-12:
            hh[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(x) - 1 / (4 * beta)) < 1e-9:
            hh[i] = (
                beta
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            hh[i] = (
                np.sin(np.pi * x * (1 - beta))
                + 4 * beta * x * np.cos(np.pi * x * (1 + beta))
            ) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    hh /= np.sqrt(np.sum(hh ** 2))
    y = ss.fftconvolve(zr, hh.astype(np.float32), mode="same").astype(np.complex64)
    spsym = fo / RS
    # 4. symbol clock: fold blocks of BS segments on the nominal 2-per-symbol grid, then search the clock
    #    offset that makes the whole record's fold coherent (every path of the channel adds, no argmax jumps)
    nsym = int((len(y) - 8) / spsym) - 4
    y2 = cubic_at(y, np.arange(2 * nsym) * (spsym / 2))
    L = 2 * SEG
    # template-free clock check: the normalised autocorrelation at one segment and one field (2 per symbol)
    p0 = np.sum(np.abs(y2) ** 2)
    for name, nom in (("segment", L), ("field", L * FIELD)):
        lags = (
            np.arange(nom - 8, nom + 9)
            if name == "segment"
            else np.arange(nom - 40, nom + 41)
        )
        cc = np.array([np.abs(np.vdot(y2[:-lg], y2[lg:])) for lg in lags]) / p0
        jb = int(np.argmax(cc))
        print(
            "autocorr at one %s: peak %.5f at lag %+d samples from nominal (= %+.2f ppm); median %.5f; profile %s"
            % (
                name,
                cc[jb],
                lags[jb] - nom,
                (lags[jb] - nom) / nom * 1e6,
                np.median(cc),
                " ".join("%.4f" % v for v in cc[max(0, jb - 4) : jb + 5]),
            )
        )
    # clock from the field repetition at growing spacings (the field sync and the segment syncs recur)
    delta = 0.0
    for kf in (1, 2, 4, 8, 16, 24):
        nom = L * FIELD * kf * (1 + delta)
        if nom + 50 >= len(y2):
            break
        base = int(round(nom))
        lags = np.arange(base - 3, base + 4)
        cc = np.array([np.abs(np.vdot(y2[:-lg], y2[lg:])) for lg in lags]) / np.sum(
            np.abs(y2[: len(y2) - base]) ** 2
        )
        jb = int(np.argmax(cc))
        if 0 < jb < len(lags) - 1:
            am, a0, ap_ = cc[jb - 1], cc[jb], cc[jb + 1]
            pk = lags[jb] + 0.5 * (am - ap_) / (am - 2 * a0 + ap_)
        else:
            pk = lags[jb]
        delta = pk / (L * FIELD * kf) - 1
        print(
            "field autocorr at %2d fields: peak %.5f at %.3f samples -> clock %+.4f ppm"
            % (kf, cc[jb], pk, delta * 1e6)
        )
    if a.delta is not None:
        delta = a.delta * 1e-6
        print("clock offset overridden: %+.4f ppm" % (delta * 1e6))
    # the band centre is f_p + Rs_true/4: correct the nominal Rs/4 shift for the clock offset
    y = (y * np.exp(2j * np.pi * (delta * RS / 4) * np.arange(len(y)) / fo)).astype(
        np.complex64
    )
    y2 = cubic_at(y, np.arange(len(y2)) * (spsym / 2))
    sb = (np.arange(len(y2) // L) + 0.5) * L
    kk = np.arange(4)
    best = None
    nseg_all = len(y2) // L
    pos = (
        np.arange(nseg_all)[:, None] * L * (1 + delta) + np.arange(L)[None, :]
    )  # aligned segment grid, nominal-y2 units
    Yseg = cubic_at(y2, pos.ravel()).reshape(nseg_all, L)
    prof = Yseg.mean(axis=0)
    for rot_name, rsign in (("(-j)^k", -1j), ("(+j)^k", 1j)):
        tpl = np.zeros(L, np.complex64)
        tpl[2 * kk] = np.array([5, -5, -5, 5]) * rsign ** kk
        TP = np.conj(sfft.fft(tpl))
        c = np.abs(sfft.ifft(sfft.fft(prof) * TP))
        print(
            "seg-sync fold with %s: peak %.2f, median %.3f"
            % (rot_name, c.max(), np.median(c))
        )
        if best is None or c.max() > best[0]:
            best = (c.max(), delta, rsign, rot_name, TP)
    _, delta, rsign, rot_name, TP = best
    cfold = sfft.ifft(sfft.fft(prof) * TP)
    j0 = int(np.argmax(np.abs(cfold)))
    top = []
    for jj in np.argsort(np.abs(cfold))[::-1]:
        if all(min(abs(jj - p), L - abs(jj - p)) > 3 for p in top):
            top.append(jj)
        if len(top) >= 8:
            break
    print(
        "rotation %s, clock %+.3f ppm; seg-sync fold paths (us rel. strongest: rel amp): %s; median %.2f"
        % (
            rot_name,
            delta * 1e6,
            " ".join(
                "%+.2f:%.2f"
                % (
                    (((p - j0 + L // 2) % L) - L // 2) / 2 / RS * 1e6,
                    np.abs(cfold[p]) / np.abs(cfold[j0]),
                )
                for p in top
            ),
            np.median(np.abs(cfold)) / np.abs(cfold[j0]),
        )
    )
    # 5. re-sample at the corrected clock, aligned to the strongest seg-sync path
    Tn = spsym * (1 + delta)
    start = (j0 / 2) * spsym
    nsym = int((len(y) - 8 - start) / Tn) - 4
    nsym = nsym // SEG * SEG
    y2c = cubic_at(y, start + np.arange(2 * nsym) * Tn / 2)
    # 6. field sync: the segment slot whose complex 2-per-symbol waveform repeats across fields
    Y = y2c.reshape(-1, L)
    nf = Y.shape[0] // FIELD
    E = np.array(
        [
            np.mean(np.abs(Y[s : nf * FIELD : FIELD].mean(axis=0)) ** 2)
            for s in range(FIELD)
        ]
    )
    fslot = int(np.argmax(E))
    fsm = Y[fslot : nf * FIELD : FIELD].mean(
        axis=0
    )  # field-averaged field-sync segment (CIR-smeared), 2 per symbol
    print(
        "field sync in segment slot %d of %d (%d fields): mean-square %.2f vs median slot %.3f"
        % (fslot, FIELD, nf, E[fslot], np.median(E))
    )
    # PN511 identity: complex correlation of the averaged segment with every LFSR phase and polarity
    ext = np.concatenate([Y[(fslot - 1) % Y.shape[0]][-L // 2 :] * 0, fsm, Y[0][:0]])
    Lx = 4096
    buf = np.zeros(Lx, np.complex64)
    buf[:L] = fsm
    FB = sfft.fft(buf)
    res = []
    for name, taps in (
        ("x9+x7+x6+x4+x3+x+1", [0, 1, 3, 4, 6, 7]),
        ("x9+x8+x6+x5+x3+x2+1", [0, 2, 3, 5, 6, 8]),
    ):
        seq = lfsr(taps, 9, [1] + [0] * 8, 511)
        for sh in range(511):
            b = np.roll(seq, -sh)
            for pol in (1, -1):
                sym = np.concatenate([[5, -5, -5, 5], pol * np.where(b > 0, 5, -5)])
                c = np.zeros(Lx, np.complex64)
                c[2 * np.arange(len(sym))] = sym * rsign ** np.arange(len(sym))
                r_ = sfft.ifft(FB * np.conj(sfft.fft(c)))
                # the PN511 starts right after the seg sync on the strongest path: lag 0 (+-1 sample)
                m0_ = np.abs(r_[[Lx - 1, 0, 1]]).max()
                res.append((m0_, name, sh, pol, 0, r_))
    res.sort(key=lambda x: -x[0])
    print(
        "PN511 at lag 0: best |r| %.1f (%s phase %d polarity %+d), next %.1f (%s phase %d polarity %+d), median over candidates %.1f"
        % (
            res[0][0],
            res[0][1],
            res[0][2],
            res[0][3],
            res[1][0],
            res[1][1],
            res[1][2],
            res[1][3],
            np.median([x[0] for x in res]),
        )
    )
    seqb = np.roll(
        lfsr(
            {
                "x9+x7+x6+x4+x3+x+1": [0, 1, 3, 4, 6, 7],
                "x9+x8+x6+x5+x3+x2+1": [0, 2, 3, 5, 6, 8],
            }[res[0][1]],
            9,
            [1] + [0] * 8,
            511,
        ),
        -res[0][2],
    )
    pn511 = res[0][3] * np.where(seqb > 0, 5, -5)
    print(
        "PN511 (as +-5, first 32): %s"
        % "".join("+" if v > 0 else "-" for v in pn511[:32])
    )
    cir = res[0][5]
    lag_us = (((np.arange(Lx) + Lx // 2) % Lx) - Lx // 2) / 2 / RS * 1e6
    order = np.argsort(np.abs(cir))[::-1]
    top = []
    for jj in order:
        if all(abs(jj - p) > 2 for p in top):
            top.append(jj)
        if len(top) >= 10:
            break
    print(
        "channel impulse response from the field sync (us: rel amp): %s; floor (median) %.3f"
        % (
            " ".join(
                "%+.2f:%.2f" % (lag_us[p], np.abs(cir[p]) / np.abs(cir[0])) for p in top
            ),
            np.median(np.abs(cir)) / np.abs(cir[0]),
        )
    )
    np.savez(
        a.out,
        y2=y2c,
        fslot=fslot,
        fsm=fsm,
        delta=delta,
        rsign=rsign,
        fo=fo,
        start=start,
        Tn=Tn,
        pilot_hz=fp + fc,
        pilot_amp=pilot_amp,
        cir=cir,
        pn511=pn511,
    )
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
