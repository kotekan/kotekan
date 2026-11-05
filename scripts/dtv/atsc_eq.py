#!/usr/bin/env python3
"""Equalise and slice the 8-VSB stream from atsc_rx.py: channel impulse response from the field sync
(seg sync + A/53 PN511, averaged over fields), a frequency-domain MMSE equaliser to the ideal raised-cosine
response over the whole record, symbol decisions, and the error rates on the known symbols (segment
syncs, PN511) and the MER of the data.

    atsc_eq.py --rx /var/tmp/bb1030/atsc/rx30_k00.npz --out /var/tmp/bb1030/atsc/eq30_k00.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import lfsr, RS, ALPHA, SEG, FIELD  # noqa: E402

W = 8
LEVELS = np.array([-7, -5, -3, -1, 1, 3, 5, 7], float)


def pn511():
    """A/53 field-sync PN511 as +-5 (binary 1 -> +5): x^9+x^7+x^6+x^4+x^3+x+1, the phase whose first
    32 bits are 0000 0001 0111 1111 1100 1010 1010 1110."""
    b = np.roll(lfsr([0, 1, 3, 4, 6, 7], 9, [1] + [0] * 8, 511), -2)
    return np.where(b > 0, 5.0, -5.0)


def rc_2sps(span=64):
    """Ideal response of one symbol at 2 samples per symbol: the matched-filter output of the RRC with
    Nyquist Rs/4 (raised cosine, zero crossings every 2T), centred at 0, length 4*span+1."""
    t = np.arange(-2 * span, 2 * span + 1) / 2.0 / 2.0  # in units of 2T
    b = ALPHA
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(np.pi * b * t) / (1 - (2 * b * t) ** 2)
    sing = np.isclose(np.abs(2 * b * t), 1.0)
    h[sing] = np.pi / 4 * np.sinc(1 / (2 * b))
    return h


def equalise(y2, h, lag, eps):
    """MMSE equaliser to the ideal 2-sps raised-cosine response, CIR h on lags lag, applied to y2 (circular)."""
    N = len(y2)
    hfull = np.zeros(N, np.complex128)
    js = np.nonzero(h)[0]
    hfull[lag[js] % N] = h[js]
    g = rc_2sps()
    gfull = np.zeros(N)
    gl = np.arange(-(len(g) // 2), len(g) // 2 + 1)
    gfull[gl % N] = g
    H = sfft.fft(hfull, workers=W)
    G = sfft.fft(gfull, workers=W)
    E = G * np.conj(H) / (np.abs(H) ** 2 + eps * np.max(np.abs(H) ** 2))
    return sfft.ifft(sfft.fft(y2, workers=W) * E, workers=W).astype(np.complex64)


def eq_piecewise(y2, dec, rsign, span, npiece, eps):
    """Per piece of the record: the CIR from the decided symbols in that piece (cross-correlation), and
    the equaliser applied to the piece with guards of 2*span samples (absorbs slow timing/phase/channel drift)."""
    N = len(y2)
    out = np.empty(N, np.complex64)
    bounds = (np.linspace(0, N // 2, npiece + 1).astype(int)) * 2
    G = 2 * span
    for p in range(npiece):
        a0, a1 = bounds[p], bounds[p + 1]
        lo, hi = max(0, a0 - G), min(N, a1 + G)
        n = hi - lo
        k = np.arange(a0 // 2, a1 // 2)
        c = np.zeros(n, np.complex128)
        c[2 * k - lo] = dec[k] * rsign ** (k % 4)
        r = sfft.ifft(
            sfft.fft(y2[lo:hi], workers=W) * np.conj(sfft.fft(c, workers=W)), workers=W
        ) / np.sum(dec[k].astype(float) ** 2)
        lg = np.arange(-span, span + 1)
        h = np.zeros(2 * span + 1, np.complex128)
        h[:] = r[lg % n]
        hfull = np.zeros(n, np.complex128)
        hfull[lg % n] = h
        g = rc_2sps()
        gfull = np.zeros(n)
        gl = np.arange(-(len(g) // 2), len(g) // 2 + 1)
        gfull[gl % n] = g
        H = sfft.fft(hfull, workers=W)
        Gf = sfft.fft(gfull, workers=W)
        E = Gf * np.conj(H) / (np.abs(H) ** 2 + eps * np.max(np.abs(H) ** 2))
        ye = sfft.ifft(sfft.fft(y2[lo:hi], workers=W) * E, workers=W)
        out[a0:a1] = ye[a0 - lo : a1 - lo]
    return out


def known_symbols(nseg, fslot, rsign, fields="even"):
    """Sparse training sequence at 2 per symbol: every segment sync, plus the PN511 of the selected fields
    ("even", "odd", "all", "none"), complex (level * rsign^k), zeros elsewhere."""
    nsym = nseg * SEG
    c = np.zeros(nsym, np.complex128)
    ss = np.array([5, -5, -5, 5], float)
    for kk in range(4):
        c[kk::SEG] = ss[kk]
    pn = pn511()
    fs_segs = np.arange(fslot, nseg, FIELD)
    for i, sgi in enumerate(fs_segs):
        if (
            fields == "all"
            or (fields == "even" and i % 2 == 0)
            or (fields == "odd" and i % 2 == 1)
        ):
            c[sgi * SEG + 4 : sgi * SEG + 515] = pn
    k = np.arange(nsym)
    c = c * rsign ** (k % 4)
    c2 = np.zeros(2 * nsym, np.complex128)
    c2[0::2] = c
    return c2


def ls_cir(y, c2, span):
    """Least-squares CIR on lags -span..span from the sparse known symbols c2 (unknown data = noise):
    sum_m' h[m'] ac(m - m') = r(m), ac(d) = sum c[n] conj(c[n-d]), r(m) = sum y[n] conj(c[n-m])."""
    n = len(y)
    C = sfft.fft(c2, workers=W)
    r = sfft.ifft(sfft.fft(y, workers=W) * np.conj(C), workers=W)
    ac = sfft.ifft(np.abs(C) ** 2, workers=W)
    lg = np.arange(-span, span + 1)
    d = lg[:, None] - lg[None, :]
    M = ac[d % n]
    h = np.linalg.solve(M, r[lg % n])
    return lg, h


def dd_cir(y2, dec, rsign, span, lag):
    """Channel from the decided symbols over the whole record: cross-correlation of y2 with the complex
    symbol sequence dec_k rsign^k (white, so this is the least-squares estimate), on +-span samples."""
    N = len(y2)
    c = np.zeros(N, np.complex128)
    k = np.arange(len(dec))
    c[0::2][: len(dec)] = dec * rsign ** (k % 4)
    r = sfft.ifft(
        sfft.fft(y2, workers=W) * np.conj(sfft.fft(c, workers=W)), workers=W
    ) / np.sum(dec.astype(float) ** 2)
    Lx = len(lag)
    h = np.zeros(Lx, np.complex128)
    keep = np.abs(lag) <= span
    h[keep] = r[lag[keep] % N]
    return h, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--span", type=int, default=600, help="CIR half-window, samples (2 per symbol)"
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=3e-3,
        help="MMSE regularisation, relative to max |H|^2",
    )
    ap.add_argument(
        "--dd",
        type=int,
        default=0,
        help="decision-directed channel re-estimates over the whole record",
    )
    ap.add_argument(
        "--dd-span",
        type=int,
        default=440,
        help="CIR half-window for the decision-directed estimates, samples",
    )
    ap.add_argument(
        "--pieces",
        type=int,
        default=1,
        help="decision-directed passes: a CIR + equaliser per piece of the record",
    )
    ap.add_argument(
        "--train",
        default="corr",
        help="first-pass channel: corr (field-sync correlation) | ls (least squares on every seg sync + the even fields' PN511)",
    )
    ap.add_argument("--ls-span", type=int, default=108)
    ap.add_argument(
        "--ls-pieces",
        type=int,
        default=1,
        help="ls training: a CIR per piece (drift check)",
    )
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.rx)
    y2, fslot, rsign = d["y2"], int(d["fslot"]), complex(d["rsign"])
    L = 2 * SEG
    nseg = len(y2) // L
    y2 = y2[: nseg * L]
    nf = nseg // FIELD
    fsm = y2.reshape(nseg, L)[fslot : nf * FIELD : FIELD].mean(axis=0)
    # known part of the field-sync segment: seg sync + PN511 (the PN63s follow; not used here)
    sym = np.concatenate([[5, -5, -5, 5], pn511()])
    Lx = 8192
    c = np.zeros(Lx, np.complex128)
    c[2 * np.arange(len(sym))] = sym * rsign ** np.arange(len(sym))
    buf = np.zeros(Lx, np.complex128)
    buf[:L] = fsm
    r = sfft.ifft(sfft.fft(buf) * np.conj(sfft.fft(c))) / np.sum(np.abs(sym) ** 2)
    lag = ((np.arange(Lx) + Lx // 2) % Lx) - Lx // 2
    jpk = int(np.argmax(np.abs(r)))
    print(
        "PN511 correlation peak at lag %+d samples (%+.2f us), |h| %.4f; median |h| %.4f"
        % (lag[jpk], lag[jpk] / 2 / RS * 1e6, np.abs(r[jpk]), np.median(np.abs(r)))
    )
    # the CIR on +-span samples around lag 0 of the grid (the channel includes the peak's offset)
    keep = np.abs(lag) <= a.span
    h = np.where(keep, r, 0)
    if a.train == "ls":
        c2 = known_symbols(nseg, fslot, rsign, "even")
        lg, hl = ls_cir(y2, c2, a.ls_span)
        h = np.zeros(len(lag), np.complex128)
        h[lg % len(lag)] = hl
        jpk = int(np.argmax(np.abs(h)))
        print(
            "LS training on %d known symbols (every seg sync + the even fields' PN511), span +-%d samples"
            % (int(np.sum(np.abs(c2) > 0)), a.ls_span)
        )
        if a.ls_pieces > 1:
            P_ = a.ls_pieces
            for p in range(P_):
                s0, s1 = (nseg * p // P_) * L, (nseg * (p + 1) // P_) * L
                cp = np.zeros_like(c2)
                cp[s0:s1] = c2[s0:s1]
                _, hp = ls_cir(y2, cp, a.ls_span)
                j = int(np.argmax(np.abs(hp)))
                cen = np.sum(lg * np.abs(hp) ** 2) / np.sum(np.abs(hp) ** 2)
                cc = np.vdot(hl, hp) / np.sqrt(
                    np.vdot(hl, hl).real * np.vdot(hp, hp).real
                )
                print(
                    "   piece %d: main tap lag %+d, phase %+.1f deg, |h| %.4g, power centroid %+.2f samples, coherence with the whole-record CIR %.3f (phase %+.1f deg)"
                    % (
                        p,
                        lg[j],
                        np.degrees(np.angle(hp[j])),
                        np.abs(hp[j]),
                        cen,
                        abs(cc),
                        np.degrees(np.angle(cc)),
                    )
                )
    top = np.argsort(np.abs(h))[::-1][:12]
    print(
        "CIR paths (us: amplitude rel. strongest): %s"
        % " ".join(
            "%+.2f:%.2f" % (lag[j] / 2 / RS * 1e6, np.abs(h[j]) / np.abs(h[jpk]))
            for j in sorted(top, key=lambda j: lag[j])
        )
    )
    pw = np.abs(h) ** 2
    print(
        "CIR power within +-1 us of the peak %.2f, +-5 us %.2f, +-20 us %.2f of the total in +-%.0f us"
        % tuple(
            [
                pw[np.abs(lag - lag[jpk]) <= x * 2 * RS * 1e-6].sum() / pw.sum()
                for x in (1, 5, 20)
            ]
            + [a.span / 2 / RS * 1e6]
        )
    )
    for it in range(a.dd + 1):
        if it > 0 and a.pieces <= 1:
            h, keep = dd_cir(y2, dec, rsign, a.dd_span, lag)
            print("--- decision-directed pass %d (span +-%d samples)" % (it, a.dd_span))
        # equaliser over the whole record: E = G conj(H) / (|H|^2 + eps max|H|^2), G the ideal 2-sps response
        if it == 0 or a.pieces <= 1:
            ye = equalise(y2, h, lag, a.eps)
        else:
            ye = eq_piecewise(y2, dec, rsign, a.dd_span, a.pieces, a.eps)
        s = ye[0::2]  # symbol instants
        k = np.arange(len(s))
        u = s * np.conj(rsign) ** (k % 4)
        S = u.reshape(nseg, SEG)
        ss = np.array([5, -5, -5, 5], float)
        beta = (S[:, :4] * ss).sum() / (len(S) * np.sum(ss ** 2))
        u = u / beta
        ar = np.real(u)
        aq = np.imag(u)
        S = ar.reshape(nseg, SEG)
        data = np.ones((nseg, SEG), bool)
        data[:, :4] = False
        data[fslot::FIELD] = False
        dc = ar.reshape(nseg, SEG)[data].mean()
        print(
            "rotation %.1f deg; residual DC (pilot remnant) %.3f; quadrature rms %.2f vs in-phase rms %.2f"
            % (np.degrees(np.angle(beta)), dc, aq.std(), ar.std())
        )
        ar = ar - dc
        S = ar.reshape(nseg, SEG)
        dec = LEVELS[np.argmin(np.abs(ar[:, None] - LEVELS[None, :]), axis=1)]
        D = dec.reshape(nseg, SEG)
        err = ar - dec
        E2 = err.reshape(nseg, SEG)
        # known symbols
        ss_ok = np.all(D[:, :4] == ss, axis=1)
        print(
            "segment syncs: %d of %d segments decoded exactly (+5 -5 -5 +5); symbol errors %.4f"
            % (ss_ok.sum(), nseg, np.mean(D[:, :4] != ss))
        )
        pn = pn511()
        fs_rows = D[fslot : nf * FIELD : FIELD, 4:515]
        print(
            "PN511 in %d fields: symbol error rate %.4f (sign errors %.4f); MER on these KNOWN symbols %.1f dB"
            % (
                fs_rows.shape[0],
                np.mean(fs_rows != pn),
                np.mean(np.sign(fs_rows) != np.sign(pn)),
                10
                * np.log10(
                    25.0 / np.mean((S[fslot : nf * FIELD : FIELD, 4:515] - pn) ** 2)
                ),
            )
        )
        print(
            "   held-out (odd fields' PN511) MER %.1f dB; sign errors %.4f"
            % (
                10
                * np.log10(
                    25.0
                    / np.mean(
                        (S[fslot + FIELD : nf * FIELD : 2 * FIELD, 4:515] - pn) ** 2
                    )
                ),
                np.mean(
                    np.sign(S[fslot + FIELD : nf * FIELD : 2 * FIELD, 4:515])
                    != np.sign(pn)
                ),
            )
        )
        dv = S[data]
        de = E2[data]
        mer = 10 * np.log10(np.mean(D[data] ** 2) / np.mean(de ** 2))
        print(
            "data symbols: MER %.1f dB (mean square level %.1f, expect 21); decision-error rms %.3f"
            % (mer, np.mean(D[data] ** 2), de.std())
        )
        hist, _ = np.histogram(dv, bins=np.arange(-9, 9.01, 0.5))
        print("histogram of equalised data symbols (0.5-wide bins from -9, thousands):")
        print(" ".join("%d" % (v // 1000) for v in hist))
        # MER per segment over time (does it hold across the record?)
        if it == a.dd:
            # rotation check on the known PN511 symbols: Re{u e^-j th}, per eighth of the record
            uc = u.reshape(nseg, SEG)[fslot : nf * FIELD : FIELD, 4:515]
            ths = np.radians(np.arange(-40, 40.1, 1.0))
            mk = [
                10
                * np.log10(
                    25.0 / np.mean((np.real(uc * np.exp(-1j * th)) - dc - pn) ** 2)
                )
                for th in ths
            ]
            jb = int(np.argmax(mk))
            print(
                "known-symbol MER vs extra rotation: best %.1f dB at %+.0f deg (0 deg: %.1f dB)"
                % (mk[jb], np.degrees(ths[jb]), mk[40])
            )
            nfr = uc.shape[0]
            for qq in range(4):
                rows_ = uc[qq * nfr // 4 : (qq + 1) * nfr // 4]
                m_ = [
                    np.mean((np.real(rows_ * np.exp(-1j * th)) - dc - pn) ** 2)
                    for th in ths
                ]
                print(
                    "   fields %d-%d: best %+.0f deg -> %.1f dB (0 deg %.1f dB); scale fit %.3f"
                    % (
                        qq * nfr // 4,
                        (qq + 1) * nfr // 4 - 1,
                        np.degrees(ths[int(np.argmin(m_))]),
                        10 * np.log10(25 / min(m_)),
                        10 * np.log10(25 / m_[40]),
                        np.sum(np.real(rows_) * pn) / np.sum(pn * pn) / rows_.shape[0],
                    )
                )
        ms = 10 * np.log10(np.mean(D[:, 4:] ** 2) / np.mean(E2[:, 4:] ** 2, axis=1))
        q = len(ms) // 8
        print(
            "MER by eighth of the record: %s dB"
            % " ".join("%.1f" % ms[i * q : (i + 1) * q].mean() for i in range(8))
        )
    np.savez(
        a.out,
        a=ar.astype(np.float32),
        dec=dec.astype(np.int8),
        h=h,
        lag=lag,
        beta=beta,
        dc=dc,
        fslot=fslot,
        mer=mer,
    )
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
