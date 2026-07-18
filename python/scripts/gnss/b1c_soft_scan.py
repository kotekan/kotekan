#!/usr/bin/env python3
"""Soft coherent matched scan for the B1C secondary overlay (rebuilt from the 07-17 design).

Per PRN: acquire + despread the pilot primary (machinery lifted from
bds_b1c_secondary_check.py, drift scan and all), giving per-record complex A_k.
Then for each candidate overlay lag sh in 0..1799 form  y_k = A_k * sec[(k+sh) mod 1800]
and FFT over k: a correct lag concentrates the series at the carrier residual -> one
sharp (lag, f) peak.  No per-record sign decisions, so the +-25 Hz alias branch and
unwrap glitches cannot scramble it (the trap that condemned C31 on 07-16).

Output per PRN: best lag, coherent fraction (peak |FFT| / sum|A|), residual f, and the
runner-up lag for margin.  Overlay lag is quoted AT RECORD k=0 of the despread window,
whose start sample in the file is T_SKIP*FS + pk (pk printed) -- so lag DIFFERENCES
between sats in the same capture are epoch-free observables to compare against BRDC.

Usage: b1c_soft_scan.py <raw.bin> <PRN> [PRN ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bds_b1c_secondary_check as bc

FS = bc.FS
IF = bc.IF
T_P = bc.T_P
T_SKIP = bc.T_SKIP
SEC_LEN = bc.SEC_LEN


def despread_series(raw, prn, n_rec):
    """Acquire + drift-scan + despread: returns (A[k] complex, pk, dop, rate)."""
    prim, sec = bc.b1c_codes(prn)
    n_samp = int(round(FS * T_P))
    snr, dop, pk = bc.acquire(raw, prim, n_samp)
    if snr < 8:
        return None, None, snr, None, None, sec
    snr, dop, pk = bc.acquire(raw, prim, n_samp, 0, np.arange(dop - 120, dop + 121, 10.0))
    spc = FS / 1.023e6
    base = 1.023e6 * dop / 1575.42e6 * spc * T_P
    off0 = int(T_SKIP * FS)
    tt = np.arange(n_samp) / FS
    rep0 = bc.replica(prim, n_samp)

    def despread(rate_spr, ks):
        out = np.empty(len(ks), complex)
        for i, k in enumerate(ks):
            o = off0 + pk + k * n_samp + int(round(rate_spr * k))
            x = raw[o:o + n_samp].astype(np.float32)
            xb = x * np.exp(-2j * np.pi * (IF + dop) * (tt + o / FS))
            out[i] = np.sum(xb * rep0)
        return out

    ks_probe = np.arange(0, n_rec, 10)
    cands = [sgn * base + d * spc * T_P
             for sgn in (1.0, -1.0) for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    pwr = [float(np.sum(np.abs(despread(c, ks_probe)) ** 2)) for c in cands]
    rate = cands[int(np.argmax(pwr))]
    cands2 = [rate + d for d in np.linspace(-0.03, 0.03, 7)]
    pwr2 = [float(np.sum(np.abs(despread(c, ks_probe)) ** 2)) for c in cands2]
    rate = cands2[int(np.argmax(pwr2))]
    A = despread(rate, np.arange(n_rec))
    return A, pk, snr, dop, rate, sec


def soft_scan(A, sec):
    """(best_lag, coh_frac, f_res_hz, runner_lag, runner_frac)."""
    n = len(A)
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    denom = float(np.sum(np.abs(A))) + 1e-12
    secf = sec.astype(np.float64)
    # all lags at once: matrix of A_k * sec[(k+sh)%1800] -> batched FFT
    best = np.empty(SEC_LEN)
    fpk = np.empty(SEC_LEN)
    B = 128  # lag batch
    k = np.arange(n)
    for s0 in range(0, SEC_LEN, B):
        shs = np.arange(s0, min(s0 + B, SEC_LEN))
        M = A[None, :] * secf[(k[None, :] + shs[:, None]) % SEC_LEN]
        F = np.fft.fft(M, nfft, axis=1)
        mag = np.abs(F)
        im = np.argmax(mag, axis=1)
        best[shs] = mag[np.arange(len(shs)), im]
        fpk[shs] = np.where(im <= nfft // 2, im, im - nfft) / (nfft * T_P)
    order = np.argsort(best)[::-1]
    b1, b2 = order[0], order[1]
    # exclude +-2 neighbours of the winner for the runner-up
    for cand in order[1:]:
        if min((cand - b1) % SEC_LEN, (b1 - cand) % SEC_LEN) > 2:
            b2 = cand
            break
    return int(b1), best[b1] / denom, float(fpk[b1]), int(b2), best[b2] / denom


def main():
    raw = np.memmap(sys.argv[1], dtype=np.int16, mode="r")
    prns = [int(a) for a in sys.argv[2:]]
    n_samp = int(round(FS * T_P))
    n_rec = min(600, int(len(raw) / n_samp) - int(T_SKIP * FS / n_samp) - 2)
    print("raw: %.1f s @ 20 MSPS -> %d records" % (len(raw) / FS, n_rec))
    print("  PRN   acq_snr   pk      lag   coh_frac  f_res     runner(frac)")
    for prn in prns:
        A, pk, snr, dop, rate, sec = despread_series(raw, prn, n_rec)
        if A is None:
            print("  C%-3d  %7.1f   -- not present, skipped" % (prn, snr))
            continue
        lag, frac, fres, lag2, frac2 = soft_scan(A, sec)
        print("  C%-3d  %7.1f   %-7d %4d   %.3f    %+7.2f Hz   %4d (%.3f)   [dop %+.0f rate %+.3f]"
              % (prn, snr, pk, lag, frac, fres, lag2, frac2, dop, rate))


if __name__ == "__main__":
    main()
