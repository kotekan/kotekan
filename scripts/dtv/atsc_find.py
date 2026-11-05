#!/usr/bin/env python3
"""Look for an 8-VSB transmitter in a stitched beam (atsc_arr.py output) without a template: narrow lines
near a reference pilot, and the field-repetition autocorrelation (field = 313 x 832 symbols) over a range of
symbol-clock offsets, at 1/2/4/8/16 field spacings.

    atsc_find.py --beam arr30r_k00.npz --pilot 566.309242e6
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

RS = 4.5e6 * 684 / 286
FIELD_SYM = 313 * 832


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", required=True)
    ap.add_argument("--pilot", type=float, required=True)
    ap.add_argument("--ppm", type=float, default=15.0)
    a = ap.parse_args()
    d = np.load(a.beam)
    z, fs, fc = d["z"].astype(np.complex128), float(d["fs"]), float(d["fc"])
    n = len(z)
    Z = np.abs(sfft.fftshift(sfft.fft(z, workers=8))) ** 2
    f = sfft.fftshift(sfft.fftfreq(n, 1 / fs)) + fc
    sel = np.abs(f - a.pilot) < 50e3
    med = np.median(Z[np.abs(f - fc) < 2.5e6])
    idx = np.nonzero(sel)[0]
    top = idx[np.argsort(Z[idx])[::-1]]
    pk = []
    for j in top:
        if all(abs(f[j] - p) > 200 for p in pk):
            pk.append(f[j])
        if len(pk) >= 6:
            break
    print("lines within 50 kHz of %.6f MHz (offset Hz: dB over the in-band median): %s" % (
        a.pilot / 1e6, "  ".join("%+.0f: %.0f" % (p - a.pilot, 10 * np.log10(Z[np.argmin(abs(f - p))] / med)) for p in pk)))
    # notch narrow lines (pilots, carriers): bins > 10 dB over the running median of +-2 kHz
    Zc = sfft.fft(z, workers=8)
    P = np.abs(Zc) ** 2
    from scipy.ndimage import median_filter
    w = int(2e3 / (fs / n)) | 1
    mloc = median_filter(P, size=w, mode="wrap")
    mask = P > 10 * mloc
    mask = np.convolve(mask.astype(float), np.ones(9), mode="same") > 0
    print("notched %.4f of the bins (narrow lines)" % mask.mean())
    z = sfft.ifft(np.where(mask, 0, Zc), workers=8)
    p0 = np.sum(np.abs(z) ** 2)
    for kf in (1, 2, 4, 8, 16):
        L0 = FIELD_SYM * kf * fs / RS
        span = int(np.ceil(L0 * a.ppm * 1e-6)) + 2
        lags = np.arange(int(L0) - span, int(L0) + span + 1)
        if lags[-1] >= n:
            break
        cc = np.array([np.abs(np.vdot(z[:-lg], z[lg:])) for lg in lags]) / p0
        j = int(np.argmax(cc))
        ppm = (lags[j] / L0 - 1) * 1e6
        o = np.argsort(cc)[::-1]
        print("field autocorr at %2d fields: peak %.5f at %+.2f ppm (median %.5f, peak/median %.1f); next peaks at %s ppm" % (
            kf, cc[j], ppm, np.median(cc), cc[j] / np.median(cc),
            " ".join("%+.2f" % ((lags[q] / L0 - 1) * 1e6) for q in o[1:4])), flush=True)


if __name__ == "__main__":
    main()
