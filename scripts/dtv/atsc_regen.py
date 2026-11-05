#!/usr/bin/env python3
"""Regenerate the 8-VSB waveform from decoded symbols and measure how much of the beam it explains.

y = beam stream (2 per symbol) derotated by the phase track; c = symbols * rsign^k; the channel h by least
squares on EVERY symbol (dense LS, lags +-span); residual y - h*c over the whole record, per field and in
total. Run for the trellis-decoded symbols and for the raw slicer decisions.

    atsc_regen.py --rx rxa30d_k00.npz --eq eq2_30_k00_w8.npz --tcm tcm30_k00.npz
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import SEG, FIELD  # noqa: E402
from atsc_eq import ls_cir  # noqa: E402

W = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx", required=True)
    ap.add_argument("--eq", required=True)
    ap.add_argument("--tcm", required=True)
    ap.add_argument("--span", type=int, default=220)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = np.load(a.rx)
    e = np.load(a.eq)
    t = np.load(a.tcm)
    y2, rsign = r["y2"].astype(np.complex128), complex(r["rsign"])
    fslot = int(t["fslot"])
    sym = t["sym"].astype(float)
    nsym = len(sym)
    nseg = nsym // SEG
    y = y2[: 2 * nsym] * np.repeat(np.exp(-1j * e["phi"].astype(float)[:nsym]), 2)
    k = np.arange(nsym)
    ar = e["a"].astype(float)[:nsym]
    # per-level statistics of the equalised symbols against the decoded ones (data only)
    filled = t["filled"]
    print("equalised vs decoded level (data symbols): level: mean error, rms error")
    print(
        "   "
        + "  ".join(
            "%+d: %+.2f/%.2f"
            % (
                L,
                np.mean(ar[filled & (sym == L)] - L),
                np.std(ar[filled & (sym == L)] - L),
            )
            for L in range(-7, 8, 2)
        )
    )
    hard = np.clip(2 * np.round((ar + 7) / 2) - 7, -7, 7)
    hard[~filled] = sym[~filled]
    for name, s in (("trellis-decoded", sym), ("raw slicer", hard)):
        c = np.zeros(2 * nsym, np.complex128)
        c[0::2] = s * rsign ** (k % 4)
        lg, h = ls_cir(y, c, a.span)
        n = len(y)
        hf = np.zeros(n, np.complex128)
        hf[lg % n] = h
        pred = sfft.ifft(sfft.fft(c, workers=W) * sfft.fft(hf, workers=W), workers=W)
        res = y - pred
        rr = np.abs(res) ** 2
        pp = np.abs(pred) ** 2
        per = [
            10
            * np.log10(
                rr[f0 * 2 * SEG : (f0 + FIELD) * 2 * SEG].sum()
                / pp[f0 * 2 * SEG : (f0 + FIELD) * 2 * SEG].sum()
            )
            for f0 in range(0, nseg - FIELD + 1, FIELD)
        ]
        print(
            "%s symbols: residual / replica power over the record %.4f (%.1f dB); per field min/median/max %.1f / %.1f / %.1f dB"
            % (
                name,
                rr.sum() / pp.sum(),
                10 * np.log10(rr.sum() / pp.sum()),
                min(per),
                np.median(per),
                max(per),
            )
        )
        if a.out and name.startswith("trellis"):
            np.savez(a.out, lg=lg, h=h, res_ratio=rr.sum() / pp.sum())
    # what a replica with NO errors would leave: the floor seen inside the known PN511 (atsc pnresid) was ~-18.5 dB


if __name__ == "__main__":
    main()
