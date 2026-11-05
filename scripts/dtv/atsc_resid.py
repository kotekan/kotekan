#!/usr/bin/env python3
"""Write per-channel residuals after subtracting the decoded main-station replica (atsc_sub's fit: own band +
the two folded-in neighbour bands, the pilot as its own regressors near it), for every channel of the
stitched set, so atsc_arr.py --resdir can beam what is left (co-channel stations).

    atsc_resid.py --out-dir /var/tmp/bb1030/atsc/res30_k00
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atsc_sub as S  # noqa: E402
import dtv_blind as db  # noqa: E402
from atsc_beam import load_channel, CS  # noqa: E402

W = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arr", default="/var/tmp/bb1030/atsc/arr30_k00.npz")
    ap.add_argument("--rx", default="/var/tmp/bb1030/atsc/rxa30d_k00.npz")
    ap.add_argument("--eq", default="/var/tmp/bb1030/atsc/eq3_30_k00_w256.npz")
    ap.add_argument("--tcm", default="/var/tmp/bb1030/atsc/tcm30_k00.npz")
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lmin", type=int, default=-48)
    ap.add_argument("--lmax", type=int, default=80)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    _, fids, rz_d, fs_z, fc_z = S.replica_channels(a.arr, a.rx, a.eq, a.tcm, pilot=0.0)
    _, _, rz_p, _, _ = S.replica_channels(a.arr, a.rx, a.eq, a.tcm, data=False)
    RZd, RZp = sfft.fft(rz_d, workers=W), sfft.fft(rz_p, workers=W)
    fz = sfft.fftfreq(len(rz_d), 1 / fs_z) + fc_z
    f_pil = float(np.load(a.rx)["pilot_hz"])
    lags = np.arange(a.lmin, a.lmax + 1)
    for fid in fids:
        fid = int(fid)
        x = load_channel(a.dir, a.chunk, fid).astype(np.complex128)
        n = x.shape[0]
        w = np.zeros(n)
        w[S.EDGE : n - S.EDGE] = 1
        regs = S.band_regressors(RZd, fz, fid, fids)
        if abs(fid * CS - f_pil) < 1.5 * CS:
            regs += S.band_regressors(RZp, fz, fid, fids)
        regs = [
            r for r in regs if np.mean(np.abs(r[S.EDGE : n - S.EDGE]) ** 2) > 1e-12
        ]  # neighbours outside the stitched set are empty
        X = np.fft.fft(x * w[:, None], db.NFFT, axis=0)
        G, Gf, Sf, model = db.fit_lagged(X, [r * w for r in regs], lags)
        res = np.fft.ifft(X - model, axis=0)[:n].astype(np.complex64)
        np.save(os.path.join(a.out_dir, "res_%d.npy" % fid), res)
        sl = slice(S.EDGE, n - S.EDGE)
        print(
            "%d: power left %.3f (%.0f s)"
            % (
                fid,
                np.sum(np.abs(res[sl]) ** 2) / np.sum(np.abs(x[sl]) ** 2),
                time.time() - t0,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
