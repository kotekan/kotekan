#!/usr/bin/env python3
"""The whole band from one recv1 N2 file: per channel (300-1500 MHz, 6145), the top eigenvalues of the
input-whitened covariance averaged over the file's bins, in noise units (the mean of eigenvalues 20..m-8, m the
inputs kept in every bin), and the share of the file's bins in which the channel was flagged. A bin with every
input flagged (a recv1 start or gap) is left out.

    vis_spectrum.py FILE.h5 --out spec.npz [--block 128]
"""
import argparse
import time

import h5py
import numpy as np

K0 = 1536  # freq_id of channel index 0 (300 MHz)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=128)
    a = ap.parse_args()
    t0 = time.time()
    with h5py.File(a.h5, "r") as f:
        prod = f["index_map/prod"][:]
        ia, ib = prod["input_a"].astype(int), prod["input_b"].astype(int)
        nf = f["vis"].shape[0]
        tins = f["bin_t_inst_ns"][:]
        lam = np.full((nf, 3), np.nan, np.float32)
        kept = np.zeros(nf, np.int16)
        flagged = np.zeros(nf, np.float32)
        for c0 in range(0, nf, a.block):
            c1 = min(nf, c0 + a.block)
            vis = f["vis"][c0:c1]  # [C, 8256, nb]
            flg = f["flags"][c0:c1] > 0  # [C, 128, nb]
            for c in range(c1 - c0):
                ok = flg[c].any(axis=0)  # bins with data
                keep = flg[c][:, ok].all(axis=1) & ok.any()
                m = int(keep.sum())
                flagged[c0 + c] = 1.0 - ok.mean()
                kept[c0 + c] = m
                if m < 64:
                    continue
                v = vis[c][:, ok].mean(axis=1)
                if not np.isfinite(v).all():
                    continue
                V = np.zeros((128, 128), np.complex128)
                V[ia, ib] = v
                V[ib, ia] = np.conj(v)
                d = np.real(np.diag(V)).copy()
                w = np.where(keep, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
                R = (V * w[:, None] * w[None, :])[np.ix_(keep, keep)]
                if not np.isfinite(R).all():
                    continue
                ev = np.linalg.eigvalsh(R)[::-1]
                lam[c0 + c] = ev[:3] / ev[20 : m - 8].mean()
            print(
                "channels %d..%d, %.0f s" % (c0, c1 - 1, time.time() - t0), flush=True
            )
    np.savez(
        a.out,
        fid=np.arange(nf) + K0,
        mhz=300.0 + np.arange(nf) * 0.1953125,
        lam=lam,
        kept=kept,
        flagged=flagged,
        t_inst_ns=tins,
        src=a.h5,
    )
    print("saved %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
