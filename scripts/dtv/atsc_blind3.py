#!/usr/bin/env python3
"""Blind subtraction with the NEIGHBOURING channels' templates as extra regressors (the PFB folds the
neighbour bands into each channel's edges): per channel k, protected eigen-beams s_{k-1}, s_k, s_{k+1} (each
from its own channel), least squares of x_k on all three (49 lags each); compared with the single-template
live recipe. Scored like dtv_blind (own-whitened eigenvalues in noise units, Tau A kept).

    atsc_blind3.py --freq 2904,2914,2924
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtv_blind as db  # noqa: E402
from dtv_modes import top_vecs, predict_p2  # noqa: E402
from atsc_beam import load_channel, CS  # noqa: E402

EDGE = 400


def template(x, Q):
    R0, lam0, V0, pw = db.whitened_eig(x)
    Pq = np.eye(128) - Q @ Q.conj().T
    _, Vd = top_vecs(Pq @ R0 @ Pq, 1)
    return (x / np.sqrt(pw)) @ np.conj(Vd[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="2904,2914,2924")
    ap.add_argument("--modes", default="/var/tmp/bb1030/dtv/modes_k00.npz")
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--lags", default="-8:40", help="lag windows to compare, lmin:lmax[,lmin:lmax...] (own template only when several)")
    a = ap.parse_args()
    m = np.load(a.modes)
    mfreq, clean, V16, mlam = m["freq"].astype(np.int64), m["clean"], m["V"], m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    windows = [tuple(int(v) for v in w.split(":")) for w in a.lags.split(",")]
    for fid in [int(v) for v in a.freq.split(",")]:
        xs, ss = {}, {}
        for kk in (fid - 1, fid, fid + 1):
            _, Q = top_vecs(predict_p2(P2, mfreq, clean, kk, 40, 0), 2)
            xs[kk] = load_channel(a.dir, a.chunk, kk).astype(np.complex128)
            ss[kk] = template(xs[kk], Q)
        _, Q = top_vecs(predict_p2(P2, mfreq, clean, fid, 40, 0), 2)
        x = xs[fid]
        n = x.shape[0]
        w = np.zeros(n)
        w[EDGE: n - EDGE] = 1
        X = np.fft.fft(x * w[:, None], db.NFFT, axis=0)
        sl = slice(EDGE + 64, n - EDGE - 64)
        out = []
        cases = ([("live recipe (own template)", [ss[fid]], np.arange(-8, 41)), ("+ neighbours' templates", [ss[fid - 1], ss[fid], ss[fid + 1]], np.arange(-8, 41))]
                 if len(windows) == 1 else [("own template, lags %+d..%+d" % w_, [ss[fid]], np.arange(w_[0], w_[1] + 1)) for w_ in windows])
        for name, regs, lags in cases:
            G, Gf, Sf, model = db.fit_lagged(X, [r * w for r in regs], lags)
            res = np.fft.ifft(X - model, axis=0)[:n]
            R, lam, V, pw = db.whitened_eig(res[sl])
            nu = lam[20:120].mean()
            ex, _ = db.tau_excess(R, Q)
            line = "%s: %s (kept %.2f)" % (name, " ".join("%.2f" % (v / nu) for v in lam[:6]), ex / ref)
            # one more protected blind source on top
            rb = db.blind(res, -lags[0], lags[-1], 1, Q, lambda m_: None, protect=True)
            R2, lam2, _, _ = db.whitened_eig(rb["r"][sl])
            ex2, _ = db.tau_excess(R2, Q)
            line += "; + 1 protected source: %s (kept %.2f)" % (" ".join("%.2f" % (v / lam2[20:120].mean()) for v in lam2[:5]), ex2 / ref)
            out.append(line)
        print("%d (%.2f MHz)\n   %s" % (fid, fid * CS / 1e6, "\n   ".join(out)), flush=True)


if __name__ == "__main__":
    main()
