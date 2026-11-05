#!/usr/bin/env python3
"""How many blind sources, and from where: the stage's sequential peel (each source = the protected top
eigen-beam of the residual after the sources before it, its own lagged fit) at a chosen lag window,
scored after every source like dtv_blind (own-whitened eigenvalues in noise units, Tau A kept).

Variants of the extra sources' weights (source 0 is always the full-band protected eigen-beam):
  full  : the residual's full-band covariance (what `start --source k` does on the GPU today)
  edge  : the residual's covariance over the PFB-edge bins only (|u| > --ufrac of a --nsub-point FFT),
          lower edge for source 1, upper edge for source 2, full band after (an eigen-start change)

    atsc_nsrc.py --freq 2904,2914 --lags -16:47 --nsrc 4 --variants full,edge
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtv_blind as db  # noqa: E402
from dtv_modes import top_vecs, predict_p2  # noqa: E402
from atsc_beam import load_channel, CS  # noqa: E402

EDGE = 400


def weights(r, pw, Q, sel_u=None, nsub=64):
    """Protected top eigenvector of the per-input-whitened covariance of r (optionally only sub-channel bins sel_u)."""
    rw = r / np.sqrt(np.maximum(pw, 1e-3))
    if sel_u is None:
        R = (rw.T @ rw.conj()) / rw.shape[0]
    else:
        T = rw.shape[0] // nsub * nsub
        F = np.fft.fftshift(
            np.fft.fft(rw[:T].reshape(-1, nsub, rw.shape[1]), axis=1), axes=1
        )
        Y = F[:, sel_u, :].reshape(-1, rw.shape[1])
        R = (Y.T @ Y.conj()) / Y.shape[0]
    Pq = np.eye(R.shape[0]) - Q @ Q.conj().T
    _, Vd = top_vecs(Pq @ R @ Pq, 1)
    return Vd[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", required=True)
    ap.add_argument("--modes", default="/var/tmp/bb1030/dtv/modes_k00.npz")
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--lags", default="-16:47")
    ap.add_argument("--nsrc", type=int, default=4)
    ap.add_argument("--variants", default="full,edge")
    ap.add_argument("--nsub", type=int, default=64)
    ap.add_argument("--ufrac", type=float, default=0.35)
    ap.add_argument(
        "--noprotect",
        action="store_true",
        help="no Tau A protection or score (channels outside --modes' band, e.g. LTE): k = nan",
    )
    a = ap.parse_args()
    m = np.load(a.modes)
    mfreq, clean, V16, mlam = m["freq"].astype(np.int64), m["clean"], m["V"], m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    l0, l1 = (int(v) for v in a.lags.split(":"))
    lags = np.arange(l0, l1 + 1)
    u = (np.arange(a.nsub) - a.nsub // 2) / a.nsub
    sels = {"lower": u < -a.ufrac, "upper": u > a.ufrac}
    for fid in [int(v) for v in a.freq.split(",")]:
        t0 = time.time()
        if a.noprotect:
            Q = np.zeros((128, 0), complex)
        else:
            _, Q = top_vecs(predict_p2(P2, mfreq, clean, fid, 40, 0), 2)
        x = load_channel(a.dir, a.chunk, fid).astype(np.complex128)
        n = x.shape[0]
        w = np.zeros(n)
        w[EDGE : n - EDGE] = 1
        sl = slice(EDGE + 64, n - EDGE - 64)
        lines = []
        for var in a.variants.split(","):
            r = x
            parts = []
            for k in range(a.nsrc):
                pw = (np.abs(r[sl]) ** 2).mean(axis=0)
                if k == 0 or var == "full" or k > 2:
                    v = weights(r[sl], pw, Q)
                else:
                    v = weights(
                        r[sl], pw, Q, sels["lower" if k == 1 else "upper"], a.nsub
                    )
                s = (r / np.sqrt(np.maximum(pw, 1e-3))) @ np.conj(v)
                X = np.fft.fft(r * w[:, None], db.NFFT, axis=0)
                _, _, _, model = db.fit_lagged(X, [s * w], lags)
                r = np.fft.ifft(X - model, axis=0)[:n]
                R, lam, _, _ = db.whitened_eig(r[sl])
                nu = lam[20:120].mean()
                ex = db.tau_excess(R, Q)[0] if Q.shape[1] else float("nan")
                parts.append(
                    "%d src %s k%.2f"
                    % (k + 1, "/".join("%.2f" % (v_ / nu) for v_ in lam[:4]), ex / ref)
                )
            lines.append("%-4s: %s" % (var, "; ".join(parts)))
        print(
            "%d chunk %d lags %+d..%+d (%.0f s)\n   %s"
            % (fid, a.chunk, l0, l1, time.time() - t0, "\n   ".join(lines)),
            flush=True,
        )


if __name__ == "__main__":
    main()
