#!/usr/bin/env python3
"""Projection with a PROTECTED Tau A subspace: if the Tau A steering vectors are known (here the
template extrapolated from the clean channels; in production the element calibration), take
the DTV modes as the top-K eigenvectors of R deflated by the template, (I - Q) R (I - Q), so
they are orthogonal to Tau A by construction, and project those out. Compare with the plain
top-K projection of dtv_modes.py on the same channels.

    dtv_protect.py --cov cov_k00.npz --modes modes_k00.npz --freq 2700-2760 [--noise-whiten] [--kmax 10]
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtv_modes import whiten, top_vecs, project, predict_p2  # noqa: E402

CS = 0.1953125


def ranges(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cov", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--freq", required=True)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--noise-whiten", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(a.cov)
    freq = z["freq"].astype(np.int64)
    cov = z["cov"]
    frame = int(z["frame"])
    m = np.load(a.modes)
    clean = m["clean"]
    V16 = m["V"]
    pw = m["pw"]
    lam = m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    want = ranges(a.freq)
    K = a.kmax + 1
    rows = []
    print(
        "%7s %5s %6s | plain top-K: Tau A kept (K=0..%d) | protected: Tau A kept | protected: DTV left = sum(lam_res-1) over modes 3..12"
        % ("MHz", "fid", "lam1", a.kmax)
    )
    for c in np.nonzero(np.isin(freq, want))[0]:
        Pp = predict_p2(P2, freq, clean, freq[c], a.win, a.gap)
        if Pp is None:
            continue
        _, Q = top_vecs(Pp, 2)
        sel = (
            clean
            & (np.abs(freq - freq[c]) <= a.win)
            & (np.abs(freq - freq[c]) >= max(a.gap, 1))
        )
        tau_ref = np.median(0.5 * (lam[sel, 0] + lam[sel, 1]))
        pw_ref = None
        if a.noise_whiten:
            idx = np.nonzero(sel)[0]
            A = np.stack(
                [np.ones(len(idx)), (freq[idx] - freq[c]).astype(np.float64)], axis=1
            )
            coef = np.linalg.lstsq(A, pw[idx], rcond=None)[0]
            pw_ref = np.maximum(coef[0], 0.3 * np.median(pw[idx], axis=0))
        _, R, _ = whiten(cov[c], frame, pw_ref)
        QQ = Q @ Q.conj().T
        Rd = (np.eye(128) - QQ) @ R @ (np.eye(128) - QQ)
        _, Vplain = top_vecs(R, a.kmax)
        _, Vprot = top_vecs(Rd, a.kmax)
        fp, fq, left = [], [], []
        for k in range(K):
            Rp = project(R, Vplain[:, :k])
            fp.append((np.real(np.trace(Q.conj().T @ Rp @ Q)) / 2 - 1) / (tau_ref - 1))
            Rq = project(R, Vprot[:, :k])
            fq.append((np.real(np.trace(Q.conj().T @ Rq @ Q)) / 2 - 1) / (tau_ref - 1))
            ev = np.linalg.eigvalsh(Rq)[::-1]
            left.append(np.sum(ev[2:12] - 1.0))
        rows.append((freq[c], fp, fq, left))
        print(
            "%7.2f %5d %6.1f | %s | %s | %s"
            % (
                freq[c] * CS,
                freq[c],
                lam[c, 0],
                " ".join("%5.2f" % v for v in fp),
                " ".join("%5.2f" % v for v in fq),
                " ".join("%5.1f" % v for v in left),
            )
        )
    if a.out:
        np.savez(
            a.out,
            freq=np.array([r[0] for r in rows]),
            plain=np.array([r[1] for r in rows]),
            protected=np.array([r[2] for r in rows]),
            left=np.array([r[3] for r in rows]),
        )


if __name__ == "__main__":
    main()
