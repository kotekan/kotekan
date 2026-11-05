#!/usr/bin/env python3
"""Replay the live DTV subtraction stage on the 2026-10-30 baseband dump (chunk 0, Tau A in the beam).

    dtv_replay.py prep    --freq 2512,2720,2912 --out x.bin                 # the channels' 4+4b bytes
    dtv_replay.py protect --freq ... --frames 0-9 --nsrc 2 --out st.bin     # protected eigen-start weights
    dtv_replay.py eval    --freq ... --res res.bin --frames 10-19           # residual vs the raw channel

prep writes uint8 [163840][n_chan][128] as the GPU replay (dtv_replay.cu) reads it. protect writes an
observe-only state with source 0's weights = the top whitened eigenvector of the frames' covariance
deflated by the Tau A template (dtv_blind.py --protect's start), so the timestream carries no Tau A.
eval scores the residual like dtv_blind.py: whitened eigenvalues in noise units (noise = mean of
eigenvalues 20..120) and the Tau A template beam's excess against the clean-channel reference.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/scripts/gnss"))
from dtv_blind import LUT, load_channel, tau_excess, whitened_eig  # noqa: E402
from dtv_modes import predict_p2, top_vecs  # noqa: E402
import dtv_state  # noqa: E402

NT, FRAME = 163840, 8192
DIR = "/var/tmp/bb1030/dtv"


def fids_of(s):
    return [int(v) for v in s.split(",")]


def frame_slice(spec):
    a, _, b = spec.partition("-")
    return slice(int(a) * FRAME, (int(b or a) + 1) * FRAME)


def template(modes, fid, win=40):
    m = np.load(modes)
    mfreq = m["freq"].astype(np.int64)
    clean = m["clean"]
    V16 = m["V"]
    mlam = m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    Pp = predict_p2(P2, mfreq, clean, fid, win, 0)
    return (None if Pp is None else top_vecs(Pp, 2)[1]), ref


def cmd_prep(a):
    fids = fids_of(a.freq)
    out = np.empty((NT, len(fids), 128), np.uint8)
    for c, fid in enumerate(fids):
        r = fid % 16
        fr = np.load("%s/k%02d_r%02d_freq.npy" % (DIR, a.chunk, r))
        col = int(np.nonzero(fr == fid)[0][0])
        raw = np.load("%s/k%02d_r%02d.npy" % (DIR, a.chunk, r), mmap_mode="r")
        out[:, c, :] = np.asarray(raw[:, col]).reshape(NT, 128)
        print(
            "freq_id %d (%.3f MHz): residue %d column %d"
            % (fid, fid * 0.1953125, r, col),
            flush=True,
        )
    out.tofile(a.out)
    print("wrote %s: %d frames x %d channels" % (a.out, NT // FRAME, len(fids)))


def cmd_protect(a):
    """Source a.source's weights: from the raw channel (source 0) or from a replay's residual bytes
    (--res, a later source), added to --state's records (or a new observe-only state)."""
    fids = fids_of(a.freq)
    st = (
        dtv_state.read_state(a.state)
        if a.state
        else dict(n_lag=a.n_lag, lag_min=a.lag_min, n_src=a.nsrc, recs={})
    )
    res = np.fromfile(a.res, np.uint8).reshape(NT, len(fids), 128) if a.res else None
    for c, fid in enumerate(fids):
        Q, _ = template(a.modes, fid)
        if Q is None:
            raise SystemExit("freq_id %d: no Tau A template" % fid)
        sl = frame_slice(a.frames)
        x = (
            LUT[res[sl, c, :]]
            if res is not None
            else load_channel(DIR, a.chunk, fid)[sl]
        )
        R, lam, V, p = whitened_eig(x)
        P = np.eye(128) - Q @ Q.conj().T
        ev, U = np.linalg.eigh(P @ R @ P)
        v = U[:, -1]
        # the GPU beam is s = sum_i w_i x_i; dtv_blind's beam is yw @ conj(v), yw = x / sqrt(p)
        w = (np.conj(v) / np.sqrt(np.maximum(p, 1e-3))).astype(np.complex64)
        rec = st["recs"].get(fid)
        if rec is None:
            rec = dict(
                n_set=0,
                subtract=0,
                w=np.zeros((st["n_src"], 128), np.complex64),
                g=np.zeros((st["n_src"], st["n_lag"], 128), np.complex64),
            )
            st["recs"][fid] = rec
        rec["w"][a.source] = w
        rec["n_set"] = max(rec["n_set"], a.source + 1)
        n = lam[20:120].mean()
        print(
            "freq_id %d source %d: %s whitened eigenvalues/noise %s; protected top eigenvalue %.1f; |v . Tau A template|^2 %.4f"
            % (
                fid,
                a.source,
                "residual" if res is not None else "raw",
                " ".join("%.1f" % (l / n) for l in lam[:5]),
                ev[-1] / n,
                np.sum(np.abs(Q.conj().T @ v) ** 2),
            ),
            flush=True,
        )
    dtv_state.write_state(a.out, st)
    print(
        "wrote %s (source %d without g: fit it from a replay with this state)"
        % (a.out, a.source)
    )


def score(x, Q, ref):
    R, lam, V, p = whitened_eig(x)
    ex, n = tau_excess(R, Q)
    return lam / n, ex / ref, p


def cmd_eval(a):
    fids = fids_of(a.freq)
    res = np.fromfile(a.res, np.uint8).reshape(NT, len(fids), 128)
    sl = frame_slice(a.frames)
    for c, fid in enumerate(fids):
        Q, ref = template(a.modes, fid)
        x0 = load_channel(DIR, a.chunk, fid)[sl]
        x1 = LUT[res[sl, c, :]]
        l0, k0, p0 = score(x0, Q, ref)
        l1, k1, p1 = score(x1, Q, ref)
        changed = np.mean(
            res[sl, c, :]
            != np.asarray(
                np.load("%s/k%02d_r%02d.npy" % (DIR, a.chunk, fid % 16), mmap_mode="r")[
                    sl,
                    int(
                        np.nonzero(
                            np.load(
                                "%s/k%02d_r%02d_freq.npy" % (DIR, a.chunk, fid % 16)
                            )
                            == fid
                        )[0][0]
                    ),
                ]
            ).reshape(-1, 128)
        )
        print("freq_id %d (%.3f MHz), frames %s:" % (fid, fid * 0.1953125, a.frames))
        print(
            "   raw     : eigenvalues/noise %s; Tau A kept %.2f"
            % (" ".join("%.2f" % v for v in l0[:6]), k0)
        )
        print(
            "   residual: eigenvalues/noise %s; Tau A kept %.2f; residual/raw power per input median %.3f; bytes changed %.2f"
            % (" ".join("%.2f" % v for v in l1[:6]), k1, np.median(p1 / p0), changed),
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--modes", default=DIR + "/../dtv/modes_k00.npz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--freq", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("protect")
    p.add_argument("--freq", required=True)
    p.add_argument("--frames", default="0-9")
    p.add_argument("--nsrc", type=int, default=2)
    p.add_argument("--lag-min", type=int, default=-8)
    p.add_argument("--n-lag", type=int, default=49)
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--state", help="state to add the source's weights to")
    p.add_argument("--res", help="replay residual bytes to start from (a later source)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_protect)
    p = sub.add_parser("eval")
    p.add_argument("--freq", required=True)
    p.add_argument("--res", required=True)
    p.add_argument("--frames", default="10-19")
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
