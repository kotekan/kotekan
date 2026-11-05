#!/usr/bin/env python3
"""Stability of the DTV Green's functions across the 80 s dump: fit the two-source blind
solution on one chunk (dtv_blind.py's method, Tau A-protected), then on later chunks apply it
FROZEN (the saved MMSE weights give the timestream, the saved g_i(tau) are subtracted) and
TRACKED (same timestream, g refitted by least squares on the new chunk, which keeps the same
timestream reference so g(chunk) is directly comparable with g(0)).

    dtv_gstab.py fit   --dir D --chunk 0  --modes modes_k00.npz --freq 2717,2723,... --out g_k00.npz
    dtv_gstab.py apply --dir D --chunk 48 --modes modes_k00.npz --g g_k00.npz [--out stab_k48.npz]

Per channel and stage the report gives the residual eigenvalues (noise units) and Tau A kept
for raw / frozen / tracked, and the change of g: the power-weighted relative difference
sum |g' - g|^2 / sum |g|^2 and the amplitude ratio and phase change of the strongest taps.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtv_modes import top_vecs, predict_p2  # noqa: E402
from dtv_blind import (
    NT,
    NFFT,
    CS,
    load_channel,
    whitened_eig,
    tau_excess,
    fit_lagged,
    ranges,
)  # noqa: E402


def mmse_weights(Gf, sig2, Ps, Q=None, pw=None):
    """MMSE combining weights [NFFT, M] (S = sum_i W_i X_i). With Q (the Tau A template in
    whitened coordinates) and pw, each frequency's weight vector is projected orthogonal to the
    template, so the timestream carries no Tau A and the subtraction cannot remove it; without
    this a weak source's iteration drifts onto Tau A (kept 0.86 -> 0.73 on channel 24)."""
    den = (np.abs(Gf) ** 2 / sig2[None, :]).sum(axis=1) + 1.0 / Ps
    W = np.conj(Gf) / sig2[None, :] / den[:, None]
    if Q is not None:
        sq = np.sqrt(np.maximum(pw, 1e-3))
        Ww = W * sq[None, :]
        Ww = Ww - (Ww @ np.conj(Q)) @ Q.T
        W = Ww / sq[None, :]
    return W


def model_spectrum(G, lags, M):
    gpad = np.zeros((NFFT, M), np.complex128)
    gpad[lags % NFFT] = G
    return np.fft.fft(gpad, axis=0)


def fit_stage(x, Q, lags, iters, protect=True):
    """One source: protected eigen start, LS g, MMSE s, iterate. Returns G, sig2, Ps, residual."""
    T, M = x.shape
    R0, lam0, V0, pw = whitened_eig(x)
    v0 = V0[:, 0]
    if protect:
        Pq = np.eye(M) - Q @ Q.conj().T
        _, Vd = top_vecs(Pq @ R0 @ Pq, 1)
        v0 = Vd[:, 0]
    s = (x / np.sqrt(pw)) @ np.conj(v0)
    X = np.fft.fft(x, NFFT, axis=0)
    for it in range(iters + 1):
        # the last pass refits g on the final MMSE timestream, so the saved (g, weights) are
        # consistent: applying the saved weights to new data reproduces this s, not the one before
        G, Gf, Sf, model = fit_lagged(X, [s], lags)
        r = np.fft.ifft(X - model, axis=0)[:T]
        sig2 = (np.abs(r) ** 2).mean(axis=0)
        if it < iters:
            Ps = (np.abs(s) ** 2).mean()
            W = mmse_weights(Gf[0], sig2, Ps, Q if protect else None, pw)
            s = np.fft.ifft((W * X).sum(axis=1))[:T]
    return G[0], sig2, (np.abs(s) ** 2).mean(), r.astype(np.complex64)


def apply_stage(x, G, sig2, Ps, lags, Q=None):
    """Frozen: s from the saved weights, subtract the saved g. Tracked: refit g on this s."""
    T, M = x.shape
    X = np.fft.fft(x, NFFT, axis=0)
    Gf = model_spectrum(G, lags, M)
    W = mmse_weights(Gf, sig2, Ps, Q, (np.abs(x) ** 2).mean(axis=0))
    S = (W * X).sum(axis=1)
    s = np.fft.ifft(S)[:T]
    r_frozen = np.fft.ifft(X - Gf * S[:, None], axis=0)[:T]
    G2, Gf2, _, model2 = fit_lagged(X, [s], lags)
    r_tracked = np.fft.ifft(X - model2, axis=0)[:T]
    return r_frozen.astype(np.complex64), r_tracked.astype(np.complex64), G2[0]


def report(label, r, Q, ref):
    R, lam, _, _ = whitened_eig(r)
    ex, n = tau_excess(R, Q)
    return (
        "%-8s lam/noise %s  Tau A kept %.2f"
        % (label, " ".join("%5.2f" % (v / n) for v in lam[:5]), ex / ref),
        lam[:8] / n,
        ex / ref,
    )


def g_change(G0, G1):
    d = np.sum(np.abs(G1 - G0) ** 2) / np.sum(np.abs(G0) ** 2)
    prof = (np.abs(G0) ** 2).sum(axis=1)
    top = np.argsort(prof)[::-1][:3]
    taps = []
    for k in top:
        amp = np.sqrt(np.sum(np.abs(G1[k]) ** 2) / np.sum(np.abs(G0[k]) ** 2))
        ph = np.angle(np.sum(G1[k] * np.conj(G0[k])))
        taps.append((int(k), amp, ph))
    return d, taps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--freq")
    ap.add_argument("--g")
    ap.add_argument("--out")
    ap.add_argument("--l1", type=int, default=8)
    ap.add_argument("--l2", type=int, default=40)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--nsrc", type=int, default=2)
    ap.add_argument("--win", type=int, default=40)
    a = ap.parse_args()
    m = np.load(a.modes)
    mfreq = m["freq"].astype(np.int64)
    clean = m["clean"]
    V16 = m["V"]
    mlam = m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    lags = np.arange(-a.l1, a.l2 + 1)
    t0 = time.time()
    if a.mode == "fit":
        out = {}
        for fid in ranges(a.freq):
            Pp = predict_p2(P2, mfreq, clean, fid, a.win, 0)
            _, Q = top_vecs(Pp, 2)
            x = load_channel(a.dir, a.chunk, fid)
            print(
                "%d (%.2f MHz) chunk %d fit, %.0f s"
                % (fid, fid * CS, a.chunk, time.time() - t0),
                flush=True,
            )
            print("  " + report("raw", x, Q, ref)[0])
            st = []
            for k in range(a.nsrc):
                G, sig2, Ps, r = fit_stage(x, Q, lags, a.iters)
                print("  " + report("src %d" % (k + 1), r, Q, ref)[0])
                st.append((G, sig2, Ps))
                x = r
            out[fid] = st
        np.savez(
            a.out,
            freq=np.array(sorted(out)),
            lags=lags,
            nsrc=a.nsrc,
            chunk=a.chunk,
            G=np.array([[s[0] for s in out[f]] for f in sorted(out)]),
            sig2=np.array([[s[1] for s in out[f]] for f in sorted(out)]),
            Ps=np.array([[s[2] for s in out[f]] for f in sorted(out)]),
        )
        print("saved", a.out)
    else:
        g = np.load(a.g)
        gfreq = g["freq"]
        lags = g["lags"]
        nsrc = int(g["nsrc"])
        rows = []
        for j, fid in enumerate(gfreq):
            fid = int(fid)
            Pp = predict_p2(P2, mfreq, clean, fid, a.win, 0)
            _, Q = top_vecs(Pp, 2)
            x = load_channel(a.dir, a.chunk, fid)
            print(
                "%d (%.2f MHz) chunk %d apply (g from chunk %d), %.0f s"
                % (fid, fid * CS, a.chunk, int(g["chunk"]), time.time() - t0),
                flush=True,
            )
            line, lam_raw, kept_raw = report("raw", x, Q, ref)
            print("  " + line)
            xf = x
            xt = x
            res = dict(fid=fid, lam_raw=lam_raw, kept_raw=kept_raw)
            for k in range(nsrc):
                G0, sig2, Ps = g["G"][j, k], g["sig2"][j, k], g["Ps"][j, k]
                rf, _, _ = apply_stage(xf, G0, sig2, Ps, lags, Q)  # frozen chain
                _, rt, G1 = apply_stage(xt, G0, sig2, Ps, lags, Q)  # tracked chain
                d, taps = g_change(G0, G1)
                lf, lam_f, kept_f = report("frozen %d" % (k + 1), rf, Q, ref)
                lt, lam_t, kept_t = report("tracked %d" % (k + 1), rt, Q, ref)
                print("  " + lf)
                print("  " + lt)
                print(
                    "    g change: sum|dg|^2/sum|g|^2 = %.4f; top taps (lag: amp ratio, phase deg): %s"
                    % (
                        d,
                        "; ".join(
                            "%+d: %.3f, %+.1f" % (lags[k2], amp, np.degrees(ph))
                            for k2, amp, ph in taps
                        ),
                    )
                )
                res["lam_frozen%d" % k] = lam_f
                res["lam_tracked%d" % k] = lam_t
                res["kept_frozen%d" % k] = kept_f
                res["kept_tracked%d" % k] = kept_t
                res["dg%d" % k] = d
                xf = rf
                xt = rt
            rows.append(res)
        if a.out:
            np.savez(
                a.out,
                chunk=a.chunk,
                gchunk=int(g["chunk"]),
                freq=np.array([r["fid"] for r in rows]),
                **{k: np.array([r[k] for r in rows]) for k in rows[0] if k != "fid"}
            )
            print("saved", a.out)


if __name__ == "__main__":
    main()
