#!/usr/bin/env python3
"""Blind single-source channel identification and subtraction, per 195 kHz channel, on the
2026-10-30 baseband dump (all 128 inputs, one chunk = 163840 samples of 5.12 us).

Model: x_i(t) = sum_tau g_i(tau) s(t - tau) + n_i(t): one transmitter timestream s, a lagged
impulse response per input (direct path, reflections, the dish's horizon response). Solve by
alternation: s0 = the top eigen-beam; g_i = least squares of x_i on the lagged copies of s
(Toeplitz normal equations, lags -L1..L2); s = MMSE deconvolution combining every input and lag,
S(f) = sum_i conj(G_i) X_i / sig_i^2 / (sum_i |G_i|^2 / sig_i^2 + 1/P_s); repeat. Then
r_i = x_i - g_i * s. The residual is judged like dtv_modes.py: whitened eigenvalues in noise
units and the Tau A template beam's excess (template from modes_k00.npz's clean channels).

    dtv_blind.py --dir /var/tmp/bb1030/dtv --chunk 0 --modes modes_k00.npz --freq 2714-2745 --out blind_ch24.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtv_modes import top_vecs, predict_p2  # noqa: E402

NT, FRAME, NFFT = 163840, 8192, 1 << 18
CS = 0.1953125
LUT = np.empty(256, dtype=np.complex64)
for b in range(256):
    LUT[b] = complex((b >> 4) - 8, (b & 0xF) - 8)


def ranges(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def load_channel(dirp, chunk, fid):
    r = fid % 16
    fr = np.load("%s/k%02d_r%02d_freq.npy" % (dirp, chunk, r))
    col = int(np.nonzero(fr == fid)[0][0])
    a = np.load("%s/k%02d_r%02d.npy" % (dirp, chunk, r), mmap_mode="r")
    return LUT[np.asarray(a[:, col]).reshape(NT, 128)]


def whitened_eig(y):
    p = (np.abs(y) ** 2).mean(axis=0)
    yw = y / np.sqrt(np.maximum(p, 1e-3))
    R = (yw.T @ yw.conj()) / y.shape[0]           # R_ik = <y_i conj(y_k)> as dtv_modes: beam b = yw @ conj(v), <|b|^2> = v^H R v
    ev, V = np.linalg.eigh(R)
    return R, ev[::-1], V[:, ::-1], p


def lagcorr(b, s, lags):
    """Normalised cross-correlation of b with s at the given lags (b(t) vs s(t - lag))."""
    B = np.fft.fft(b, NFFT); S = np.fft.fft(s, NFFT)
    c = np.fft.ifft(B * np.conj(S))[lags % NFFT]
    return np.abs(c) / np.sqrt((np.abs(b) ** 2).sum() * (np.abs(s) ** 2).sum())


def fit_lagged(X, regs, lags):
    """Least squares of every input on the lagged copies of the regressor timestreams.
    X [NFFT, M] input spectra; regs: list of timestreams. Returns G [P, L, M] and the model spectrum."""
    L = len(lags); P = len(regs); M = X.shape[1]
    Sf = [np.fft.fft(s, NFFT) for s in regs]
    C = np.concatenate([np.fft.ifft(X * np.conj(S)[:, None], axis=0)[lags % NFFT] for S in Sf], axis=0)   # [P*L, M]
    A = np.zeros((P * L, P * L), np.complex128)
    for p in range(P):
        for q in range(P):
            a = np.fft.ifft(Sf[q] * np.conj(Sf[p]))                       # a_pq(k) = sum_t s_q(t) conj(s_p(t-k))
            A[p * L:(p + 1) * L, q * L:(q + 1) * L] = a[(lags[:, None] - lags[None, :]) % NFFT]
    G = np.linalg.solve(A, C).reshape(P, L, M)
    model = np.zeros_like(X); Gf = []
    for p in range(P):
        gpad = np.zeros((NFFT, M), np.complex128); gpad[lags % NFFT] = G[p]
        gf = np.fft.fft(gpad, axis=0); Gf.append(gf)
        model += gf * Sf[p][:, None]
    return G, Gf, Sf, model


def tau_excess(R, Q):
    """Template beam's excess over the mean per-input power, in units of the noise block."""
    n = np.linalg.eigvalsh(R)[::-1][20:120].mean()
    return (np.real(np.trace(Q.conj().T @ R @ Q)) / 2 - np.real(np.trace(R)) / 128) / n, n


def blind(x, L1, L2, iters, Q, log, cubic=False, protect=False):
    T, M = x.shape
    R0, lam0, V0, pw = whitened_eig(x)
    v0 = V0[:, 0]
    if protect:
        # start from the top eigenvector of the covariance deflated by the Tau A template, so the
        # initial timestream carries no Tau A (eigenvectors of near-equal eigenvalues mix otherwise)
        Pq = np.eye(M) - Q @ Q.conj().T
        _, Vd = top_vecs(Pq @ R0 @ Pq, 1)
        v0 = Vd[:, 0]
    s = (x / np.sqrt(pw)) @ np.conj(v0)                    # top eigen-beam, power lam0[0]
    X = np.fft.fft(x, NFFT, axis=0)
    lags = np.arange(-L1, L2 + 1)
    hist = []
    for it in range(iters):
        regs = [s] + ([s * (np.abs(s) ** 2 - (np.abs(s) ** 2).mean())] if cubic else [])
        G, Gf, Sf, model = fit_lagged(X, regs, lags)
        r = np.fft.ifft(X - model, axis=0)[:T]
        sig2 = (np.abs(r) ** 2).mean(axis=0)
        Rr, lamr, Vr, _ = whitened_eig(r)
        exr, nr = tau_excess(Rr, Q)
        hist.append((lamr[:8].copy(), exr, nr))
        log("    iter %d: residual eigenvalues (own-whitened) %s; Tau A excess %.2f (noise block %.2f)"
            % (it, " ".join("%.2f" % v for v in lamr[:6]), exr, nr))
        if it < iters - 1:
            Ps = (np.abs(s) ** 2).mean()
            num = (np.conj(Gf[0]) * X / sig2[None, :]).sum(axis=1)
            den = (np.abs(Gf[0]) ** 2 / sig2[None, :]).sum(axis=1)
            s = np.fft.ifft(num / (den + 1.0 / Ps))[:T]
    Ps = (np.abs(s) ** 2).mean()
    prof = (np.abs(G[0]) ** 2 * Ps / sig2[None, :]).sum(axis=1)                # DTV power per lag, noise units, all inputs
    # per-frame residual: own-whitened top eigenvalues per 8192-sample frame
    fr = []
    for j in range(T // FRAME):
        _, l, _, _ = whitened_eig(r[j * FRAME:(j + 1) * FRAME])
        fr.append(l[:4])
    # what is the residual's top mode? its inputs, and its beam's correlation with s and with the cubic term
    u = Vr[:, 0]
    rw = r / np.sqrt(np.maximum(sig2, 1e-3))
    br = rw @ np.conj(u)
    cl = np.arange(-60, 61)
    c_lin = lagcorr(br, s, cl); c_cub = lagcorr(br, s * (np.abs(s) ** 2 - (np.abs(s) ** 2).mean()), cl)
    rail = ((np.abs(x.real) >= 7) | (np.abs(x.imag) >= 7)).mean(axis=0)
    top_in = np.argsort(np.abs(u) ** 2)[::-1][:8]
    return dict(lam0=lam0, pw=pw, G=G[0], lags=lags, sig2=sig2, lamr=hist[-1][0], ex_after=hist[-1][1], n_after=hist[-1][2],
                prof=prof, frame_lam=np.median(np.array(fr), axis=0), hist=hist, R0=R0, Rr=Rr, r=r.astype(np.complex64),
                top_in=top_in, top_w=np.abs(u[top_in]) ** 2, rail=rail, c_lin=c_lin, c_cub=c_cub, cl=cl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--modes", required=True); ap.add_argument("--freq", required=True)
    ap.add_argument("--l1", type=int, default=8, help="taps before the reference lag")
    ap.add_argument("--l2", type=int, default=40, help="taps after (5.12 us each)")
    ap.add_argument("--iters", type=int, default=3); ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--cubic", action="store_true", help="also regress on lagged s|s|^2 (odd-order clipping distortion)")
    ap.add_argument("--nsrc", type=int, default=1, help="sources to peel in turn (each from the residual's top eigen-beam)")
    ap.add_argument("--protect", action="store_true", help="initialise each source's timestream orthogonal to the Tau A template")
    ap.add_argument("--out")
    a = ap.parse_args()
    m = np.load(a.modes); mfreq = m["freq"].astype(np.int64); clean = m["clean"]; V16 = m["V"]; mlam = m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])     # clean Tau A excess, noise units
    log = lambda msg: print(msg, flush=True)
    log("clean-channel Tau A excess reference: %.2f noise units" % ref)
    results = {}
    t0 = time.time()
    for fid in ranges(a.freq):
        c = np.searchsorted(mfreq, fid)
        Pp = predict_p2(P2, mfreq, clean, fid, a.win, 0)
        if Pp is None:
            log("%d: no template" % fid); continue
        _, Q = top_vecs(Pp, 2)
        x = load_channel(a.dir, a.chunk, fid)
        log("%d (%.2f MHz): loaded %.0f s" % (fid, fid * CS, time.time() - t0))
        stages = []
        for k in range(a.nsrc):
            res = blind(x, a.l1, a.l2, a.iters, Q, log, cubic=a.cubic, protect=a.protect)
            ex0, n0 = tau_excess(res["R0"], Q)
            res["ex_before"] = ex0; res["n_before"] = n0
            top = np.argsort(res["prof"])[::-1]
            taps = [(int(res["lags"][j]), res["prof"][j]) for j in top[:8] if res["prof"][j] > 0.01 * res["prof"][top[0]]]
            log("  source %d before: eigenvalues/noise %s; Tau A excess %.2f (kept %.2f)"
                % (k + 1, " ".join("%.1f" % (v / n0) for v in res["lam0"][:6]), ex0, ex0 / ref))
            log("  source %d after : eigenvalues/noise %s; Tau A excess %.2f (kept %.2f); per-frame residual top-4 %s"
                % (k + 1, " ".join("%.2f" % (v / res["n_after"]) for v in res["lamr"][:6]), res["ex_after"], res["ex_after"] / ref,
                   " ".join("%.2f" % v for v in res["frame_lam"])))
            log("  delay profile (lag samples: power in noise units, summed over inputs): %s"
                % "; ".join("%+d (%+.1f us, %+.1f km): %.1f" % (l, l * 5.12, l * 1.536, p) for l, p in taps))
            rs = res["sig2"] / res["pw"]
            log("  residual/original power per input: median %.2f, min %.2f, inputs with > 0.9 (little removed): %d, with < 0.3: %d"
                % (np.median(rs), rs.min(), (rs > 0.9).sum(), (rs < 0.3).sum()))
            log("  residual top mode: inputs %s with |u|^2 %s (rail fraction %s); its beam vs s: max |corr| %.3f at lag %+d; vs s|s|^2: %.3f at lag %+d"
                % (res["top_in"].tolist(), np.round(res["top_w"], 2).tolist(), np.round(res["rail"][res["top_in"]], 2).tolist(),
                   res["c_lin"].max(), res["cl"][res["c_lin"].argmax()], res["c_cub"].max(), res["cl"][res["c_cub"].argmax()]))
            stages.append({kk: v for kk, v in res.items() if kk not in ("R0", "Rr", "hist", "r")})
            x = res["r"]
        out = dict(stages[0])                       # a copy: the per-stage arrays below must not see the overrides
        out["lam_stage"] = np.array([s["lamr"][:8] / s["n_after"] for s in stages])
        out["kept_stage"] = np.array([s["ex_after"] / ref for s in stages])
        out["prof2"] = stages[-1]["prof"]
        out["kept_before"] = stages[0]["ex_before"] / ref
        out["kept_after"] = stages[-1]["ex_after"] / ref
        out["lamr"] = stages[-1]["lamr"]; out["n_after"] = stages[-1]["n_after"]
        out["frame_lam"] = stages[-1]["frame_lam"]
        results[fid] = out
    if a.out:
        fids = sorted(results)
        np.savez(a.out, freq=np.array(fids), ref=ref, lags=results[fids[0]]["lags"],
                 lam0=np.array([results[f]["lam0"][:16] for f in fids]), n_before=np.array([results[f]["n_before"] for f in fids]),
                 lamr=np.array([results[f]["lamr"][:16] for f in fids]), n_after=np.array([results[f]["n_after"] for f in fids]),
                 kept_before=np.array([results[f]["kept_before"] for f in fids]), kept_after=np.array([results[f]["kept_after"] for f in fids]),
                 prof=np.array([results[f]["prof"] for f in fids]), G=np.array([results[f]["G"] for f in fids]),
                 sig2=np.array([results[f]["sig2"] for f in fids]), pw=np.array([results[f]["pw"] for f in fids]),
                 frame_lam=np.array([results[f]["frame_lam"] for f in fids]),
                 lam_stage=np.array([results[f]["lam_stage"] for f in fids]), kept_stage=np.array([results[f]["kept_stage"] for f in fids]),
                 prof2=np.array([results[f]["prof2"] for f in fids]))
        log("saved %s" % a.out)


if __name__ == "__main__":
    main()
