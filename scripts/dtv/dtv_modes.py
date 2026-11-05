#!/usr/bin/env python3
"""Can a K-mode projection recover Tau A under the DTV channels? From dtv_cov.py's per-frame
covariances (one chunk, 0.84 s, 20 frames of 8192 samples).

Per channel: whiten each input by its chunk-mean power, R_f (frame) and R (chunk mean), the
eigenvalues (noise = 1, Tau A = two modes of ~3.2 on a clean channel, one per sky polarisation).
Clean channels (lam3 < clean_lam3) define the Tau A template: the projector onto their top-2
subspace, P2 = V V^H, whose element phases are per-input-pair delays, so P2 at any channel is
predicted from the clean channels within +-win by a per-element phase slope (from adjacent clean
pairs, no unwrapping) and intercept, and its top-2 eigenvectors are the predicted Tau A subspace
Q at that channel. Then for K = 0..kmax modes projected out (eigenvectors of R, chunk level, or
of R_f per frame):
    tau(K)      = Tr(Q R_proj) / 2            Tau A power left (noise = 1; clean channels ~3.2)
    frac(K)     = (tau(K) - 1) / (tau_ref - 1) relative to the clean neighbours' interpolated value
    lam_res(K)  = top eigenvalues of R_proj   what else is left
    ovl(K)      = Tr(Q Q_res2) / 2            does the residual's own top-2 subspace agree with Tau A
On a clean channel (leave-one-out template) frac(0) measures the template's fidelity.

    dtv_modes.py --cov /var/tmp/bb1030/dtv/cov_k00.npz --out /var/tmp/bb1030/dtv/modes_k00.npz [--kmax 8] [--win 40]
"""
import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

CS = 0.1953125


def whiten(cov, frame, pw_ref=None):
    """cov [nfr, 128, 128] raw sums -> R_f [nfr,128,128] whitened by the chunk-mean power (or by
    pw_ref, e.g. the clean neighbours' noise power), R, pw."""
    pw = np.real(np.einsum("fii->i", cov)).astype(np.float64) / (cov.shape[0] * frame)
    pr = pw if pw_ref is None else pw_ref
    w = np.where(pr > 1e-3, 1.0 / np.sqrt(np.maximum(pr, 1e-3)), 0.0)
    Rf = (cov / frame) * w[None, :, None] * w[None, None, :]
    return Rf.astype(np.complex128), Rf.mean(axis=0).astype(np.complex128), pw


def top_vecs(R, k):
    ev, V = np.linalg.eigh(R)
    return ev[::-1], V[:, ::-1][:, :k]


def project(R, V):
    """(I - V V^H) R (I - V V^H)."""
    if V.shape[1] == 0:
        return R
    P = np.eye(R.shape[0]) - V @ V.conj().T
    return P @ R @ P


def predict_p2(P2, freq, clean, f0, win, gap=0):
    """P2 at freq f0 from the clean channels within +-win but outside +-gap (channel units):
    per-element phase slope from adjacent clean pairs, intercept = circular mean after removing
    the slope. gap > 0 on a clean channel is the control for extrapolating across a DTV channel."""
    sel = clean & (np.abs(freq - f0) <= win) & (np.abs(freq - f0) >= max(gap, 1))
    idx = np.nonzero(sel)[0]
    if len(idx) < 6:
        return None
    Ps = P2[idx]; fs = freq[idx].astype(np.float64)
    adj = np.nonzero(np.diff(fs) == 1)[0]
    if len(adj) < 3:
        return None
    d = Ps[adj + 1] * np.conj(Ps[adj])                   # phase step per channel, weighted by |P|^2
    slope = np.angle(d.sum(axis=0))                      # radians per channel, per element
    derot = Ps * np.exp(-1j * slope[None] * (fs - f0)[:, None, None])
    amp = np.abs(Ps).mean(axis=0)
    ph = np.angle(derot.sum(axis=0))
    Ppred = amp * np.exp(1j * ph)
    Ppred = 0.5 * (Ppred + Ppred.conj().T)
    return Ppred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cov", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kmax", type=int, default=8); ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--clean-lam3", type=float, default=1.6)
    ap.add_argument("--nvec", type=int, default=16, help="eigenvectors of R to save per channel")
    ap.add_argument("--gap", type=int, default=0, help="exclude +-gap channels from the template fit (control)")
    ap.add_argument("--noise-whiten", action="store_true",
                    help="whiten by the clean neighbours' per-input power (linear in f) instead of the channel's own")
    a = ap.parse_args()
    z = np.load(a.cov)
    freq = z["freq"].astype(np.int64); cov = z["cov"]; frame = int(z["frame"])
    nch, nfr = cov.shape[:2]
    t0 = time.time()
    lam = np.zeros((nch, 128)); V16 = np.zeros((nch, 128, a.nvec), np.complex64)
    lam_f = np.zeros((nch, nfr, 16)); pw = np.zeros((nch, 128))
    P2 = np.zeros((nch, 128, 128), np.complex128)
    Rs = []; Rfs = []
    for c in range(nch):
        Rf, R, p = whiten(cov[c], frame)
        pw[c] = p
        ev, V = top_vecs(R, a.nvec)
        lam[c] = ev; V16[c] = V
        P2[c] = V[:, :2] @ V[:, :2].conj().T
        for j in range(nfr):
            lam_f[c, j] = np.linalg.eigvalsh(Rf[j])[::-1][:16]
        Rs.append(R); Rfs.append(Rf)
    clean = (lam[:, 2] < a.clean_lam3) & (lam[:, 0] < 10.0) & (pw.max(axis=1) > 0)
    print("eigen done %.0f s: %d/%d clean channels (lam3 < %.2f); clean lam1/lam2/lam3 median %.2f %.2f %.2f"
          % (time.time() - t0, clean.sum(), nch, a.clean_lam3, *np.median(lam[clean, :3], axis=0)), flush=True)
    K = a.kmax + 1
    tau_c = np.full((nch, K), np.nan); tau_f = np.full((nch, K), np.nan)
    lam_res_c = np.full((nch, K, 6), np.nan); lam_res_f = np.full((nch, K, 6), np.nan)
    ovl_c = np.full((nch, K), np.nan); ovl_f = np.full((nch, K), np.nan)
    stab = np.full((nch, K), np.nan)        # mean over frames of Tr(P_K,f P_K,chunk)/K
    tau_ref = np.full(nch, np.nan); tmpl_ok = np.zeros(nch, bool)
    lam_nw = np.full((nch, 16), np.nan)      # eigenvalues under the noise whitening (--noise-whiten)
    for c in range(nch):
        if pw[c].max() <= 0:
            continue
        Pp = predict_p2(P2, freq, clean, freq[c], a.win, a.gap)
        if Pp is None:
            continue
        tmpl_ok[c] = True
        _, Q = top_vecs(Pp, 2)
        # reference Tau A level: clean neighbours' (lam1+lam2)/2, nearest in frequency
        sel = clean & (np.abs(freq - freq[c]) <= a.win) & (np.abs(freq - freq[c]) >= max(a.gap, 1))
        tau_ref[c] = np.median(0.5 * (lam[sel, 0] + lam[sel, 1]))
        R = Rs[c]; Rf = Rfs[c]
        if a.noise_whiten:
            # whiten by the clean neighbours' per-input power, linear in frequency across the gap
            idx = np.nonzero(sel)[0]
            A = np.stack([np.ones(len(idx)), (freq[idx] - freq[c]).astype(np.float64)], axis=1)
            coef = np.linalg.lstsq(A, pw[idx], rcond=None)[0]
            pw_ref = np.maximum(coef[0], 0.3 * np.median(pw[idx], axis=0))
            Rf, R, _ = whiten(cov[c], frame, pw_ref)
            lam_nw[c] = np.linalg.eigvalsh(R)[::-1][:16]
        _, Vc = top_vecs(R, a.kmax)
        Vfs = [top_vecs(Rf[j], a.kmax)[1] for j in range(nfr)]
        for k in range(K):
            Rp = project(R, Vc[:, :k])
            tau_c[c, k] = np.real(np.trace(Q.conj().T @ Rp @ Q)) / 2
            evp, Vp = top_vecs(Rp, 2)
            lam_res_c[c, k] = evp[:6]
            ovl_c[c, k] = np.real(np.trace(Q.conj().T @ Vp @ Vp.conj().T @ Q)) / 2
            tf = 0.0; lr = np.zeros(6); ov = 0.0; st = 0.0
            for j in range(nfr):
                Rpf = project(Rf[j], Vfs[j][:, :k])
                tf += np.real(np.trace(Q.conj().T @ Rpf @ Q)) / 2
                evpf, Vpf = top_vecs(Rpf, 2)
                lr += evpf[:6]
                ov += np.real(np.trace(Q.conj().T @ Vpf @ Vpf.conj().T @ Q)) / 2
                if k:
                    st += np.real(np.trace(Vfs[j][:, :k].conj().T @ Vc[:, :k] @ Vc[:, :k].conj().T @ Vfs[j][:, :k])) / k
            tau_f[c, k] = tf / nfr; lam_res_f[c, k] = lr / nfr; ovl_f[c, k] = ov / nfr
            stab[c, k] = st / nfr if k else np.nan
        if c % 50 == 0:
            print("  %d/%d %.0f s" % (c, nch, time.time() - t0), flush=True)
    frac_c = (tau_c - 1) / (tau_ref[:, None] - 1); frac_f = (tau_f - 1) / (tau_ref[:, None] - 1)
    np.savez(a.out, freq=freq, mhz=freq * CS, lam=lam, lam_f=lam_f, V=V16, pw=pw, clean=clean, tmpl_ok=tmpl_ok,
             tau_ref=tau_ref, tau_c=tau_c, tau_f=tau_f, frac_c=frac_c, frac_f=frac_f, lam_res_c=lam_res_c,
             lam_res_f=lam_res_f, ovl_c=ovl_c, ovl_f=ovl_f, stab=stab, kmax=a.kmax, win=a.win, frame=frame,
             gap=a.gap, noise_whiten=a.noise_whiten, lam_nw=lam_nw)
    print("saved %s (%.0f s)" % (a.out, time.time() - t0))
    # summary
    print("template fidelity on clean channels: frac(0) median %.3f, 5th pct %.3f; ovl(0) median %.3f"
          % (np.nanmedian(frac_c[clean, 0]), np.nanpercentile(frac_c[clean, 0], 5), np.nanmedian(ovl_c[clean, 0])))
    hot = ~clean & tmpl_ok
    print("\n%7s %5s %6s %6s %6s %6s | chunk-level frac(K) K=0..%d | frame-level frac(K) | stab(K=1,2,4)"
          % ("MHz", "fid", "lam1", "lam2", "lam3", "lam4", a.kmax))
    for c in np.nonzero(hot)[0]:
        print("%7.2f %5d %6.1f %6.1f %6.1f %6.1f | %s | %s | %s"
              % (freq[c] * CS, freq[c], *lam[c, :4],
                 " ".join("%5.2f" % v for v in frac_c[c]), " ".join("%5.2f" % v for v in frac_f[c]),
                 " ".join("%4.2f" % stab[c, k] for k in (1, 2, 4) if k <= a.kmax)))


if __name__ == "__main__":
    main()
