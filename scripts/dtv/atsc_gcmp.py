#!/usr/bin/env python3
"""Compare the impulse responses g fitted against the DECODED replica (atsc_sub, own band + the two folded-in
neighbour bands) with those of the BLIND method (protected eigen-beam, one least-squares fit = the live recipe).

1. delay profiles (power per lag summed over inputs, inverse-noise weighted);
2. per fine frequency inside the channel: the blind template is the transmitted signal through the beam's own
   response B(u), so G_dec,i(u) ~ B(u) G_blind,i(u); fit the common B(u) per bin and report the per-input
   disagreement left (fraction of |G_dec|^2) and the cross-input coherence, by position in the channel;
3. the subtracted DTV models m_i = g_i * s per input: their difference, in noise units (eigenvalues of the
   whitened difference covariance) and its Tau A template content.

    atsc_gcmp.py --freq 2904,2914,2924
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
from dtv_modes import top_vecs, predict_p2  # noqa: E402
from atsc_beam import load_channel, CS  # noqa: E402

W = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="2904,2914,2924")
    ap.add_argument("--arr", default="/var/tmp/bb1030/atsc/arr30_k00.npz")
    ap.add_argument("--rx", default="/var/tmp/bb1030/atsc/rxa30d_k00.npz")
    ap.add_argument("--eq", default="/var/tmp/bb1030/atsc/eq2_30_k00_w8.npz")
    ap.add_argument("--tcm", default="/var/tmp/bb1030/atsc/tcm30_k00.npz")
    ap.add_argument("--modes", default="/var/tmp/bb1030/dtv/modes_k00.npz")
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    C, fids, rz, fs_z, fc_z = S.replica_channels(a.arr, a.rx, a.eq, a.tcm)
    RZ = sfft.fft(rz, workers=W)
    fz = sfft.fftfreq(len(rz), 1 / fs_z) + fc_z
    m = np.load(a.modes)
    mfreq, clean, V16, mlam = m["freq"].astype(np.int64), m["clean"], m["V"], m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    lags = np.arange(-8, 41)
    NU = 512
    u = (np.arange(NU) - NU // 2) / NU
    E = np.exp(
        -2j * np.pi * u[:, None] * lags[None, :]
    )  # [NU, L]: G(u) = sum_tau g(tau) e^{-2pi i u tau}
    save = {}
    for fid in [int(v) for v in a.freq.split(",")]:
        Pp = predict_p2(P2, mfreq, clean, fid, 40, 0)
        _, Q = top_vecs(Pp, 2)
        x = load_channel(a.dir, a.chunk, fid).astype(np.complex128)
        n = x.shape[0]
        w = np.zeros(n)
        w[S.EDGE : n - S.EDGE] = 1
        X = np.fft.fft(x * w[:, None], db.NFFT, axis=0)
        sl = slice(S.EDGE + 64, n - S.EDGE - 64)
        # decoded: own band + the two folded-in neighbour bands
        regs = S.band_regressors(RZ, fz, fid, fids)
        Gd, _, _, model_d = db.fit_lagged(X, [r * w for r in regs], lags)
        # blind, live recipe: protected eigen-beam, one LS fit
        R0, lam0, V0, pw = db.whitened_eig(x)
        Pq = np.eye(128) - Q @ Q.conj().T
        _, Vd = top_vecs(Pq @ R0 @ Pq, 1)
        sb = (x / np.sqrt(pw)) @ np.conj(Vd[:, 0])
        Gb, _, _, model_b = db.fit_lagged(X, [sb * w], lags)
        md = np.fft.ifft(model_d, axis=0)[:n][sl]
        mb = np.fft.ifft(model_b, axis=0)[:n][sl]
        rd = x[sl] - md
        sig2 = (np.abs(rd) ** 2).mean(
            axis=0
        )  # per-input noise (decoded residual) for weighting/whitening
        iw = 1 / sig2
        # 1. delay profiles (template power x |g|^2 / noise, summed over inputs)
        Pd = np.mean(np.abs(regs[1][sl]) ** 2)
        Pb = np.mean(np.abs(sb[sl]) ** 2)
        prof_d = (np.abs(Gd[1]) ** 2 * iw[None, :]).sum(axis=1) * Pd
        prof_b = (np.abs(Gb[0]) ** 2 * iw[None, :]).sum(axis=1) * Pb

        def top(p):
            o = np.argsort(p)[::-1][:6]
            return " ".join(
                "%+d:%.2f" % (lags[j], p[j] / p.max())
                for j in sorted(o, key=lambda j: lags[j])
            )

        print("%d (%.2f MHz)" % (fid, fid * CS / 1e6))
        print(
            "  delay profile (lag frames of 5.12 us: power rel. peak), decoded own band: %s"
            % top(prof_d)
        )
        print(
            "                                                      blind: %s"
            % top(prof_b)
        )
        print(
            "  profile spread: power outside lags -2..+3: decoded %.3f, blind %.3f"
            % (
                prof_d[(lags < -2) | (lags > 3)].sum() / prof_d.sum(),
                prof_b[(lags < -2) | (lags > 3)].sum() / prof_b.sum(),
            )
        )
        # 2. per fine frequency, after the best common filter B(u)
        Fd = E @ Gd[1]  # [NU, M]
        Fb = E @ Gb[0]
        num = (np.conj(Fb) * Fd * iw[None, :]).sum(axis=1)
        den = (np.abs(Fb) ** 2 * iw[None, :]).sum(axis=1)
        B = num / den
        resid = (np.abs(Fd - B[:, None] * Fb) ** 2 * iw[None, :]).sum(axis=1)
        tot = (np.abs(Fd) ** 2 * iw[None, :]).sum(axis=1)
        coh = np.abs(num) / np.sqrt(den * tot)
        dis = resid / tot
        pos = ((u + 0.5) * 16).astype(int)
        print(
            "  per-input disagreement after a common filter (fraction of |G_dec|^2) by position in the channel, edge..centre..edge:"
        )
        print("     " + " ".join("%.3f" % np.mean(dis[pos == i]) for i in range(16)))
        print(
            "  cross-input coherence |<G_blind, G_dec>|:  "
            + " ".join("%.3f" % np.mean(coh[pos == i]) for i in range(16))
        )
        print(
            "  inner half of the channel: disagreement %.4f (%.1f dB), coherence %.4f; |B(u)| ripple (std/mean) %.2f"
            % (
                np.mean(dis[np.abs(u) < 0.25]),
                10 * np.log10(np.mean(dis[np.abs(u) < 0.25])),
                np.mean(coh[np.abs(u) < 0.25]),
                np.std(np.abs(B[np.abs(u) < 0.45]))
                / np.mean(np.abs(B[np.abs(u) < 0.45])),
            )
        )
        # 3. the subtracted models: difference in noise units, and Tau A in it
        dlt = (md - mb) / np.sqrt(sig2)[None, :]
        Rdl = dlt.T @ np.conj(dlt) / dlt.shape[0]
        rw = rd / np.sqrt(sig2)[None, :]
        Rn = rw.T @ np.conj(rw) / rw.shape[0]
        nunit = np.linalg.eigvalsh(Rn)[::-1][20:120].mean()
        evd = np.linalg.eigvalsh(Rdl)[::-1] / nunit
        tau_d = np.real(np.trace(Q.conj().T @ Rdl @ Q)) / 2 / nunit
        pm = np.mean(np.abs(md) ** 2)
        print(
            "  models: |m_dec - m_blind|^2 / |m_dec|^2 = %.4f (%.1f dB); difference eigenvalues (noise units) %s; Tau A template power in the difference %.2f (clean Tau A excess %.2f)"
            % (
                np.mean(np.abs(md - mb) ** 2) / pm,
                10 * np.log10(np.mean(np.abs(md - mb) ** 2) / pm),
                " ".join("%.2f" % v for v in evd[:6]),
                tau_d,
                ref,
            ),
            flush=True,
        )
        # where does the model difference live: inner part of the channel vs the outer edges
        D = sfft.fft(md - mb, axis=0, workers=W)
        fu = sfft.fftfreq(D.shape[0])
        for name, msk in (
            ("inner |u|<0.375", np.abs(fu) < 0.375),
            ("edges |u|>=0.375", np.abs(fu) >= 0.375),
        ):
            dd = (
                sfft.ifft(np.where(msk[:, None], D, 0), axis=0, workers=W)
                / np.sqrt(sig2)[None, :]
            )
            ev_ = np.linalg.eigvalsh(dd.T @ np.conj(dd) / dd.shape[0])[::-1] / nunit
            print(
                "     model difference, %s (%.0f%% of the band): eigenvalues %s; share of the difference power %.2f"
                % (
                    name,
                    100 * msk.mean(),
                    " ".join("%.2f" % v for v in ev_[:5]),
                    np.sum(np.abs(dd) ** 2) / np.sum(np.abs(dlt) ** 2),
                )
            )
        save[fid] = dict(prof_d=prof_d, prof_b=prof_b, dis=dis, coh=coh, B=B, evd=evd)
    if a.out:
        np.savez(
            a.out,
            lags=lags,
            u=u,
            **{"%d_%s" % (f, k): v for f, r in save.items() for k, v in r.items()}
        )
    print("done (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
