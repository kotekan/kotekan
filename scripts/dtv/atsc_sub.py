#!/usr/bin/env python3
"""Subtract a DECODED 8-VSB replica from the raw PFB channels of every input, and score it like dtv_blind.py.

Replica: the trellis-decoded symbols (atsc_tcm.py; syncs known; +1.25 pilot on every symbol) through the
transmit root raised cosine on the receiver's symbol grid (atsc_rx: start, Tn, clock), times the tracked
phase (atsc_eq2 phi), mapped back to the stitched wideband frame (undo the band-centre and pilot shifts,
including the pilot's own 5 ms phase track), then channelised by the measured F-engine PFB model (4-tap
sinc x Hamming, atsc_ipfb). Per input and channel: least squares of x_i on the lagged replica channel
(lags -8..40 frames, dtv_blind.fit_lagged), subtract. Scores: own-whitened eigenvalues in noise units and
the Tau A template excess (modes_k00.npz), raw / after / after one extra protected blind source.

    atsc_sub.py --freq 2899,2904,2909,2914,2919,2924 --out /var/tmp/bb1030/atsc/sub30_k00.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import RS, ALPHA, cubic_at  # noqa: E402
from atsc_ipfb import pfb_sim  # noqa: E402
from atsc_beam import load_channel, CS  # noqa: E402
import dtv_blind as db  # noqa: E402
from dtv_modes import top_vecs, predict_p2  # noqa: E402

W = 8
EDGE = 400  # frames excluded at each end (replica support, PFB/lag transients)


def rrc_2sps(span=48):
    """Transmit RRC (Nyquist Rs/4) at 2 samples per symbol, t in units of 2T."""
    t = np.arange(-4 * span, 4 * span + 1) / 4.0
    b = ALPHA
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-12:
            h[i] = 1 - b + 4 * b / np.pi
        elif abs(abs(x) - 1 / (4 * b)) < 1e-9:
            h[i] = b / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - b)) + 4 * b * x * np.cos(np.pi * x * (1 + b))) / (np.pi * x * (1 - (4 * b * x) ** 2))
    return h


def replica_channels(arr, rx, eq, tcm, use_phi=True, data=True, pilot=1.25):
    z = np.load(arr)
    fs, fc = float(z["fs"]), float(z["fc"])
    zz = z["z"].astype(np.complex128)
    fids = z["fids"]
    r = np.load(rx)
    start, Tn, fo, delta, rsign = float(r["start"]), float(r["Tn"]), float(r["fo"]), float(r["delta"]), complex(r["rsign"])
    fp = float(r["pilot_hz"]) - fc
    nz = len(zz)
    t = np.arange(nz) / fs
    # the pilot's 5 ms phase track, exactly as atsc_rx
    blk = int(fs * 0.005)
    nb = nz // blk
    amp = (zz[: nb * blk] * np.exp(-2j * np.pi * fp * t[: nb * blk])).reshape(nb, blk).mean(axis=1)
    ph = np.unwrap(np.angle(amp))
    php = np.interp(t, (np.arange(nb) + 0.5) * blk / fs, ph)
    del zz
    sym = np.load(tcm)["sym"].astype(float)
    nsym = len(sym)
    k = np.arange(nsym)
    c2 = np.zeros(2 * nsym, np.complex128)
    c2[0::2] = ((sym if data else 0.0) + pilot) * rsign ** (k % 4)
    r2 = sfft.ifft(sfft.fft(c2, workers=W) * sfft.fft(np.roll(np.pad(rrc_2sps(), (0, 2 * nsym - len(rrc_2sps()))), -(len(rrc_2sps()) // 2)), workers=W), workers=W)
    if use_phi:
        phi = np.load(eq)["phi"].astype(float)[:nsym]
        r2 *= np.exp(1j * np.interp(np.arange(2 * nsym) / 2.0, k, phi))
    # 4x band-limited upsampling (exact), then cubic onto the stitched time base
    R2 = sfft.fft(r2, workers=W)
    n2 = len(r2)
    R8 = np.zeros(4 * n2, np.complex128)
    R8[: n2 // 2] = R2[: n2 // 2]
    R8[-n2 // 2:] = R2[-n2 // 2:]
    r8 = sfft.ifft(R8, workers=W) * 4
    pos8 = 4 * (t * fo - start) / (Tn / 2)
    ok = (pos8 > 2) & (pos8 < len(r8) - 3)
    rc = np.zeros(nz, np.complex128)
    rc[ok] = cubic_at(r8, pos8[ok])
    rz = rc * np.exp(1j * (2 * np.pi * fp * t + php + 2 * np.pi * (RS / 4) * (1 - delta) * t))
    C = pfb_sim(rz, len(fids), "hamming")  # [frames, n_chan], channel j <-> fids[j]
    return C, fids, rz, fs, fc


def band_regressors(RZ, fz, fid, fids, comps=(-1, 0, 1)):
    """The replica channel `fid` split by source band: for each offset in comps, the part of the wideband
    replica in the nominal band of channel fid+offset, channelised (what the PFB folds into channel fid)."""
    j = list(fids).index(fid)
    out = []
    for o in comps:
        kk = fid + o
        band = (fz >= kk * CS - CS / 2) & (fz < kk * CS + CS / 2)
        out.append(pfb_sim(sfft.ifft(np.where(band, RZ, 0), workers=W), len(fids), "hamming")[:, j].astype(np.complex128))
    return out


def score(R, Q, ref):
    ev = np.linalg.eigvalsh(R)[::-1]
    n = ev[20:120].mean()
    ex, _ = db.tau_excess(R, Q)
    return ev[:6] / n, ex / ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arr", default="/var/tmp/bb1030/atsc/arr30_k00.npz")
    ap.add_argument("--rx", default="/var/tmp/bb1030/atsc/rxa30d_k00.npz")
    ap.add_argument("--eq", default="/var/tmp/bb1030/atsc/eq2_30_k00_w8.npz")
    ap.add_argument("--tcm", default="/var/tmp/bb1030/atsc/tcm30_k00.npz")
    ap.add_argument("--modes", default="/var/tmp/bb1030/dtv/modes_k00.npz")
    ap.add_argument("--dir", default="/var/tmp/bb1030/dtv")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--freq", default="2899,2904,2909,2914,2919,2924")
    ap.add_argument("--no-phi", action="store_true", help="replica without the tracked phase (control)")
    ap.add_argument("--pilot-reg", action="store_true", help="the pilot as its own (band-split) regressors in the channels near it")
    ap.add_argument("--alias", action="store_true", help="separate regressors for the replica's own band and the two neighbours' bands folded in by the PFB")
    ap.add_argument("--lmin", type=int, default=-8, help="first lag, frames (5.12 us)")
    ap.add_argument("--lmax", type=int, default=40, help="last lag, frames")
    ap.add_argument("--peel", type=int, default=1, help="protected blind sources to peel after the decoded one")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    C, fids, rz, fs_z, fc_z = replica_channels(a.arr, a.rx, a.eq, a.tcm, use_phi=not a.no_phi)
    RZ = sfft.fft(rz, workers=W) if a.alias else None
    if a.pilot_reg:
        _, _, rz_d, _, _ = replica_channels(a.arr, a.rx, a.eq, a.tcm, use_phi=not a.no_phi, pilot=0.0)
        _, _, rz_p, _, _ = replica_channels(a.arr, a.rx, a.eq, a.tcm, use_phi=not a.no_phi, data=False)
        RZd, RZp = sfft.fft(rz_d, workers=W), sfft.fft(rz_p, workers=W)
        f_pil = float(np.load(a.rx)["pilot_hz"])
    fz = sfft.fftfreq(len(rz), 1 / fs_z) + fc_z
    print("replica channelised: %s frames x %d channels (%.0f s)" % (C.shape[0], C.shape[1], time.time() - t0), flush=True)
    m = np.load(a.modes)
    mfreq = m["freq"].astype(np.int64)
    clean = m["clean"]
    V16 = m["V"]
    mlam = m["lam"]
    P2 = np.einsum("cik,cjk->cij", V16[:, :, :2], np.conj(V16[:, :, :2]))
    n_clean = mlam[:, 20:120].mean(axis=1)
    ref = np.median((0.5 * (mlam[clean, 0] + mlam[clean, 1]) - 1.0) / n_clean[clean])
    lags = np.arange(a.lmin, a.lmax + 1)
    out = {}
    for fid in [int(v) for v in a.freq.split(",")]:
        Pp = predict_p2(P2, mfreq, clean, fid, 40, 0)
        _, Q = top_vecs(Pp, 2)
        x = load_channel(a.dir, a.chunk, fid).astype(np.complex128)
        j = list(fids).index(fid)
        if a.pilot_reg and abs(fid * CS - f_pil) < 1.5 * CS:
            regs = band_regressors(RZd, fz, fid, fids) + band_regressors(RZp, fz, fid, fids)
            s = None
        elif a.alias:
            regs = band_regressors(RZ, fz, fid, fids)
            s = None
        else:
            regs = None
            s = C[:, j].astype(np.complex128)
        n = x.shape[0]
        w = np.zeros(n)
        w[EDGE: n - EDGE] = 1
        xw = x * w[:, None]
        X = np.fft.fft(xw, db.NFFT, axis=0)
        G, Gf, Sf, model = db.fit_lagged(X, [r_ * w for r_ in regs] if regs is not None else [s * w], lags)
        res = np.fft.ifft(X - model, axis=0)[:n]
        sl = slice(EDGE + 64, n - EDGE - 64)
        R0 = (lambda y: (y.T @ y.conj()) / y.shape[0])(x[sl] / np.sqrt(np.maximum((np.abs(x[sl]) ** 2).mean(axis=0), 1e-3)))
        lam0, k0 = score(R0, Q, ref)
        # whiten the residual by the RAW per-input power (same units as the raw), and by its own power (as dtv_blind)
        yr = res[sl]
        R1 = (yr / np.sqrt(np.maximum((np.abs(yr) ** 2).mean(axis=0), 1e-3))).T @ np.conj(yr / np.sqrt(np.maximum((np.abs(yr) ** 2).mean(axis=0), 1e-3))) / yr.shape[0]
        lam1, k1 = score(R1, Q, ref)
        pr = (np.abs(yr) ** 2).sum() / (np.abs(x[sl]) ** 2).sum()
        prof = np.sum(np.abs(G[0]) ** 2, axis=1)
        top = np.argsort(prof)[::-1][:4]
        print("%d (%.2f MHz): raw %s (Tau A kept %.2f) | decoded replica: power left %.3f, %s (kept %.2f); g lags %s" % (
            fid, fid * CS / 1e6, " ".join("%.1f" % v for v in lam0), k0, pr, " ".join("%.2f" % v for v in lam1), k1,
            " ".join("%+d" % lags[j] for j in top)), flush=True)
        rec = dict(lam0=lam0, kept0=k0, lam1=lam1, kept1=k1, power_left=pr, G=G, sig2=(np.abs(yr) ** 2).mean(axis=0))
        if a.peel:
            rb = db.blind(res, 8, 40, 1, Q, lambda m_: None, protect=True)
            yb = rb["r"][sl]
            if yb is not None:
                R2_ = (yb / np.sqrt(np.maximum((np.abs(yb) ** 2).mean(axis=0), 1e-3))).T @ np.conj(yb / np.sqrt(np.maximum((np.abs(yb) ** 2).mean(axis=0), 1e-3))) / yb.shape[0]
                lam2, k2 = score(R2_, Q, ref)
                print("      + 1 protected blind source: %s (kept %.2f)" % (" ".join("%.2f" % v for v in lam2), k2), flush=True)
                rec.update(lam2=lam2, kept2=k2)
        out[fid] = rec
    if a.out:
        np.savez(a.out, **{"%d_%s" % (f, kk): v for f, rr in out.items() for kk, v in rr.items()})
    print("done (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
