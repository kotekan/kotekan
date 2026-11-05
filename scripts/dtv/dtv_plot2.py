#!/usr/bin/env python3
"""Noise-referenced figure for dtv_modes.py output. The own-power whitening scales everything in
a DTV channel by the noise's share n_c of the input power (the F-engine's per-channel gain was
levelled with the DTV present), so eigenvalues are shown as lambda / n_c (noise = 1) and the
Tau A statistic is the template beam's excess over the average residual power per input,
    excess(K) = Tr(Q R_proj)/2 - Tr(R_proj)/128,   kept(K) = excess(K) / n_c / <clean excess(0) / n_c>,
with n_c = mean of eigenvalues 20..120 (the noise block).

    dtv_plot2.py --modes modes_k00.npz --out dtv_modes_20261102.png [--table]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = [
    "#1f5f8b",
    "#c0392b",
    "#2e8b57",
    "#8e44ad",
    "#d68910",
    "#17a2b8",
    "#555555",
    "#999999",
]
DTV = {17: 488, 19: 500, 24: 530, 30: 566, 31: 572, 33: 584, 35: 596}


def dtv_grid(ax, lo, hi):
    for n in range(14, 37):
        f0 = 470 + 6 * (n - 14)
        if lo - 6 < f0 < hi:
            ax.axvline(f0, color="#bbbbbb", lw=0.5, zorder=0)
            ax.text(
                f0 + 3,
                ax.get_ylim()[1],
                "%d" % n,
                ha="center",
                va="top",
                fontsize=6,
                color="#777777",
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="0,1,2,3,4,5,6")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    z = np.load(a.modes)
    mhz = z["mhz"]
    lam = z["lam"]
    clean = z["clean"]
    ok = z["tmpl_ok"]
    tau_c = z["tau_c"]
    lam_res = z["lam_res_c"]
    kmax = int(z["kmax"])
    ks = [int(k) for k in a.ks.split(",") if int(k) <= kmax]
    n_c = lam[:, 20:120].mean(axis=1)
    cum = np.concatenate(
        [np.zeros((lam.shape[0], 1)), np.cumsum(lam, axis=1)], axis=1
    )  # sum of top-K
    tr = (128.0 - cum[:, : kmax + 1]) / 128.0  # Tr(R_proj)/128
    excess = (tau_c - tr) / n_c[:, None]
    ref = np.nanmedian(excess[clean, 0])
    kept = excess / ref
    lam_n = lam / n_c[:, None]
    res_n = lam_res[:, :, 2] / n_c[:, None]
    lo, hi = mhz.min(), mhz.max()
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 13), sharex=True, gridspec_kw=dict(hspace=0.12)
    )
    ax = axes[0]
    for i in range(8):
        ax.plot(mhz, lam_n[:, i], lw=0.8, color=COL[i], label=r"$\lambda_%d$" % (i + 1))
    ax.axhline(5.63 / 0.92, color="k", lw=0.6, ls="--")
    ax.axhline(5.04 / 0.92, color="k", lw=0.6, ls="--")
    ax.text(lo + 1, 5.63 / 0.92 * 1.1, "Tau A pair (clean channels)", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.9, max(300, np.nanmax(lam_n[:, 0]) * 1.3))
    ax.set_ylabel("eigenvalue / noise\n(128 inputs, 0.84 s)")
    ax.legend(ncol=8, fontsize=8, loc="upper left")
    ax.set_title(
        "2026-10-30 baseband chunk 0, 470-600 MHz: input covariance eigenvalues in noise units; ATSC channels numbered at the top",
        fontsize=10,
    )
    dtv_grid(ax, lo, hi)
    ax = axes[1]
    for j, k in enumerate(ks):
        ax.plot(
            mhz[ok], kept[ok, k], lw=0.8, color=COL[j % len(COL)], label="K = %d" % k
        )
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Tau A kept after projecting\nK modes (noise-referenced)")
    ax.legend(ncol=len(ks), fontsize=8, loc="lower left")
    dtv_grid(ax, lo, hi)
    ax = axes[2]
    for j, k in enumerate(ks):
        ax.plot(
            mhz[ok], res_n[ok, k], lw=0.8, color=COL[j % len(COL)], label="K = %d" % k
        )
    ax.axhline(5.04 / 0.92, color="k", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1.0, max(100, np.nanmax(res_n[ok, 0]) * 1.3))
    ax.set_ylabel(
        r"$\lambda_3$ of the residual / noise"
        + "\n(the third mode left after projection)"
    )
    ax.legend(ncol=len(ks), fontsize=8, loc="upper left")
    dtv_grid(ax, lo, hi)
    ax = axes[3]
    ax.plot(
        mhz, 1 - n_c / 0.92, lw=0.8, color=COL[0], label="DTV share of the input power"
    )
    ax.plot(
        mhz,
        n_c / 0.92,
        lw=0.8,
        color=COL[1],
        label="noise share = the F-engine's relative gain in that channel",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of per-input power")
    ax.legend(fontsize=8, loc="center left")
    dtv_grid(ax, lo, hi)
    ax.set_xlabel("MHz")
    ax.set_xlim(lo, hi)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)
    if a.table:
        print(
            "ATSC ch  MHz range   lam1/n  modes>TauA  DTV share | Tau A kept at K=0 1 2 3 4 5 6 (median over the channel's middle half) | lam3_res/n at K=4"
        )
        for n, f0 in sorted(DTV.items()):
            sel = (mhz > f0 + 1.5) & (mhz < f0 + 4.5)
            if not sel.any():
                continue
            above = (lam_n[sel] > 5.63 / 0.92).sum(axis=1)
            print(
                "  %2d   %3d-%3d MHz  %6.1f   %4.1f       %4.2f   | %s | %5.1f"
                % (
                    n,
                    f0,
                    f0 + 6,
                    np.median(lam_n[sel, 0]),
                    np.median(above),
                    np.median(1 - n_c[sel] / 0.92),
                    " ".join("%5.2f" % np.nanmedian(kept[sel, k]) for k in ks),
                    np.nanmedian(res_n[sel, 4]),
                )
            )


if __name__ == "__main__":
    main()
