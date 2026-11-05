#!/usr/bin/env python3
"""Figure for dtv_modes.py output: eigenvalue spectrum across the band, the Tau A fraction
kept after projecting K modes (chunk- and frame-level eigenvectors), and what is left.

    dtv_plot.py --modes /var/tmp/bb1030/dtv/modes_k00.npz --out ~/gnss/bb1030/dtv_modes_20261102.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = ["#1f5f8b", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8", "#555555", "#999999"]


def dtv_grid(ax, lo, hi):
    for n in range(14, 37):
        f0 = 470 + 6 * (n - 14)
        if lo - 6 < f0 < hi:
            ax.axvline(f0, color="#bbbbbb", lw=0.5, zorder=0)
            ax.text(f0 + 3, ax.get_ylim()[1], "%d" % n, ha="center", va="top", fontsize=6, color="#777777")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="0,1,2,3,4,6,8")
    a = ap.parse_args()
    z = np.load(a.modes)
    mhz = z["mhz"]; lam = z["lam"]; clean = z["clean"]; ok = z["tmpl_ok"]
    frac_c = z["frac_c"]; frac_f = z["frac_f"]; lam_res_c = z["lam_res_c"]; lam_res_f = z["lam_res_f"]
    stab = z["stab"]; kmax = int(z["kmax"])
    ks = [int(k) for k in a.ks.split(",") if int(k) <= kmax]
    lo, hi = mhz.min(), mhz.max()
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True, gridspec_kw=dict(hspace=0.12))
    ax = axes[0]
    for i in range(6):
        ax.plot(mhz, lam[:, i], lw=0.8, color=COL[i], label=r"$\lambda_%d$" % (i + 1))
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_yscale("log"); ax.set_ylim(0.9, max(60, np.nanmax(lam[:, 0]) * 1.3))
    ax.set_ylabel("whitened eigenvalue\n(noise = 1)")
    ax.legend(ncol=6, fontsize=8, loc="upper left")
    ax.set_title("2026-10-30 baseband chunk 0 (0.84 s), 128 inputs whitened: Tau A = the pair at 4.5-5.5; DTV channels (numbers at top)", fontsize=10)
    dtv_grid(ax, lo, hi)
    for ax, fr, lab in ((axes[1], frac_c, "chunk-level eigenvectors (0.84 s)"), (axes[2], frac_f, "frame-level eigenvectors (42 ms), mean of 20 frames")):
        for j, k in enumerate(ks):
            ax.plot(mhz[ok], fr[ok, k], lw=0.8, color=COL[j % len(COL)], label="K = %d" % k)
        ax.axhline(1.0, color="k", lw=0.5, ls=":"); ax.axhline(0.0, color="k", lw=0.5, ls=":")
        ax.set_ylim(-0.35, 1.15); ax.set_ylabel("Tau A power kept\nafter projecting K modes")
        ax.legend(ncol=len(ks), fontsize=8, loc="lower left"); ax.text(0.99, 0.95, lab, ha="right", va="top", transform=ax.transAxes, fontsize=9)
        dtv_grid(ax, lo, hi)
    ax = axes[3]
    for j, k in enumerate(ks):
        ax.plot(mhz[ok], lam_res_f[ok, k, 2], lw=0.8, color=COL[j % len(COL)], label="K = %d" % k)
    ax.set_yscale("log"); ax.set_ylim(1.0, max(30, np.nanmax(lam_res_f[ok, :, 2]) * 1.3))
    ax.set_ylabel(r"$\lambda_3$ after projection" + "\n(frame level; what is left beside Tau A)")
    ax.legend(ncol=len(ks), fontsize=8, loc="upper left"); dtv_grid(ax, lo, hi)
    ax = axes[4]
    for j, k in enumerate([k for k in ks if k > 0]):
        ax.plot(mhz[ok], stab[ok, k], lw=0.8, color=COL[(j + 1) % len(COL)], label="K = %d" % k)
    ax.set_ylim(0, 1.05); ax.set_ylabel("frame vs chunk top-K\nsubspace overlap")
    ax.legend(ncol=len(ks), fontsize=8, loc="lower left"); dtv_grid(ax, lo, hi)
    ax.set_xlabel("MHz"); ax.set_xlim(lo, hi)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
