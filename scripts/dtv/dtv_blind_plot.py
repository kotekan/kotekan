#!/usr/bin/env python3
"""Figure + table for dtv_blind.py outputs: eigenvalues in noise units before and after each
peeled source, Tau A kept, and the delay profile (Green's function magnitude per lag) of the
first source across the DTV channel's sub-channels.

    dtv_blind_plot.py --main blind_ch24.npz --others blind_ch30.npz,blind_ch17.npz --out dtv_blind_20261102.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CS = 0.1953125
COL = ["#1f5f8b", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]


def table(path):
    z = np.load(path)
    f = z["freq"]; lam0 = z["lam0"][:, :6] / z["n_before"][:, None]; ls = z["lam_stage"]; kb = z["kept_before"]; ks = z["kept_stage"]
    print("%s: %d sub-channels %.2f-%.2f MHz" % (path.split("/")[-1], len(f), f[0] * CS, f[-1] * CS))
    print("   before          : lam/noise %s; Tau A kept %.2f" % (" ".join("%6.1f" % v for v in np.median(lam0, axis=0)), np.median(kb)))
    for k in range(ls.shape[1]):
        print("   after source %d  : lam/noise %s; Tau A kept %.2f (min %.2f)"
              % (k + 1, " ".join("%6.2f" % v for v in np.median(ls[:, k, :6], axis=0)), np.median(ks[:, k]), ks[:, k].min()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True); ap.add_argument("--others", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in [a.main] + [o for o in a.others.split(",") if o]:
        table(p)
    z = np.load(a.main)
    f = z["freq"]; mhz = f * CS; lam0 = z["lam0"] / z["n_before"][:, None]; ls = z["lam_stage"]
    kb = z["kept_before"]; ks = z["kept_stage"]; prof = z["prof"]; lags = z["lags"]
    nst = ls.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw=dict(hspace=0.3))
    ax = axes[0]
    for i in range(4):
        ax.plot(mhz, lam0[:, i], color=COL[0], lw=0.8, alpha=1 - 0.2 * i, label="before" if i == 0 else None)
    for k in range(nst):
        for i in range(4):
            ax.plot(mhz, ls[:, k, i], color=COL[k + 1], lw=0.8, alpha=1 - 0.2 * i, label="after source %d" % (k + 1) if i == 0 else None)
    ax.axhline(5.63 / 0.92, color="k", lw=0.6, ls="--"); ax.axhline(5.04 / 0.92, color="k", lw=0.6, ls="--")
    ax.axhline(1.38 / 0.92, color="k", lw=0.6, ls=":")
    ax.text(mhz[0], 5.63 / 0.92 * 1.15, "Tau A pair on clean channels", fontsize=8)
    ax.text(mhz[0], 1.38 / 0.92 * 1.1, "clean-channel third mode", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("top-4 eigenvalues / noise"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("DTV channel 24 (530-536 MHz), 0.84 s: blind single-source channel identification and subtraction, peeled in turn", fontsize=10)
    ax = axes[1]
    ax.plot(mhz, kb, color=COL[0], lw=1, marker="o", ms=3, label="before")
    for k in range(nst):
        ax.plot(mhz, ks[:, k], color=COL[k + 1], lw=1, marker="o", ms=3, label="after source %d" % (k + 1))
    ax.axhline(1.0, color="k", lw=0.5, ls=":"); ax.set_ylim(0, 1.15); ax.set_ylabel("Tau A kept\n(template excess, noise-referenced)")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlabel("MHz")
    ax = axes[2]
    P = prof / prof.max(axis=1, keepdims=True)
    im = ax.imshow(10 * np.log10(np.maximum(P, 1e-4)).T, aspect="auto", origin="lower", cmap="viridis", vmin=-30, vmax=0,
                   extent=[mhz[0] - 0.1, mhz[-1] + 0.1, lags[0] * 5.12 - 2.56, lags[-1] * 5.12 + 2.56])
    ax.set_ylabel("lag relative to the dominant arrival (us)"); ax.set_xlabel("MHz")
    ax.set_title("source 1 delay profile: power per lag summed over the 128 inputs, dB below the peak (5.12 us = 1.54 km of path)", fontsize=9)
    plt.colorbar(im, ax=ax, label="dB")
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
