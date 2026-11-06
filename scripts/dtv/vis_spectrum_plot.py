#!/usr/bin/env python3
"""Two vis_spectrum.py outputs over the whole band, before and after: per channel the top eigenvalue of the
input-whitened N2 in noise units, with the DTV subtraction's targets shaded, and a zoom on 340-620 MHz with the
second eigenvalue too. Prints per target the median top eigenvalue before and after.

    vis_spectrum_plot.py --before "11-01 07:05Z, no subtraction=pre.npz" --after "11-06 07:06Z=post.npz" --out f.png
"""
import argparse

import numpy as np

# freq_id ranges (vis_dtv_extract / the manifest's dtv-targets)
TARGETS = (
    ("MUOS", 1845, 1943),
    ("17", 2499, 2529),
    ("19", 2560, 2590),
    ("24", 2714, 2744),
    ("30", 2898, 2928),
    ("31", 2929, 2959),
    ("33", 2991, 3020),
    ("35", 3052, 3082),
)
C_BEFORE, C_AFTER, C_BAND = "#9ca3af", "#2563eb", "#fde68a"


def mhz(fid):
    return 300.0 + (fid - 1536) * 0.1953125


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--before", required=True, help="LABEL=NPZ")
    ap.add_argument("--after", required=True, help="LABEL=NPZ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (lb, pb), (la, pa) = (s.split("=", 1) for s in (a.before, a.after))
    B, A = np.load(pb), np.load(pa)
    f, x = A["fid"], A["mhz"]
    print("%-6s %9s %9s %9s %9s" % ("target", "before", "after", "ratio", "chan"))
    for name, lo, hi in TARGETS:
        sel = (f >= lo) & (f <= hi)
        b, af = np.nanmedian(B["lam"][sel, 0]), np.nanmedian(A["lam"][sel, 0])
        print("%-6s %9.1f %9.2f %9.0f %9d" % (name, b, af, b / af, sel.sum()))
    clean = np.ones(len(f), bool)
    for _, lo, hi in TARGETS:
        clean &= ~((f >= lo - 2) & (f <= hi + 2))
    for lo, hi in ((350, 359), (381, 470), (606, 614)):
        s = clean & (x >= lo) & (x <= hi)
        print(
            "clean %d-%d MHz: before %.2f, after %.2f"
            % (lo, hi, np.nanmedian(B["lam"][s, 0]), np.nanmedian(A["lam"][s, 0]))
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9.5), height_ratios=[1, 1.15], constrained_layout=True
    )
    for ax, (x0, x1) in ((ax1, (300, 1500)), (ax2, (340, 620))):
        for name, lo, hi in TARGETS:
            ax.axvspan(
                mhz(lo) - 0.1, mhz(hi) + 0.1, color=C_BAND, alpha=0.55, lw=0, zorder=0
            )
        ax.semilogy(
            B["mhz"], B["lam"][:, 0], color=C_BEFORE, lw=0.8, label=lb, zorder=2
        )
        ax.semilogy(x, A["lam"][:, 0], color=C_AFTER, lw=0.9, label=la, zorder=3)
        if ax is ax2:
            ax.semilogy(
                B["mhz"], B["lam"][:, 1], color=C_BEFORE, lw=0.6, ls=":", zorder=2
            )
            ax.semilogy(x, A["lam"][:, 1], color=C_AFTER, lw=0.7, ls=":", zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(1, 2e3)
        ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylabel("top eigenvalue, noise units")
    ax1.set_title(
        "recv1 N2, whole band: each channel's top mode (input-whitened), before and after the DTV / MUOS "
        "subtraction; shaded: the targets",
        loc="left",
        fontsize=11,
    )
    ax1.legend(loc="upper right", frameon=False, fontsize=10)
    ax2.set_title(
        "340-620 MHz: MUOS and the seven ATSC stations (dotted: second eigenvalue)",
        loc="left",
        fontsize=11,
    )
    for name, lo, hi in TARGETS:
        ax2.text(
            mhz((lo + hi) / 2),
            1.25e3,
            name if name == "MUOS" else "ATSC " + name,
            ha="center",
            fontsize=8,
            color="#374151",
        )
    ax2.set_xlabel("MHz")
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
