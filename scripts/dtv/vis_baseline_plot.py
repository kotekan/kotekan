#!/usr/bin/env python3
"""Waterfalls of one whitened baseline (vis_baseline.py) around a source's transit, before and after: Re R_ab
against frequency and minutes from the transit (default Tau A), the whole band and 340-620 MHz, one colour scale
for every panel set by the source's fringes in the clean channels. The DTV subtraction's targets are marked.

    vis_baseline_plot.py --night "11-01 no subtraction=bl1.npz" --night "11-06 subtracted=bl6.npz" \
        [--pair 0] [--half 30] [--label "dish 0 - dish 11, 44 m E-W, pol 0"] --out f.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_dtv_plot import hour_angle_min  # noqa: E402
from vis_spectrum_plot import TARGETS, mhz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--night", action="append", required=True, help="LABEL=NPZ, in row order"
    )
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument(
        "--half", type=float, default=30.0, help="minutes either side of the transit"
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nights = []
    for spec in a.night:
        label, path = spec.split("=", 1)
        d = np.load(path)
        ha = hour_angle_min(d["t_inst_ns"])
        sel = np.abs(ha) <= a.half
        nights.append(
            dict(
                label=label,
                ha=ha[sel],
                re=np.real(d["R"][a.pair][sel]),
                x=d["mhz"],
                fid=d["fid"],
            )
        )
    # one scale: the source's fringes in channels away from every target, within 10 min of the transit
    fid = nights[-1]["fid"]
    away = np.ones(len(fid), bool)
    for _, lo, hi in TARGETS:
        away &= ~((fid >= lo - 4) & (fid <= hi + 4))
    n = nights[-1]
    # quiet channels: no steady correlation outside the transit (RFI elsewhere would set the scale)
    quiet = away & (np.nanmedian(np.abs(n["re"][np.abs(n["ha"]) > 20]), axis=0) < 0.03)
    vmax = float(np.nanpercentile(np.abs(n["re"][np.abs(n["ha"]) < 10][:, quiet]), 99))
    print(
        "colour scale +-%.3f (99th percentile of |Re R| in %d quiet channels away from the "
        "targets, |HA| < 10 min, last night)" % (vmax, quiet.sum())
    )
    for nt in nights:
        for name, lo, hi in TARGETS:
            s = (fid >= lo) & (fid <= hi)
            print(
                "%-30s %-5s median |R| %.3f (in transit %.3f)"
                % (
                    nt["label"],
                    name,
                    np.nanmedian(np.abs(nt["re"][:, s])),
                    np.nanmedian(np.abs(nt["re"][np.abs(nt["ha"]) < 10][:, s])),
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#d1d5db")
    nn = len(nights)
    fig, axs = plt.subplots(
        nn,
        2,
        figsize=(22, 4.2 * nn + 0.8),
        width_ratios=[2.2, 1],
        constrained_layout=True,
        squeeze=False,
    )
    for r, nt in enumerate(nights):
        for c, (x0, x1) in enumerate(((300, 1500), (340, 620))):
            ax = axs[r, c]
            s = (nt["x"] >= x0) & (nt["x"] <= x1)
            im = ax.pcolormesh(
                nt["x"][s],
                nt["ha"],
                nt["re"][:, s],
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                shading="nearest",
                rasterized=True,
            )
            for name, lo, hi in TARGETS:
                if mhz(hi) < x0 or mhz(lo) > x1:
                    continue
                ax.plot(
                    [mhz(lo), mhz(hi)],
                    [a.half * 1.02] * 2,
                    color="#b45309",
                    lw=3,
                    clip_on=False,
                    solid_capstyle="butt",
                )
                if c == 1:
                    ax.text(
                        mhz((lo + hi) / 2),
                        a.half * 1.07,
                        name if name == "MUOS" else "ATSC " + name,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#78350f",
                    )
            ax.axhline(0, color="#111827", lw=0.5, ls=":")
            ax.set_ylim(-a.half, a.half)
            ax.set_xlim(x0, x1)
            ax.set_ylabel("min from transit" if c == 0 else "")
            ax.set_title(
                "%s%s" % (nt["label"], "" if c == 0 else "  (340-620 MHz)"),
                loc="left",
                fontsize=10,
                pad=14 if c == 1 else 6,
            )
            if r == nn - 1:
                ax.set_xlabel("MHz")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("Re of the whitened correlation V_ab / sqrt(V_aa V_bb)")
    fig.suptitle(
        "Tau A transit on one baseline%s, whole band, before and after the DTV / MUOS subtraction (orange bars: the targets)"
        % (", " + a.label if a.label else ""),
        fontsize=12,
    )
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
