#!/usr/bin/env python3
"""The DTV residual between refits: the recv1 N2 top eigenvalue (noise units) of the cleaned channels against
time, with the refit-loop rounds marked, and against the time since the last refit, for several nights.
Inputs: vis_dtv_extract.py outputs and per-night refit times (the "before the refit" HH:MM:SSZ stamps of one
node's dtv_live.py log, one per line; the load follows the stamp by ~LOAD_DELAY s).

    vis_dtv_sawtooth.py --night "11-03, rev1=dtv_20261103.npz=refits_20261103_cx51.txt" \
        --night "11-04, rev2=dtv_20261104.npz=refits_20261104_cx51.txt" --out dtv_sawtooth_20261104.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_dtv_plot import hour_angle_min, C_CLEAN, C_A24, C_A30  # noqa: E402

LOAD_DELAY = 15.0
GROUPS = (
    ("ATSC 24 (2720-2735)", C_A24, lambda f: (f >= 2720) & (f <= 2735)),
    ("ATSC 30 (2898-2911)", C_A30, lambda f: (f >= 2898) & (f <= 2911)),
)


def refit_times(path, t_ref):
    """HH:MM:SSZ stamps -> UNIX s on the UTC day of t_ref (s), + the load delay."""
    day = np.floor(t_ref / 86400.0) * 86400.0
    out = []
    for line in open(path):
        s = line.strip().rstrip("Z")
        if s:
            h, m, x = (int(v) for v in s.split(":"))
            out.append(day + h * 3600 + m * 60 + x + LOAD_DELAY)
    return np.array(sorted(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--night",
        action="append",
        required=True,
        help="LABEL=NPZ=REFITS, repeatable, in order",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--span", type=float, default=60.0)
    ap.add_argument(
        "--transit",
        type=float,
        default=20.0,
        help="|hour angle| (min) where Tau A dominates the top mode",
    )
    a = ap.parse_args()
    nights = []
    for spec in a.night:
        label, npz, rf = spec.split("=")
        d = np.load(npz)
        t = d["t_inst_ns"] / 1e9
        ha = hour_angle_min(d["t_inst_ns"])
        sel = np.abs(ha) <= a.span
        r = refit_times(rf, np.median(t))
        idx = np.searchsorted(r, t, side="right") - 1
        since = np.where(idx >= 0, t - r[np.maximum(idx, 0)], np.nan)
        f = d["fids"]
        cl = np.isin(f // 16 * 16, d["clean"])
        nights.append(
            dict(
                label=label,
                t=t[sel],
                ha=ha[sel],
                since=since[sel],
                lam=d["lam"][:, 0, sel],
                fids=f,
                clean=cl,
                refit_ha=hour_angle_min(r * 1e9),
            )
        )
    styles = ("--", "-", ":")
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
    for row, (name, color, sel_f) in enumerate(GROUPS):
        ax = fig.add_subplot(gs[row, :])
        for nt, ls in zip(nights, styles):
            y = np.median(nt["lam"][sel_f(nt["fids"])], axis=0)
            yc = np.median(nt["lam"][nt["clean"]], axis=0)
            ax.plot(
                nt["ha"],
                y,
                ls=ls,
                color=color,
                lw=1.6,
                label="%s, %s" % (name, nt["label"]),
            )
            if nt is nights[-1]:
                ax.plot(
                    nt["ha"],
                    yc,
                    ls="-",
                    color=C_CLEAN,
                    lw=1.2,
                    label="clean neighbours, %s" % nt["label"],
                )
                for x in nt["refit_ha"][np.abs(nt["refit_ha"]) <= a.span]:
                    ax.axvline(x, color=C_CLEAN, lw=0.5, alpha=0.5)
        ax.axvspan(-a.transit, a.transit, color="#e5e7eb", alpha=0.6, lw=0)
        ax.text(
            0,
            0.97,
            "Tau A in the beam: its two modes are the top eigenvalues here",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
        )
        ax.set_xlim(-a.span, a.span)
        ax.set_ylim(1.0, 9.0)
        ax.set_ylabel("top eigenvalue (noise units)")
        ax.set_title(
            "%s: channel median, 10 s bins; thin grey lines = %s refit loads"
            % (name, nights[-1]["label"]),
            fontsize=10,
        )
        ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)
        ax.grid(color="#e5e7eb", lw=0.6)
    ax.set_xlabel("Tau A hour angle (minutes)")
    # against the time since the last refit, Tau A out of the beam
    edges = np.arange(0, 451, 30)
    mid = 0.5 * (edges[1:] + edges[:-1]) / 60.0
    for col, (name, color, sel_f) in enumerate(GROUPS):
        ax = fig.add_subplot(gs[2, col])
        for nt, ls in zip(nights, styles):
            y = np.median(nt["lam"][sel_f(nt["fids"])], axis=0)
            off = (np.abs(nt["ha"]) > a.transit) & np.isfinite(nt["since"])
            med = [
                np.median(y[off & (nt["since"] >= e0) & (nt["since"] < e1)])
                if (off & (nt["since"] >= e0) & (nt["since"] < e1)).sum() >= 5
                else np.nan
                for e0, e1 in zip(edges[:-1], edges[1:])
            ]
            ax.plot(
                mid, med, ls=ls, color=color, lw=2, marker="o", ms=4, label=nt["label"]
            )
            yc = np.median(nt["lam"][nt["clean"]], axis=0)
            print(
                "%-34s %-20s top eigenvalue, Tau A out of the beam: %s; clean %.2f"
                % (
                    nt["label"],
                    name,
                    " ".join(
                        "%.0f-%.0fs %.2f" % (e0, e1, m)
                        for e0, e1, m in zip(edges[:-1:2], edges[1::2], med[::2])
                    ),
                    np.median(yc[off]),
                )
            )
        ax.axhline(
            np.median(np.median(nights[-1]["lam"][nights[-1]["clean"]], axis=0)),
            color=C_CLEAN,
            lw=1.2,
            label="clean neighbours",
        )
        ax.set_xlabel("minutes since the last refit load")
        ax.set_ylabel("top eigenvalue (noise units)")
        ax.set_ylim(1.0, 7.0)
        ax.set_title(
            "%s, Tau A out of the beam (|HA| > %.0f min): median per 30 s"
            % (name, a.transit),
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.grid(color="#e5e7eb", lw=0.6)
    fig.suptitle(
        "DTV residual between refits: recv1 N2 subset, cleaned channels", fontsize=12
    )
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
