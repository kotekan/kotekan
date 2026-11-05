#!/usr/bin/env python3
"""Several nights of the DTV residual side by side, from vis_dtv_extract.py outputs (recv1 N2, 10 s bins,
whitened eigenvalues in noise units): per station, a waterfall of each channel's top eigenvalue against UTC, one
row per night, then the station medians against the sky (the clean channels' top eigenvalue) for every night.
Prints, per night and station, the median top eigenvalue, the excess over the sky, the share of bins in which
more than half the channels carry an extra mode (above the clean channels' 99th percentile), and the same around
the Tau A transit.

    vis_dtv_nights.py --night "11-02 stage off=night2.npz" --night "11-05 rolling g=night5.npz" \
        [--from 3.83 --to 12.25] [--out fig.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_dtv_plot import hour_angle_min  # noqa: E402

STATIONS = (
    ("ATSC 24", 2716, 2742, "Blues", "#2563eb"),
    ("ATSC 30", 2899, 2927, "Oranges", "#ea580c"),
)
C_SKY = "#6b7280"
STYLE = [
    dict(color=C_SKY, lw=1.0, ls="-", alpha=0.9),  # first night: the reference, neutral
    dict(lw=0.9, ls=":", alpha=0.55),
    dict(lw=1.0, ls="--", alpha=0.75),
    dict(lw=1.7, ls="-", alpha=1.0),
]


def run_med(x, n):
    """Running median over n bins (NaN-aware, edges shrink)."""
    out = np.full(len(x), np.nan)
    h = n // 2
    for i in range(len(x)):
        out[i] = np.nanmedian(x[max(0, i - h) : i + h + 1])
    return out


def load(path, t0, t1):
    d = np.load(path)
    t = np.asarray(d["t_inst_ns"], np.float64)
    o = np.argsort(t)
    t = t[o]
    th = t / 1e9 % 86400 / 3600
    sel = (th >= t0) & (th <= t1)
    lam = d["lam"][:, :, o][:, :, sel]
    fids = d["fids"]
    good = np.isfinite(lam).all(axis=1).all(axis=1)
    clean = np.isin(fids // 16 * 16, d["clean"]) & good
    ha = hour_angle_min(t[sel])
    day = np.datetime64(int(np.median(t)), "ns").astype("datetime64[D]")
    return dict(
        th=th[sel],
        ha=ha,
        lam=lam,
        fids=fids,
        good=good,
        clean=clean,
        day=str(day),
        sky=np.median(lam[clean, 0, :], axis=0),
        q99=np.percentile(lam[clean, 0, :], 99, axis=0),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--night",
        action="append",
        required=True,
        help="LABEL=NPZ, repeatable, in row order",
    )
    ap.add_argument("--from", dest="t0", type=float, default=3.83, help="UTC hours")
    ap.add_argument("--to", dest="t1", type=float, default=12.25)
    ap.add_argument(
        "--transit",
        type=float,
        default=20.0,
        help="|Tau A hour angle| (min) counted as the transit",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=18,
        help="running median (bins) for the line panels",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    nights = []
    for spec in a.night:
        label, path = spec.split("=", 1)
        n = load(path, a.t0, a.t1)
        n["label"] = label
        nights.append(n)

    print(
        "%-20s %-8s %7s %7s %7s %8s | %7s %7s %8s"
        % (
            "night",
            "station",
            "top",
            "sky",
            "excess",
            "extra %",
            "top@TA",
            "sky@TA",
            "extra %",
        )
    )
    for n in nights:
        tr = np.abs(n["ha"]) < a.transit
        for name, lo, hi, _, _ in STATIONS:
            sel = (n["fids"] >= lo) & (n["fids"] <= hi) & n["good"]
            L = n["lam"][sel]
            med = np.median(L[:, 0, :], axis=0)
            extra = ((L > n["q99"][None, None, :]).sum(axis=1) >= 1).mean(axis=0) > 0.5
            n[name] = dict(sel=sel, med=med, extra=extra)
            ot = ~tr
            print(
                "%-20s %-8s %7.2f %7.2f %7.2f %7.0f%% | %7.2f %7.2f %7.0f%%"
                % (
                    n["label"],
                    name,
                    np.median(med[ot]),
                    np.median(n["sky"][ot]),
                    np.median(med[ot] - n["sky"][ot]),
                    100 * extra[ot].mean(),
                    np.median(med[tr]) if tr.any() else np.nan,
                    np.median(n["sky"][tr]) if tr.any() else np.nan,
                    100 * extra[tr].mean() if tr.any() else np.nan,
                )
            )
    print(
        "(top = channel median of the top eigenvalue, noise units, outside |HA| < %.0f min; @TA = inside)"
        % a.transit
    )
    print("\nhourly channel-median top eigenvalue (UTC hour start):")
    hrs = np.arange(int(a.t0), int(np.ceil(a.t1)))
    print("%-20s %-8s " % ("", "") + " ".join("%5d" % h for h in hrs))
    for n in nights:
        for name, *_ in STATIONS:
            v = [
                np.median(n[name]["med"][(n["th"] >= h) & (n["th"] < h + 1)])
                if ((n["th"] >= h) & (n["th"] < h + 1)).any()
                else np.nan
                for h in hrs
            ]
            print("%-20s %-8s " % (n["label"], name) + " ".join("%5.2f" % x for x in v))
        v = [
            np.median(n["sky"][(n["th"] >= h) & (n["th"] < h + 1)])
            if ((n["th"] >= h) & (n["th"] < h + 1)).any()
            else np.nan
            for h in hrs
        ]
        print("%-20s %-8s " % (n["label"], "sky") + " ".join("%5.2f" % x for x in v))
    if not a.out:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nn = len(nights)
    fig = plt.figure(figsize=(15, 2.3 * nn + 3.6), constrained_layout=True)
    gs = fig.add_gridspec(nn + 1, 2, height_ratios=[1] * nn + [1.5])
    norm = matplotlib.colors.LogNorm(1.2, 40)
    nice = (1.5, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000)
    for col, (name, lo, hi, cmap, hue) in enumerate(STATIONS):
        shared, im_shared = [], None
        for row, n in enumerate(nights):
            ax = fig.add_subplot(gs[row, col])
            f = n["fids"][n[name]["sel"]]
            Z = n["lam"][n[name]["sel"], 0, :]
            own = (
                float(np.median(n[name]["med"])) > 40
            )  # off the shared scale: this row gets its own
            if own:
                v0, v1 = np.nanpercentile(Z, [1, 99.5])
                nrm = matplotlib.colors.LogNorm(v0, v1)
            im = ax.pcolormesh(
                n["th"],
                np.arange(len(f)),
                Z,
                shading="nearest",
                cmap=cmap,
                norm=nrm if own else norm,
                rasterized=True,
            )
            if own:
                for sp in ax.spines.values():
                    sp.set_linewidth(2.2)
                    sp.set_linestyle((0, (5, 2.5)))
                    sp.set_color("#111827")
                cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01)
                tk = [v for v in nice if v0 <= v <= v1]
                cb.set_ticks(tk)
                cb.set_ticklabels(["%g" % v for v in tk])
                cb.minorticks_off()
                cb.outline.set_linestyle((0, (5, 2.5)))
                cb.outline.set_linewidth(1.6)
                cb.set_label(
                    "THIS ROW ONLY\n%.0f-%.0f" % (v0, v1), fontsize=7, fontweight="bold"
                )
            else:
                shared.append(ax)
                im_shared = im
            i0 = np.argmin(np.abs(n["ha"]))
            ax.axvline(n["th"][i0], color="k", lw=0.7, ls=":")
            ax.set_xlim(a.t0, a.t1)
            ax.set_yticks(np.arange(len(f))[::6])
            ax.set_yticklabels(["%d" % x for x in f[::6]], fontsize=7)
            ax.tick_params(axis="x", labelsize=8)
            if row < nn - 1:
                ax.set_xticklabels([])
            if own:
                ax.set_title(
                    "%s  %s   OWN COLOUR SCALE: %.0f-%.0f (rows below: 1.2-40)"
                    % (name, n["label"], v0, v1),
                    fontsize=9,
                    loc="left",
                    fontweight="bold",
                )
            else:
                ax.set_title("%s  %s" % (name, n["label"]), fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel("freq_id", fontsize=8)
        if shared:
            cb = fig.colorbar(im_shared, ax=shared, shrink=0.95, pad=0.01)
            cb.set_ticks([1.5, 2, 3, 5, 10, 20, 40])
            cb.set_ticklabels(["1.5", "2", "3", "5", "10", "20", "40"])
            cb.minorticks_off()
            cb.set_label(
                "top eigenvalue (noise units)\nshared by these rows", fontsize=7
            )
        ax = fig.add_subplot(gs[nn, col])
        rest = (
            STYLE[1:][-(nn - 1) :] if nn > 1 else []
        )  # the last night gets the bold solid line
        for k, n in enumerate(nights):
            st = dict(STYLE[0] if k == 0 else rest[k - 1])
            st.setdefault("color", hue)
            ax.plot(
                n["th"],
                run_med(n[name]["med"], a.smooth),
                label="%s (%s)" % (n["label"], n["day"][5:]),
                **st
            )
        sk = nights[-1]
        ax.plot(
            sk["th"],
            run_med(sk["sky"], a.smooth),
            color="k",
            lw=0.8,
            ls="-.",
            alpha=0.6,
            label="sky: clean channels, %s" % sk["label"],
        )
        for n in nights:
            ax.axvline(
                n["th"][np.argmin(np.abs(n["ha"]))],
                color="k",
                lw=0.5,
                ls=":",
                alpha=0.6,
            )
        ax.set_yscale("log")
        top = max(
            60.0,
            1.5 * max(np.nanmax(run_med(n[name]["med"], a.smooth)) for n in nights),
        )
        ax.set_ylim(1.2, top)
        yt = [v for v in (1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500) if v < top]
        ax.set_yticks(yt)
        ax.set_yticklabels(["%g" % v for v in yt])
        ax.minorticks_off()
        ax.set_xlim(a.t0, a.t1)
        ax.grid(color="#e5e7eb", lw=0.6)
        ax.set_xlabel("UTC hour (dotted: Tau A transit, ~4 min earlier each night)")
        ax.set_ylabel(
            "top eigenvalue, channel median\n(%d-bin running median)" % a.smooth,
            fontsize=8,
        )
        ax.set_title(
            "%s: residual against the sky, all nights" % name, fontsize=9, loc="left"
        )
        ax.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(0.0, 0.6),
            frameon=False,
            ncol=2,
        )
    fig.suptitle(
        "DTV residual by night (recv1 N2, whitened top eigenvalue per channel, 10 s bins)",
        fontsize=11,
    )
    fig.savefig(a.out, dpi=100)
    print("saved", a.out)


if __name__ == "__main__":
    main()
