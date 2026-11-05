#!/usr/bin/env python3
"""Waterfalls and transit curves of the DTV subtraction around the Tau A transit, from vis_dtv_extract.py
outputs of two nights: one before the subtraction stage existed (uncleaned) and one with it running.

    vis_dtv_plot.py --before dtv_20261102.npz --after dtv_20261103.npz --out ~/gnss/bb1030/dtv_taua_waterfall_20261103.png
    vis_dtv_plot.py --night "11-02, stage off=dtv_20261102.npz" --night "11-03, rev1=dtv_20261103.npz" \
        --night "11-04, rev2=dtv_20261104.npz" --out ...   (any number of nights, rows in order)
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skyfield.api import load, wgs84  # noqa: E402

SITE = wgs84.latlon(49.3199845, -119.62262691, 528.4)
CS = 0.1953125

# categorical slots in fixed order (validated default palette: blue, orange, then the neutral reference)
C_CLEAN, C_A24, C_A30 = "#6b7280", "#2563eb", "#ea580c"


def hour_angle_min(t_ns):
    """Tau A's hour angle in minutes of time: the site's local sidereal time minus Tau A's right ascension
    precessed to 2026.84 (05:34:31.9 J2000 + 97 s; nutation and aberration are ~1 s, below a 10 s bin)."""
    ts = load.timescale()
    tt = ts.utc(1970, 1, 1, 0, 0, np.asarray(t_ns, np.float64) / 1e9)
    ra_date = 5 + 34 / 60 + (31.9 + 97.0) / 3600
    ha = (SITE.lst_hours_at(tt) - ra_date + 12.0) % 24.0 - 12.0
    return ha * 60.0


def groups(fids, clean, cleaned):
    c16 = fids // 16 * 16
    g = {
        "clean": np.isin(c16, clean),
        "ATSC 24 (2720-2735)": (c16 == 2720),
        "ATSC 30 (2898-2911)": (c16 == 2896) & (fids >= 2898),
    }
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument(
        "--night",
        action="append",
        default=[],
        help="LABEL=PATH, repeatable, in row order (replaces --before/--after)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--span", type=float, default=60.0, help="minutes either side of the transit"
    )
    a = ap.parse_args()
    spec = [tuple(s.split("=", 1)) for s in a.night] or [
        ("11-02, stage off", a.before),
        ("11-03, stage on", a.after),
    ]
    nights = []
    for label, path in spec:
        d = np.load(path)
        ha = hour_angle_min(d["t_inst_ns"])
        sel = np.abs(ha) <= a.span
        nights.append(
            dict(
                label=label,
                ha=ha[sel],
                lam=d["lam"][:, :, sel],
                ex=d["excess"][:, sel],
                fids=d["fids"],
                clean=d["clean"],
                cleaned=d["cleaned"],
                uncleaned=d["uncleaned"],
            )
        )
    # nights extracted with different chunk lists: keep the channels every night has
    common = nights[0]["fids"]
    for nt in nights[1:]:
        common = np.intersect1d(common, nt["fids"])
    for nt in nights:
        k = np.isin(nt["fids"], common)
        nt.update(fids=nt["fids"][k], lam=nt["lam"][k], ex=nt["ex"][k])
    fids = nights[0]["fids"]
    chunks = sorted(set(fids // 16 * 16))
    status = {
        c: (
            "cleaned"
            if c in nights[0]["cleaned"]
            else "uncleaned DTV"
            if c in nights[0]["uncleaned"]
            else "clean"
        )
        for c in chunks
    }

    nn = len(nights)
    fig = plt.figure(figsize=(15, 4.4 * nn + 4.2), constrained_layout=True)
    gs = fig.add_gridspec(nn + 1, 2, height_ratios=[1] * nn + [0.95])
    # x: chunks side by side with a gap column between them
    xs, xt, xl = [], [], []
    x = 0
    for c in chunks:
        xs.append(np.arange(16) + x)
        xt.append(x + 7.5)
        xl.append(
            "%.1f\n%s"
            % (
                (c + 8) * CS,
                {"cleaned": "CLEANED", "uncleaned DTV": "DTV", "clean": "clean"}[
                    status[c]
                ],
            )
        )
        x += 18
    for col, (key, title, vmin, vmax, log) in enumerate(
        (
            ("lam", "top eigenvalue of the whitened N2 (noise units)", 1.0, 60.0, True),
            (
                "ex",
                "Tau A strength: power in the Tau A template (noise units)",
                -1.0,
                6.0,
                False,
            ),
        )
    ):
        for row, nt in enumerate(nights):
            ax = fig.add_subplot(gs[row, col])
            for k, c in enumerate(chunks):
                s = (fids // 16 * 16) == c
                Z = nt["lam"][s, 0, :] if key == "lam" else nt["ex"][s, :]
                norm = (
                    matplotlib.colors.LogNorm(vmin, vmax)
                    if log
                    else matplotlib.colors.Normalize(vmin, vmax)
                )
                im = ax.pcolormesh(
                    np.append(xs[k], xs[k][-1] + 1) - 0.5,
                    np.append(nt["ha"], nt["ha"][-1] + 1 / 6),
                    Z.T,
                    cmap="Blues",
                    norm=norm,
                    shading="flat",
                    rasterized=True,
                )
            ax.set_xticks(xt)
            ax.set_xticklabels(xl, fontsize=7)
            ax.set_ylabel("Tau A hour angle (minutes)")
            ax.axhline(0, color="#111827", lw=0.6, ls=":")
            ax.set_title("%s: %s" % (nt["label"], title), fontsize=10)
            ax.set_ylim(-a.span, a.span)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    # transit curves
    ax = fig.add_subplot(gs[nn, :])
    styles, snames = (
        (((0, (1, 1.5)), "--", "-")[-nn:], ("dotted", "dashed", "solid")[-nn:])
        if nn <= 3
        else (["-"] * nn, ["solid"] * nn)
    )
    for nt, ls in zip(nights, styles):
        g = groups(nt["fids"], nt["clean"], nt["cleaned"])
        for name, color in (
            ("clean", C_CLEAN),
            ("ATSC 24 (2720-2735)", C_A24),
            ("ATSC 30 (2898-2911)", C_A30),
        ):
            y = np.nanmedian(nt["ex"][g[name]], axis=0)
            k = np.ones(6) / 6  # 1 min smoothing
            ys = np.convolve(np.nan_to_num(y), k, mode="same")
            ax.plot(
                nt["ha"],
                ys,
                ls=ls,
                color=color,
                lw=2,
                label="%s, %s"
                % (name if name != "clean" else "clean neighbours", nt["label"]),
            )
    ax.axhline(0, color="#111827", lw=0.6)
    ax.set_xlim(-a.span, a.span)
    ax.set_xlabel("Tau A hour angle (minutes)")
    ax.set_ylabel("Tau A strength (noise units)")
    ax.set_title(
        "Tau A through the beam: channel medians, 1-min smoothing (%s)"
        % "; ".join("%s: %s" % (sn, nt["label"]) for nt, sn in zip(nights, snames)),
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=nn, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6)
    fig.suptitle(
        "CHORD DTV subtraction at the Tau A transit: recv1 N2 subset, 128 inputs, 10 s bins",
        fontsize=12,
    )
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)
    # numbers for the report: peak Tau A strength (median over +-5 min of the peak) per group per night
    for nt in nights:
        g = groups(nt["fids"], nt["clean"], nt["cleaned"])
        ref = None
        for name in ("clean", "ATSC 24 (2720-2735)", "ATSC 30 (2898-2911)"):
            y = np.nanmedian(nt["ex"][g[name]], axis=0)
            near = np.abs(nt["ha"]) <= 5
            pk = float(np.nanmedian(y[near]))
            off = (
                float(np.nanmedian(y[np.abs(nt["ha"]) >= 40]))
                if (np.abs(nt["ha"]) >= 40).any()
                else float("nan")
            )
            lam1 = float(np.nanmedian(nt["lam"][g[name], 0][:, near]))
            lam3 = float(np.nanmedian(nt["lam"][g[name], 2][:, near]))
            far = np.abs(nt["ha"]) >= 40
            lam1_off = (
                float(np.nanmedian(nt["lam"][g[name], 0][:, far]))
                if far.any()
                else float("nan")
            )
            ref = pk if name == "clean" else ref
            print(
                "%-22s %-20s Tau A strength at transit %.2f (%.2f of clean), off transit %.2f; at transit top %.1f, "
                "3rd %.2f; off-transit top %.2f"
                % (
                    nt["label"],
                    name,
                    pk,
                    pk / ref if ref else float("nan"),
                    off,
                    lam1,
                    lam3,
                    lam1_off,
                )
            )


if __name__ == "__main__":
    main()
