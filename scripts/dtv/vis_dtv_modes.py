#!/usr/bin/env python3
"""Residual modes the DTV subtraction leaves, over a night, from vis_dtv_extract.py output (recv1 N2, 10 s bins).

The sky reference in each bin is the clean channels' whitened eigenvalues (the same sky, Tau A included, and no
DTV). A cleaned channel's eigenvalue counts as an EXTRA mode when it exceeds the 99th percentile of the clean
channels' top eigenvalue in that bin. A station has an episode while more than half of its channels carry at
least one extra mode. Prints the episode list and the fraction of the night per station; with --out, draws the
waterfall of each cleaned channel's top residual eigenvalue and the station medians against the sky.

    vis_dtv_modes.py --night dtv_night_20261103.npz [--refits st_dir] [--out fig.png]
"""
import argparse
import glob
import os
import re

import numpy as np

STATIONS = (
    ("ATSC 24", 2716, 2742),
    ("ATSC 30", 2899, 2927),
)  # band edges carry little DTV
C_CLEAN, C_A24, C_A30 = "#6b7280", "#2563eb", "#ea580c"


def hours(t_ns):
    return (np.asarray(t_ns, np.float64) / 1e9 % 86400) / 3600


def episodes(on, th, min_bins=2):
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j - i + 1 >= min_bins:
                out.append((th[i], th[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def refit_hours(state_dir):
    """Source-0 and source-1 refit times (hours UTC) from one node's state file names."""
    r = {0: [], 1: []}
    for p in glob.glob(os.path.join(state_dir, "st_g0_*_r[01].bin")):
        m = re.search(r"_(\d\d)(\d\d)(\d\d)_r(\d)\.bin$", p)
        if m:
            r[int(m.group(4))].append(
                int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600
            )
    return {k: np.sort(v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--night", required=True)
    ap.add_argument("--refits", help="a node's state directory, for the refit times")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = np.load(a.night)
    th = hours(d["t_inst_ns"])
    o = np.argsort(th)
    th = th[o]
    lam = d["lam"][:, :, o]  # [C, 8, nb], noise units
    fids = d["fids"]
    clean = np.isin(fids // 16 * 16, d["clean"])
    good = np.isfinite(lam).all(axis=1).all(axis=1)
    clean &= good
    q99 = np.percentile(lam[clean, 0, :], 99, axis=0)  # [nb]
    sky = np.median(lam[clean, :4, :], axis=0)  # [4, nb]
    print(
        "%d bins %.2f-%.2fh UTC; clean channels %d; sky top eigenvalue median %.2f (p99 over clean channels %.2f)"
        % (len(th), th[0], th[-1], clean.sum(), np.median(sky[0]), np.median(q99))
    )
    res = {}
    for name, lo, hi in STATIONS:
        sel = (fids >= lo) & (fids <= hi) & good
        L = lam[sel]  # [c, 8, nb]
        n_extra = (L > q99[None, None, :]).sum(axis=1)  # [c, nb]
        frac_ch = (n_extra >= 1).mean(axis=0)
        on = frac_ch > 0.5
        eps = episodes(on, th)
        med = np.median(L[:, :4, :], axis=0)
        res[name] = dict(sel=sel, L=L, n_extra=n_extra, on=on, med=med)
        print(
            "\n%s: %d channels; residual top eigenvalue median %.2f (sky %.2f); median extra modes per channel %.1f"
            % (
                name,
                sel.sum(),
                np.median(med[0]),
                np.median(sky[0]),
                np.median(n_extra),
            )
        )
        print(
            "  bins with >half the channels carrying an extra mode: %.1f %% of the night; with two or more: %.1f %%"
            % (100 * on.mean(), 100 * (np.median(n_extra, axis=0) >= 2).mean())
        )
        print("  episodes (>= 2 bins): %d" % len(eps))
        for t0, t1, n in eps[:40]:
            k = (th >= t0) & (th <= t1)
            print(
                "    %02d:%02d-%02d:%02d  %3d bins  top residual eigenvalue (channel median) peak %.1f"
                % (
                    int(t0),
                    int(t0 % 1 * 60),
                    int(t1),
                    int(t1 % 1 * 60),
                    n,
                    med[0][k].max(),
                )
            )
        if len(eps) > 40:
            print("    ... %d more" % (len(eps) - 40))
        hrs = [
            100 * on[(th >= h) & (th < h + 1)].mean()
            if ((th >= h) & (th < h + 1)).any()
            else np.nan
            for h in range(2, 14)
        ]
        print(
            "  hourly %% with an extra mode, 02..13Z: "
            + " ".join("%.0f" % x for x in hrs)
        )
    if a.refits:
        rf = refit_hours(a.refits)
        for name, _, _ in STATIONS:
            r = res[name]
            for k in (0, 1):
                if len(rf[k]) < 2:
                    continue
                idx = np.searchsorted(rf[k], th) - 1
                ok = idx >= 0
                since = np.full(len(th), np.nan)
                since[ok] = (th[ok] - rf[k][idx[ok]]) * 60
                edges = [0, 1, 2, 3, 4, 5] if k == 0 else [0, 4, 8, 12, 16, 20]
                parts = []
                for e0, e1 in zip(edges[:-1], edges[1:]):
                    s = (since >= e0) & (since < e1)
                    if s.any():
                        parts.append(
                            "%d-%d min %.2f"
                            % (e0, e1, np.median(r["med"][0][s] - sky[0][s]))
                        )
                print(
                    "%s: top residual excess over the sky vs minutes since the last source-%d refit: %s"
                    % (name, k, ", ".join(parts))
                )
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 0.45])
        for col, (name, lo, hi) in enumerate(STATIONS):
            r = res[name]
            ax = fig.add_subplot(gs[0, col])
            f = fids[r["sel"]]
            cmap = "Blues" if col == 0 else "Oranges"
            im = ax.pcolormesh(
                th,
                np.arange(len(f)),
                r["L"][:, 0, :],
                shading="nearest",
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(1.5, 40),
                rasterized=True,
            )
            ax.set_yticks(np.arange(len(f))[::4])
            ax.set_yticklabels(["%d" % x for x in f[::4]], fontsize=7)
            ax.set_ylabel("freq_id")
            ax.set_title(
                "%s: top residual eigenvalue per channel (noise units)" % name,
                fontsize=10,
            )
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
            cb.set_ticks([2, 3, 5, 10, 20, 40])
            cb.set_ticklabels(["2", "3", "5", "10", "20", "40"])
            cb.minorticks_off()
            ax = fig.add_subplot(gs[1, col])
            c = C_A24 if col == 0 else C_A30
            ax.plot(
                th,
                sky[0],
                color=C_CLEAN,
                lw=1.2,
                label="sky: clean channels, top eigenvalue",
            )
            ax.plot(
                th,
                q99,
                color=C_CLEAN,
                lw=0.8,
                ls=":",
                label="clean channels' 99th percentile",
            )
            ax.plot(
                th,
                r["med"][0],
                color=c,
                lw=1.4,
                label="%s residual, 1st (channel median)" % name,
            )
            ax.plot(
                th,
                r["med"][1],
                color=c,
                lw=1.0,
                ls="--",
                label="%s residual, 2nd" % name,
            )
            ax.set_yscale("log")
            ax.set_ylabel("eigenvalue (noise units)")
            ax.legend(fontsize=7, loc="upper left", frameon=False)
            ax.grid(color="#e5e7eb", lw=0.6)
            ax = fig.add_subplot(gs[2, col])
            ax.fill_between(
                th,
                0,
                (r["n_extra"] >= 1).mean(axis=0),
                color=c,
                alpha=0.5,
                step="mid",
                lw=0,
            )
            ax.axhline(0.5, color="#111827", lw=0.6, ls=":")
            ax.set_ylim(0, 1)
            ax.set_ylabel("channels with\nan extra mode", fontsize=8)
            ax.set_xlabel(
                "UTC hour, %s"
                % np.datetime64(int(np.median(d["t_inst_ns"])), "ns").astype(
                    "datetime64[D]"
                )
            )
            if a.refits:
                for x in refit_hours(a.refits)[1]:
                    ax.axvline(x, color="#111827", lw=0.4, alpha=0.4)
            for axx in fig.axes[-3:]:
                axx.set_xlim(th[0], th[-1])
        fig.suptitle(
            "DTV subtraction overnight: residual modes against the sky (recv1 N2, 10 s bins; ticks in the "
            "bottom row are source-1 refits)",
            fontsize=11,
        )
        fig.savefig(a.out, dpi=110)
        print("saved", a.out)


if __name__ == "__main__":
    main()
