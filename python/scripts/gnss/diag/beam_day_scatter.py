#!/usr/bin/env python3
"""Beam-map DAY-TO-DAY scatter: repeatability of the per-pixel mean C/N0 across days.

The render's own "per-pixel scatter" panel is the WITHIN-map spread -- every sample that
ever landed in a pixel, so it folds together thermal noise, multipath, per-satellite EIRP
offsets and geometry. This tool answers the orthogonal question: given each DAY's own map,
how much does a pixel's MEAN move from day to day? That isolates slow/systematic terms
(front-end gain drift, antenna, calibration) from the fast within-day scatter -- if a pixel's
day-to-day std is small but its within-day std is large, the spread is fast (noise/multipath);
if day-to-day is large, something is drifting between days.

Per day d and pixel p: mean_dp = s1/n (n > --min-n). Across the days a pixel appears in
(>= --min-days), report the unbiased std of mean_dp = the day-to-day scatter, plus the count
of days. Emits a 3-panel sky map (mean-of-day-means / day-to-day std / #days) and a text
summary; optionally dumps the per-pixel table.

    python3 beam_day_scatter.py --band L1 --tag G \
        --labels d2026-07-27,d2026-07-28,d2026-07-29,d2026-07-30,d2026-07-31
    # or let it auto-discover every d<date> label carrying this band/tag/quantity:
    python3 beam_day_scatter.py --band L1 --tag G --auto

Reads only the coadd accumulators under data/beam/maps/<label>/ -- no node contact.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gnss_beam_map as bm  # NSIDE + the exact sky projection the renders use


def _map_path(beamdir, label, band, tag, quant):
    return os.path.join(beamdir, "maps", label, f"{band}_{tag}_{quant}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="L1", help="L1 / L2C / L5")
    ap.add_argument("--tag", default="G", help="constellation tag G/E/C or ALL")
    ap.add_argument("--quantity", default="coh", choices=["coh", "inc"],
                    help="coh (default) or inc (BOC-biased); never mixed")
    ap.add_argument("--labels", default=None,
                    help="comma-separated day labels; omit with --auto")
    ap.add_argument("--auto", action="store_true",
                    help="discover every d<date> label that has this band/tag/quantity map")
    ap.add_argument("--beamdir", default="data/beam")
    ap.add_argument("--min-n", type=int, default=30,
                    help="min samples for a pixel to count THAT day (a thin day shouldn't "
                         "inject a noisy daily mean into the day-to-day spread)")
    ap.add_argument("--min-days", type=int, default=3,
                    help="min days a pixel must appear in to report a day-to-day std")
    ap.add_argument("--npx", type=int, default=480)
    ap.add_argument("--stdmax", type=float, default=4.0)
    ap.add_argument("--vmin", type=float, default=20.0)
    ap.add_argument("--vmax", type=float, default=52.0)
    ap.add_argument("--png", default=None)
    ap.add_argument("--dump", default=None, help="optional per-pixel npz")
    args = ap.parse_args()

    if args.auto:
        pat = _map_path(args.beamdir, "d*", args.band, args.tag, args.quantity)
        labels = sorted(os.path.basename(os.path.dirname(p))
                        for p in glob.glob(pat))
    else:
        if not args.labels:
            sys.exit("need --labels or --auto")
        labels = args.labels.split(",")

    day_means = []   # (label, per-pixel mean array with NaN where n<min_n)
    npix = None
    for lab in labels:
        p = _map_path(args.beamdir, lab, args.band, args.tag, args.quantity)
        if not os.path.exists(p):
            print(f"  skip {lab}: no {os.path.basename(p)}", file=sys.stderr)
            continue
        z = np.load(p)
        n, s1 = z["n"], z["s1"]
        if npix is None:
            npix = len(n)
        mean = np.full(len(n), np.nan)
        m = n >= args.min_n
        mean[m] = s1[m] / n[m]
        day_means.append((lab, mean, int(m.sum()), int(n.sum())))

    if len(day_means) < args.min_days:
        sys.exit(f"only {len(day_means)} usable day maps for {args.band}_{args.tag}_"
                 f"{args.quantity} (need >= --min-days {args.min_days})")

    print(f"=== day-to-day scatter: {args.band}_{args.tag}_{args.quantity} "
          f"({len(day_means)} days) ===")
    for lab, mean, npx_lab, nsamp in day_means:
        print(f"  {lab}: {npx_lab:>5d} pixels, {nsamp:>9d} samples, "
              f"map mean {np.nanmean(mean):.2f} dB-Hz")

    M = np.vstack([dm[1] for dm in day_means])          # (days, pixels)
    valid = np.isfinite(M)
    ndays = valid.sum(axis=0)
    keep = ndays >= args.min_days

    mean_of_means = np.full(npix, np.nan)
    d2d_std = np.full(npix, np.nan)
    with np.errstate(invalid="ignore"):
        mean_of_means[keep] = np.nanmean(M[:, keep], axis=0)
        # unbiased (ddof=1) std across the days each pixel actually has
        d2d_std[keep] = np.nanstd(M[:, keep], axis=0, ddof=1)
    ndays_f = np.where(keep, ndays, np.nan).astype(float)

    good = np.isfinite(d2d_std)
    print(f"\npixels with >= {args.min_days} days: {int(good.sum())}")
    if good.any():
        v = d2d_std[good]
        print(f"day-to-day per-pixel std (dB):  median {np.median(v):.2f}  "
              f"mean {v.mean():.2f}  p90 {np.percentile(v, 90):.2f}  max {v.max():.2f}")
        # split by elevation: zenith pixels should be the most repeatable
        th, ph = _pix_angles(npix)
        el_pix = 90.0 - np.degrees(th)
        for lo, hi in ((60, 90), (30, 60), (0, 30)):
            mm = good & (el_pix >= lo) & (el_pix < hi)
            if mm.any():
                print(f"   el {lo:>2d}-{hi:<2d} deg: median d2d std "
                      f"{np.median(d2d_std[mm]):.2f} dB  ({int(mm.sum())} px)")

    png = args.png or os.path.join(args.beamdir, "plots",
                                   f"day_scatter_{args.band}_{args.tag}_{args.quantity}.png")
    _render(mean_of_means, d2d_std, ndays_f, args, png, len(day_means))
    print(png)
    if args.dump:
        np.savez_compressed(args.dump, mean=mean_of_means, d2d_std=d2d_std,
                            ndays=ndays_f, labels=[dm[0] for dm in day_means])
        print(args.dump)


def _pix_angles(npix):
    import healpy as hp
    return hp.pix2ang(bm.NSIDE, np.arange(npix))


def _render(mean, std, ndays, args, png, nlab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    el, az, inside = bm._sky_grid(args.npx)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    panels = [(mean, "mean of daily means (dB-Hz)", "viridis", args.vmin, args.vmax),
              (std, f"DAY-TO-DAY std (dB), >={args.min_days}d", "magma", 0.0, args.stdmax),
              (ndays, "# days", "cividis", None, None)]
    for ax, (vals, title, cmap, lo, hi) in zip(axs, panels):
        img = bm._sky_image(vals, el, az, inside, args.npx)
        im = ax.imshow(img, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap,
                       vmin=lo, vmax=hi, interpolation="nearest")
        bm._sky_axes(ax, title)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"{args.band} {args.tag} ({args.quantity}) day-to-day repeatability "
                 f"over {nlab} days", fontsize=11)
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
