#!/usr/bin/env python3
"""RADIAL PROFILES of a GNSS beam map: response vs angle from boresight.

A healpix coadd is the right thing to STORE (nights and constellations combine by addition
forever after) but the wrong thing to READ: a dish beam is circularly symmetric to first
order, so the informative cut is one dimensional. This turns coadd accumulators into
    * a radial profile -- median with a 16-84 percentile band, in ~log-spaced bins
    * the same split into azimuth sectors, which is where a mispointing or a squint shows
    * an optional AIRY OVERLAY for a circular aperture, so the measurement is read against a
      prediction rather than against itself

WHAT THE PREDICTION IS (CHORD pathfinder, from the pointing memo):
    boresight az 180.0, el 81.41  -- the dishes sit 8.59 deg SOUTH of zenith (dec +40.73).
    ⚠️ `telescope.dish_coelev_deg` in the generated configs reads -27.3 and does NOT give this.
    6 m dish at L5 (lambda 0.2548 m): FWHM 2.48 deg, first null 2.97 deg, first sidelobe
    3.97 deg at -17.6 dB, deep sidelobes ~-40 dB.

⚠️ THE CORE IS NOT MEASURABLE THE EASY WAY. A satellite within ~5 deg of boresight rails the
4+4b quantiser for EVERY chain at once, so the samples that define the main lobe are the
corrupted ones (survivors read +2-3 dB high and the tracked population drops 7 -> 4 per epoch;
see gnss_beam_veto.py). Pass BOTH the vetoed and unvetoed maps and the plot will show the two
profiles together -- their divergence inside ~8 deg IS the railing, measured.

⚠️ AND THE PIXELISATION IS A FLOOR ON THE WIDTH. nside 64 is ~0.92 deg/pixel against a 2.48
deg FWHM, so the main lobe is barely three pixels across: a measured width at or below ~1 deg
is the map's resolution talking, not the dish. Do not quote a fitted FWHM from this without
saying so.

Usage:
    gnss_beam_radial.py --map V1176.npz:vetoed P2_1176.npz:all-epochs \\
        --png radial_1176.png --freq 1176.45e6 --title "1176 MHz"
    gnss_beam_radial.py --map P2_1176.npz:1176 P2_1207.npz:1207 --png bands.png --sectors

@author Keith Vanderlinde
"""
import argparse
import sys

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_LIGHT = 299792458.0
EL0_DEF, AZ0_DEF = 81.41, 180.0     # the pointing memo, NOT dish_coelev_deg
DISH_M_DEF = 6.0


def uvec(el_deg, az_deg):
    e, a = np.radians(el_deg), np.radians(az_deg)
    return np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])


def load(path, min_hits):
    d = np.load(path)
    n = d["n"]
    ok = n >= min_hits
    m = np.full(n.size, np.nan)
    m[ok] = d["s1"][ok] / n[ok]
    return m, n, ok, int(d["nside"])


def airy_db(theta_deg, freq_hz, dish_m):
    """Circular-aperture power pattern, normalised to 0 dB on axis."""
    from scipy.special import j1
    lam = C_LIGHT / freq_hz
    x = np.pi * dish_m / lam * np.sin(np.radians(theta_deg))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
    return 20.0 * np.log10(np.maximum(np.abs(a), 1e-6))


def profile(ang, val, edges):
    """Median and 16/84 percentiles per bin, plus the count."""
    cen, med, lo, hi, cnt = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (ang >= a) & (ang < b) & np.isfinite(val)
        if s.sum() < 3:
            continue
        cen.append(np.sqrt(a * b) if a > 0 else b / 2.0)
        v = val[s]
        med.append(np.median(v))
        lo.append(np.percentile(v, 16))
        hi.append(np.percentile(v, 84))
        cnt.append(int(s.sum()))
    return map(np.array, (cen, med, lo, hi, cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="+", required=True, metavar="PATH[:LABEL]")
    ap.add_argument("--png", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--el0", type=float, default=EL0_DEF)
    ap.add_argument("--az0", type=float, default=AZ0_DEF)
    ap.add_argument("--freq", type=float, default=1176.45e6)
    ap.add_argument("--dish", type=float, default=DISH_M_DEF)
    ap.add_argument("--min-hits", type=int, default=3,
                    help="pixels with fewer samples are dropped: n=1 pixels are single "
                         "samples and dominate any max/percentile taken over them")
    ap.add_argument("--no-airy", action="store_true")
    ap.add_argument("--sectors", action="store_true",
                    help="add the azimuth-sector panel (mispointing / squint check)")
    ap.add_argument("--fit-centroid", action="store_true",
                    help="use the map's own bright centroid instead of --el0/--az0")
    args = ap.parse_args()

    specs = []
    for s in args.map:
        path, _, label = s.partition(":")
        specs.append((path, label or path.rsplit("/", 1)[-1]))

    b = uvec(args.el0, args.az0)
    ncol = 2 if args.sectors else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 5.6), squeeze=False)
    ax = axes[0]

    edges = np.concatenate([[0.0], np.geomspace(0.8, 90.0, 34)])
    peak_ref = None
    _ylo = []

    for i, (path, label) in enumerate(specs):
        m, n, ok, nside = load(path, args.min_hits)
        th, ph = hp.pix2ang(nside, np.arange(m.size))
        el, az = 90.0 - np.degrees(th), np.degrees(ph)
        bb = b
        if args.fit_centroid:
            hi_ = ok & (m >= np.nanpercentile(m[ok], 98))
            w = 10.0 ** (m[hi_] / 10.0)
            v = uvec(el[hi_], az[hi_]) * w
            c = v.sum(axis=1) / w.sum()
            bb = c / np.linalg.norm(c)
            print("%-14s fitted centroid: el %.2f az %.2f"
                  % (label, np.degrees(np.arcsin(bb[2])),
                     np.degrees(np.arctan2(bb[0], bb[1])) % 360))
        u = uvec(el, az)
        ang = np.degrees(np.arccos(np.clip(bb @ u, -1.0, 1.0)))

        val = np.where(ok, m, np.nan)
        cen, med, lo, hi, cnt = profile(ang, val, edges)
        if peak_ref is None:
            peak_ref = med[0]          # first map's innermost bin sets the 0 dB reference
        med, lo, hi = med - peak_ref, lo - peak_ref, hi - peak_ref
        col = "C%d" % i
        _ylo.append(lo)
        ax[0].fill_between(cen, lo, hi, color=col, alpha=0.18, lw=0)
        ax[0].plot(cen, med, "-o", color=col, ms=3.5, lw=1.6,
                   label="%s  (%d px)" % (label, int(ok.sum())))
        for a_, m_, c_ in zip(cen, med, cnt):
            if c_ < 10:
                ax[0].plot([a_], [m_], "x", color=col, ms=7, mew=1.6)

        if args.sectors and i == 0:
            for lab, (a0, a1) in (("N (315-45)", (315, 45)), ("E (45-135)", (45, 135)),
                                  ("S (135-225)", (135, 225)), ("W (225-315)", (225, 315))):
                sel = ((az >= a0) | (az < a1)) if a0 > a1 else ((az >= a0) & (az < a1))
                c2, m2, _, _, n2 = profile(ang[sel], val[sel], edges)
                ax[1].plot(c2, m2 - peak_ref, "-o", ms=3, lw=1.4, label=lab)

    if not args.no_airy:
        t = np.geomspace(0.05, 90.0, 2000)
        ax[0].plot(t, airy_db(t, args.freq, args.dish), "k--", lw=1.2, alpha=0.75,
                   label="Airy, %.1f m at %.0f MHz" % (args.dish, args.freq / 1e6))
        lam = C_LIGHT / args.freq
        fwhm = np.degrees(1.02 * lam / args.dish)
        null = np.degrees(1.22 * lam / args.dish)
        for x_, lab, c_ in ((fwhm / 2.0, "HWHM %.2f°" % (fwhm / 2), "0.45"),
                            (null, "first null %.2f°" % null, "0.6")):
            ax[0].axvline(x_, color=c_, ls=":", lw=1.1)
            ax[0].text(x_, 3, lab, rotation=90, fontsize=7, va="bottom", ha="right",
                       color="0.35")

    ax[0].set_xscale("log")
    # CLIP TO THE DATA. The Airy nulls run to -inf; letting them set the y-range compresses
    # every measured point into the top tenth of the axis, which is how a plot hides its own
    # subject. The overlay is a reference, not the content.
    ymin = min(np.nanmin(lo) for lo in _ylo) if _ylo else -40.0
    ax[0].set_ylim(ymin - 4.0, 5.0)
    ax[0].set_xlabel("angle from boresight (deg)   [az %.1f, el %.1f]" % (args.az0, args.el0))
    ax[0].set_ylabel("response (dB, innermost bin = 0)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("radial profile (median, 16-84%)")
    if args.sectors:
        ax[1].set_xscale("log")
        ax[1].set_xlabel("angle from boresight (deg)")
        ax[1].grid(alpha=0.3, which="both")
        ax[1].legend(fontsize=8)
        ax[1].set_title("by azimuth sector -- asymmetry / mispointing check")
    fig.suptitle(args.title or "GNSS beam radial profile")
    fig.tight_layout()
    fig.savefig(args.png, dpi=115)
    print("wrote %s" % args.png)

    # The numbers, so the plot never has to be read off by eye.
    m, n, ok, nside = load(specs[0][0], args.min_hits)
    th, ph = hp.pix2ang(nside, np.arange(m.size))
    ang = np.degrees(np.arccos(np.clip(
        b @ uvec(90.0 - np.degrees(th), np.degrees(ph)), -1.0, 1.0)))
    val = np.where(ok, m, np.nan)
    cen, med, lo, hi, cnt = profile(ang, val, edges)
    print("\n%-10s %10s %8s %8s %7s   (%s)" % ("deg", "median dB", "p16", "p84", "n px",
                                               specs[0][1]))
    for a_, m_, l_, h_, c_ in zip(cen, med, lo, hi, cnt):
        print("%-10.2f %10.1f %8.1f %8.1f %7d" % (a_, m_, l_, h_, c_))


if __name__ == "__main__":
    sys.exit(main())
