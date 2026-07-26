#!/usr/bin/env python3
"""Beam-map pipeline from GNSS obs logs (2026-07-22, Keith's multi-day beam buildup).

Design: per-satellite TRANSIT intermediates stay in raw (t, az, el, C/N0) form --
transits are 1-D tracks, a map pixelization would only blur them; the healpix layer
(~1 deg, nside 64) enters at the COADD, and coadds are stored as raw ACCUMULATORS
(hits, sum, sum-of-squares per pixel) so nights/constellations combine by addition
forever after -- the "build up resolution across days" workflow.

    transits  : obs.jsonl -> data/beam/transits/<night>/<tag>_G<prn>_t<k>.npz
    coadd     : transit npzs -> healpix accumulator npz (n, s1, s2 in dB-Hz)
    combine   : sum accumulator npzs (nights, constellations -- anything)
    render    : accumulator npz -> PNG (mean map + scatter + hits, zenith sky view)
    movie     : transit npzs -> mp4 of satellites tracking across the sky, leaving
                the cumulative beam map behind them

Quantity mapped: cn0_coh_dbhz by default; --quantity inc selects cn0_inc_dbhz. The
incoherent estimator is BOC-biased (E1C -7 dB, B1C -13 dB, see memory/docs) so the
two are NOT interchangeable and must never be coadded together -- keep them in
separate maps and compare. Transits carry BOTH, so the choice is made at coadd time.

Coadds average in dB: per-satellite EIRP offsets become additive and show up in the
per-pixel scatter (s2), which is the outlier/systematics handle. el <= 0 and
geometry-less rows are dropped (the obs logger loses az/el when its BRDC fetch fails).

Sky projection (render + movie): zenith-centred orthographic, r = cos(el), NORTH UP,
EAST RIGHT -- i.e. the sky as seen looking up, and cardinal directions are drawn on
the frame. (The pre-2026-07-24 renders used healpy's orthview, which produced the
same projection but with no orientation marks at all; west is where the house is,
so which side is which is not a detail.)

Examples:
    gnss_beam_map.py transits --obs obs_gps_l1.jsonl --tag G --night n2026-07-21 \
        --tmin 1784680000 --tmax 1784730000 --outdir data/beam
    gnss_beam_map.py coadd --transits 'data/beam/transits/n2026-07-21/G_*.npz' \
        --out data/beam/maps/n2026-07-21_G.npz --quantity coh
    gnss_beam_map.py combine --inputs a.npz b.npz --out both.npz
    gnss_beam_map.py render --map both.npz --png both.png --title 'G, both nights'
    gnss_beam_map.py movie --transits 'data/beam/transits/n2026-07-2*/G_*.npz' \
        --out g.mp4 --step 60 --title 'GPS L1 C/A'
"""
import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np

NSIDE = 64
MIN_TRANSIT_PTS = 50
GAP_SPLIT_S = 1800.0
QUANTITY = {"coh": "cn0", "inc": "cn0_inc"}


def cmd_transits(args):
    per = {}
    n_rows = n_geo = n_cn0 = 0
    for path in args.obs:
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("t")
            if t is None or not (args.tmin <= t <= args.tmax):
                continue
            n_rows += 1
            az, el = d.get("az"), d.get("el")
            if az is None or el is None or el <= 0.0:
                continue
            n_geo += 1
            cn0 = d.get("cn0_coh_dbhz")
            if cn0 is None:
                continue
            n_cn0 += 1
            # cn0_inc may be absent/NaN on a row that has cn0_coh; carry NaN so the
            # coherent map is never thinned by the incoherent estimator's gaps.
            inc = d.get("cn0_inc_dbhz")
            per.setdefault(int(d["prn"]), []).append(
                (t, float(az), float(el), float(cn0),
                 float(inc) if inc is not None else np.nan,
                 float(d.get("sig") or 0.0)))
    outdir = os.path.join(args.outdir, "transits", args.night)
    os.makedirs(outdir, exist_ok=True)
    for stale in glob.glob(os.path.join(outdir, f"{args.tag}_*.npz")):
        os.remove(stale)  # a re-run must not leave orphan segments from an older window
    n_tr = 0
    for prn, rows in sorted(per.items()):
        rows.sort()
        A = np.array(rows)
        brk = np.where(np.diff(A[:, 0]) > GAP_SPLIT_S)[0]
        for k, seg in enumerate(np.split(A, brk + 1)):
            if len(seg) < MIN_TRANSIT_PTS:
                continue
            path = os.path.join(outdir, f"{args.tag}_{prn:02d}_t{k}.npz")
            np.savez_compressed(path, t=seg[:, 0], az=seg[:, 1], el=seg[:, 2],
                                cn0=seg[:, 3], cn0_inc=seg[:, 4], sig=seg[:, 5],
                                tag=args.tag, prn=prn, night=args.night)
            n_tr += 1
    print(f"{args.night}/{args.tag}: {n_rows} rows in window, {n_geo} w/ geometry, "
          f"{n_cn0} w/ cn0 -> {n_tr} transits ({len(per)} sats)")


def _new_acc():
    npix = 12 * NSIDE * NSIDE
    return (np.zeros(npix, np.int64), np.zeros(npix), np.zeros(npix))


def _load_q(z, quantity, path=""):
    key = QUANTITY[quantity]
    if key not in z.files:
        raise SystemExit(f"{path}: no '{key}' -- transit predates the two-estimator "
                         f"format (2026-07-24); re-run `transits` for this night")
    return np.asarray(z[key], float)


def cmd_coadd(args):
    import healpy as hp
    n, s1, s2 = _new_acc()
    files = sorted(glob.glob(args.transits))
    n_nan = 0
    for f in files:
        z = np.load(f)
        q = _load_q(z, args.quantity, f)
        good = np.isfinite(q)
        n_nan += int((~good).sum())
        pix = hp.ang2pix(NSIDE, np.radians(90.0 - z["el"][good]),
                         np.radians(z["az"][good]))
        q = q[good]
        np.add.at(n, pix, 1)
        np.add.at(s1, pix, q)
        np.add.at(s2, pix, q ** 2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, n=n, s1=s1, s2=s2, nside=NSIDE,
                        n_transits=len(files), quantity=args.quantity)
    print(f"{args.out}: {len(files)} transits, {int(n.sum())} samples, "
          f"{int((n > 0).sum())} pixels hit"
          + (f", {n_nan} dropped (no {args.quantity})" if n_nan else ""))


def cmd_combine(args):
    n, s1, s2 = _new_acc()
    ntr = 0
    quant = None
    for f in args.inputs:
        z = np.load(f)
        assert int(z["nside"]) == NSIDE, f"{f}: nside mismatch"
        q = str(z["quantity"]) if "quantity" in z.files else "coh"
        if quant is None:
            quant = q
        # coh and inc differ by up to 13 dB on BOC signals -- summing them would be
        # a silent physical error, so refuse rather than warn.
        assert q == quant, f"{f}: quantity '{q}' != '{quant}' -- cannot combine"
        n += z["n"]; s1 += z["s1"]; s2 += z["s2"]
        ntr += int(z["n_transits"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, n=n, s1=s1, s2=s2, nside=NSIDE, n_transits=ntr,
                        quantity=quant)
    print(f"{args.out}: combined {len(args.inputs)} maps, {int(n.sum())} samples, "
          f"{int((n > 0).sum())} pixels hit")


def _sky_grid(npx):
    """Zenith orthographic grid: north up, east right, r = cos(el).

    Returns (el, az, inside) on an npx x npx grid spanning [-1, 1]^2, to be shown
    with imshow(origin='lower', extent=[-1, 1, -1, 1]).
    """
    u = np.linspace(-1.0, 1.0, npx)
    X, Y = np.meshgrid(u, u)          # +X east, +Y north
    R = np.hypot(X, Y)
    inside = R <= 1.0
    el = np.degrees(np.arccos(np.clip(R, 0.0, 1.0)))
    az = np.degrees(np.arctan2(X, Y)) % 360.0
    return el, az, inside


def _sky_xy(az, el):
    """Same projection as _sky_grid, for plotting satellite positions."""
    r = np.cos(np.radians(el))
    a = np.radians(az)
    return r * np.sin(a), r * np.cos(a)


def _sky_image(vals, el, az, inside, npx):
    import healpy as hp
    img = np.full((npx, npx), np.nan)
    pix = hp.ang2pix(NSIDE, np.radians(90.0 - el[inside]), np.radians(az[inside]))
    img[inside] = vals[pix]
    return img


def _sky_axes(ax, title):
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    th = np.linspace(0, 2 * np.pi, 361)
    for el_ring in (0, 30, 60):
        r = np.cos(np.radians(el_ring))
        ax.plot(r * np.sin(th), r * np.cos(th), color="0.5", lw=0.5, alpha=0.6)
    for a in range(0, 360, 45):
        r = np.array([0.0, 1.0])
        ax.plot(r * np.sin(np.radians(a)), r * np.cos(np.radians(a)),
                color="0.5", lw=0.4, alpha=0.4)
    for lab, (x, y) in (("N", (0, 1.045)), ("E", (1.045, 0)),
                        ("S", (0, -1.045)), ("W", (-1.045, 0))):
        ax.text(x, y, lab, ha="center", va="center", fontsize=9, color="0.25")


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(args.map)
    n, s1, s2 = z["n"], z["s1"], z["s2"]
    quant = str(z["quantity"]) if "quantity" in z.files else "coh"
    mean = np.full(len(n), np.nan)
    m = n > 0
    mean[m] = s1[m] / n[m]
    std = np.full(len(n), np.nan)
    m2 = n > 3
    std[m2] = np.sqrt(np.maximum(s2[m2] / n[m2] - (s1[m2] / n[m2]) ** 2, 0.0))
    hits = np.full(len(n), np.nan)
    hits[m] = n[m]
    npx = args.npx
    el, az, inside = _sky_grid(npx)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    est = "coherent" if quant == "coh" else "incoherent"
    panels = [(mean, f"mean C/N0 ({est}, dB-Hz)", "viridis", args.vmin, args.vmax),
              (std, "per-pixel scatter (dB)", "magma", 0.0, args.stdmax),
              (hits, "hits", "cividis", None, None)]
    for ax, (vals, title, cmap, lo, hi) in zip(axs, panels):
        img = _sky_image(vals, el, az, inside, npx)
        im = ax.imshow(img, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap,
                       vmin=lo, vmax=hi, interpolation="nearest")
        _sky_axes(ax, title)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"{args.title}   ({int(n.sum())} samples, "
                 f"{int(m.sum())} pixels, nside {NSIDE})", fontsize=11)
    os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
    fig.savefig(args.png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(args.png)


def _load_points(pattern, quantity):
    """All transit samples under a glob, time-sorted, with per-sample prn."""
    files = sorted(glob.glob(pattern)) if isinstance(pattern, str) else \
        sorted(sum([glob.glob(p) for p in pattern], []))
    if not files:
        raise SystemExit(f"no transits matched {pattern}")
    T, AZ, EL, Q, PRN, TAG = [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        q = _load_q(z, quantity, f)
        good = np.isfinite(q)
        T.append(z["t"][good]); AZ.append(z["az"][good]); EL.append(z["el"][good])
        Q.append(q[good])
        PRN.append(np.full(int(good.sum()), int(z["prn"])))
        TAG.append(np.full(int(good.sum()), str(z["tag"])))
    T = np.concatenate(T); o = np.argsort(T)
    return dict(t=T[o], az=np.concatenate(AZ)[o], el=np.concatenate(EL)[o],
                q=np.concatenate(Q)[o], prn=np.concatenate(PRN)[o],
                tag=np.concatenate(TAG)[o], n_files=len(files))


def cmd_movie(args):
    import healpy as hp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import matplotlib.patheffects as pe

    P = _load_points(args.transits, args.quantity)
    t, npx = P["t"], args.npx
    el_g, az_g, inside = _sky_grid(npx)
    pix_g = np.full((npx, npx), -1)
    pix_g[inside] = hp.ang2pix(NSIDE, np.radians(90.0 - el_g[inside]),
                               np.radians(az_g[inside]))
    pix_pt = hp.ang2pix(NSIDE, np.radians(90.0 - P["el"]), np.radians(P["az"]))
    sx, sy = _sky_xy(P["az"], P["el"])

    # One frame per --step of DATA time, but only for steps that actually contain
    # samples: the record has multi-hour gaps (restarts, BRDC lapses) and holding a
    # frozen frame through them would be most of the movie.
    #
    # SPEED RAMP (--ramp-s): a multi-day record at a fixed cadence is mostly a slow
    # crawl once the sky is full, so the step DOUBLES every --ramp-s seconds of VIDEO
    # (1x, 2x, 4x, ... up to --ramp-max). Tiers are counted in RENDERED frames, not
    # elapsed data time, so gap-skipping cannot smear the tier boundaries -- the
    # viewer really does get --ramp-s seconds at each speed.
    fps = float(args.fps)
    per_tier = max(1, int(round(args.ramp_s * fps))) if args.ramp_s > 0 else 0
    edge_t, spans = [float(t[0])], []
    cur, k = float(t[0]), 0
    tend = float(t[-1])
    while cur < tend:
        tier = min(k // per_tier, args.ramp_max) if per_tier else 0
        width = args.step * (2.0 ** tier)
        nxt = cur + width
        lo = int(np.searchsorted(t, cur, "left"))
        hi = int(np.searchsorted(t, nxt, "left"))
        if hi > lo:                       # frame has samples: spend one
            edge_t.append(nxt)
            spans.append(width)
            k += 1
            cur = nxt
        else:                             # empty: jump the gap without spending a frame
            if lo >= len(t):
                break
            cur = float(t[lo])
    edges = np.searchsorted(t, np.array(edge_t), "left")
    edges[-1] = len(t)
    n_frames = len(edges) - 1

    # Per-sample DWELL: a "sample" is one obs row, i.e. one combiner emit that the
    # logger kept, so the time it represents is the per-PRN cadence -- NOT the run
    # length and not the coherent integration. Median of the per-PRN inter-sample
    # gaps (robust to restarts, which inject huge single gaps), so the header can say
    # what the sample count is WORTH in integration time.
    dts = []
    for key in set(zip(P["tag"], P["prn"])):
        sel = (P["tag"] == key[0]) & (P["prn"] == key[1])
        tt = t[sel]
        if len(tt) > 8:
            d = np.diff(tt)
            d = d[(d > 0) & (d < 60.0)]   # drop restart gaps
            if len(d):
                dts.append(np.median(d))
    dwell = float(np.median(dts)) if dts else float("nan")

    n_acc, s1_acc = np.zeros(12 * NSIDE ** 2, np.int64), np.zeros(12 * NSIDE ** 2)
    # A per-frame autoscale would make the colour scale crawl as the map fills, so
    # fix it once from the whole record.
    vmin = args.vmin if args.vmin is not None else np.percentile(P["q"], 1.0)
    vmax = args.vmax if args.vmax is not None else np.percentile(P["q"], 99.0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    n_vis = len(np.unique(pix_g[inside]))
    tmp = tempfile.mkdtemp(prefix="beammov_")
    fig = plt.figure(figsize=(args.npx / 100.0 + 1.4, args.npx / 100.0 + 0.7), dpi=100)
    ax = fig.add_axes([0.01, 0.01, 0.80, 0.88])
    cax = fig.add_axes([0.855, 0.10, 0.025, 0.72])
    # ONCE, not per frame: a fresh colorbar each frame stacks transforms on cax and
    # blows the recursion limit a few hundred frames in. The norm is fixed anyway.
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label="mean C/N0 (dB-Hz)")
    try:
        for k in range(n_frames):
            lo, hi = edges[k], edges[k + 1]
            np.add.at(n_acc, pix_pt[lo:hi], 1)
            np.add.at(s1_acc, pix_pt[lo:hi], P["q"][lo:hi])
            mean = np.full(len(n_acc), np.nan)
            m = n_acc > 0
            mean[m] = s1_acc[m] / n_acc[m]
            img = np.full((npx, npx), np.nan)
            img[inside] = mean[pix_g[inside]]
            ax.clear()
            ax.imshow(img, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap,
                      norm=norm, interpolation="nearest")
            # Trail: everything within --tail of the frame's last sample, so a
            # satellite shows where it just came from; head marker + PRN label.
            t_now = t[hi - 1]
            tail = slice(np.searchsorted(t, t_now - args.tail), hi)
            # Two-tone: the map underneath spans dark blue to bright yellow, so a
            # single-colour overlay vanishes over half of it.
            ax.scatter(sx[tail], sy[tail], s=6.0, c="black", alpha=0.30, lw=0)
            ax.scatter(sx[tail], sy[tail], s=1.5, c="white", alpha=0.55, lw=0)
            # Head marker = each PRN's latest sample, provided it is recent enough to
            # still be "here" (per-PRN cadence is coarser than --step for weak sats).
            head = {}
            for i in range(tail.start, hi):
                head[(P["tag"][i], P["prn"][i])] = i
            fresh = max(3.0 * args.step, 120.0)
            for (tag, prn), i in head.items():
                if t_now - t[i] > fresh:
                    continue
                ax.plot(sx[i], sy[i], "o", ms=6, mfc="white", mec="black", mew=1.0)
                ax.text(sx[i] + 0.035, sy[i] + 0.025, f"{tag}{prn:02d}",
                        color="white", fontsize=7,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
            when = np.datetime64(int(t_now), "s")
            # Integration = samples x per-sample dwell (see `dwell` above), plus the
            # current playback speed so the ramp is legible rather than mysterious.
            n_s = int(n_acc.sum())
            integ = n_s * dwell / 3600.0
            speed = spans[k] / args.step if k < len(spans) else 1.0
            _sky_axes(ax, f"{args.title}\n{str(when).replace('T', ' ')} UTC   "
                          f"{n_s} samples x {dwell:.1f}s = {integ:.1f} h integrated   "
                          f"{100.0 * m.sum() / n_vis:.0f}% of sky   "
                          f"[{speed:.0f}x]")
            fig.savefig(os.path.join(tmp, f"f{k:05d}.png"))
            if k % 100 == 0:
                print(f"  frame {k}/{n_frames}", flush=True)
        plt.close(fig)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                        str(args.fps), "-i", os.path.join(tmp, "f%05d.png"),
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        args.out], check=True)
    finally:
        for f in glob.glob(os.path.join(tmp, "*.png")):
            os.remove(f)
        os.rmdir(tmp)
    print(f"{args.out}: {n_frames} frames from {P['n_files']} transits "
          f"({len(t)} samples, dwell {dwell:.1f}s -> {len(t) * dwell / 3600.0:.1f} h), "
          f"{n_frames / args.fps:.0f} s video, "
          f"data span {(t[-1] - t[0]) / 3600.0:.1f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("transits")
    p.add_argument("--obs", required=True, nargs="+",
                   help="one or more obs jsonl (concatenated; rows are re-sorted)")
    p.add_argument("--tag", required=True, help="constellation tag (G/E/C/L)")
    p.add_argument("--night", required=True)
    p.add_argument("--tmin", type=float, required=True)
    p.add_argument("--tmax", type=float, required=True)
    p.add_argument("--outdir", default="data/beam")
    p.set_defaults(fn=cmd_transits)
    p = sub.add_parser("coadd")
    p.add_argument("--transits", required=True, help="glob of transit npzs")
    p.add_argument("--out", required=True)
    p.add_argument("--quantity", choices=list(QUANTITY), default="coh")
    p.set_defaults(fn=cmd_coadd)
    p = sub.add_parser("combine")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_combine)
    p = sub.add_parser("render")
    p.add_argument("--map", required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--stdmax", type=float, default=6.0)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--npx", type=int, default=420)
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("movie")
    p.add_argument("--transits", required=True, nargs="+", help="glob(s) of transit npzs")
    p.add_argument("--out", required=True)
    p.add_argument("--quantity", choices=list(QUANTITY), default="coh")
    p.add_argument("--step", type=float, default=60.0, help="data seconds per frame")
    p.add_argument("--tail", type=float, default=1800.0, help="trail length, seconds")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ramp-s", type=float, default=15.0,
                   help="VIDEO seconds at each speed tier before the step doubles "
                        "(0 = constant cadence)")
    p.add_argument("--ramp-max", type=int, default=2,
                   help="max doublings (2 = 1x, 2x, 4x)")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--npx", type=int, default=700)
    p.add_argument("--title", default="")
    p.set_defaults(fn=cmd_movie)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
