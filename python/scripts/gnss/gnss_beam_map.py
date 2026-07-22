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
    render    : accumulator npz -> PNG (mean map + hit map, zenith orthographic)

Quantity mapped: cn0_coh_dbhz (the coherent estimator -- the incoherent one is
biased for BOC signals, see memory/docs). Coadds average in dB: per-satellite EIRP
offsets become additive and show up in the per-pixel scatter (s2), which is the
outlier/systematics handle. el <= 0 and geometry-less rows are dropped (the obs
logger loses az/el when its BRDC fetch fails).

Examples:
    gnss_beam_map.py transits --obs obs_gps_l1.jsonl --tag G --night n2026-07-21 \
        --tmin 1784680000 --tmax 1784730000 --outdir data/beam
    gnss_beam_map.py coadd --transits 'data/beam/transits/n2026-07-21/G_*.npz' \
        --out data/beam/maps/n2026-07-21_G.npz
    gnss_beam_map.py combine --inputs a.npz b.npz --out both.npz
    gnss_beam_map.py render --map both.npz --png both.png --title 'G, both nights'
"""
import argparse
import glob
import json
import os

import numpy as np

NSIDE = 64
MIN_TRANSIT_PTS = 50
GAP_SPLIT_S = 1800.0


def cmd_transits(args):
    per = {}
    n_rows = n_geo = n_cn0 = 0
    for line in open(args.obs):
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
        per.setdefault(int(d["prn"]), []).append(
            (t, float(az), float(el), float(cn0), float(d.get("sig") or 0.0)))
    outdir = os.path.join(args.outdir, "transits", args.night)
    os.makedirs(outdir, exist_ok=True)
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
                                cn0=seg[:, 3], sig=seg[:, 4],
                                tag=args.tag, prn=prn, night=args.night)
            n_tr += 1
    print(f"{args.night}/{args.tag}: {n_rows} rows in window, {n_geo} w/ geometry, "
          f"{n_cn0} w/ cn0 -> {n_tr} transits ({len(per)} sats)")


def _new_acc():
    npix = 12 * NSIDE * NSIDE
    return (np.zeros(npix, np.int64), np.zeros(npix), np.zeros(npix))


def cmd_coadd(args):
    import healpy as hp
    n, s1, s2 = _new_acc()
    files = sorted(glob.glob(args.transits))
    for f in files:
        z = np.load(f)
        pix = hp.ang2pix(NSIDE, np.radians(90.0 - z["el"]), np.radians(z["az"]))
        np.add.at(n, pix, 1)
        np.add.at(s1, pix, z["cn0"])
        np.add.at(s2, pix, z["cn0"] ** 2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, n=n, s1=s1, s2=s2, nside=NSIDE,
                        n_transits=len(files))
    print(f"{args.out}: {len(files)} transits, {int(n.sum())} samples, "
          f"{int((n > 0).sum())} pixels hit")


def cmd_combine(args):
    n, s1, s2 = _new_acc()
    ntr = 0
    for f in args.inputs:
        z = np.load(f)
        assert int(z["nside"]) == NSIDE, f"{f}: nside mismatch"
        n += z["n"]; s1 += z["s1"]; s2 += z["s2"]
        ntr += int(z["n_transits"])
    np.savez_compressed(args.out, n=n, s1=s1, s2=s2, nside=NSIDE, n_transits=ntr)
    print(f"{args.out}: combined {len(args.inputs)} maps, {int(n.sum())} samples, "
          f"{int((n > 0).sum())} pixels hit")


def cmd_render(args):
    import healpy as hp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(args.map)
    n, s1, s2 = z["n"], z["s1"], z["s2"]
    mean = np.full(len(n), hp.UNSEEN)
    m = n > 0
    mean[m] = s1[m] / n[m]
    std = np.full(len(n), hp.UNSEEN)
    m2 = n > 3
    std[m2] = np.sqrt(np.maximum(s2[m2] / n[m2] - (s1[m2] / n[m2]) ** 2, 0.0))
    hits = np.full(len(n), hp.UNSEEN)
    hits[m] = n[m]
    fig = plt.figure(figsize=(15, 5))
    # Zenith-centered orthographic: rot to the north pole (el=90 -> theta=0).
    hp.orthview(mean, rot=(0, 90, 0), half_sky=True, title=f"{args.title}: mean C/N0 (dB-Hz)",
                sub=(1, 3, 1), fig=fig.number, cmap="viridis")
    hp.orthview(std, rot=(0, 90, 0), half_sky=True, title="per-pixel scatter (dB)",
                sub=(1, 3, 2), fig=fig.number, cmap="magma", max=args.stdmax)
    hp.orthview(hits, rot=(0, 90, 0), half_sky=True, title="hits",
                sub=(1, 3, 3), fig=fig.number, cmap="cividis")
    for ax in fig.axes:
        pass
    hp.graticule(dpar=30, dmer=45, alpha=0.3)
    os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
    plt.savefig(args.png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(args.png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("transits")
    p.add_argument("--obs", required=True)
    p.add_argument("--tag", required=True, help="constellation tag (G/E/C/L)")
    p.add_argument("--night", required=True)
    p.add_argument("--tmin", type=float, required=True)
    p.add_argument("--tmax", type=float, required=True)
    p.add_argument("--outdir", default="data/beam")
    p.set_defaults(fn=cmd_transits)
    p = sub.add_parser("coadd")
    p.add_argument("--transits", required=True, help="glob of transit npzs")
    p.add_argument("--out", required=True)
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
    p.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
