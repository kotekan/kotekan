#!/usr/bin/env python3
"""The receiver antenna's beam map from beam_log.py's samples, on a HEALPix grid of the local
sky (theta = 90 deg - elevation, phi = azimuth).

Each present sample's power is its per-record SNR, 10log10(p_pow / p_med - 1) dB. By default the
map is the relative gain: satellites differ in transmit power, so each takes an offset, and the
fit is SNR = G(pixel) + b(satellite), solved by alternating weighted means over the (satellite,
pixel) averages, with G's mean above 75 deg set to 0 dB. A gain that varies with elevation alone
cannot be told from the satellites' own patterns and range; azimuth structure can.

--cn0 maps the median C/N0 per pixel instead, with no offsets: the per-record SNR plus the median
of cn0_coh_db minus it over the L5, E5a and B2a samples (about 32 dB for 1 ms records).
GuoWang's own cn0_coh_db is not used: its coherent estimator normalizes each record, which the
half-zeroed frame makes noisy.

The plot is orthographic: zenith at the centre, north up, east to the right.

    beam_map.py beam.jsonl [...] [--const GW] [--nside 64] [--el-min 5] [--cn0] [--png beam.png]
"""

import argparse
import json
from collections import defaultdict

import healpy as hp
import numpy as np


def load(paths, el_min):
    """The present samples at or above el_min, as (satellite, az, el, snr_db, cn0_coh_db), the
    satellite being (const, name) where the log names it (GuoWang) and (const, prn) where not.
    """
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("present") or r["el"] < el_min or not r.get("p_med"):
                    continue
                x = r["p_pow"] / r["p_med"] - 1.0
                if x > 0:
                    rows.append(
                        (
                            (r["const"], r.get("name") or r["prn"]),
                            r["az"],
                            r["el"],
                            10 * np.log10(x),
                            r.get("cn0_coh_db"),
                        )
                    )
    return rows


def fit(sat, pix, y, npix, el_pix, n_iter=100):
    """G[pixel] (dB, NaN where unseen) and b {satellite: dB} for y = G[pix] + b[sat]."""
    cell = defaultdict(list)
    for s, p, v in zip(sat, pix, y):
        cell[s, p].append(v)
    keys = list(cell)
    sats = sorted({k[0] for k in keys})
    index = {s: j for j, s in enumerate(sats)}
    js = np.array([index[k[0]] for k in keys])
    ip = np.array([k[1] for k in keys])
    ym = np.array([np.mean(cell[k]) for k in keys])
    w = np.array([len(cell[k]) for k in keys], dtype=float)
    G = np.zeros(npix)
    for _ in range(n_iter):
        b = np.bincount(js, w * (ym - G[ip]), len(sats)) / np.bincount(js, w, len(sats))
        num = np.bincount(ip, w * (ym - b[js]), npix)
        den = np.bincount(ip, w, npix)
        G = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    top = G[el_pix >= 75]
    ref = np.nanmean(top) if np.isfinite(top).any() else np.nanmax(G)
    return G - ref, dict(zip(sats, b + ref))


def median_map(pix, y, npix):
    """The median of y per pixel, NaN where unseen."""
    cell = defaultdict(list)
    for p, v in zip(pix, y):
        cell[p].append(v)
    m = np.full(npix, np.nan)
    for p, v in cell.items():
        m[p] = np.median(v)
    return m


def plot(path, nside, m, label, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = np.meshgrid(np.linspace(-1, 1, 801), np.linspace(-1, 1, 801))
    r = np.hypot(x, y)
    el = np.degrees(np.arccos(np.clip(r, 0, 1)))
    az = np.degrees(np.arctan2(x, y)) % 360
    img = m[hp.ang2pix(nside, np.radians(90 - el), np.radians(az))]
    img[r > 1] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(
        np.ma.masked_invalid(img), origin="lower", extent=(-1, 1, -1, 1), cmap="viridis"
    )
    t = np.linspace(0, 2 * np.pi, 361)
    for e in (0, 30, 60):
        c = np.cos(np.radians(e))
        ax.plot(c * np.sin(t), c * np.cos(t), color="0.75", lw=0.6)
    ax.plot([0, 0], [-1, 1], color="0.75", lw=0.6)
    ax.plot([-1, 1], [0, 0], color="0.75", lw=0.6)
    for s, xy in {
        "N": (0, 1.07),
        "E": (1.07, 0),
        "S": (0, -1.07),
        "W": (-1.07, 0),
    }.items():
        ax.text(*xy, s, ha="center", va="center")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="beam_log.py output")
    ap.add_argument("--const", default="", help="letters to use, e.g. GW (default all)")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--el-min", type=float, default=5.0)
    ap.add_argument("--cn0", action="store_true", help="map C/N0, not the fitted gain")
    ap.add_argument("--png", help="write a plot here")
    a = ap.parse_args()
    rows = load(a.logs, a.el_min)
    cal = [r[4] - r[3] for r in rows if r[0][0] in "GEC" and r[4] is not None]
    if a.const:
        rows = [r for r in rows if r[0][0] in a.const]
    if not rows:
        raise SystemExit("no present samples")
    npix = hp.nside2npix(a.nside)
    th, ph = hp.pix2ang(a.nside, np.arange(npix))
    el_pix = 90 - np.degrees(th)
    pix = hp.ang2pix(
        a.nside, np.radians([90 - r[2] for r in rows]), np.radians([r[1] for r in rows])
    )
    y = np.array([r[3] for r in rows])
    n_sat = len({r[0] for r in rows})
    if a.cn0:
        if not cal:
            raise SystemExit("no L5, E5a or B2a cn0_coh_db to calibrate against")
        off = float(np.median(cal))
        m = median_map(pix, y + off, npix)
        what, label = "C/N0 (dB-Hz, SNR + %.1f dB)" % off, "C/N0 (dB-Hz)"
    else:
        m, _ = fit([r[0] for r in rows], pix, y, npix, el_pix)
        what, label = "relative gain (dB)", "relative gain (dB)"
    print(
        "%d samples, %d satellites, nside %d; %s by elevation, median over seen pixels:"
        % (len(rows), n_sat, a.nside, what)
    )
    for e0 in range(0, 90, 10):
        band = (el_pix >= e0) & (el_pix < e0 + 10)
        g = m[band & np.isfinite(m)]
        if g.size:
            print(
                "  el %2d-%2d: %+6.1f  (%d of %d pixels)"
                % (e0, e0 + 10, np.median(g), g.size, band.sum())
            )
    if a.png:
        title = "%s: %d satellites, nside %d" % (a.const or "all", n_sat, a.nside)
        plot(a.png, a.nside, m, label, title)


if __name__ == "__main__":
    main()
