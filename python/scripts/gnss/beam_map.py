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
                [--since 2026-11-22T18:00 --until 2026-11-23T06:00] [--dpi 120]

--since/--until take ISO times, local unless they carry an offset, or unix seconds. They may
repeat, pairing in order, to map several windows together (such as nights).
"""

import argparse
import datetime
import json
import re
from collections import defaultdict

import healpy as hp
import numpy as np


def when(s):
    """Unix seconds from unix seconds or an ISO time (local unless it carries an offset)."""
    try:
        return float(s)
    except ValueError:
        return datetime.datetime.fromisoformat(s).timestamp()


_T = re.compile(r'"t": ?([0-9.]+)')


def load(paths, el_min, windows=()):
    """The present samples at or above el_min, inside one of the (t0, t1) unix-time windows when
    given, as
    (satellite, az, el, snr_db, cn0_coh_db), the satellite being (const, name) where the log names
    it (GuoWang) and (const, prn) where not.
    """
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if windows:
                    # rows open with "t": a row outside the windows is never parsed
                    m = _T.search(line, 0, 32)
                    t = float(m.group(1)) if m else None
                    if t is None or not any(t0 <= t < t1 for t0, t1 in windows):
                        continue
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
    # str: a log from before GuoWang names were logged keys some by code (int), some by name
    sats = sorted({k[0] for k in keys}, key=str)
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


def smooth(m, fwhm_deg, min_frac=0.2):
    """m smoothed on the sphere by a Gaussian of fwhm_deg, weighting seen pixels only: the map
    times its coverage and the coverage are smoothed apart and divided, so empty pixels do not
    pull it toward zero. NaN where the smoothed coverage is under min_frac of a fully seen
    patch's, i.e. far from any data."""
    seen = np.isfinite(m).astype(float)
    fwhm = np.radians(fwhm_deg)
    num = hp.smoothing(np.where(seen > 0, m, 0.0), fwhm=fwhm)
    den = hp.smoothing(seen, fwhm=fwhm)
    return np.where(den > min_frac, num / np.where(den > min_frac, den, 1.0), np.nan)


def plot(path, nside, m, label, title, dpi=120, vrange=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the render grid resolves the pixels: ~6 image points per HEALPix pixel across the sky
    n = max(801, 12 * nside + 1)
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    r = np.hypot(x, y)
    el = np.degrees(np.arccos(np.clip(r, 0, 1)))
    az = np.degrees(np.arctan2(x, y)) % 360
    img = m[hp.ang2pix(nside, np.radians(90 - el), np.radians(az))]
    img[r > 1] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vmin, vmax = vrange if vrange else (None, None)
    im = ax.imshow(
        np.ma.masked_invalid(img),
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
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
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="beam_log.py output")
    ap.add_argument("--const", default="", help="letters to use, e.g. GW (default all)")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--el-min", type=float, default=5.0)
    ap.add_argument("--cn0", action="store_true", help="map C/N0, not the fitted gain")
    ap.add_argument("--png", help="write a plot here")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument(
        "--since", action="append", help="first sample time (ISO, local, or unix s)"
    )
    ap.add_argument("--until", action="append", help="end of the window (exclusive)")
    ap.add_argument(
        "--smooth-deg",
        type=float,
        help="smooth the plotted map (FWHM, deg), seen pixels only",
    )
    ap.add_argument("--vrange", nargs=2, type=float, help="colour scale limits (dB)")
    a = ap.parse_args()
    since, until = a.since or [None], a.until or [None]
    if len(since) != len(until):
        raise SystemExit(
            "--since and --until pair up: give them the same number of times"
        )
    windows = []
    if a.since or a.until:
        for s, u in zip(since, until):
            windows.append((when(s) if s else -np.inf, when(u) if u else np.inf))
    rows = load(a.logs, a.el_min, windows)
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
        if windows:
            title += "\n" + "; ".join(
                "%s to %s" % (s or "start", u or "end") for s, u in zip(since, until)
            )
        if a.smooth_deg:
            m = smooth(m, a.smooth_deg)
            title += ", smoothed %.1f deg" % a.smooth_deg
        plot(a.png, a.nside, m, label, title, a.dpi, a.vrange)


if __name__ == "__main__":
    main()
