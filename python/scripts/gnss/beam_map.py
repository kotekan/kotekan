#!/usr/bin/env python3
"""The receiver antenna's beam map from beam_log.py's samples.

Each sample's power is its per-record SNR, 10log10(p_pow / p_med - 1) dB, from present
satellites only. Satellites differ in transmit power, so each takes an offset: the fit is
SNR = G(cell) + b(satellite) on an azimuth-elevation grid, solved by alternating weighted means
over the (satellite, cell) averages, with G's mean above 75 deg set to 0 dB. A satellite's
transmit pattern and range that vary with elevation alone cannot be told from G; azimuth
structure can.

    beam_map.py beam.jsonl [...] [--const GW] [--az-step 15] [--el-step 10] [--el-min 5]
                [--png beam.png]
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def load(paths, consts, el_min):
    """The present samples at or above el_min, as (const, prn, az, el, snr_db)."""
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if consts and r["const"] not in consts:
                    continue
                if not r.get("present") or r["el"] < el_min or not r.get("p_med"):
                    continue
                x = r["p_pow"] / r["p_med"] - 1.0
                if x > 0:
                    rows.append(
                        (r["const"], r["prn"], r["az"], r["el"], 10 * np.log10(x))
                    )
    return rows


def fit(rows, az_step, el_step, n_iter=50):
    """G[i_el, i_az] (dB, NaN where unseen), n[i_el, i_az] samples and b {satellite: dB}."""
    n_az, n_el = int(round(360 / az_step)), int(np.ceil(90 / el_step))
    cell = defaultdict(list)
    for c, prn, az, el, y in rows:
        k = (int(az // az_step) % n_az, min(int(el // el_step), n_el - 1))
        cell[(c, prn), k].append(y)
    keys = list(cell)
    sat = [k[0] for k in keys]
    ia = np.array([k[1][0] for k in keys])
    ie = np.array([k[1][1] for k in keys])
    y = np.array([np.mean(cell[k]) for k in keys])
    w = np.array([len(cell[k]) for k in keys], dtype=float)
    sats = sorted(set(sat))
    js = np.array([sats.index(s) for s in sat])
    G = np.zeros((n_el, n_az))
    b = np.zeros(len(sats))
    for _ in range(n_iter):
        num = np.bincount(js, w * (y - G[ie, ia]), len(sats))
        b = num / np.bincount(js, w, len(sats))
        flat = ie * n_az + ia
        num = np.bincount(flat, w * (y - b[js]), n_el * n_az)
        den = np.bincount(flat, w, n_el * n_az)
        G = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).reshape(
            n_el, n_az
        )
    n = np.bincount(ie * n_az + ia, w, n_el * n_az).reshape(n_el, n_az)
    top = G[int(75 // el_step) :]
    ref = np.nanmean(top) if np.isfinite(top).any() else np.nanmax(G)
    return G - ref, n, dict(zip(sats, b + ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="beam_log.py output")
    ap.add_argument("--const", default="", help="letters to use, e.g. GW (default all)")
    ap.add_argument("--az-step", type=float, default=15.0)
    ap.add_argument("--el-step", type=float, default=10.0)
    ap.add_argument("--el-min", type=float, default=5.0)
    ap.add_argument("--png", help="write a polar plot here")
    a = ap.parse_args()
    rows = load(a.logs, set(a.const), a.el_min)
    if not rows:
        raise SystemExit("no present samples")
    G, n, b = fit(rows, a.az_step, a.el_step)
    n_el = G.shape[0]
    print(
        "%d samples, %d satellites; gain by elevation (dB, median over azimuth):"
        % (len(rows), len(b))
    )
    for i in range(n_el):
        g = G[i][np.isfinite(G[i])]
        if g.size:
            print(
                "  el %2d-%2d: %+5.1f  (%d of %d cells, %d samples)"
                % (
                    i * a.el_step,
                    (i + 1) * a.el_step,
                    np.median(g),
                    g.size,
                    G.shape[1],
                    n[i].sum(),
                )
            )
    if a.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        th = np.radians(np.arange(G.shape[1] + 1) * a.az_step)
        r = 90 - np.arange(n_el + 1) * a.el_step
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(False)
        m = ax.pcolormesh(
            th, r, np.ma.masked_invalid(G), cmap="viridis", shading="flat"
        )
        ax.grid(True)
        ax.set_rlim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90", "60", "30", "0"])
        fig.colorbar(m, ax=ax, shrink=0.7, label="relative gain (dB)")
        ax.set_title("%s: %d satellites" % (a.const or "all", len(b)))
        fig.savefig(a.png, dpi=120, bbox_inches="tight")
        print("wrote", a.png)


if __name__ == "__main__":
    main()
