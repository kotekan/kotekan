#!/usr/bin/env python3
"""TEC sky movie: differential slant TEC interpolated over the local sky dome, one frame
per time bin, satellites marked at their true (az, el), stitched to mp4.

    python3 gnss_tec.py --pair ... --dump-series --out /tmp/tec_run
    python3 gnss_tec_movie.py /tmp/tec_run_series.npz --outdir /tmp/tec_movie \
            [--frame-s 120] [--window-s 300] [--highpass-s 1200] [--video out.mp4]

LEVELING. Carrier arcs are relative (each re-zeros at its start), so the per-arc constant is
unknown until leveling. Two modes:
  * default (no --ionex): each arc is HIGH-PASSED (running mean over --highpass-s removed).
    The map then shows medium-scale structure -- wave activity / TIDs -- honestly, with zero
    absolute claim. This is the prototype mode.
  * --ionex FILE (hook, Monday): level each arc's mean onto the IONEX map value at its
    pierce points, giving absolute TEC. fetch: ftp://gssc.esa.int/gnss/products/ionex/
    (anonymous, same host as the DCBs). Not implemented yet -- the function raises with
    a pointer, so the plumbing is visible and testable today.

RENDERING. Sky dome projected az/el -> (x, y) = (sin az, cos az) * (90 - el)/90 (zenith at
center, north up). RBF (thin-plate) interpolation over the measurement points per frame,
masked where the nearest measurement is > --mask-deg away (no data invented over holes).
"""
import argparse
import os
import subprocess
import sys

import numpy as np


def load_series(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def highpass(d, tau):
    """Per (sys, prn, arc): subtract a running mean of width tau seconds."""
    out = np.array(d["tecu"], dtype=float)
    key = np.char.add(np.char.add(d["sys"].astype(str), d["prn"].astype(str)),
                      np.char.add("_", d["arc"].astype(str)))
    for k in np.unique(key):
        m = key == k
        t = d["t"][m]
        x = d["tecu"][m]
        sm = np.empty_like(x)
        for i, ti in enumerate(t):
            w = np.abs(t - ti) <= tau / 2
            sm[i] = x[w].mean()
        out[m] = x - sm
    return out


def sky_xy(az, el):
    r = (90.0 - el) / 90.0
    a = np.deg2rad(az)
    return np.sin(a) * r, np.cos(a) * r


def _in_hull(grid, pts):
    """Boolean: which grid points fall inside the convex hull of pts (for --voronoi masking).
    Delaunay.find_simplex >= 0 iff the point is inside the triangulated hull."""
    from scipy.spatial import Delaunay
    try:
        return Delaunay(pts).find_simplex(grid) >= 0
    except Exception:  # degenerate (collinear) point set
        return np.zeros(len(grid), dtype=bool)



# ---------------- IONEX (absolute leveling) ----------------

def fetch_ionex(day):
    """Fetch (with cache) one UTC day's GIM: rapid first, CODE 1-day predicted fallback."""
    import subprocess
    import urllib.request
    cache = "/tmp/ionex_cache"
    os.makedirs(cache, exist_ok=True)
    y, j = day.year, int(day.strftime("%j"))
    base = f"ftp://gssc.esa.int/gnss/products/ionex/{y}/{j:03d}"
    names = [f"COD0OPSRAP_{y}{j:03d}0000_01D_01H_GIM.INX.gz",
             f"ESA0OPSRAP_{y}{j:03d}0000_01D_02H_GIM.INX.gz",
             f"JPL0OPSRAP_{y}{j:03d}0000_01D_02H_GIM.INX.gz",
             f"c1pg{j:03d}0.{y%100:02d}i.Z",
             f"c2pg{j:03d}0.{y%100:02d}i.Z"]
    for n in names:
        raw = os.path.join(cache, n)
        txt = raw.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(txt):
            return txt, n
        try:
            urllib.request.urlretrieve(f"{base}/{n}", raw)
        except Exception:
            continue
        with open(txt, "wb") as f:
            subprocess.run(["gzip", "-dc", raw], stdout=f, check=True)
        print(f"IONEX: fetched {n}")
        return txt, n
    raise RuntimeError(f"no IONEX product for {day:%Y-%j} (tried rapid + predicted)")


def parse_ionex(path):
    """-> (epochs unix[], lats[], lons[], maps[n_ep, n_lat, n_lon] TECU, shell height km)."""
    from datetime import datetime, timezone
    eps, maps = [], []
    lats = lons = None
    hgt = 450.0
    exp = -1
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        L = lines[i]
        lab = L[60:].strip()
        if lab == "EXPONENT":
            exp = int(L.split()[0])
        elif lab == "HGT1 / HGT2 / DHGT":
            hgt = float(L.split()[0])
        elif lab == "START OF TEC MAP":
            i += 1
            ep = None
            grid_rows = {}
            while lines[i][60:].strip() != "END OF TEC MAP":
                lab2 = lines[i][60:].strip()
                if lab2 == "EPOCH OF CURRENT MAP":
                    v = [int(x) for x in lines[i].split()[:6]]
                    ep = datetime(*v, tzinfo=timezone.utc).timestamp()
                elif lab2 == "LAT/LON1/LON2/DLON/H":
                    lat = float(lines[i][2:8])
                    lo1 = float(lines[i][8:14])
                    lo2 = float(lines[i][14:20])
                    dlo = float(lines[i][20:26])
                    n_lon = int(round((lo2 - lo1) / dlo)) + 1
                    vals = []
                    while len(vals) < n_lon:
                        i += 1
                        vals += [int(lines[i][k:k + 5]) for k in range(0, len(lines[i]), 5)
                                 if lines[i][k:k + 5].strip()]
                    grid_rows[lat] = np.array(vals[:n_lon], float) * (10.0 ** exp)
                    if lons is None:
                        lons = lo1 + dlo * np.arange(n_lon)
                i += 1
            if grid_rows and ep is not None:
                if lats is None:
                    lats = np.array(sorted(grid_rows, reverse=True))
                eps.append(ep)
                maps.append(np.array([grid_rows[la] for la in lats]))
        i += 1
    return np.array(eps), lats, lons, np.array(maps), hgt


def load_ionex_days(spec, t):
    """spec = 'auto' (fetch every UTC day t covers) or a local file path."""
    from datetime import datetime, timezone, timedelta
    if spec != "auto":
        e, la, lo, mp, h = parse_ionex(spec)
        return e, la, lo, mp, h
    d0 = datetime.fromtimestamp(t.min(), tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(t.max(), tz=timezone.utc).date()
    eps, mps = [], []
    la = lo = None
    h = 450.0
    day = d0
    while day <= d1:
        try:
            path, name = fetch_ionex(datetime(day.year, day.month, day.day,
                                              tzinfo=timezone.utc))
            e, la, lo, mp, h = parse_ionex(path)
            eps.append(e)
            mps.append(mp)
        except Exception as ex:
            print(f"IONEX {day}: {ex}", file=sys.stderr)
        day += timedelta(days=1)
    if not eps:
        raise RuntimeError("no IONEX coverage for the series window")
    e = np.concatenate(eps)
    mp = np.concatenate(mps)
    o = np.argsort(e)
    # de-dup overlapping day-boundary epochs (keep first)
    e, idx = np.unique(e[o], return_index=True)
    return e, la, lo, mp[o][idx], h


def ipp(az, el, lat0, lon0, hgt_km):
    """Pierce point (deg) + slant mapping factor at shell height."""
    R = 6371.0
    azr, elr = np.deg2rad(az), np.deg2rad(el)
    psi = np.pi / 2 - elr - np.arcsin(R / (R + hgt_km) * np.cos(elr))  # earth angle
    la0, lo0 = np.deg2rad(lat0), np.deg2rad(lon0)
    la = np.arcsin(np.sin(la0) * np.cos(psi) + np.cos(la0) * np.sin(psi) * np.cos(azr))
    lo = lo0 + np.arcsin(np.sin(psi) * np.sin(azr) / np.cos(la))
    M = 1.0 / np.sqrt(1.0 - (R / (R + hgt_km) * np.cos(elr)) ** 2)
    return np.rad2deg(la), np.rad2deg(lo), M


def ionex_at(eps, lats, lons, maps, t, la, lo):
    """Bilinear in (lat, lon), linear in time. lats descending; lons ascending."""
    out = np.empty(len(t))
    ti = np.clip(np.searchsorted(eps, t) - 1, 0, len(eps) - 2)
    w_t = np.clip((t - eps[ti]) / (eps[ti + 1] - eps[ti]), 0, 1)
    li = np.clip(np.searchsorted(-lats, -la) - 1, 0, len(lats) - 2)
    w_la = np.clip((lats[li] - la) / (lats[li] - lats[li + 1]), 0, 1)
    lo = (lo - lons[0]) % 360.0 + lons[0]
    oi = np.clip(np.searchsorted(lons, lo) - 1, 0, len(lons) - 2)
    w_lo = np.clip((lo - lons[oi]) / (lons[oi + 1] - lons[oi]), 0, 1)
    for k in range(len(t)):
        v = 0.0
        for dt_, wt in ((0, 1 - w_t[k]), (1, w_t[k])):
            m = maps[ti[k] + dt_]
            v += wt * ((1 - w_la[k]) * ((1 - w_lo[k]) * m[li[k], oi[k]]
                                        + w_lo[k] * m[li[k], oi[k] + 1])
                       + w_la[k] * ((1 - w_lo[k]) * m[li[k] + 1, oi[k]]
                                    + w_lo[k] * m[li[k] + 1, oi[k] + 1]))
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--outdir", default="/tmp/tec_movie")
    ap.add_argument("--frame-s", type=float, default=120.0, help="one frame per this many s")
    ap.add_argument("--window-s", type=float, default=300.0,
                    help="frame uses measurements within +-window/2")
    ap.add_argument("--highpass-s", type=float, default=1200.0,
                    help="per-arc running-mean removal (0 = off; raw arc-relative levels)")
    ap.add_argument("--mask-deg", type=float, default=18.0,
                    help="blank map cells farther than this from any measurement")
    ap.add_argument("--el-mask", type=float, default=10.0)
    ap.add_argument("--vmax", type=float, default=0.0,
                    help="color scale (TECU); 0 = auto (95th pct of |values| AFTER gating)")
    ap.add_argument("--max-abs", type=float, default=30.0,
                    help="drop measurements with |dTEC| above this before interpolating "
                         "(instrument-junk gate: one wild arc otherwise fabricates a "
                         "sky-wide gradient; 0 = off)")
    ap.add_argument("--ionex", default=None,
                    help="'auto' (fetch rapid GIM, predicted fallback, per covered UTC day) "
                         "or a local IONEX file: level arcs to the GIM -> ABSOLUTE vertical-"
                         "equivalent TEC")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--video", default="tec_sky.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-hold-s", type=float, default=0.0,
                    help="bridge per-sat dropouts: if a sat has no measurement in the frame "
                         "window but its samples bracket the frame time by <= this gap, linearly "
                         "interpolate its position+TEC to the frame (0=off). Kills the flicker "
                         "from tracks briefly dropping/re-acquiring; only sensible in --ionex "
                         "absolute mode, where arcs share one GIM level so bridging is continuous. "
                         "NOTE the window already bridges gaps <= window_s, so use hold > window_s.")
    ap.add_argument("--smooth-frames", type=int, default=1,
                    help="temporal smoothing: display a trailing nanmean of the last N rendered "
                         "fields (1=off). Damps the frame-to-frame RBF jump when a sat "
                         "appears/disappears -- the dominant flicker once dropouts are bridged. "
                         "Safe for absolute VTEC (slowly varying); N=3 ~ 3 min at frame_s=60.")
    ap.add_argument("--voronoi", action="store_true",
                    help="render nearest-satellite (Voronoi) cells over the covered sky (convex "
                         "hull of the sat points) instead of RBF discs. Each pixel takes its "
                         "nearest sat's TEC -> the whole covered dome is tiled; with "
                         "--smooth-frames the moving cell boundaries blend into gradients.")
    args = ap.parse_args()

    d = load_series(args.series)
    ok = np.isfinite(d["az"]) & np.isfinite(d["el"]) & (d["el"] > args.el_mask)
    for k in d:
        d[k] = d[k][ok]
    if args.ionex:
        # ABSOLUTE MODE: level each arc's mean onto the IONEX GIM prediction at its pierce
        # points, then display VERTICAL-equivalent TEC (slant / mapping) -- the elevation
        # trend removed so the map shows ionosphere, not geometry.
        eps, lats, lons, maps, hgt = load_ionex_days(args.ionex, d["t"])
        ipla, iplo, M = ipp(d["az"], d["el"], args.lat, args.lon, hgt)
        vpred = ionex_at(eps, lats, lons, maps, d["t"], ipla, iplo)
        spred = vpred * M
        key = np.char.add(np.char.add(d["sys"].astype(str), d["prn"].astype(str)),
                          np.char.add("_", d["arc"].astype(str)))
        x = np.array(d["tecu"], dtype=float)
        for kk in np.unique(key):
            m = key == kk
            x[m] += np.median(spred[m] - x[m])   # per-arc carrier ambiguity -> IONEX level
        # junk gate on the residual-to-IONEX (same spirit as the relative gate)
        if args.max_abs > 0:
            g = np.abs(x - spred) <= args.max_abs
            print(f"IONEX residual gate: kept {g.sum()}/{len(x)}")
            for k in d:
                d[k] = d[k][g]
            x, M = x[g], M[g]
        x = x / M          # vertical-equivalent TECU
        absolute = True
    else:
        x = highpass(d, args.highpass_s) if args.highpass_s > 0 else np.array(d["tecu"])
        absolute = False
    if args.max_abs > 0 and not absolute:
        g = np.abs(x) <= args.max_abs
        print(f"junk gate |dTEC|<={args.max_abs}: kept {g.sum()}/{len(x)} epochs")
        for k in d:
            d[k] = d[k][g]
        x = x[g]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.interpolate import RBFInterpolator, NearestNDInterpolator

    os.makedirs(args.outdir, exist_ok=True)
    t0, t1 = d["t"].min(), d["t"].max()
    n_frames = int((t1 - t0) / args.frame_s)
    if absolute:
        # Anchor the scale at ZERO TECU: the colormap then reads as absolute fraction
        # (a wiggle of X TECU on a Y-TECU field is visibly X/Y of the ramp), instead of
        # a stretched slice that exaggerates structure.
        vhi = np.percentile(x, 98)
        vmin_, vmax = 0.0, vhi * 1.15
    else:
        vmax = args.vmax or max(np.percentile(np.abs(x), 95), 1e-3)
        vmin_ = -vmax
    # map grid (cartesian over the dome)
    g = np.linspace(-1, 1, 181)
    GX, GY = np.meshgrid(g, g)
    horizon = GX**2 + GY**2 <= 1.0
    from datetime import datetime, timezone
    # Per-sat time-sorted (t, value, sky-x, sky-y) for window-median + short-gap bridging. sky_xy
    # is precomputed per measurement so the hold interpolates CARTESIAN sky position (no azimuth
    # wraparound) and TEC together.
    _pxall, _pyall = sky_xy(d["az"], d["el"])
    _satid = np.char.add(d["sys"].astype(str), d["prn"].astype(str))
    sat_series = {}
    for sid in np.unique(_satid):
        mm = _satid == sid
        tt = d["t"][mm]
        o = np.argsort(tt)
        sat_series[sid] = (tt[o], x[mm][o], _pxall[mm][o], _pyall[mm][o],
                           str(d["sys"][mm][0]), int(d["prn"][mm][0]))
    print(f"{n_frames} frames, {len(x)} epochs, vmax {vmax:.2f} TECU"
          + (f", max-hold {args.max_hold_s:.0f}s" if args.max_hold_s > 0 else ""))
    written = 0
    import collections
    import warnings
    zbuf = collections.deque(maxlen=max(1, args.smooth_frames))  # trailing fields for --smooth-frames
    for fi in range(n_frames):
        tc = t0 + (fi + 0.5) * args.frame_s
        # one point per sat: median over the frame window, OR (if the sat dropped out but its
        # samples bracket tc within --max-hold-s) a linear bridge so the patch doesn't flicker.
        pts, vals, labs = [], [], []
        half = args.window_s / 2
        for sid, (tt, xx, pxs, pys, s_, p_) in sat_series.items():
            lo = np.searchsorted(tt, tc - half)
            hi = np.searchsorted(tt, tc + half)
            if hi > lo:  # in-window: median (kills per-epoch noise)
                pts.append((np.median(pxs[lo:hi]), np.median(pys[lo:hi])))
                vals.append(np.median(xx[lo:hi]))
                labs.append(f"{s_}{p_}")
            elif args.max_hold_s > 0:  # bridge a short dropout by interpolating to tc
                i = np.searchsorted(tt, tc)
                if 0 < i < len(tt) and (tt[i] - tt[i - 1]) <= args.max_hold_s:
                    f = (tc - tt[i - 1]) / (tt[i] - tt[i - 1])
                    pts.append((pxs[i - 1] + f * (pxs[i] - pxs[i - 1]),
                                pys[i - 1] + f * (pys[i] - pys[i - 1])))
                    vals.append(xx[i - 1] + f * (xx[i] - xx[i - 1]))
                    labs.append(f"{s_}{p_}")
        if len(pts) < 4:
            continue
        pts = np.array(pts)
        vals = np.array(vals)
        if not absolute:
            vals = vals - np.median(vals)   # arc-relative levels are arbitrary; the
                                            # frame's spatial median is the honest zero
        grid = np.column_stack([GX.ravel(), GY.ravel()])
        if args.voronoi:
            # Nearest-satellite (Voronoi) tiling, masked to the convex hull of the sat points so
            # we tile the COVERED sky, not empty dome. Temporal --smooth-frames then blends the
            # moving cell boundaries (a pixel that flips A->B between frames nanmeans to a gradient).
            try:
                Z = NearestNDInterpolator(pts, vals)(grid).reshape(GX.shape)
            except Exception:
                continue
            inside = _in_hull(grid, pts).reshape(GX.shape) if len(pts) >= 3 else np.ones_like(horizon)
            Z[~inside | ~horizon] = np.nan
        else:
            try:
                rbf = RBFInterpolator(pts, vals, kernel="linear", smoothing=1e-2)
                Z = rbf(grid).reshape(GX.shape)
            except Exception:
                continue
            # mask: no data invented far from measurements
            dist = np.min(np.sqrt((GX[..., None] - pts[:, 0])**2
                                  + (GY[..., None] - pts[:, 1])**2), axis=-1)
            Z[(dist > args.mask_deg / 90.0) | ~horizon] = np.nan
        # temporal smoothing: trailing nanmean of the last N fields -> the displayed field damps
        # the RBF jump when a sat enters/leaves. Per-pixel nanmean so a briefly-uncovered pixel
        # holds its recent value instead of blinking out.
        zbuf.append(Z)
        if len(zbuf) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels -> NaN
                Zdisp = np.nanmean(np.stack(zbuf), axis=0)
        else:
            Zdisp = Z

        fig, ax = plt.subplots(figsize=(7.2, 7.6))
        im = ax.pcolormesh(GX, GY, Zdisp, cmap="viridis" if absolute else "RdBu_r",
                           vmin=vmin_, vmax=vmax, shading="auto")
        th = np.linspace(0, 2 * np.pi, 361)
        for el_ring in (0, 30, 60):
            rr = (90 - el_ring) / 90.0
            ax.plot(np.sin(th) * rr, np.cos(th) * rr, color="0.6", lw=0.5)
        for lab_az in (0, 90, 180, 270):
            xx, yy = sky_xy(lab_az, 0)
            ax.annotate("NESW"[lab_az // 90], (xx * 1.06, yy * 1.06),
                        ha="center", va="center", color="0.4")
        colors = {"G": "tab:blue", "E": "tab:orange", "C": "tab:red"}
        for (px, py), v, lab in zip(pts, vals, labs):
            cmap_ = cm.viridis if absolute else cm.RdBu_r
            ax.plot(px, py, "o", ms=5,
                    mfc=cmap_(np.clip((v - vmin_) / (vmax - vmin_ + 1e-9), 0, 1)),
                    mec=colors.get(lab[0], "k"), mew=1.4)
            ax.annotate(lab, (px, py), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=colors.get(lab[0], "k"))
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        ts = datetime.fromtimestamp(tc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if absolute:
            mode = "IONEX-leveled, vertical-equivalent"
            ax.set_title(f"TEC -- {ts}\n({mode}; {len(pts)} sats)")
        else:
            mode = "arc-relative, high-passed %.0f min" % (args.highpass_s / 60) \
                if args.highpass_s > 0 else "arc-relative"
            ax.set_title(f"differential slant TEC -- {ts}\n({mode}; {len(pts)} sats)")
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label("vTEC (TECU)" if absolute else "dTEC (TECU)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"frame_{written:05d}.png"), dpi=100)
        plt.close(fig)
        written += 1
    print(f"wrote {written} frames -> {args.outdir}")
    if written > 3:
        out = os.path.join(args.outdir, args.video)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", os.path.join(args.outdir, "frame_%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
