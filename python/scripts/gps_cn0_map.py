#!/usr/bin/env python3
"""Incoherent-C/N0 beam map from the gps_status_logger JSONL -- the commercial-receiver-style
observable (per-record power ratio: needs only the 1 ms despread, NO carrier coherence), averaged
per sky pixel so the estimator floor integrates down as the dwell accumulates.

Per emit (one ~1 s combiner window, K=1000 records):
    x = s^2/N   (per-1-ms-record signal-to-noise power ratio)
      = (unbiased_amplitude^2) / (amplitude^2 - unbiased_amplitude^2)   [exact, new logs]
      = amp_snr / K^(1/4)                                               [fallback, older logs]
NO per-emit thresholding (thresholding pins medians at the single-emit floor); x is averaged per
pixel/bin first, THEN converted: C/N0 = 10*log10((<x> - pedestal)/T_rec), T_rec = 1 ms. The
PEDESTAL is the estimator's positive bias on pure noise (the moment debias clips at zero),
calibrated from BELOW-HORIZON samples (elevation < mask via skyfield) -- the noise-floor
calibration. Zero-point convention matches the rest of the pipeline (~ +3 dB of true C/N0).

Usage:
  python3 gps_cn0_map.py /tmp/gpswipe/status_log.jsonl --lat 43.968697 --lon -79.252106 --alt 260
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gps_beamtrack sibling

# Per-constellation signal geometry (must match the live configs): record period T_rec and
# combiner integration_length K. x = s2/N is a PER-RECORD power ratio, so cross-constellation
# merging happens in C/N0-density units (x - pedestal)/T_rec, each tag carrying its OWN
# pedestal/noise scale (the clip-bias statistics depend on K and the record stats).
CONST = {
    "G": dict(name="GPS L1 C/A",  t_rec=1e-3,  K=1000,
              tle=None),  # None -> gps_beamtrack.DEFAULT_TLE_URL
    "E": dict(name="Galileo E1C", t_rec=4e-3,  K=250,
              tle="https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"),
    "C": dict(name="BeiDou B1C",  t_rec=10e-3, K=100,
              tle="https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle"),
}
DEFAULT_LOGS = {  # run_live's per-constellation status logger outputs
    "G": "/tmp/gpswipe/status_log.jsonl",
    "E": "/tmp/gpswipe/status_log_gal.jsonl",
    "C": "/tmp/gpswipe/status_log_bds.jsonl",
}


def load(path, tag):
    """One constellation's status log -> (tag array, prn, t, x, dop). x = s2/N per record."""
    K4 = CONST[tag]["K"] ** 0.25
    prn, t, x, dop = [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = r.get("amplitude") or 0.0
        u = r.get("unbiased_amplitude") or 0.0
        if a > 0.0 and a > u:  # exact: N = a^2 - u^2 (new logs)
            v = (u * u) / (a * a - u * u)
        else:                   # fallback: invert the normalized significance (older logs)
            v = (r.get("amp_snr") or 0.0) / K4
        prn.append(int(r["prn"]))
        t.append(float(r["t"]))
        x.append(v)
        dop.append(r.get("doppler_hz") or 0.0)
    n = len(prn)
    return (np.full(n, tag), np.array(prn, int), np.array(t, "<f8"), np.array(x, float),
            np.array(dop, float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*",
                    help="status logs as TAG=PATH (G=..., E=..., C=...); bare paths are G. "
                         "Default: every existing " + ", ".join(DEFAULT_LOGS.values()))
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--ped-mask", type=float, default=10.0,
                    help="samples BELOW this elevation calibrate the noise pedestal (deg)")
    ap.add_argument("--map-mask", type=float, default=5.0, help="map elevation mask (deg)")
    ap.add_argument("--tbin", type=float, default=60.0, help="time-series bin (s)")
    ap.add_argument("--naz", type=int, default=48)
    ap.add_argument("--nel", type=int, default=17)
    ap.add_argument("--min-bin", type=int, default=10, help="min emits per map cell")
    ap.add_argument("--dop-jump", type=float, default=5.0,
                    help="a seed-Doppler step above this (Hz) between consecutive emits marks a "
                         "RE-ANCHOR -- the code is briefly off-peak while the DLL re-trims")
    ap.add_argument("--reanchor-pad", type=float, default=2.0,
                    help="exclude emits within this many seconds of a re-anchor (STATE-flagged, "
                         "value-blind: no selection bias on the map)")
    ap.add_argument("--ped-tbin", type=float, default=600.0,
                    help="pedestal time bin (s), per constellation")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args(argv)

    # ---- gather logs (composite: all constellations sharing the 1575.42 tune) ----
    logs = {}
    if args.logs:
        for spec in args.logs:
            tag, _, path = spec.rpartition("=")
            tag = tag or "G"
            if tag not in CONST:
                ap.error("unknown constellation tag %r (use G/E/C)" % tag)
            logs[tag] = path
    else:
        logs = {tag: p for tag, p in DEFAULT_LOGS.items() if os.path.exists(p)}
    if not logs:
        ap.error("no status logs found")

    from gps_beamtrack import load_gps_satellites, DEFAULT_TLE_URL
    from skyfield.api import load as sky_load, wgs84
    ts = sky_load.timescale()
    obs = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt)

    # ---- per-constellation: load, excise re-anchors, sky positions, pedestal, -> density y ----
    # y = (x - pedestal_tag(t)) / T_rec_tag : a C/N0 DENSITY (Hz), comparable across signals --
    # the per-record ratio x is NOT (records are 1/4/10 ms), and the clip-bias pedestal + noise
    # scale are per-signal statistics.
    TAG, PRN, T, Y, ALT, AZ, YSIG = [], [], [], [], [], [], []
    for tag, path in sorted(logs.items()):
        tags, prn, t, x, dop = load(path, tag)
        if len(x) == 0:
            print("%s: empty log %s" % (tag, path))
            continue
        # state-flagged re-anchor excision (value-blind)
        keep = np.ones(len(x), bool)
        for p in np.unique(prn):
            idx = np.where(prn == p)[0]
            o = idx[np.argsort(t[idx])]
            if len(o) < 3:
                continue
            jumps = t[o][1:][np.abs(np.diff(dop[o])) > args.dop_jump]
            bad = np.zeros(len(o), bool)
            for tj in jumps:
                bad |= np.abs(t[o] - tj) <= args.reanchor_pad
            keep[o[bad]] = False
        n_cut = int((~keep).sum())
        prn, t, x = prn[keep], t[keep], x[keep]
        # sky positions
        sats = load_gps_satellites(CONST[tag]["tle"] or DEFAULT_TLE_URL)
        alt = np.full(len(prn), np.nan)
        az = np.full(len(prn), np.nan)
        for p in np.unique(prn):
            sat = sats.get(int(p))
            if sat is None:
                continue
            sel = prn == p
            times = ts.from_datetimes(
                [datetime.fromtimestamp(u, tz=timezone.utc) for u in t[sel]])
            topo = (sat - obs).at(times).altaz()
            alt[sel] = topo[0].degrees
            az[sel] = topo[1].degrees
        ok = np.isfinite(alt)
        prn, t, x, alt, az = prn[ok], t[ok], x[ok], alt[ok], az[ok]
        # per-tag, per-epoch pedestal from below-horizon sats
        low = alt < args.ped_mask
        glob = (float(np.median(x[low])) if low.sum() >= 200
                else float(np.percentile(x, 20)) if len(x) else 0.0)
        tb = ((t - t.min()) // args.ped_tbin).astype(int) if len(t) else np.array([], int)
        ped_t = np.full((tb.max() + 1) if len(tb) else 1, glob)
        for b in np.unique(tb):
            sel = low & (tb == b)
            if sel.sum() >= 100:
                ped_t[b] = np.median(x[sel])
        sig1 = (float(np.std(x[low])) if low.sum() >= 200
                else float(np.std(x[x < np.percentile(x, 50)])) if len(x) else 1.0)
        t_rec = CONST[tag]["t_rec"]
        y = (x - ped_t[tb]) / t_rec          # C/N0 density, Hz
        ysig = sig1 / t_rec                  # single-emit noise scale in the same units
        print("%s (%s): %d emits (%d re-anchor-cut), ped x=%.4f, single-emit floor %.1f dB-Hz"
              % (tag, CONST[tag]["name"], len(x), n_cut, glob,
                 10 * np.log10(max(2 * sig1 / t_rec, 1e-9))))
        TAG.append(np.full(len(y), tag)); PRN.append(prn); T.append(t); Y.append(y)
        ALT.append(alt); AZ.append(az); YSIG.append(np.full(len(y), ysig))
    tag = np.concatenate(TAG); prn = np.concatenate(PRN); t = np.concatenate(T)
    y = np.concatenate(Y); alt = np.concatenate(ALT); az = np.concatenate(AZ)
    ysig = np.concatenate(YSIG)
    sid = np.array([a + str(b) for a, b in zip(tag, prn)])   # "G12"/"E25"/"C19"
    span_h = (t.max() - t.min()) / 3600.0
    print("composite: %d emits, %d sats, span %.1f h" % (len(y), len(set(sid)), span_h))

    def to_db(mean_y, mean_sig, nsamp):
        lim = 2.0 * mean_sig / np.sqrt(np.maximum(nsamp, 1))
        return np.where(mean_y > lim, 10.0 * np.log10(np.maximum(mean_y, 1e-12)), np.nan)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag_color = {"G": "tab:blue", "E": "tab:orange", "C": "tab:red"}
    t0 = t.min()

    # (1) per-sat C/N0 vs time (constellation-colored)
    fig, ax = plt.subplots(figsize=(11, 5))
    for s_ in sorted(set(sid)):
        m = sid == s_
        if m.sum() < 20:
            continue
        tb = ((t[m] - t0) // args.tbin).astype(int)
        ub = np.unique(tb)
        my = np.array([y[m][tb == b].mean() for b in ub])
        ms = np.array([ysig[m][tb == b].mean() for b in ub])
        ns = np.array([(tb == b).sum() for b in ub])
        ax.plot(ub * args.tbin / 3600.0, to_db(my, ms, ns), ".", ms=3,
                color=tag_color[s_[0]], alpha=0.7)
    for k, c in tag_color.items():
        if k in set(tag):
            ax.plot([], [], "o", color=c, label=CONST[k]["name"])
    ax.set_xlabel("hours since %s UTC"
                  % datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
    ax.set_ylabel("incoherent C/N0 (dB-Hz, est.)")
    ax.set_title("Composite incoherent C/N0 (%.0f s bins) -- %.1f h, %d sats"
                 % (args.tbin, span_h, len(set(sid))))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "cn0_inc_vs_time.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)

    # (2) elevation response, constellation-colored
    fig, ax = plt.subplots(figsize=(8, 5))
    eb = np.arange(args.map_mask, 90.1, 2.0)
    for s_ in sorted(set(sid)):
        m = (sid == s_) & (alt > args.map_mask)
        if m.sum() < 50:
            continue
        idx = np.digitize(alt[m], eb) - 1
        ub = np.unique(idx)
        my = np.array([y[m][idx == b].mean() for b in ub])
        ms = np.array([ysig[m][idx == b].mean() for b in ub])
        ns = np.array([(idx == b).sum() for b in ub])
        ax.plot(eb[ub] + 1.0, to_db(my, ms, ns), ".-", ms=4, lw=0.6,
                color=tag_color[s_[0]], alpha=0.6)
    ax.set_xlabel("elevation (deg)")
    ax.set_ylabel("incoherent C/N0 (dB-Hz, est.)")
    ax.set_title("Elevation response (per-sat; blue=GPS orange=Galileo red=BeiDou)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(args.outdir, "cn0_inc_vs_elev.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)

    # (3) COMPOSITE polar beam map: all constellations pooled per (az, el) cell in density
    # units. Different satellites' EIRP differences smear a pixel a little, but the beam
    # SHAPE dominates -- and the 3x track density fills the sky 3x faster.
    m = alt > args.map_mask
    az_e = np.linspace(0, 360, args.naz + 1)
    el_e = np.linspace(args.map_mask, 90, args.nel + 1)
    ai = np.clip(np.digitize(az[m], az_e) - 1, 0, args.naz - 1)
    ei = np.clip(np.digitize(alt[m], el_e) - 1, 0, args.nel - 1)
    ys, ss = y[m], ysig[m]
    grid = np.full((args.nel, args.naz), np.nan)
    for i in range(args.nel):
        for j in range(args.naz):
            sel = (ei == i) & (ai == j)
            n = sel.sum()
            if n >= args.min_bin:
                grid[i, j] = to_db(np.array([ys[sel].mean()]),
                                   np.array([ss[sel].mean()]), np.array([n]))[0]
    TH, RG = np.meshgrid(np.deg2rad(az_e), 90.0 - el_e)
    fig = plt.figure(figsize=(8.4, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    vlo, vhi = (np.nanpercentile(grid, [2, 98]) if np.isfinite(grid).any() else (0, 1))
    pc = ax.pcolormesh(TH, RG, np.ma.masked_invalid(grid), cmap="turbo", vmin=vlo, vmax=vhi,
                       shading="flat")
    ax.set_rlim(0, 90)
    ax.set_rticks([30, 60, 90])
    ax.set_yticklabels(["60", "30", "0"])
    ax.set_title("Composite incoherent C/N0 beam map (G+E+C; %.1f h, %d sats)"
                 % (span_h, len(set(sid))), pad=18)
    fig.colorbar(pc, ax=ax, label="C/N0 (dB-Hz, est.)", shrink=0.8, pad=0.09)
    fig.tight_layout()
    f3 = os.path.join(args.outdir, "cn0_inc_beammap.png")
    fig.savefig(f3, dpi=120)
    plt.close(fig)

    if np.isfinite(grid).any():
        print("map: %d cells; C/N0 %.1f .. %.1f dB-Hz"
              % (int(np.isfinite(grid).sum()), np.nanmin(grid), np.nanmax(grid)))
    print("wrote:\n  %s\n  %s\n  %s" % (f1, f2, f3))


if __name__ == "__main__":
    main()
