#!/usr/bin/env python3
"""Beam survey from recorded GPS deep-integration records (/tmp/gpswipe/level_*.raw).

Reads the COMBINER output records (one frame per ~1 s integration, n_prn PRN-slots each), recovers
each satellite's deep amplitude Â over time, recomputes its apparent alt/az at every sample via
skyfield + current GPS TLEs, and makes:

  1. amp_vs_time.png   -- Â(t) per PRN over the whole run
  2. amp_vs_elev.png   -- Â vs elevation (the antenna/beam elevation cut), per PRN
  3. skymap.png        -- alt/az polar map, points coloured by response -> the partially-filled beam

WHAT'S IN THE FILE (gnssRecord.hpp combiner slots): prn, doppler, code_phase, |A|_incoh, <A>re,
<A>im, |A|_coh, nchan, DEEP Â (slot 8), UTC (slots 9-10 f64), E/L power, dll_disc, carrier_resid.
NOTE: deep_snr and coherence_s are NOT written to disk (REST/browser only), so an ABSOLUTE C/N0 can't
be reconstructed from the record. Â is recorded and is a faithful RELATIVE signal-quality proxy
(Â ~ sqrt(C/N0)), so this reports a relative response in dB = 20*log10(Â / floor). For an absolute,
persisted C/N0 going forward, run a get_status logger alongside the pipeline (see --help notes).

Usage:
  python3 gps_beam_survey.py /tmp/gpswipe --lat 43.968697 --lon -79.252106 --alt 260
"""
import numpy as np, struct, os, sys, glob, argparse
from datetime import datetime, timezone

# Reuse the (schema-agnostic) skyfield + TLE helpers from the live beam-track tool.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kotekan",
                                "python", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "scripts"))

RECORD_SLOTS = 15
RECORD_BYTES = RECORD_SLOTS * 4
UTC_SLOT = 9
# Current combiner record layout (lib/stages/gnss/gnssRecord.hpp CMB_* constants).
SLOT = {"prn": 0, "dop": 1, "cp": 2, "a_incoh": 3, "mean_re": 4, "mean_im": 5,
        "a_coh": 6, "nchan": 7, "deep": 8, "e_pow": 11, "l_pow": 12, "dll": 13, "carrier_resid": 14}


def infer_n_prn(path):
    size = os.path.getsize(path)
    # frames are [uint32 meta][n_prn * RECORD_BYTES]; for a multi-frame file, find the n_prn that
    # divides the whole file into equal frames (meta is a constant 0 here).
    for n in range(1, 40):
        stride = 4 + n * RECORD_BYTES
        if size % stride == 0:
            return n
    raise ValueError("could not infer n_prn from size %d of %s" % (size, path))


def read_records(paths, n_prn=None):
    """Read all combiner records -> dict of flat arrays keyed by SLOT names, plus 'utc'."""
    if n_prn is None:
        n_prn = infer_n_prn(paths[0])
    stride = 4 + n_prn * RECORD_BYTES
    cols = {k: [] for k in SLOT}
    utc = []
    for path in paths:
        buf = open(path, "rb").read()
        nframe = len(buf) // stride
        # Drop the last frame: the pipeline may be writing live, so the final frame can be partial.
        for fi in range(max(0, nframe - 1)):
            base = fi * stride + 4
            block = np.frombuffer(buf, dtype="<f4", count=n_prn * RECORD_SLOTS, offset=base)
            block = block.reshape(n_prn, RECORD_SLOTS)
            for k, s in SLOT.items():
                cols[k].append(block[:, s])
            # UTC is a float64 aliased over slots 9-10 of each record.
            for p in range(n_prn):
                utc.append(struct.unpack_from("<d", buf, base + p * RECORD_BYTES + UTC_SLOT * 4)[0])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["utc"] = np.array(utc, dtype="<f8")
    return out, n_prn


def read_status_log(path):
    """Load the gps_status_logger JSONL (has real wall-clock t + deep_snr + coherence_s -> a real
    C/N0, unlike the raw record which only has Â). Returns (prn, abs_unix, deep_amplitude, cn0_dbhz,
    deep_records) -- deep_records lets us drop the auto-coherence ladder's short-window emits, whose
    deep_snr is extreme-value-inflated and whose C/N0 = deep_snr^2/coherence_s is then amplified by
    the small coherence_s -> the spurious upward spikes over the true ~1 s-window baseline."""
    import json
    prn, t, da, cn0, dr = [], [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        c = r.get("cn0_dbhz")
        if c is None:
            continue
        prn.append(int(r["prn"])); t.append(float(r["t"]))
        da.append(float(r.get("deep_amplitude") or 0.0)); cn0.append(float(c))
        dr.append(int(r.get("deep_records") or 0))
    return (np.array(prn, int), np.array(t, "<f8"),
            np.array(da, float), np.array(cn0, float), np.array(dr, int))


def attach_altaz(prn, utc, lat, lon, alt_m, tle):
    """(alt_deg, az_deg) per record via skyfield + current GPS TLEs (accurate for a day-old run:
    GPS orbits move <1 km/day at TLE precision -> <0.1 deg in alt/az)."""
    from skyfield.api import load, wgs84
    from gps_beamtrack import load_gps_satellites
    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = load_gps_satellites(tle)
    alt = np.full(len(prn), np.nan)
    az = np.full(len(prn), np.nan)
    ip = prn.astype(int)
    for p in np.unique(ip):
        sat = by_prn.get(int(p))
        if sat is None:
            continue
        sel = ip == p
        times = ts.from_datetimes([datetime.fromtimestamp(u, tz=timezone.utc) for u in utc[sel]])
        topo = (sat - observer).at(times).altaz()
        alt[sel] = topo[0].degrees
        az[sel] = topo[1].degrees
    return alt, az


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="/tmp/gpswipe", help="record dir/file/glob")
    ap.add_argument("--status", default=None,
                    help="read a gps_status_logger JSONL instead of the raw records: real C/N0 "
                         "(deep_snr-based) + real wall-clock time, no anchoring needed (RECOMMENDED)")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--tle", default="https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle")
    ap.add_argument("--mask", type=float, default=3.0, help="elevation mask, deg (skymap/elev plots)")
    ap.add_argument("--start-utc", type=float, default=None,
                    help="unix time of the FIRST sample (overrides the file-mtime anchor for sky pos)")
    ap.add_argument("--pctile", type=float, default=50.0,
                    help="per-cell statistic for the binned beam map (percentile; default 50 = median, "
                         "robust to residual scatter -- NOT the upper envelope: the up-spikes are "
                         "short-window artifacts, not extra signal)")
    ap.add_argument("--min-bin", type=int, default=8, help="min samples per alt/az cell to fill it")
    ap.add_argument("--skip-warmup", type=float, default=0.0,
                    help="drop the first N minutes (the deep integration + carrier loop ramping up: "
                         "Â/C/N0 read high until the coherent window converges). Try 8.")
    ap.add_argument("--min-coh-frac", type=float, default=0.6,
                    help="status log only: drop emits whose deep_records < this * the run's max "
                         "(the auto-coherence ladder's short-window picks -> the spurious C/N0 spikes). "
                         "1.0 keeps everything.")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n-prn", type=int, default=None)
    args = ap.parse_args(argv)

    # --- Load: prefer the status JSONL (real C/N0 + real time) over the raw record (Â only). ---
    if args.status:
        prn, abs_unix, deep, cn0, drec = read_status_log(args.status)
        if not len(prn):
            ap.error("no usable samples in %s" % args.status)
        keep = np.isfinite(deep) & (deep > 0)
        # Drop the auto-coherence short-window emits: their deep_snr is extreme-value-inflated and the
        # C/N0 = deep_snr^2/coherence_s normalization amplifies it -> the spurious up-spikes. Keep only
        # emits near the run's dominant (converged, ~1 s) window.
        if args.min_coh_frac < 1.0 and drec.max() > 0:
            keep &= drec >= args.min_coh_frac * drec.max()
        prn, abs_unix, deep, cn0 = prn[keep], abs_unix[keep], deep[keep], cn0[keep]
        n_drop = int((~keep).sum())
        print("status: dropped %d short-window emits (deep_records < %.0f%% of %d)"
              % (n_drop, 100 * args.min_coh_frac, drec.max()))
        src = "status log (real C/N0)"
    else:
        if os.path.isdir(args.path):
            paths = sorted(glob.glob(os.path.join(args.path, "*.raw")))
        else:
            paths = sorted(glob.glob(args.path))
        if not paths:
            ap.error("no record files matched: %s (or pass --status)" % args.path)
        rec, n_prn = read_records(paths, args.n_prn)
        prn = rec["prn"].astype(int)
        deep = rec["deep"]
        rel = rec["utc"]   # RELATIVE capture time (config capture_utc0=1.0 -> 1.0 + elapsed_seconds)
        # Keep only real, active samples: a covered slot (nchan>0), finite positive Â, sane timestamp.
        good = ((rec["nchan"] > 0) & np.isfinite(deep) & (deep > 0) & (prn > 0)
                & np.isfinite(rel) & (rel > 0.5) & (rel < 2.0e5))
        prn, deep, rel = prn[good], deep[good], rel[good]
        # Anchor relative capture time to wall-clock UTC (for sky positions). One continuous run (a
        # restart wipes the dir + resets to 1.0), so map the LAST sample to the file's last-write
        # mtime and advance 1:1. --start-utc overrides. GPS TLEs are day-stable -> ~min error <~1 deg.
        mtime = max(os.path.getmtime(p) for p in paths)
        abs_unix = (args.start_utc + (rel - rel.min())) if args.start_utc else (mtime - (rel.max() - rel))
        cn0 = None
        src = "raw records (Â-derived, relative dB)"

    # Drop the warm-up: the first minutes read high while the deep integration + shared carrier loop
    # ramp up (the coherent window grows from ~0.1 s to ~1 s -> the short-window noise floor shrinks).
    if args.skip_warmup > 0 and len(abs_unix):
        cut = abs_unix.min() + args.skip_warmup * 60.0
        w = abs_unix >= cut
        prn, abs_unix, deep = prn[w], abs_unix[w], deep[w]
        if cn0 is not None:
            cn0 = cn0[w]
        print("skipped first %.0f min warm-up (%d samples dropped, %d kept)"
              % (args.skip_warmup, int((~w).sum()), int(w.sum())))

    prns = sorted(np.unique(prn).tolist())
    span_h = (abs_unix.max() - abs_unix.min()) / 3600.0 if len(abs_unix) else 0.0
    t0dt = datetime.fromtimestamp(abs_unix.min(), tz=timezone.utc)
    print("source: %s -- %d samples, %d PRNs %s, span %.1f h"
          % (src, len(prn), len(prns), prns, span_h))
    print("time: %s .. %s UTC" % (t0dt.strftime("%Y-%m-%d %H:%M"),
          datetime.fromtimestamp(abs_unix.max(), tz=timezone.utc).strftime("%m-%d %H:%M")))

    # Alt/az per sample.
    alt = az = None
    try:
        alt, az = attach_altaz(prn, abs_unix, args.lat, args.lon, args.alt, args.tle)
        print("alt/az computed (skyfield). elevation range %.0f..%.0f deg"
              % (np.nanmin(alt), np.nanmax(alt)))
    except Exception as e:
        print("alt/az unavailable (%s) -- time plots only" % e, file=sys.stderr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("turbo")
    colors = {p: cmap(i / max(1, len(prns) - 1)) for i, p in enumerate(prns)}

    # The plotted "response": real C/N0 (dB-Hz) from the status log, else an Â-derived relative dB
    # (20*log10(Â / floor), floor = 10th pctile above the mask -> ~0 dB at floor).
    if cn0 is not None:
        resp_db = cn0
        resp_label = "C/N0 (dB-Hz, est.)"
    else:
        resp_label = "relative response (dB, Â-derived)"
        ref_sel = (alt > args.mask) if alt is not None else np.ones(len(deep), bool)
        floor = np.percentile(deep[ref_sel], 10) if ref_sel.any() else np.percentile(deep, 10)
        resp_db = 20.0 * np.log10(np.clip(deep / floor, 1e-3, None))
        print("Â floor (10th pctile) = %.4f" % floor)
    print("response: %.1f .. %.1f (%s)" % (resp_db.min(), resp_db.max(), resp_label))

    th = (abs_unix - abs_unix.min()) / 3600.0   # hours since start

    # (1) Â vs time.
    fig, axarr = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for p in prns:
        s = prn == p
        axarr[0].plot(th[s], deep[s], ".", ms=1.5, color=colors[p], label="PRN %d" % p)
        axarr[1].plot(th[s], resp_db[s], ".", ms=1.5, color=colors[p])
    axarr[0].set_ylabel("Â (deep |A|)"); axarr[0].set_ylim(bottom=0)
    axarr[1].set_ylabel(resp_label)
    axarr[1].set_xlabel("hours since %s UTC" % t0dt.strftime("%Y-%m-%d %H:%M"))
    axarr[0].legend(ncol=min(len(prns), 8), fontsize=8, markerscale=4, loc="upper right")
    axarr[0].set_title("GPS deep amplitude vs time  (%d PRNs, %.1f h)" % (len(prns), span_h))
    for a in axarr:
        a.grid(alpha=0.3)
    fig.tight_layout(); f1 = os.path.join(args.outdir, "amp_vs_time.png")
    fig.savefig(f1, dpi=110); plt.close(fig)

    # (2) Â (and response dB) vs elevation.
    if alt is not None:
        fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
        for p in prns:
            s = (prn == p) & (alt > args.mask)
            axarr[0].plot(alt[s], deep[s], ".", ms=1.5, color=colors[p], label="PRN %d" % p)
            axarr[1].plot(alt[s], resp_db[s], ".", ms=1.5, color=colors[p])
        axarr[0].set_xlabel("elevation (deg)"); axarr[0].set_ylabel("Â (deep |A|)")
        axarr[0].set_ylim(bottom=0)
        axarr[1].set_xlabel("elevation (deg)"); axarr[1].set_ylabel(resp_label)
        axarr[0].legend(ncol=2, fontsize=8, markerscale=4)
        for a in axarr:
            a.grid(alpha=0.3)
        fig.suptitle("Elevation cut of the beam response")
        fig.tight_layout(); f2 = os.path.join(args.outdir, "amp_vs_elev.png")
        fig.savefig(f2, dpi=110); plt.close(fig)

        # (3) Sky map: polar (az, el), zenith at centre, horizon at rim, N up, E right, coloured by
        # response dB -> the partially-filled beam map traced out by the visible-PRN passes.
        m = alt > args.mask
        theta = np.deg2rad(az[m])                 # 0 = North, clockwise
        r = 90.0 - alt[m]                          # 0 at zenith, 90 at horizon
        # Colour range from the DATA's own spread (2..98 pctile), not 0..max: an absolute C/N0
        # (~15-18 dB-Hz) never reaches 0, so a 0-based scale slams every cell to the top of the
        # colourmap. Spanning the actual range makes the few-dB beam variation visible.
        vlo, vhi = np.percentile(resp_db[m], [2, 98])
        fig = plt.figure(figsize=(8.4, 7.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)  # N up, clockwise (E right)
        sc = ax.scatter(theta, r, c=resp_db[m], s=4, cmap="turbo", vmin=vlo, vmax=vhi)
        ax.set_rlim(0, 90); ax.set_rticks([30, 60, 90])
        ax.set_yticklabels(["60°", "30°", "0°"])   # elevation labels (r=90-el)
        ax.set_title("GPS beam map  (deep-Â response, dB; %d PRN passes, %.1f h)"
                     % (len(prns), span_h), pad=18)
        fig.colorbar(sc, ax=ax, label=resp_label, shrink=0.8, pad=0.09)
        fig.tight_layout(); f3 = os.path.join(args.outdir, "skymap.png")
        fig.savefig(f3, dpi=120); plt.close(fig)

        # (4) BINNED beam map: per alt/az cell, the MEDIAN response (default; --pctile to change).
        # NOT the upper envelope: the per-emit up-spikes are the auto-coherence ladder's short-window
        # picks, whose deep_snr is extreme-value-inflated and whose C/N0 = deep_snr^2/coherence_s is
        # then amplified by the small coherence_s -- a processing artifact, not extra signal (the true
        # C/N0 is the steady ~1 s-window baseline). --min-coh-frac already drops most of them from the
        # status source; the median then reads the physical response per direction.
        naz, nel = 60, 18
        az_edges = np.linspace(0, 360, naz + 1)
        el_edges = np.linspace(args.mask, 90, nel + 1)
        ai = np.clip(np.digitize(az[m], az_edges) - 1, 0, naz - 1)
        ei = np.clip(np.digitize(alt[m], el_edges) - 1, 0, nel - 1)
        rr = resp_db[m]
        grid = np.full((nel, naz), np.nan)
        for i in range(nel):
            for j in range(naz):
                sel = (ei == i) & (ai == j)
                if sel.sum() >= args.min_bin:
                    grid[i, j] = np.percentile(rr[sel], args.pctile)
        TH, RG = np.meshgrid(np.deg2rad(az_edges), 90.0 - el_edges)
        fig = plt.figure(figsize=(8.4, 7.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        # Colour range from the filled cells' own 2..98 pctile (NOT 0..max): the per-cell C/N0 sits
        # in a narrow absolute band (~15-18 dB-Hz), so a 0-based scale saturates every cell -> pick
        # the range from the data so the beam's few-dB variation actually shows.
        if np.isfinite(grid).any():
            gvlo, gvhi = np.nanpercentile(grid, [2, 98])
            if gvhi - gvlo < 1e-6:
                gvlo, gvhi = gvlo - 0.5, gvhi + 0.5
        else:
            gvlo, gvhi = 0.0, 1.0
        pc = ax.pcolormesh(TH, RG, np.ma.masked_invalid(grid), cmap="turbo", vmin=gvlo, vmax=gvhi,
                           shading="flat")
        ax.set_rlim(0, 90); ax.set_rticks([30, 60, 90]); ax.set_yticklabels(["60°", "30°", "0°"])
        ax.set_title("GPS beam map -- %gth-pctile response per cell  (%.1f h)"
                     % (args.pctile, span_h), pad=18)
        fig.colorbar(pc, ax=ax, label=resp_label, shrink=0.8, pad=0.09)
        fig.tight_layout(); f4 = os.path.join(args.outdir, "beammap_binned.png")
        fig.savefig(f4, dpi=120); plt.close(fig)
        print("wrote:\n  %s\n  %s\n  %s\n  %s" % (f1, f2, f3, f4))
    else:
        print("wrote:\n  %s" % f1)


if __name__ == "__main__":
    main()
