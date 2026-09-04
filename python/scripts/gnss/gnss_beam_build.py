#!/usr/bin/env python3
"""Drive the whole beam-map pipeline for one time window (2026-07-24).

Replaces the by-hand 3-commands-per-band-signal workflow: finds the obs logs that
cover the window across the rotated archive + the live dir, builds transits for
every constellation of every band, coadds BOTH C/N0 estimators, and renders.

    gnss_beam_build.py --label d2026-07-24 --tmin '2026-07-24 00:00' --tmax now

Windows are DAY boundaries here, not the old 18:00->10:00 "night" ones: those
silently dropped 10:00-18:00 every day (~8 h of sky). Labels are therefore d<date>,
which also keeps them from colliding with the older n<date> sets on disk.

Layout (all under --beamdir, default data/beam):
    transits/<label>_<band>/<tag>_<prn>_t<k>.npz
    maps/<label>/<band>_<tag>_<quantity>.npz     accumulators -- combine forever
    plots/<label>/<band>_<tag>_<quantity>.png

Bands are never mixed (L1/L2C/L5 sit ~11 dB apart) and neither are the two
estimators (the incoherent one is BOC-biased); both facts are enforced downstream
by gnss_beam_map.py's combine, which refuses a quantity mismatch.
"""
import argparse
import concurrent.futures as cf
import datetime
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_MAP = os.path.join(HERE, "gnss_beam_map.py")

# band -> (kotekan output dir name, {constellation tag: obs file})
BANDS = {
    "L1": ("gpswipe", {"G": "obs_gps_l1.jsonl", "E": "obs_gal_e1.jsonl",
                       "C": "obs_bds_b1c.jsonl", "L": "obs_gps_l1c.jsonl"}),
    "L2C": ("gps_l2c_gpu", {"G": "obs_gps_l2c.jsonl"}),
    "L5": ("gps_l5_gpu", {"G": "obs_gps_l5.jsonl", "E": "obs_gal_e5a.jsonl",
                          "C": "obs_bds_b2a.jsonl"}),
}
SIGNAL = {("L1", "G"): "GPS L1 C/A", ("L1", "E"): "Galileo E1C",
          ("L1", "C"): "BeiDou B1C", ("L1", "L"): "GPS L1C",
          ("L2C", "G"): "GPS L2C", ("L5", "G"): "GPS L5",
          ("L5", "E"): "Galileo E5a", ("L5", "C"): "BeiDou B2a"}


def parse_time(s):
    if s == "now":
        return datetime.datetime.now().timestamp()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return float(s)  # raw unix


def obs_files(args, band, fname):
    """Archive + live obs logs whose data can fall in [tmin, tmax].

    A rotated archive dir holds data ENDING near its file mtime and starting where
    the previous rotation left off, so a file is in scope when its own mtime is
    after tmin and its predecessor's mtime is before tmax. (Verified 2026-07-24:
    consecutive archives are disjoint in t -- no dedup needed.)
    """
    subdir = BANDS[band][0]
    cands = [(os.path.getmtime(p), p) for p in
             glob.glob(os.path.join(args.archive, "run_*", subdir, fname))]
    cands.sort()
    live = os.path.join(args.live_root, subdir, fname)  # /tmp/gpswipe, /tmp/gps_l5_gpu, ...
    if os.path.exists(live):
        cands.append((float("inf"), live))
    out, prev = [], float("-inf")
    for mt, p in cands:
        if mt > args.tmin and prev < args.tmax:
            out.append(p)
        prev = mt
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return f"FAILED {' '.join(cmd[1:4])}\n{r.stderr.strip()[-800:]}"
    return r.stdout.strip()


def build_one(args, band, tag):
    fname = BANDS[band][1][tag]
    files = obs_files(args, band, fname)
    if not files:
        return f"{band}/{tag}: no obs logs in window"
    night = f"{args.label}_{band}"
    lines = [run([sys.executable, BEAM_MAP, "transits", "--obs", *files,
                  "--tag", tag, "--night", night, "--tmin", str(args.tmin),
                  "--tmax", str(args.tmax), "--outdir", args.beamdir])]
    tr = os.path.join(args.beamdir, "transits", night, f"{tag}_*.npz")
    if not glob.glob(tr):
        return "\n".join(lines + [f"{band}/{tag}: no transits -> no map"])
    for q in ("coh", "inc"):
        mp = os.path.join(args.beamdir, "maps", args.label, f"{band}_{tag}_{q}.npz")
        png = os.path.join(args.beamdir, "plots", args.label, f"{band}_{tag}_{q}.png")
        lines.append(run([sys.executable, BEAM_MAP, "coadd", "--transits", tr,
                          "--out", mp, "--quantity", q]))
        lines.append(run([sys.executable, BEAM_MAP, "render", "--map", mp,
                          "--png", png, "--title",
                          f"{SIGNAL[(band, tag)]} -- {args.label} ({q})"]))
    return "\n".join(lines)


def merge(args):
    """Sum day accumulators into one label, then render everything on ONE colour
    scale per signal so the days are actually comparable to each other."""
    import numpy as np
    out = []
    for band in args.bands.split(","):
        tags = list(BANDS[band][1])
        for q in ("coh", "inc"):
            groups = [(t, [t]) for t in tags]
            if len(tags) > 1:
                groups.append(("ALL", tags))
            for gname, gtags in groups:
                ins = [os.path.join(args.beamdir, "maps", lab, f"{band}_{t}_{q}.npz")
                       for lab in args.merge.split(",") for t in gtags]
                ins = [p for p in ins if os.path.exists(p)]
                if not ins:
                    continue
                mp = os.path.join(args.beamdir, "maps", args.label,
                                  f"{band}_{gname}_{q}.npz")
                r = run([sys.executable, BEAM_MAP, "combine",
                         "--inputs", *ins, "--out", mp])
                out.append(r)
                if r.startswith("FAILED"):
                    continue
                # Colour limits from the merged map -- the days inherit them.
                z = np.load(mp)
                m = z["n"] > 0
                mean = z["s1"][m] / z["n"][m]
                lo, hi = np.percentile(mean, [2.0, 99.5])
                lo, hi = float(np.floor(lo)), float(np.ceil(hi))
                sig = SIGNAL.get((band, gname), f"{band} all constellations")
                for lab in [args.label] + args.merge.split(","):
                    src = os.path.join(args.beamdir, "maps", lab,
                                       f"{band}_{gname}_{q}.npz")
                    if not os.path.exists(src):
                        continue
                    png = os.path.join(args.beamdir, "plots", lab,
                                       f"{band}_{gname}_{q}.png")
                    out.append(run([sys.executable, BEAM_MAP, "render", "--map", src,
                                    "--png", png, "--vmin", str(lo), "--vmax", str(hi),
                                    "--title", f"{sig} -- {lab} ({q})"]))
    print("\n".join(x for x in out if x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="e.g. d2026-07-24")
    ap.add_argument("--merge", help="comma-separated labels to sum into --label "
                                    "(skips the build; also adds per-band ALL maps)")
    ap.add_argument("--tmin")
    ap.add_argument("--tmax", help="'now' or 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--bands", default="L1,L2C,L5")
    ap.add_argument("--archive", default="/home/lwlab/gnss_archive")
    ap.add_argument("--live-root", default="/tmp",
                    help="holds the LIVE run's dirs (gpswipe/, gps_l2c_gpu/, ...)")
    ap.add_argument("--beamdir", default="data/beam")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.merge:
        return merge(args)
    if not (args.tmin and args.tmax):
        ap.error("--tmin/--tmax required unless --merge")
    args.tmin, args.tmax = parse_time(args.tmin), parse_time(args.tmax)
    f = lambda u: datetime.datetime.fromtimestamp(u).strftime("%Y-%m-%d %H:%M")
    print(f"{args.label}: {f(args.tmin)} -> {f(args.tmax)} "
          f"({(args.tmax - args.tmin) / 3600.0:.1f} h)")
    jobs = [(b, t) for b in args.bands.split(",") for t in BANDS[b][1]]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for out in ex.map(lambda bt: build_one(args, *bt), jobs):
            print(out, flush=True)


if __name__ == "__main__":
    main()
