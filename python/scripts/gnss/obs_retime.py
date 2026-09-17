#!/usr/bin/env python3
"""Re-time observables files whose writer carried a stale F-engine sample-0 epoch.

    obs_retime.py census gal_e5a 20260913 --out-map maps/20260913.json
    obs_retime.py apply --map maps/gal_e5a_20260913.json --map maps/bds_b2a_20260913.json \\
                          fixtures/obs/*_20260913.jsonl --out-dir fixtures/obs_retimed

THE FAULT. The writer stamps every row t = frame0 + hop * (samples_per_hop / sample_rate) with
frame0 read once at startup. An F-engine re-base restarts the hop counter without moving that
latched value, so a writer that outlives it files the new session onto the OLD epoch: rows land
tens of hours in the past, several sessions stack into one day file (t non-monotonic in file
order), and every geometry column -- az, el, range, range-rate, sat clock, eph age, the carrier
residual -- is evaluated at the wrong instant. Nothing the BROKER published is affected: its
anchor is re-read and it restarts itself. The hop axis in the rows is intact.

WHAT IS RECOVERABLE. Within one F-engine session the error is ONE constant: the difference
between the true session anchor and the latched one. So a file is cut into monotonic segments
(a re-base shows as t stepping back), each segment gets one offset, t and t_gps shift by it,
and the geometry is recomputed from the ephemeris at the corrected epoch. The broker-side
columns are copied through untouched.

HOW THE OFFSET IS FOUND, AND WHY IT IS THEN SNAPPED. Each strong row carries the replica's own
Doppler at its own record hop (dop_rec_hz, rec_hop). The ephemeris Doppler of every strong PRN
matches those numbers at exactly one time shift (receiver clock removed as the common mode);
the scan finds it to about a second. A session anchor is an integer second plus the F-engine's
constant fraction, and the brokers log every anchor they ever latched, so the scan result is
snapped to the nearest KNOWN anchor when one lies within --snap-s. A segment that snaps to
nothing is written with the raw scan value and flagged `anchor_src: scan` in the map; a
segment with too few strong rows to date is flagged and NOT rewritten.

⚠️ CENSUS ON GAL OR BDS, NEVER ON gps_l5. A genuine match has rms 0.02-0.05 Hz; anything above
--max-rms-hz is a false minimum and is discarded. The L5 chain's replicas are deliberately run
on secondary-code lobes, whole multiples of ~100 Hz off the sky Doppler (the lobe-coherent fleet
DLL), so their dop_rec_hz cannot be matched to an ephemeris at any shift -- the scan finds
plausible-looking minima at 0.3-1.6 Hz that disagree with the other bands by tens of hours.

NEVER IN PLACE. Output goes to --out-dir, named by each row's TRUE day, with the raw epoch kept
beside the new one (t_raw, frame0_raw, retime_s) so a row can always be traced back. The
originals are not opened for writing.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_ephemeris import parse_rinex_nav, predict_all, gpst_of_utc, C_LIGHT  # noqa: E402

HPS = 3.2e9 / 16384.0            # F-engine hops per second
FRAC = 0.000002861               # the constant sub-second part of every CHORD anchor seen so far
SITE = (49.32001414, -119.62262691, 545.0)
CACHE = os.path.expanduser("~/.cache/kotekan_gps")
SYSOF = {"gps": "G", "gal": "E", "bds": "C"}
GEOM_KEYS = ("az", "el", "range_m", "range_rate_mps", "sat_clk_s", "eph_age_s")


def load_eph(days_back=12):
    """One merged ephemeris from every cached daily of the last `days_back` days: predict_all
    picks per epoch inside its toe window, so one object serves any instant in the span."""
    now = time.time()
    files = [f for f in glob.glob(os.path.join(CACHE, "BRDC00WRD_*_01D_MN.rnx.gz"))
             if now - os.path.getmtime(f) < days_back * 86400.0]
    if not files:
        raise SystemExit("no cached BRDC dailies under %s" % CACHE)
    return parse_rinex_nav(sorted(files))


def rows_of(path):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if len(line) < 50:
                continue
            try:
                yield i, json.loads(line)
            except ValueError:
                continue


def segments(path):
    """[(first_idx, last_idx_exclusive, t_first, t_last, frame0_latched)] in file order; a new
    segment starts wherever t steps back by more than 60 s."""
    segs, cur = [], None
    for i, d in rows_of(path):
        t = d.get("t")
        h = d.get("fadr_hop") or d.get("rec_hop")
        if t is None:
            continue
        if cur is None or t < cur["t_last"] - 60.0:
            if cur is not None:
                segs.append(cur)
            cur = {"i0": i, "i1": i + 1, "t0": t, "t_last": t, "f0": []}
        cur["i1"] = i + 1
        cur["t_last"] = t
        if h and len(cur["f0"]) < 2000:
            cur["f0"].append(t - h / HPS)
    if cur is not None:
        segs.append(cur)
    for s in segs:
        if s["f0"]:
            # t is stamped from fleet_hop while the row keeps fadr_hop/rec_hop, so the median
            # sits a fraction of a second off the latched anchor; an anchor is an integer
            # second plus FRAC, so snap -- the retime then comes out as whole seconds and an
            # unaffected file re-times by exactly zero
            est = float(np.median(s["f0"]))
            s["frame0_latched"] = round(est - FRAC) + FRAC
            s["frame0_latched_est"] = est
        else:
            s["frame0_latched"] = None
        del s["f0"]
    return segs


def scan_offset(path, seg, eph, sysid, lam, probes=3, cn0_min=30.0, max_rms_hz=0.2):
    """Time shift (s) that matches the ephemeris Doppler to dop_rec_hz for the strong rows at
    `probes` instants across the segment; None when no instant has 4 strong PRNs."""
    span = seg["t_last"] - seg["t0"]
    if span > 900:
        centres = [seg["t0"] + max(300.0, 0.1 * span), seg["t0"] + 0.5 * span,
                   seg["t_last"] - max(300.0, 0.1 * span)]
    else:
        centres = [seg["t0"] + 0.5 * span]
    picks = {c: {} for c in centres[:probes]}
    for i, d in rows_of(path):
        if i < seg["i0"]:
            continue
        if i >= seg["i1"]:
            break
        t, dop, hop, cn = d.get("t"), d.get("dop_rec_hz"), d.get("rec_hop"), d.get("cn0_kcoh_dbhz")
        if dop is None or not hop or cn is None or cn < cn0_min:
            continue
        for c in picks:
            if abs(t - c) < 3.0 and d["prn"] not in picks[c]:
                # the record's own epoch on the latched axis, not the poll instant
                picks[c][int(d["prn"])] = (seg["frame0_latched"] + hop / HPS, float(dop))

    def rms_at(pk, delta):
        r = []
        for prn, (t_rec, dop) in pk.items():
            v = predict_all(eph, SITE[0], SITE[1], SITE[2],
                            datetime.fromtimestamp(t_rec + delta, tz=timezone.utc),
                            mask_deg=-90.0, max_age=7200.0).get((sysid, prn))
            if v is not None:
                r.append(dop + v["range_rate_mps"] / lam)
        if len(r) < 4:
            return None
        r = np.array(r)
        r -= np.median(r)
        return float(np.sqrt(np.mean(r * r)))

    def best_in(pk, lo, hi, step):
        best = (1e9, None)
        for dl in np.arange(lo, hi, step):
            v = rms_at(pk, dl)
            if v is not None and v < best[0]:
                best = (v, float(dl))
        return best

    results = []
    for c, pk in picks.items():
        if len(pk) < 4:
            continue
        # coarse over +-4 days, then two refinements: the residual is a smooth function of
        # the shift (Doppler rate ~1 Hz/s), so the coarse minimum brackets the true one
        best = best_in(pk, -4 * 86400, 4 * 86400, 300.0)
        if best[1] is None:
            continue
        best = best_in(pk, best[1] - 400, best[1] + 400, 10.0)
        best = best_in(pk, best[1] - 15, best[1] + 15, 1.0)
        rec = {"t_row": c, "n_prn": len(pk), "offset_s": best[1], "rms_hz": best[0]}
        if best[0] > max_rms_hz:
            rec["rejected"] = "rms %.2f Hz > %.2f: a false minimum, not a match" % (best[0], max_rms_hz)
        results.append(rec)
    return results


def census(args):
    band = args.band
    if band == "gps_l5":
        raise SystemExit("gps_l5 cannot date a segment: its replicas run on +-100 Hz lobes (see the module note); census a GAL or BDS band")
    sysid = SYSOF[band.split("_")[0]]
    lam = C_LIGHT / float(args.carrier_hz or {"gps_l5": 1176.45e6, "gal_e5a": 1176.45e6,
                                                 "bds_b2a": 1176.45e6, "gps_l2c": 1227.6e6,
                                                 "gal_e5b": 1207.14e6, "bds_b2b": 1207.14e6,
                                                 "gal_e6": 1278.75e6, "bds_b3i": 1268.52e6}[band])
    path = os.path.join(args.obs_dir, "%s_%s.jsonl" % (band, args.day))
    eph = load_eph()
    anchors = sorted(float(a) for a in args.anchor)
    out = {"day": args.day, "band": band, "path": path, "segments": []}
    for k, seg in enumerate(segments(path)):
        rec = {"seg": k, "i0": seg["i0"], "i1": seg["i1"], "t0": seg["t0"], "t1": seg["t_last"],
               "frame0_latched": seg["frame0_latched"], "probes": []}
        if seg["frame0_latched"] is None or seg["i1"] - seg["i0"] < 50:
            rec["status"] = "skipped: no hop axis or too short"
            out["segments"].append(rec)
            print("seg %d rows %d-%d: %s" % (k, seg["i0"], seg["i1"], rec["status"]))
            continue
        res = scan_offset(path, seg, eph, sysid, lam, max_rms_hz=args.max_rms_hz)
        rec["probes"] = res
        good = [r for r in res if "rejected" not in r]
        if not good:
            rec["status"] = ("undated: fewer than 4 strong PRNs at every probe" if not res
                             else "undated: every probe rejected (rms > %.2f Hz)" % args.max_rms_hz)
        else:
            offs = np.array([r["offset_s"] for r in good])
            spread = float(offs.max() - offs.min())
            off = float(np.median(offs))
            f0_new = seg["frame0_latched"] + off
            near = [a for a in anchors if abs(a - f0_new) <= args.snap_s]
            if near:
                a = min(near, key=lambda a: abs(a - f0_new))
                rec.update({"frame0_true": a, "anchor_src": "logged anchor",
                            "retime_s": a - seg["frame0_latched"]})
            else:
                a = math.floor(f0_new) + FRAC
                rec.update({"frame0_true": a, "anchor_src": "scan (no logged anchor within %.0f s)" % args.snap_s,
                            "retime_s": a - seg["frame0_latched"]})
            rec["scan_spread_s"] = spread
            rec["status"] = "ok" if spread <= 6.0 else "inconsistent: probes disagree by %.0f s" % spread
        out["segments"].append(rec)
        print("seg %d rows %d-%d  t %s..%s  latched %.0f  -> %s  retime %+.3f h  (%s; probes %s)"
              % (k, seg["i0"], seg["i1"],
                 time.strftime("%m-%d %H:%M", time.gmtime(seg["t0"])),
                 time.strftime("%m-%d %H:%M", time.gmtime(seg["t_last"])),
                 seg["frame0_latched"],
                 ("%.6f" % rec["frame0_true"]) if "frame0_true" in rec else "-",
                 rec.get("retime_s", float("nan")) / 3600.0, rec["status"],
                 ["%+.0f s rms %.2f Hz n%d%s" % (r["offset_s"], r["rms_hz"], r["n_prn"], " REJECTED" if "rejected" in r else "") for r in res]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_map)), exist_ok=True)
    json.dump(out, open(args.out_map, "w"), indent=1)
    print("wrote", args.out_map)


def apply(args):
    # several maps per day: the segment a band could not date (no strong rows -- the broker was
    # blind on that chain) is usually dated by another band of the same writer generation
    maps = [json.load(open(m)) for m in args.map]
    day = maps[0]["day"]
    segs = []
    for m in maps:
        for s in m["segments"]:
            if s.get("status") != "ok":
                continue
            if any(abs(s["frame0_latched"] - q["frame0_latched"]) < 2.0
                   and s["t0"] < q["t1"] + 120 and q["t0"] < s["t1"] + 120 for q in segs):
                continue        # the same session already dated by an earlier map
            segs.append(s)
    eph = load_eph()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"maps": [os.path.abspath(m) for m in args.map], "segments": segs,
                "inputs": [], "outputs": {}}
    handles = {}
    # predict_all evaluates the whole sky for one epoch; every PRN of a poll shares its epoch
    # to the microsecond, so one call serves ~20 rows. Without this the rewrite is hours.
    geo_cache = collections.OrderedDict()

    def sky_at(t_utc):
        k = round(t_utc, 4)
        v = geo_cache.get(k)
        if v is None:
            try:
                v = predict_all(eph, SITE[0], SITE[1], SITE[2],
                                datetime.fromtimestamp(t_utc, tz=timezone.utc),
                                mask_deg=-90.0, max_age=21600.0)
            except Exception:
                v = {}
            geo_cache[k] = v
            if len(geo_cache) > 4096:
                geo_cache.popitem(last=False)
        return v

    def out_for(band, t_true):
        day = time.strftime("%Y%m%d", time.gmtime(t_true))
        key = "%s_%s.jsonl" % (band, day)
        if key not in handles:
            p = os.path.join(args.out_dir, key)
            if os.path.exists(p) and not args.append:
                raise SystemExit("%s exists; pass --append to add to it" % p)
            handles[key] = open(p, "a")
            manifest["outputs"][key] = 0
        return handles[key], key

    for path in args.inputs:
        band = os.path.basename(path).rsplit("_", 1)[0]
        sysid = SYSOF[band.split("_")[0]]
        n_in = n_out = n_skip = 0
        # ⚠️ the map was censused on ONE band; the other bands' writers started together and
        # latched the same anchor, so segments are matched by LATCHED frame0 and t-range, not
        # by row index (row counts differ per band).
        for i, d in rows_of(path):
            n_in += 1
            t = d.get("t")
            h = d.get("fadr_hop") or d.get("rec_hop")
            if t is None or not h:
                n_skip += 1
                continue
            f0 = t - h / HPS
            seg = None
            for s in segs:
                if abs(s["frame0_latched"] - f0) < 2.0 and s["t0"] - 120 <= t <= s["t1"] + 120:
                    seg = s
                    break
            if seg is None:
                n_skip += 1
                continue
            dt = seg["retime_s"]
            t_new = round(t + dt, 4)
            d["t_raw"], d["frame0_raw"], d["frame0"], d["retime_s"] = d["t"], seg["frame0_latched"], seg["frame0_true"], dt
            d["retime_src"] = seg["anchor_src"]
            d["t"] = t_new
            d["t_gps"] = round(gpst_of_utc(t_new), 4)
            # geometry at the corrected epoch; the same call and window the writer used
            v = sky_at(t_new).get((sysid, int(d["prn"])))
            if v is None:
                for k in GEOM_KEYS:
                    d.pop(k, None)
                d["carr_resid_m"] = None
                d["carr_resid_src"] = None
            else:
                d["az"], d["el"] = round(v["az"], 3), round(v["el"], 3)
                d["range_m"] = round(v["range_m"], 3)
                d["range_rate_mps"] = round(v["range_rate_mps"], 4)
                d["sat_clk_s"] = v["sat_clk_s"]
                d["eph_age_s"] = round(v["toe_age_s"], 1)
                lam = C_LIGHT / float(d["carrier_hz"])
                # carr_resid_m as the writer builds it: the fleet ADR at ITS hop against the
                # range at that hop; the single-instance adr form at the row epoch as fallback
                fh, fd = d.get("fadr_hop"), d.get("fadr_dop_cycles")
                if fh and fd is not None:
                    va = sky_at(seg["frame0_true"] + fh / HPS).get((sysid, int(d["prn"])))
                    if va is not None:
                        d["carr_resid_m"] = -fd * lam - va["range_m"]
                        d["carr_resid_src"] = "fadr"
                elif d.get("adr_cycles") is not None:
                    d["carr_resid_m"] = -d["adr_cycles"] * lam - v["range_m"]
                    d["carr_resid_src"] = "adr"
            fh_out, key = out_for(band, t_new)
            fh_out.write(json.dumps(d, separators=(",", ":")) + "\n")
            manifest["outputs"][key] += 1
            n_out += 1
        manifest["inputs"].append({"path": path, "rows": n_in, "written": n_out, "skipped": n_skip})
        print("%s: %d rows, %d re-timed, %d skipped (no dated segment)" % (os.path.basename(path), n_in, n_out, n_skip))
    for fh in handles.values():
        fh.close()
    if args.sort:
        for key in handles:
            p = os.path.join(args.out_dir, key)
            lines = open(p).read().splitlines()
            lines.sort(key=lambda s: json.loads(s)["t"])
            with open(p, "w") as fh:
                fh.write("\n".join(lines) + "\n")
    tag = "+".join(sorted({os.path.basename(p).rsplit("_", 1)[0] for p in args.inputs}))
    mp = os.path.join(args.out_dir, "retime_manifest_%s_%s.json" % (day, tag))
    json.dump(manifest, open(mp, "w"), indent=1)
    print("wrote", mp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("band")
    c.add_argument("day")
    c.add_argument("--obs-dir", default="/home/kvand/gnss/fixtures/obs")
    c.add_argument("--carrier-hz", type=float, default=None)
    c.add_argument("--anchor", action="append", default=[],
                   help="a known F-engine anchor (s); repeatable. Scan results within --snap-s snap to it")
    c.add_argument("--snap-s", type=float, default=10.0)
    c.add_argument("--max-rms-hz", type=float, default=0.2,
                   help="a probe whose best fit is worse than this is a false minimum")
    c.add_argument("--out-map", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--map", action="append", required=True, help="census map(s) for ONE day; repeatable")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--append", action="store_true")
    a.add_argument("--no-sort", dest="sort", action="store_false")
    args = ap.parse_args()
    (census if args.cmd == "census" else apply)(args)


if __name__ == "__main__":
    main()
