"""elem_<chain>_<YYYYMMDD>.jsonl  ->  beam-map obs rows (gnss_beam_map.py transits).

The element archive is the offline consumer the `--element-archive-dir` flag was written for
("the beam map and peel solves are offline consumers and a combined number cannot be
un-combined"). It is also, as of 2026-08-26, the ONLY continuous record that survived the
broker-log truncations: one append-per-day file per chain, unbroken across every restart.

WHAT IS IN A ROW: per (t, chain, prn, instance) the raw per-element parts --
`u[e]` complex correlation, `q[e]` its normalisation, `p2[e]` power -- plus `probe`, which
marks the NOISE-ANCHOR PRNs (satellites deliberately followed where nothing is transmitting).

⚠️⚠️ WHICH FIELD IS THE BEAM. `GnssCoherentCombiner::get_elements_callback` defines the four
parts exactly, and only ONE of them is a beam:

    u  = <A_e * conj(LOO ref)>   the per-element complex GAIN, "phase relative to the
                                 array-mean convention"
    p2 = <|A_e|^2>               "the incoherent beam-map power, still biased -- its debias
                                 is the broker's job, from the probes"
    q  = <|LOO ref|^2>           the NORMALISER: amp_e ~ |u|/q "in array-mean units"

`|u|/q` is therefore a RATIO TO THE ARRAY MEAN, and a ratio to the array mean is flat across
the sky BY CONSTRUCTION -- every element rises and falls with the reference. Mapping it
produced a believable-looking but meaningless ~6 dB envelope on the first attempt here; the
tell was physical (a 6 m dish at 1176 MHz cannot have a 6 dB全-sky pattern). `p2` is the
un-normalised power, and `gnssRecord.hpp` calls its record-side twin CMB_ELEM_AMP_INCOH
"THE BEAM MAP VALUE for this antenna ... what a transit traces out".

TWO QUANTITIES COME OUT, AND THEY ARE NOT THE SAME MAP:

  * `cn0_inc_dbhz` = 10 log10( median_e ( p2_e - floor_e ) )  -- ⚡ THE BEAM. Per-element
    received power, PROBE-DEBIASED: `p2` includes the noise power, and the probe PRNs (real
    satellites deliberately followed where nothing is transmitting) measure exactly that
    pedestal per (instance, element). Subtracting in POWER is the only correct domain.
  * `cn0_coh_dbhz` = 20 log10( |SUM_e u_e| / SUM_e q_e )  -- NOT a beam: the raw-parts phase
    alignment (see gnss_beam_coh.py). Kept because it is free and diagnostic.

⚠️ They ride the fields `gnss_beam_map.py` calls "coh" and "inc" because that is the transport
it already has, and the pipeline's own rule -- never coadd the two together -- is exactly the
rule these need. `--quantity coh` renders the array map, `--quantity inc` the element map.
Their DIFFERENCE is the interesting one: it is the array coherence in dB, so a region of sky
where the elements stop phasing up shows as the two maps disagreeing.

⚠️ dB, ARBITRARY ZERO. `u` is not flux-calibrated, so only the SHAPE is meaningful, and only
within one chain (bands differ in gain). Coadding in dB is what the pipeline does anyway, so
per-satellite EIRP offsets land in the per-pixel scatter, which is the systematics handle.

RANGE IS DIVIDED OUT by default (--no-range-norm to keep it): a satellite at 20,000 km is
~2.5 dB stronger overhead than at the horizon purely from spreading loss, which would
masquerade as beam. BRDC supplies the range at the row epoch, so this costs nothing.

INSTANCES ARE MEDIANED, NOT SUMMED. The sky does not care which GPU served a record, and a
wedged instance serving a ten-minute-old sky would otherwise pull the mean
([[chord-nothing-is-per-node]]: any per-instance structure is a bug, never a feature).

Usage:
    gnss_beam_elem2obs.py --elem elem_gps_l5_20260825.jsonl elem_gps_l5_20260826.jsonl \\
        --sys G --out obs_gps_l5.jsonl [--tmin ... --tmax ...] [--every 30]

@author Keith Vanderlinde
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")
from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all  # noqa: E402

LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
R_REF = 20.2e6          # m, the GPS semi-synchronous slant range -- the dB zero for --range-norm
MIN_ELEMS = 8           # an instance with fewer live elements says nothing about the beam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elem", nargs="+", required=True)
    ap.add_argument("--sys", required=True, choices=["G", "E", "C"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4e9)
    ap.add_argument("--every", type=float, default=0.0,
                    help="decimate: keep at most one row per PRN per this many seconds")
    ap.add_argument("--no-range-norm", action="store_true")
    ap.add_argument("--mask-deg", type=float, default=0.0)
    args = ap.parse_args()

    # ---- pass 1: fold (prn, tick) x instance -> per-instance array + element amplitudes ----
    # Keyed on the archive tick (rows written in one pass share `t` to the centisecond), so
    # instances combine only with the instances of the SAME poll.
    # PASS 0: the probe pedestal, per (instance, element, 5-min bin). Per ELEMENT, never
    # medianed across them -- elements differ in gain by design, and a floor averaged over
    # elements would over-subtract the quiet ones into negative power.
    floors = defaultdict(lambda: defaultdict(list))    # (inst, tbin) -> {e: [p2 ...]}
    for path in args.elem:
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"probe": true' not in line and '"probe":true' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("t")
                if t is None or not (args.tmin <= t <= args.tmax) or not d.get("probe"):
                    continue
                for e, v in enumerate(d.get("p2") or []):
                    if v > 0.0:
                        floors[(d["inst"], int(t // 300))][e].append(float(v))
    floor = {k: {e: statistics.median(v) for e, v in d.items()} for k, d in floors.items()}
    print("probe pedestal: %d (instance, 5-min) bin(s)" % len(floor))

    per = defaultdict(dict)        # (prn, t) -> {inst: (arr, beam, sig)}
    n_rows = n_probe = 0

    for path in args.elem:
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("t")
                if t is None or not (args.tmin <= t <= args.tmax):
                    continue
                u, q, p2 = d.get("u") or [], d.get("q") or [], d.get("p2") or []
                if len(u) < MIN_ELEMS:
                    continue
                n_rows += 1
                if d.get("probe"):
                    n_probe += 1
                    continue              # a probe is the FLOOR, never a beam sample
                fl = floor.get((d["inst"], int(t // 300))) or {}
                pw, rat, sre, sim, sq = [], [], 0.0, 0.0, 0.0
                for e, pair in enumerate(u):
                    qe = q[e] if e < len(q) else 0.0
                    re_, im_ = float(pair[0]), float(pair[1])
                    if re_ == 0.0 and im_ == 0.0:
                        continue          # a DARK element, not a zero measurement
                    if qe > 0.0:
                        sre += re_
                        sim += im_
                        sq += qe
                    p2e = float(p2[e]) if e < len(p2) else 0.0
                    f = fl.get(e)
                    if p2e > 0.0 and f is not None and f > 0.0:
                        # DEBIAS IN POWER. A sample below its own pedestal is a
                        # non-detection, not negative power -- dropped, not clamped, so it
                        # cannot pile up at an artificial floor and fake a beam edge.
                        if p2e > f:
                            pw.append(p2e - f)
                            rat.append(p2e / f)
                if len(pw) < MIN_ELEMS or sq <= 0.0:
                    continue
                per[(int(d["prn"]), round(float(t), 1))][d["inst"]] = (
                    math.hypot(sre, sim) / sq, statistics.median(pw),
                    statistics.median(rat))

    print("rows %d (%d probe) -> %d (prn, tick) sample(s)" % (n_rows, n_probe, len(per)))
    if not per:
        sys.exit("no usable rows")

    # ---- geometry: BRDC once per (prn, minute), interpolation-free like mkobs.py ----------
    days = sorted({int(t // 86400) for _, t in per})
    eph = None
    for dnum in days:                     # one fetch covers the span; last wins if it spans
        eph = parse_rinex_nav(fetch_brdc(
            datetime.fromtimestamp(dnum * 86400 + 43200, tz=timezone.utc)))
    geo = {}

    def geom(prn, t):
        key = (prn, int(t // 60))
        if key not in geo:
            try:
                v = predict_all(eph, LAT, LON, ALT, key[1] * 60 + 30,
                                mask_deg=-90.0, max_age=86400.0).get((args.sys, prn))
            except Exception:
                v = None
            geo[key] = (v["el"], v["az"], v["range_m"]) if v else None
        return geo[key]

    last = {}
    n_out = n_nogeo = n_low = 0
    with open(args.out, "w") as fh:
        for (prn, t), insts in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if args.every > 0.0 and t - last.get(prn, -1e18) < args.every:
                continue
            g = geom(prn, t)
            if not g or g[0] is None:
                n_nogeo += 1
                continue
            el, az, rng = g
            if el <= args.mask_deg:
                n_low += 1
                continue
            last[prn] = t
            # MEDIAN over instances -- see the module note.
            arr = statistics.median([v[0] for v in insts.values()])
            beam = statistics.median([v[1] for v in insts.values()])
            sig = statistics.median([v[2] for v in insts.values()])
            if arr <= 0.0 or beam <= 0.0:
                continue
            # POWER domain: 10 log10, and the range correction doubles to 20 log10 of the
            # ratio ... which is the same 20 log10 in amplitude. Spelled out because mixing
            # the two conventions is the classic factor-of-two in a beam map.
            corr = 0.0 if args.no_range_norm else 20.0 * math.log10(rng / R_REF)
            fh.write(json.dumps({
                "t": t, "prn": prn, "sys": args.sys, "az": az, "el": el,
                "range_m": rng, "n_inst": len(insts),
                "cn0_inc_dbhz": 10.0 * math.log10(beam) + corr,
                "cn0_coh_dbhz": 20.0 * math.log10(arr) + corr,
                "sig": round(sig, 3)}) + "\n")
            n_out += 1
    print("wrote %d row(s) -> %s   (dropped: %d no-geometry, %d below %.0f deg)"
          % (n_out, args.out, n_nogeo, n_low, args.mask_deg))


if __name__ == "__main__":
    main()
