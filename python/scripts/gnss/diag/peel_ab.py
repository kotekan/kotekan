#!/usr/bin/env python3
"""Fused-peel A/B: read one PEELED GnssCoherentCombiner and report, per PRN,
    depth_dB = 20*log10(deep_amplitude / peel_deep)
straight from the record slots the device add-back and the combiner already produced
(gnssRecord.hpp CMB_DEEP vs CMB_PEEL_DEEP). No second chain, no twin tracker: the fused
peel measures its own depth in one pass (docs/gnss_voltage_peel_live.md).

Two things this checks that a single number cannot:
  * the TRACKING products are unchanged. deep_amplitude here must match a baseline (un-peeled)
    run of the same capture to within noise -- that is the "recover everything we have now" gate,
    and it is the whole point of the analytic add-back. Pass --baseline <url> to compare live.
  * the residual is a LOWER BOUND when it sits at the combiner floor: peel_deep <= deep_floor
    means the true depth is deeper than reported (exactly as diag/peel_depth.py flags offline).

Usage:
    ./peel_ab.py --url http://localhost:12062 --prns 10,32,31,23,25,28 --samples 20
    ./peel_ab.py --url http://localhost:12062 --baseline http://localhost:12063 --prns 10,32
"""
import argparse
import json
import math
import statistics
import time
import urllib.request


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _by_prn(rows, prns):
    out = {}
    for r in rows:
        try:
            p = int(r.get("prn", -1))
        except (TypeError, ValueError):
            continue
        if p in prns:
            out[p] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12062", help="PEELED combiner REST base")
    ap.add_argument("--combiner", default="gps_combiner")
    ap.add_argument("--baseline", default="", help="optional un-peeled combiner base for a "
                                                   "products A/B (deep_amplitude must match)")
    ap.add_argument("--prns", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=6.0)
    args = ap.parse_args()

    prns = [int(p) for p in args.prns.split(",") if p.strip()]
    u = "%s/%s/get_status" % (args.url.rstrip("/"), args.combiner.strip("/"))
    ub = ("%s/%s/get_status" % (args.baseline.rstrip("/"), args.combiner.strip("/"))
          if args.baseline else "")

    if args.warmup > 0:
        time.sleep(args.warmup)

    acc = {p: {"deep": [], "peel": [], "floor": [], "base": []} for p in prns}
    for _ in range(args.samples):
        try:
            rows = _by_prn(_get(u), set(prns))
            base = _by_prn(_get(ub), set(prns)) if ub else {}
        except Exception as e:
            print("  poll failed (%s)" % e, flush=True)
            time.sleep(args.interval)
            continue
        for p in prns:
            r = rows.get(p)
            if not r:
                continue
            dp = r.get("deep_amplitude", 0.0) or 0.0
            pk = r.get("peel_deep", 0.0) or 0.0
            if dp > 0:
                acc[p]["deep"].append(float(dp))
            if pk > 0:
                acc[p]["peel"].append(float(pk))
            acc[p]["floor"].append(float(r.get("deep_floor", 0.0) or 0.0))
            b = base.get(p)
            if b and (b.get("deep_amplitude", 0.0) or 0.0) > 0:
                acc[p]["base"].append(float(b["deep_amplitude"]))
        time.sleep(args.interval)

    print("\nFUSED PEEL DEPTH (from the record slots -- one pass, self-measured)")
    hdr = "  PRN   deep_amp   peel_deep   depth_dB   note"
    if ub:
        hdr += "     base_amp  d(prod)%"
    print(hdr)
    for p in prns:
        a = acc[p]
        if not a["deep"]:
            print("  %3d   (no deep samples)" % p)
            continue
        md = statistics.median(a["deep"])
        floored = a["peel"] and statistics.median(a["peel"]) <= (
            statistics.median(a["floor"]) * 1.05 if a["floor"] else 0.0)
        note = ""
        if not a["peel"]:
            mp = float("nan")
            depth = float("nan")
            note = "residual not populated (peel not engaged?)"
        else:
            mp = statistics.median(a["peel"])
            depth = 20.0 * math.log10(md / mp) if mp > 0 else float("inf")
            if floored:
                note = "resid AT FLOOR -> depth is a LOWER BOUND"
        line = "  %3d   %8.4f   %9.5f   %7.2f   %s" % (p, md, mp, depth, note)
        if ub:
            if a["base"]:
                mb = statistics.median(a["base"])
                dprod = 100.0 * (md - mb) / mb if mb else float("nan")
                line = ("  %3d   %8.4f   %9.5f   %7.2f   %-38s %8.4f  %+.2f"
                        % (p, md, mp, depth, note, mb, dprod))
            else:
                line += "   (no baseline)"
        print(line)


if __name__ == "__main__":
    main()
