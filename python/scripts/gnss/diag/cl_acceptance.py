#!/usr/bin/env python3
"""L2C CL sibling-chain acceptance: CL vs CM, time-averaged, judged against the in-run control.

Polls both combiners' get_status for --minutes, then reports per-PRN MEDIANS (the 07-27
lesson twice over: deep_snr swings 3-5x between samples on healthy chains, and a snapshot
inside a warm-up transient reads as a wedge). Acceptance shape:

  - CL deep_snr within a few dB of CM's (equal TDM power split) on CM-certified sats;
  - CL coherence_s >= CM's (the floor CL exists to remove is CM's, not physics);
  - CL certifies sats CM cannot (the prize: 2.2-sigma floor vs 11.3) -- report, don't gate;
  - CM medians unchanged vs a pre-CL baseline (run with --baseline before deploying, or
    just eyeball the CM columns against the viewer's history).

usage: cl_acceptance.py [--url http://localhost:12048] [--cm l2c_gps_combiner]
                        [--cl l2c_cl_combiner] [--minutes 5] [--interval 5]
"""
import argparse
import json
import time
import urllib.request

import numpy as np


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--cm", default="l2c_gps_combiner")
    ap.add_argument("--cl", default="l2c_cl_combiner")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    t_end = time.time() + args.minutes * 60.0
    hist = {}  # prn -> {"cm_deep": [], "cl_deep": [], "cm_coh": [], "cl_coh": [], "cl_amp": []}
    n_poll = 0
    while time.time() < t_end:
        try:
            cm = {int(r["prn"]): r for r in get("%s/%s/get_status" % (args.url, args.cm))}
            cl = {int(r["prn"]): r for r in get("%s/%s/get_status" % (args.url, args.cl))}
        except Exception as e:
            print("poll failed: %s" % e)
            time.sleep(args.interval)
            continue
        n_poll += 1
        for prn in set(cm) | set(cl):
            h = hist.setdefault(prn, {k: [] for k in
                                      ("cm_deep", "cl_deep", "cm_coh", "cl_coh",
                                       "cm_amp", "cl_amp")})
            for tag, src in (("cm", cm), ("cl", cl)):
                r = src.get(prn) or {}
                h[tag + "_deep"].append(float(r.get("deep_snr") or 0.0))
                h[tag + "_coh"].append(float(r.get("coherence_s") or 0.0))
                h[tag + "_amp"].append(float(r.get("amp_snr") or 0.0))
        time.sleep(args.interval)

    print("%d polls over %.1f min; medians (0 rows dropped only if never present)"
          % (n_poll, args.minutes))
    print("%4s %9s %9s %9s | %9s %9s %9s | %s" %
          ("prn", "cm_amp", "cm_deep", "cm_coh", "cl_amp", "cl_deep", "cl_coh", "verdict"))
    live = 0
    for prn in sorted(hist):
        h = hist[prn]
        med = {k: (np.median([x for x in v if x > 0]) if any(x > 0 for x in v) else 0.0)
               for k, v in h.items()}
        if med["cm_deep"] < 5 and med["cl_deep"] < 5:
            continue  # neither chain sees it; skip the noise rows
        live += 1
        verdict = []
        if med["cm_deep"] >= 20:  # CM solidly locked: CL should be within a few dB
            ratio_db = (20 * np.log10(med["cl_deep"] / med["cm_deep"])
                        if med["cl_deep"] > 0 else float("-inf"))
            verdict.append("cl/cm %+.1f dB" % ratio_db)
            if med["cl_deep"] < 0.25 * med["cm_deep"]:
                verdict.append("WEAK-CL(k wrong? seed gap?)")
        if med["cl_coh"] > 0 and med["cm_coh"] == 0.0:
            verdict.append("CL-CERTIFIES-WHERE-CM-CANNOT")
        if med["cl_coh"] > med["cm_coh"] > 0:
            verdict.append("coh %+.2fs" % (med["cl_coh"] - med["cm_coh"]))
        print("%4d %9.1f %9.1f %9.2f | %9.1f %9.1f %9.2f | %s" %
              (prn, med["cm_amp"], med["cm_deep"], med["cm_coh"],
               med["cl_amp"], med["cl_deep"], med["cl_coh"], "; ".join(verdict) or "-"))
    if not live:
        print("no live satellites in either chain -- is the node up?")


if __name__ == "__main__":
    main()
