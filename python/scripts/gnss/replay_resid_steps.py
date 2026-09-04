#!/usr/bin/env python3
"""Measure the carrier-residual step at each RELEASE/ESCAPE event of a replay leg.

Needs the timestamped broker log (2ec49e04). For every RELEASE/ESCAPE with a PRN,
finds that sat's |carrier_hz_resid| in the status stream just before the event and the
max within the following window; the difference is the step the event injected.
The precomp matrix's primary metric: a working pre-compensation drives steps to ~0.

Usage: replay_resid_steps.py <legname> [--window 6]
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

SP = os.environ.get("REPLAY_OUT",
                    "/tmp/claude-1000/-home-lwlab-airspy-gps/"
                    "19ffc4a5-abfb-4bf5-970c-216ad32aea52/scratchpad")

ap = argparse.ArgumentParser()
ap.add_argument("leg")
ap.add_argument("--window", type=float, default=6.0)
args = ap.parse_args()

# status series per prn: (t, resid, coh)
raw = open("%s/replay_%s_status.jsonl" % (SP, args.leg)).read()
series = {}
day0 = None
for rec in re.split(r'(?=\{"t":1)', raw):
    rec = rec.strip()
    if not rec:
        continue
    try:
        d = json.loads(rec)
    except Exception:
        continue
    if day0 is None:
        day0 = datetime.fromtimestamp(d["t"]).replace(hour=0, minute=0, second=0,
                                                      microsecond=0).timestamp()
    for r in d.get("status") or []:
        series.setdefault(r["prn"], []).append(
            (d["t"], abs(r.get("carrier_hz_resid") or 0.0),
             (r.get("coherence_s") or 0) > 0))

# events from the timestamped broker log
events = []
pat = re.compile(r"\[broker (\d\d):(\d\d):(\d\d\.\d+)\] (RELEASE|ESCAPE) PRN (\d+)")
mdop = re.compile(r"ddop ([+-]\d+)")
for line in open("%s/replay_%s_broker.log" % (SP, args.leg)):
    m = pat.search(line)
    if not m:
        continue
    t = day0 + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    dd = mdop.search(line)
    events.append((t, m.group(4), int(m.group(5)),
                   float(dd.group(1)) if dd else float("nan")))

steps = []
for t, kind, prn, ddop in events:
    s = series.get(prn)
    if not s:
        continue
    pre = [v for tt, v, _ in s if t - 4.0 < tt <= t]
    post = [v for tt, v, _ in s if t < tt <= t + args.window]
    if not pre or not post:
        continue
    steps.append((kind, prn, ddop, min(pre), max(post)))
    print("  %-7s C%-3d ddop %+5.0f  resid %5.2f -> max %6.2f Hz"
          % (kind, prn, ddop, min(pre), max(post)))
if steps:
    import statistics
    jumps = [mx - pre for _, _, _, pre, mx in steps]
    print("%s: %d events | post-event resid jump med %.2f  p90 %.2f  max %.2f Hz"
          % (args.leg, len(steps), statistics.median(jumps),
             sorted(jumps)[int(len(jumps) * 0.9)] if len(jumps) > 1 else jumps[0],
             max(jumps)))
else:
    print("%s: no scorable events" % args.leg)
