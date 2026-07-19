#!/usr/bin/env python3
"""Extract the commanded carrier-trim history from a (timestamped) broker log.

The shared carrier loop's trim integrates directly into the exported ADR (phi integrates
f_nco = ctrim + ff), so every trim move -- REACQ re-pull, BOOTSTRAP convergence, watchdog
re-seed -- is a known, commanded phase-ramp change contaminating gf/TEC arcs at the
~200 TECU/Hz/100s scale (docs/adr_trim_subtraction.md). This tool recovers trim(t) per
sat as a piecewise-constant series from the CAR log lines (timestamped since 2ec49e04),
resetting the level at FIRST SEED lines (car_trim is deleted on re-seed; a fresh seed
starts at the fleet median, which the next CAR line reports).

Output: jsonl rows {"t": unix, "prn": n, "trim_hz": x} sorted by t.

Usage: gnss_trim_journal.py <broker.log> [--date YYYY-MM-DD] [--out trims.jsonl]
"""
import argparse
import json
import re
import sys
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--date", default=None,
                help="date of the log's first line (default: today); day rollovers "
                     "inside the log are handled by timestamp monotonicity")
ap.add_argument("--out", default="-")
args = ap.parse_args()

day0 = datetime.strptime(args.date, "%Y-%m-%d") if args.date else \
    datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
day0 = day0.timestamp()

stamp = re.compile(r"^\[broker (\d\d):(\d\d):(\d\d\.\d+)\] (.*)$")
car = re.compile(r"PRN (\d+) resid [+-][\d.]+ Hz trim ([+-][\d.]+)")
first_seed = re.compile(r"PRN (\d+) FIRST SEED .* trim ([+-][\d.]+)")

rows = []
last_t = 0.0
day_off = 0.0
for line in open(args.log, errors="replace"):
    m = stamp.match(line)
    if not m:
        continue
    t = day0 + day_off + int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if t < last_t - 43200:      # clock wrapped past midnight
        day_off += 86400.0
        t += 86400.0
    last_t = t
    body = m.group(4)
    if body.startswith("CAR:"):
        for pm in car.finditer(body):
            rows.append({"t": round(t, 3), "prn": int(pm.group(1)),
                         "trim_hz": float(pm.group(2))})
    else:
        fm = first_seed.search(body)
        if fm:
            rows.append({"t": round(t, 3), "prn": int(fm.group(1)),
                         "trim_hz": float(fm.group(2)), "reseed": True})

rows.sort(key=lambda r: r["t"])
out = sys.stdout if args.out == "-" else open(args.out, "w")
for r in rows:
    out.write(json.dumps(r) + "\n")
if out is not sys.stdout:
    out.close()
    n_prn = len(set(r["prn"] for r in rows))
    print("%d trim samples, %d sats -> %s" % (len(rows), n_prn, args.out), file=sys.stderr)
