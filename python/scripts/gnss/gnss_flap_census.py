#!/usr/bin/env python3
"""FLAP CENSUS: classify every coherence-loss event by cause, one line per death.

The 2026-07-19 methodology shift (Keith): every tracking failure this week traced to a
DISCRETE mechanism visible in +-10 s of context -- rate deltas never diagnosed anything.
So: stop integrating rates; name every death. An UNKNOWN row is an immediate autopsy
target with its evidence already captured; a rate is just the census total.

Classes (from the week's confirmed taxonomy):
  SNR-FADE    amp and detection both gone -- physics (fade/set), not a bug
  RELEASE     hold release within the window (seed-dop step; carB kills its carrier step)
  ESCAPE      escape re-anchor within the window (referee accusation)
  REACQ       carrier demotion within the window (recovery machinery, not a cause per se)
  WATCHDOG    watchdog re-seed within the window
  CARRIER-STEP resid stepped > 3 Hz at death with amp surviving (un-precomped step class)
  LATCH       deep at floor, amp or detection still strong, resid quiet -- the
              strong-search/no-track class (alias capture / walked trim / overlay mislock)
  EPHEMERIS   prediction loss markers near the event (BRDC cliff class)
  UNKNOWN     none of the above -- the interesting ones

Usage:
  gnss_flap_census.py --status <status_log.jsonl> [--broker <broker.log>] \
      [--since HH:MM] [--min-sig 8] [--window 6]
Broker log must be the timestamped format (2ec49e04) for event correlation.
"""
import argparse
import json
import re
import statistics
import sys
from datetime import datetime, date

ap = argparse.ArgumentParser()
ap.add_argument("--status", required=True)
ap.add_argument("--broker", default=None)
ap.add_argument("--since", default=None, help="HH:MM today")
ap.add_argument("--min-sig", type=float, default=8.0,
                help="only census deaths of sats that were at least this significant "
                     "(amp or deep) in the 10 s before the death")
ap.add_argument("--window", type=float, default=6.0)
ap.add_argument("--resid-step-hz", type=float, default=3.0)
args = ap.parse_args()

day0 = datetime.combine(date.today(), datetime.min.time()).timestamp()
t_cut = 0.0
if args.since:
    hh, mm = args.since.split(":")
    t_cut = day0 + int(hh) * 3600 + int(mm) * 60

rows = {}
for line in open(args.status):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("t", 0) < t_cut or "prn" not in d:
        continue
    rows.setdefault(d["prn"], []).append(d)

events = []   # (t, kind, prn)
if args.broker:
    pat = re.compile(r"\[broker (\d\d):(\d\d):(\d\d\.\d+)\] "
                     r"(RELEASE|ESCAPE|CARRIER REACQ|CARRIER ALIAS REACQ|WATCHDOG RESEED"
                     r"|drop|PREDICTION COLLAPSE)[ :]*(?:PRN )?(\d+)?")
    for line in open(args.broker, errors="replace"):
        m = pat.search(line)
        if m:
            t = (day0 + int(m.group(1)) * 3600 + int(m.group(2)) * 60
                 + float(m.group(3)))
            if t >= t_cut:
                events.append((t, m.group(4), int(m.group(5)) if m.group(5) else -1))

census = []
for prn, r in sorted(rows.items()):
    r.sort(key=lambda d: d["t"])
    for i in range(1, len(r)):
        was = (r[i - 1].get("coherence_s") or 0) > 0
        now = (r[i].get("coherence_s") or 0) > 0
        if not (was and not now):
            continue
        t = r[i]["t"]
        pre = [x for x in r if t - 10 <= x["t"] < t]
        post = [x for x in r if t <= x["t"] <= t + args.window]
        sig_pre = max((max(x.get("amp_snr") or 0, x.get("deep_snr") or 0) for x in pre),
                      default=0)
        if sig_pre < args.min_sig:
            continue
        amp_post = max((x.get("amp_snr") or 0 for x in post), default=0)
        resid_pre = statistics.median([abs(x.get("carrier_hz_resid") or 0) for x in pre]) \
            if pre else 0
        resid_post = max((abs(x.get("carrier_hz_resid") or 0) for x in post), default=0)
        near = [(k, p) for (te, k, p) in events
                if abs(te - t) <= args.window and p in (prn, -1)]
        cls = None
        if any(k == "PREDICTION COLLAPSE" or (k == "drop" and p == -1) for k, p in near):
            cls = "EPHEMERIS"
        elif any(k == "WATCHDOG RESEED" for k, p in near):
            cls = "WATCHDOG"
        elif any(k == "ESCAPE" for k, p in near):
            cls = "ESCAPE"
        elif any(k == "RELEASE" for k, p in near):
            cls = "RELEASE"
        elif resid_post - resid_pre > args.resid_step_hz and amp_post > 5:
            cls = "CARRIER-STEP"
        elif any("REACQ" in k for k, p in near):
            cls = "REACQ"
        elif amp_post < 3:
            cls = "SNR-FADE"
        elif amp_post > args.min_sig and resid_post < 2.0:
            cls = "LATCH"
        else:
            cls = "UNKNOWN"
        census.append((t, prn, cls, sig_pre, amp_post, resid_pre, resid_post))
        print("%s  PRN %-3d %-12s sig_pre %6.1f  amp_post %6.1f  resid %5.2f->%5.2f"
              % (datetime.fromtimestamp(t).strftime("%H:%M:%S"), prn, cls,
                 sig_pre, amp_post, resid_pre, resid_post))

print("\nCENSUS (%d deaths):" % len(census))
hist = {}
for _, _, cls, *_ in census:
    hist[cls] = hist.get(cls, 0) + 1
for cls, n in sorted(hist.items(), key=lambda kv: -kv[1]):
    print("  %-12s %4d  (%d%%)" % (cls, n, 100 * n // max(1, len(census))))
unk = [c for c in census if c[2] == "UNKNOWN"]
if unk:
    print("UNKNOWNS (autopsy targets): %s"
          % ", ".join("PRN%d@%s" % (c[1], datetime.fromtimestamp(c[0]).strftime("%H:%M:%S"))
                      for c in unk[:10]))
