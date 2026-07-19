#!/usr/bin/env python3
"""Summarize a replay leg: per-sat deep/coherence timeline + broker event counts.
Usage: leg_report.py <legname> [prn ...]"""
import json
import os
import re
import sys

SP = os.environ.get("REPLAY_OUT", "/tmp/replay_legs")
leg = sys.argv[1]
focus = [int(a) for a in sys.argv[2:]]

raw = open("%s/replay_%s_status.jsonl" % (SP, leg)).read()
recs = [r for r in re.split(r'(?=\{"t":1)', raw) if r.strip()]
t0 = None
per = {}
for rec in recs:
    try:
        d = json.loads(rec)
    except Exception:
        continue
    t0 = t0 or d["t"]
    dt = d["t"] - t0
    for r in d.get("status") or []:
        p = r["prn"]
        e = per.setdefault(p, {"deep_max": 0.0, "coh_n": 0, "first_coh": None,
                               "cn0_max": 0.0, "amp_max": 0.0, "last_coh": None})
        deep = r.get("deep_snr") or 0
        e["deep_max"] = max(e["deep_max"], deep)
        e["amp_max"] = max(e["amp_max"], r.get("amp_snr") or 0)
        e["cn0_max"] = max(e["cn0_max"], r.get("cn0_dbhz") or 0)
        if (r.get("coherence_s") or 0) > 0:
            e["coh_n"] += 1
            if e["first_coh"] is None:
                e["first_coh"] = dt
            e["last_coh"] = dt
print("%d snapshots" % len(recs))
print("  PRN  deep_max amp_max cn0_max  coh_n  first_coh  last_coh")
for p in sorted(per):
    e = per[p]
    if e["deep_max"] < 7 and not e["coh_n"] and p not in focus:
        continue
    print("  C%-3d %7.1f %7.1f %7.1f  %5d   %8s  %8s"
          % (p, e["deep_max"], e["amp_max"], e["cn0_max"], e["coh_n"],
             "%.0fs" % e["first_coh"] if e["first_coh"] is not None else "--",
             "%.0fs" % e["last_coh"] if e["last_coh"] is not None else "--"))
log = open("%s/replay_%s_broker.log" % (SP, leg)).read()
for pat in ("WATCHDOG RESEED", "CARRIER ALIAS REACQ", "CARRIER REACQ", "ESCAPE PRN",
            "TRIM FORCE", "FIRST SEED", "nh-assist POST failed"):
    n = log.count(pat)
    if n:
        print("broker: %-22s x%d" % (pat, n))
for line in log.splitlines():
    if "WATCHDOG RESEED" in line or "TRIM FORCE" in line:
        print("   ", line.strip()[:120])
