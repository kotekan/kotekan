#!/usr/bin/env python3
"""Log the fleet-combined per-PRN state at high cadence across a satellite pass.

WHY: a snapshot cannot distinguish "weak signal" from "wrong code phase". The SHAPE across a
pass can. As a satellite climbs toward boresight its power rises by a known amount; if q and
the discriminator follow it into a lock, the limit was sensitivity, and if p_pow climbs by
orders of magnitude while q stays pinned at 1.0, the tracker is in the wrong place and no
amount of signal will fix it (see docs/CHORD_GNSS_SHARED_DLL.md section 8a).

Samples the SAME fleet_dll() the broker closes its loop with, so the log and the loop cannot
disagree, plus the boresight separation from the almanac so power can be plotted against
geometry rather than against wall clock.

usage:  passwatch.py --prn 3 --minutes 40 [--interval 5] [--out FILE]
"""
import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import time

K = "/home/kvand/gnss/kotekan"
sys.path.insert(0, K + "/python/scripts/gnss")
spec = importlib.util.spec_from_file_location("b", K + "/python/scripts/gnss/gps_distributed_broker.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b._log_rl = lambda k, m: None

import gnss_ephemeris as ge

LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
BOR_AZ, BOR_EL = 180.0, 90.0 - 8.59      # dishes 8.59 deg south of zenith; see the memory note
EPS = (["http://cx19:12049/gnss0_combine", "http://cx51:12049/gnss0_combine"]
       + ["http://%s:12049/gnss%d_combine" % (n, g)
          for n in ("cx27", "cx42", "cx43", "cx44", "cx47", "cx52") for g in (0, 1)])

ap = argparse.ArgumentParser()
ap.add_argument("--prn", type=int, required=True)
ap.add_argument("--minutes", type=float, default=40.0)
ap.add_argument("--interval", type=float, default=5.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out_path = a.out or "/tmp/passwatch_prn%d.jsonl" % a.prn

eph = ge.parse_rinex_nav(ge.fetch_brdc(dt.datetime.now(dt.timezone.utc)))


def sep_now(t):
    try:
        v = ge.predict_all(eph, LAT, LON, ALT, t, mask_deg=-90.0, max_age=6 * 3600).get(("G", a.prn))
    except Exception:
        return None, None, None
    if not v:
        return None, None, None
    a1, e1, a2, e2 = map(math.radians, (v["az"], v["el"], BOR_AZ, BOR_EL))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))), v["el"], v["az"]


print("PRN %d -> %s  (%.0f min at %.0fs)" % (a.prn, out_path, a.minutes, a.interval))
print("%-8s %6s %6s %8s %8s %9s %6s  %s"
      % ("utc", "sep", "el", "p/noise", "q", "disc", "n_src", "state"))
t_end = time.time() + a.minutes * 60.0
with open(out_path, "a") as fh:
    while time.time() < t_end:
        t = time.time()
        sep, el, az = sep_now(t)
        try:
            fleet = b.fleet_dll(EPS, 976562, 2, 3.0, 2.2)
        except Exception as e:
            fleet = {}
        v = fleet.get(a.prn)
        rec = {"t": t, "prn": a.prn, "sep_deg": sep, "el_deg": el, "az_deg": az}
        if v:
            rec.update({k: v[k] for k in ("q", "disc", "p_pow", "p_med", "n_src", "n_chan",
                                          "hop", "present", "q_floor")})
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if v:
            print("%-8s %6.2f %6.1f %7.1fx %8.3f %+9.4f %6d  %s"
                  % (dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S"),
                     sep if sep is not None else float("nan"),
                     el if el is not None else float("nan"),
                     v["p_pow"] / v["p_med"] if v.get("p_med") else 0.0,
                     v["q"], v["disc"], v["n_src"],
                     "PRESENT" if v["present"] else ""))
        else:
            print("%-8s %6.2f %6.1f %s"
                  % (dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S"),
                     sep if sep is not None else float("nan"),
                     el if el is not None else float("nan"), "  -- not in the fleet set"))
        sys.stdout.flush()
        time.sleep(max(0.5, a.interval - (time.time() - t)))
