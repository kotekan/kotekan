#!/usr/bin/env python3
"""Predicted B1C overlay indices at the 07-18 17:37 capture epoch, differenced vs C19.

k_i = round((gpst - range_i/c + clk_i)/0.01) mod 1800.  The capture start is only known
to ~seconds (file mtime), which shifts all k_i commonly (100 chips/s); DIFFERENCES
k_i - k_19 change by <0.01 chip over that uncertainty, so they are the observable to
compare against the soft-scan's measured lag differences.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge

C = 299792458.0
PERIOD = 0.01
SEC_LEN = 1800
REF = 19

t0 = datetime(2026, 7, 18, 21, 37, 30, tzinfo=timezone.utc)   # + T_SKIP is common-mode
eph = ge.parse_rinex_nav(ge.fetch_brdc(t0))
pa = ge.predict_all(eph, 43.968697, -79.252106, 260.0, t0, mask_deg=-90.0)
gpst = ge.gpst_of_utc(t0.timestamp())

ks = {}
for (s, prn), v in sorted(pa.items()):
    if s != "C" or prn < 19:
        continue
    if v["el"] < 0:
        continue
    ks[prn] = ((gpst - v["range_m"] / C + v["sat_clk_s"]) / PERIOD) % SEC_LEN
    print("C%-3d el %5.1f   k = %8.2f" % (prn, v["el"], ks[prn]))

if REF in ks:
    print("\ndifferences vs C%d (the epoch-free observable):" % REF)
    for prn, k in sorted(ks.items()):
        print("  C%-3d  dk = %+8.2f  (mod 1800: %7.2f)"
              % (prn, k - ks[REF], (k - ks[REF]) % SEC_LEN))
