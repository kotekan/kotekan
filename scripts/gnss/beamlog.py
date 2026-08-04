#!/usr/bin/env python3
"""Log every detected satellite's strength against its sky position -- the actual beam map.

WHY, 2026-08-04. Everything in the lock chase has rested on a premise: that CHORD's satellites
are weak because they sit in deep dish sidelobes (~-40 dB), and that a near-boresight pass would
be ~10^4x stronger and lock trivially. PRN 19 passed 0.40 deg off the assumed boresight at 08:04
UTC and peaked at search snr 74, while PRN 4 and PRN 9 reached 10000-12000 at 5-13 deg off, and
PRN 28 reaches thousands at 4-22 deg ELEVATION. Separation from boresight does not predict
strength, so either the pointing is not what we think, the tap is not seeing the dish beam, or
the sidelobe model is wrong -- and all three change what the rest of the work list means.

This logs the raw material to settle it: for every detection, its search SNR and the fleet's
combined power, paired IN TIME with its almanac az/el and its separation from the assumed
boresight. Hours of that is a beam map, which is the instrument this whole project exists to
build -- so it is worth having regardless of how the lock question resolves.

Deliberately records az/el rather than only the separation, so the beam can be FITTED (where
does the response actually peak?) instead of only tested against an assumed centre.

usage:  beamlog.py [--interval 20] [--out FILE]
"""
import argparse
import datetime as dt
import json
import math
import sys
import time
import urllib.request

sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")
import gnss_ephemeris as ge

LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
BOR_AZ, BOR_EL = 180.0, 81.41   # dishes 8.59 deg south of zenith; ASSUMED, and under test here

ap = argparse.ArgumentParser()
ap.add_argument("--interval", type=float, default=20.0)
ap.add_argument("--out", default="/tmp/beamlog.jsonl")
ap.add_argument("--search", default="http://localhost:12050/gps_search/get_detections")
ap.add_argument("--fleet", default="http://localhost:12060/get_status")
a = ap.parse_args()

eph = ge.parse_rinex_nav(ge.fetch_brdc(dt.datetime.now(dt.timezone.utc)))
eph_t = time.time()


def sep(az, el):
    a1, e1, a2, e2 = map(math.radians, (az, el, BOR_AZ, BOR_EL))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def get(url):
    try:
        return json.load(urllib.request.urlopen(url, timeout=8))
    except Exception:
        return None


print("logging to %s every %.0f s" % (a.out, a.interval))
while True:
    t = time.time()
    if t - eph_t > 3 * 3600:          # refresh BRDC so the geometry never goes stale
        try:
            eph = ge.parse_rinex_nav(ge.fetch_brdc(dt.datetime.now(dt.timezone.utc)))
            eph_t = t
        except Exception:
            pass
    dets = get(a.search) or []
    fleet = {int(r["prn"]): r for r in (get(a.fleet) or [])}
    try:
        pos = ge.predict_all(eph, LAT, LON, ALT, t, mask_deg=-90.0, max_age=12 * 3600)
    except Exception:
        pos = {}
    n = 0
    with open(a.out, "a") as fh:
        for r in dets:
            prn = int(r["prn"])
            v = pos.get(("G", prn))
            if not v:
                continue
            f = fleet.get(prn, {})
            fh.write(json.dumps({
                "t": t, "prn": prn,
                "search_snr": r.get("snr"), "nh": r.get("nh"),
                "az": v["az"], "el": v["el"], "sep": sep(v["az"], v["el"]),
                "range_m": v["range_m"],
                # fleet-combined, full 20.46 MHz -- the honest strength measure
                "fleet_amp_snr": f.get("amp_snr"),
                "fleet_p_over_noise": f.get("fleet_p_over_noise"),
                "fleet_q": f.get("fleet_q"), "fleet_instances": f.get("fleet_instances"),
            }) + "\n")
            n += 1
    print("%s  %d sats logged" % (dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S"), n))
    sys.stdout.flush()
    time.sleep(max(1.0, a.interval - (time.time() - t)))
