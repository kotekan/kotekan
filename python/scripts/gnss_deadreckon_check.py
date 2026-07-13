#!/usr/bin/env python3
"""Dead-reckoning validation + receiver-time bootstrap: compare the LIVE search's measured
code phases against broadcast-ephemeris predictions across every tracked satellite.

    measured_cp0 - predicted_cp0  =  chip_rate * (receiver clock error)   [mod code period]

so per constellation the differences should be CONSTANT across sats: the constant is the
receiver clock (the one-time bootstrap that lets everything else dead-reckon), and the
per-sat scatter is the honest error budget (BRDC orbit ~2 m + sat clock ~5 ns + iono/tropo
~10-30 m + measurement noise => expect well under 1 chip). The fractional-millisecond part
of the clock must also AGREE ACROSS CONSTELLATIONS (independent code periods 1/4/10 ms) --
a strong internal consistency check.

Usage: python3 gnss_deadreckon_check.py [--url http://localhost:12048]
"""
import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gnss_ephemeris import (fetch_brdc, parse_rinex_nav, predict_all, gpst_of_utc,
                            C_LIGHT, GPS_UTC_LEAP)

CHIP = 1.023e6
SEARCHES = [("G", "search", 1023, 1e-3), ("E", "gal_search", 4092, 4e-3),
            ("C", "bds_search", 10230, 10e-3)]


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--airspy", default="airspy_in")
    args = ap.parse_args()

    utc0 = get("%s/%s/adcstat" % (args.url, args.airspy))["utc0_sample0"]
    eph = parse_rinex_nav(fetch_brdc())
    t0 = datetime.fromtimestamp(utc0, tz=timezone.utc)
    frac_ms = {}
    for tag, stage, L, t_code in SEARCHES:
        try:
            dets = get("%s/%s/get_detections" % (args.url, stage))
        except Exception as e:
            print("%s: no detections (%s)" % (tag, e))
            continue
        rows = []
        for d in dets:
            key = (tag, int(d["prn"]))
            if d.get("snr", 0) < 10:
                continue
            # The detection's cp0 is a CURRENCY construct: the physical measurement,
            # back-referenced to sample 0 through the CURRENT doppler held over the whole
            # capture age -- per-sat errors of hundreds of chips at multi-hour ages. UNDO
            # it: the physical code phase AT THE DETECTION HOP is
            #   cp_local = cp0 + t_abs * f_chip * sign * dop / f_carrier   (mod L)
            # and the prediction is evaluated at that same instant. The receiver sample
            # clock's cumulative drift (l-a, ~ms over hours) is COMMON across sats and
            # lands in the solved clock constant, as it should.
            t_abs = d["ref_hop"] / 1e6            # hops_per_sec = 1e6 at the 20 MSPS front end
            cp_local = (d["code_phase_chips"]
                        + t_abs * CHIP * d["doppler_hz"] / 1575.42e6) % L
            t_i = datetime.fromtimestamp(utc0 + t_abs, tz=timezone.utc)
            pred = predict_all(eph, args.lat, args.lon, args.alt, t_i, mask_deg=0.0)
            if key not in pred:
                continue
            v = pred[key]
            t_tx_sv = gpst_of_utc(utc0 + t_abs) - v["range_m"] / C_LIGHT + v["sat_clk_s"]
            cp_pred = (t_tx_sv % t_code) / t_code * L
            dcp = (cp_local - cp_pred) % L
            rows.append((d["prn"], d["snr"], cp_local, cp_pred, dcp))
        if len(rows) < 2:
            print("%s: only %d usable detections" % (tag, len(rows)))
            continue
        # Circular median of the offsets (they live mod L).
        ref = rows[0][4]
        cent = [((r[4] - ref + L / 2) % L) - L / 2 for r in rows]
        med = sorted(cent)[len(cent) // 2] + ref
        print("== %s: %d sats ==" % (tag, len(rows)))
        for prn, snr, meas, cp_pred, dcp in sorted(rows):
            resid = ((dcp - med + L / 2) % L) - L / 2
            print("  PRN %2d snr %5.1f  meas %8.2f  pred %8.2f  offset %9.2f  resid %+7.2f chips"
                  % (prn, snr, meas, cp_pred, dcp, resid))
        rms = math.sqrt(sum((((r[4] - med + L / 2) % L) - L / 2) ** 2
                            for r in rows) / len(rows))
        clk_us = med / CHIP * 1e6
        frac_ms[tag] = med / CHIP * 1e3 % 1.0
        print("  -> common offset %.2f chips = receiver clock %.3f us (mod %.0f ms); "
              "scatter rms %.2f chips (%.0f m)"
              % (med, clk_us, t_code * 1e3, rms, rms / CHIP * C_LIGHT))
    if len(frac_ms) >= 2:
        print("cross-constellation clock consistency (fractional ms, must agree): %s"
              % ", ".join("%s=%.4f" % kv for kv in sorted(frac_ms.items())))


if __name__ == "__main__":
    main()
