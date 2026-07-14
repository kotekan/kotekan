#!/usr/bin/env python3
"""Receiver-time bootstrap + dead-reckoning validation: compare LIVE tracked code phases
against broadcast-ephemeris predictions. Each locked sat measures the receiver clock
independently:

    measured_cp - predicted_cp = chip_rate * (rx_clock_err)  (mod code period)
                                   + per-sat model/tracking residual

so across sats the differences must CLUSTER at one common offset; the cluster median IS
the receiver clock error (the one-time bootstrap that makes full code dead-reckoning
possible), the scatter is the joint eph+tracking quality, and outliers are the integrity
report (stale eph / maneuver / false lock). Cross-constellation code periods (1/4/10 ms)
extend the clock ambiguity: agreement across G+E+C pins rx_clock mod 10 ms.

Conventions (pipeline internal vs physical) are resolved EMPIRICALLY: both cp signs are
tried and the tighter cluster wins -- same approach that validated the peel bench.

Usage: python3 gnss_time_bootstrap.py   (needs the live pipeline on :12048)
"""
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all, gpst_of_utc, \
    best_eph, sat_pos_clk, C_LIGHT

LAT, LON, ALT = 43.968697, -79.252106, 260.0
REST = "http://localhost:12048"
CHAINS = [("G", "combiner", 1023, 1.023e6, 1e-3),
          ("E", "gal_combiner", 4092, 1.023e6, 4e-3),
          ("C", "bds_combiner", 10230, 1.023e6, 10e-3)]


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    adc = get(REST + "/airspy_in/adcstat")
    utc0 = adc["utc0_sample0"]
    print("capture utc0_sample0 = %.6f (host-clock anchored: ms-accurate)" % utc0)
    eph = parse_rinex_nav(fetch_brdc())
    now = datetime.now(timezone.utc)
    t_now = now.timestamp()
    pred = predict_all(eph, LAT, LON, ALT, now, mask_deg=0.0)

    # sample index 'now' (approx; the smooth l-a corrected sample clock is close enough
    # for a chips-level check): hop rate 1e6, fs 20e6
    fs = 20e6
    n_now = (t_now - utc0) * fs

    rows = []
    for tag, comb, L, chip, t_rec in CHAINS:
        try:
            status = get("%s/%s/get_status" % (REST, comb))
        except Exception:
            continue
        for r in status:
            prn = int(r.get("prn") or 0)
            amp = r.get("amp_snr") or 0
            if prn <= 0 or amp < 30:
                continue
            key = (tag, prn)
            if key not in pred:
                continue
            p = pred[key]
            # measured cp at 'now': the status cp is the seed cp0 (sample-0 anchored, seed
            # currency); the tracked code phase at sample n advances at the doppler-scaled
            # chip rate. For a chips-level cluster test, propagate at the nominal rate +
            # geometric code doppler from the PREDICTED range rate (exact enough over the
            # sub-second status latency).
            cp0 = float(r.get("code_phase_chips") or 0.0)
            dop_geo = -p["range_rate_mps"] / C_LIGHT * 1.023e6  # chips/s geometric
            cp_meas = (cp0 + (chip + dop_geo) * (n_now / fs)) % L
            # predicted: transmit-time code phase of the signal arriving at sample n_now:
            # t_tx(sat time) = t_rx_gpst - tau - sat_clk ; code phase = frac of code period
            t_rx = gpst_of_utc(t_now)
            t_tx_sv = t_rx - p["range_m"] / C_LIGHT - p["sat_clk_s"]
            cp_pred = (t_tx_sv % (L / chip)) * chip
            rows.append((tag, prn, amp, cp_meas, cp_pred, L))

    if not rows:
        print("no locked sats above amp 30 -- is the pipeline up?")
        return
    print("%d locked sats with predictions" % len(rows))
    for sign in (+1, -1):
        print("=== convention: measured_cp sign %+d ===" % sign)
        # per-constellation clusters (each has its own period ambiguity)
        for tag in "GEC":
            sel = [(p_, a, (sign * m - q) % L_) for (t_, p_, a, m, q, L_) in rows
                   if t_ == tag]
            if len(sel) < 2:
                continue
            # circular clustering: histogram the offsets mod L
            L_ = [x[2] for x in sel]
            offs = sorted(L_)
            med = offs[len(offs) // 2]
            # wrapped deviation from the median
            per = {"G": 1023, "E": 4092, "C": 10230}[tag]
            dev = [((o - med + per / 2) % per) - per / 2 for o in offs]
            spread = sorted(abs(d) for d in dev)[len(dev) // 2]
            print("  %s: n=%d  cluster median %8.2f chips (= %.3f ms rx clock)  "
                  "median |dev| %7.2f chips" % (tag, len(sel), med,
                                                med / 1.023e6 * 1e3, spread))
            for (p_, a, o) in sorted(sel, key=lambda x: -x[1])[:8]:
                d = ((o - med + per / 2) % per) - per / 2
                print("     PRN %2d amp %5.0f  offset %8.2f  dev %+8.2f chips"
                      % (p_, a, o, d))


if __name__ == "__main__":
    main()
