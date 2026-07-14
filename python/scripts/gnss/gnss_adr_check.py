#!/usr/bin/env python3
"""ACCEPTANCE TEST for the accumulated-carrier-phase observable.

The claim being tested: adr_cycles is the true received carrier phase, continuously
accumulated. If so, its TIME DERIVATIVE must equal the satellite's Doppler -- and we have an
utterly independent prediction of that from the broadcast ephemeris (range-rate / lambda).
So:

    d(adr)/dt   vs   -range_rate / lambda + f_clock

agreed to a fraction of a Hz is a hard, external confirmation that (a) the commanded-phase
export is right, (b) the f_ref re-pin steps genuinely cancel, (c) the bit/overlay-robust
increment isn't slipping, and (d) the sign convention is what we think it is.

f_clock (the receiver's frequency offset, common to every satellite) is solved here as the
mean discrepancy -- it is real and expected (a free-running TCXO is ~1e-6 of 1.5 GHz = kHz).
What must be small is the per-satellite SCATTER about it: that is the actual measurement.

Sanity thresholds: scatter << 1 Hz means the phase is tracking. A satellite whose residual is
metres-per-second off is slipping cycles. A residual that grows linearly with arc length is a
frequency error, not a phase error.

Usage: python3 gnss_adr_check.py [--combiner gps_combiner] [--secs 30]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_stages import resolve_stage, capture_clock
from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all, C_LIGHT


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def snap(url, stage, sysletter, eph, lat, lon, alt, to_unix):
    """One sample per PRN: (adr, arc, CAPTURE utc, predicted range-rate AT THAT utc, sig).

    The time base is the emit's CAPTURE UTC, never wall-clock-at-poll. Pipeline latency and
    emit-cadence jitter of even 0.1 s divide a 2 kHz Doppler into a ~6 Hz phantom residual --
    which is precisely what the first run of this test reported, before the combiner exported
    the epoch. The phase and the geometry must be evaluated at the SAME instant, and that
    instant belongs to the data, not to the poller."""
    st = get("%s/%s/get_status" % (url, stage))
    out = {}
    for r in st:
        prn = int(r.get("prn") or 0)
        utc = r.get("utc") or 0.0
        if not prn or utc <= 0.0 or (r.get("adr_records") or 0) < 2:
            continue
        utc = to_unix(utc)   # capture clock -> unix, so the geometry is evaluated at the
                             # same instant the phase was measured (gnss_stages.capture_clock)
        v = predict_all(eph, lat, lon, alt, datetime.fromtimestamp(utc, tz=timezone.utc),
                        mask_deg=-90.0).get((sysletter, prn))
        if v is None:
            continue
        sig = (max(r.get("deep_snr") or 0.0, r.get("amp_snr") or 0.0)
               if (r.get("coherence_s") or 0) > 0 else (r.get("amp_snr") or 0.0))
        out[prn] = (r.get("adr_cycles") or 0.0, r.get("adr_arc") or 0, utc,
                    v["range_m"], v["sat_clk_s"], sig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--combiner", default="gps_combiner")
    ap.add_argument("--sys", default="G", choices=("G", "E", "C"))
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6)
    ap.add_argument("--secs", type=float, default=30.0, help="baseline over which to difference")
    ap.add_argument("--min-sig", type=float, default=10.0,
                    help="skip sats below this significance: an unlocked PRN's despread is "
                         "noise, and the phase of noise is a random walk -- including them "
                         "measures the noise floor, not the instrument")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    args = ap.parse_args()

    stage = resolve_stage(args.url, args.combiner)
    carrier = args.carrier_hz
    lam = C_LIGHT / args.carrier_hz
    eph = parse_rinex_nav(fetch_brdc())
    to_unix = capture_clock(args.url)

    a0 = snap(args.url, stage, args.sys, eph, args.lat, args.lon, args.alt, to_unix)
    print("baseline %.0f s on %s (%d sats with phase)..." % (args.secs, stage, len(a0)))
    time.sleep(args.secs)
    a1 = snap(args.url, stage, args.sys, eph, args.lat, args.lon, args.alt, to_unix)

    rows = []
    for prn in sorted(set(a0) & set(a1)):
        adr0, arc0, utc0, rng0, clk0, _ = a0[prn]
        adr1, arc1, utc1, rng1, clk1, sig = a1[prn]
        if arc0 != arc1:
            print("  PRN %2d  ARC BROKE during the baseline (%d -> %d) -- excluded"
                  % (prn, arc0, arc1))
            continue
        dt = utc1 - utc0                                  # CAPTURE-time baseline (see snap())
        if dt <= 0:
            continue
        if sig < args.min_sig:
            continue    # no signal -> the "phase" is the phase of noise, and means nothing
        f_meas = (adr1 - adr0) / dt                       # Hz, straight from the phase
        # PREDICTED phase change = -(delta range)/lambda + f*(delta sat clock). Use the RANGE
        # DIFFERENCE, not the mean of the endpoint range-RATES: the Doppler curves, so the
        # trapezoid of its endpoints misses the true mean by ~f''T^2/12 -- several Hz over a
        # 90 s baseline on an MEO, which is larger than the effect being measured. The range
        # difference has no quadrature error at all; it is the integral, exactly.
        dphi_pred = -(rng1 - rng0) / lam + (clk1 - clk0) * carrier
        f_pred = dphi_pred / dt
        rows.append((prn, f_meas, f_pred, f_meas - f_pred, sig))

    if len(rows) < 2:
        print("not enough continuous arcs to test (%d)" % len(rows))
        return 1
    # The receiver's own frequency offset is common to all sats: solve and remove it.
    clk = statistics.median(r[3] for r in rows)
    print("\n  PRN   f_meas(ADR)   f_pred(BRDC)      resid    resid-clk     sig")
    resid = []
    for prn, fm, fp, d, sig in rows:
        r = d - clk
        resid.append(r)
        flag = "   <-- SLIPPING" if abs(r) > 1.0 else ""
        print("  %3d  %+11.3f  %+12.3f  %+10.3f  %+10.3f  %6.1f%s"
              % (prn, fm, fp, d, r, sig, flag))
    rms = math.sqrt(sum(x * x for x in resid) / len(resid))
    print("\n  receiver clock offset %+.3f Hz (%.3f ppm of %.3f MHz)"
          % (clk, clk / args.carrier_hz * 1e6, args.carrier_hz / 1e6))
    print("  per-sat scatter about it: %.4f Hz rms  (= %.2f mm/s of range rate)"
          % (rms, rms * lam * 1e3))
    print("  VERDICT: %s" % ("PHASE IS TRACKING (scatter << 1 Hz)" if rms < 1.0
                             else "FAIL -- cycle slips or a broken convention"))
    return 0 if rms < 1.0 else 2


if __name__ == "__main__":
    sys.exit(main())
