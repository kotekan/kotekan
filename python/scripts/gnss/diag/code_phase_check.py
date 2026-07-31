#!/usr/bin/env python3
"""Code-phase observable check: SEARCH cp0 (control) vs COMBINER code_phase_chips (device under test).

Both are run through the SAME orbit-cancelled reconstruction the TEC chain uses
(gnss_observables.cn0/code_resid): propagate the reported code phase to its epoch, subtract the
broadcast-ephemeris prediction, remove the common receiver clock (across-sat median), and look at
the per-satellite residual. What's left should be ionosphere + multipath -- a few metres.

    cp_local   = (cp + t_abs * f_chip * (1 + dop/f_carrier)) mod L      # physical code phase
    cp_pred    = (t_tx mod T_code)/T_code * L                           # from the orbit
    residual   = wrap(cp_local - cp_pred) - median(all sats)            # -> chips (metres)

The SEARCH's cp0 is back-referenced to sample 0 (search subtracts the nominal advance + Doppler
drift), so it reconstructs CLEAN -- it is the positive control that proves the reconstruction,
orbit, epoch handling and constants are all correct. The COMBINER's code_phase_chips is REC_CP,
the tracker's "commanded prompt code phase" at the tracking WINDOW (its job is to command the
despread), which is NOT the same sample-0 currency -- so it reconstructs with a large per-sat
scatter (measured ~90 chips / 25 km at L1, ~150 km at L2C, 2026-07-30). That scatter is the
window->sample-0 back-reference the observable never undoes, and it is why absolute VTEC is blocked.

Use this to (a) reproduce the split and (b) regression-check any fix: a fix that gives the combiner
a proper sample-0 cp0 currency should bring the COMBINER rms down to the SEARCH rms (< ~1 chip).

Usage:
    python3 code_phase_check.py                       # GPS L1 on the merged 3-band node
    python3 code_phase_check.py --band l2c --sys G --code-length 20460 --t-code 20e-3 \
            --airspy l2c_airspy_in --search-stage l2c_gps_search --combiner-stage l2c_gps_combiner
"""
import argparse
import json
import math
import os
import sys
import urllib.request
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnss_ephemeris import (fetch_brdc, parse_rinex_nav, predict_all, gpst_of_utc, C_LIGHT)

CHIP = 1.023e6  # GPS/GAL/BDS civil chip rate (Hz)


def get(url):
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.loads(r.read().decode())


def reconstruct(cp, dop, t_abs, prn, eph, utc0, args):
    """physical code phase at t_abs, minus the orbit prediction -> chips (pre-clock-removal)."""
    L = args.code_length
    cp_local = (cp + t_abs * CHIP * (1.0 + dop / args.carrier_hz)) % L
    t_unix = utc0 + t_abs
    when = dt.datetime.fromtimestamp(t_unix, tz=dt.timezone.utc)
    pred = predict_all(eph, args.lat, args.lon, args.alt, when, mask_deg=0.0)
    v = pred.get((args.sys, prn))
    if not v:
        return None
    t_tx = gpst_of_utc(t_unix) - v["range_m"] / C_LIGHT + v["sat_clk_s"]
    cp_pred = (t_tx % args.t_code) / args.t_code * L
    return ((cp_local - cp_pred) % L)


def center(rows):
    """remove the common receiver clock (circular median); return {prn: resid_chips}, median."""
    if len(rows) < 2:
        return {}, 0.0
    L = _L
    vals = list(rows.values())
    ref = vals[0]
    cent = sorted(((v - ref + L / 2) % L) - L / 2 for v in vals)
    med = cent[len(cent) // 2] + ref
    return {p: (((v - med + L / 2) % L) - L / 2) for p, v in rows.items()}, med


def rms(d):
    if len(d) < 2:
        return float("nan")
    m = sum(d.values()) / len(d)
    return math.sqrt(sum((x - m) ** 2 for x in d.values()) / len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--band", default="l1", help="informational label")
    ap.add_argument("--airspy", default="l1_airspy_in")
    ap.add_argument("--search-stage", default="l1_gps_search")
    ap.add_argument("--combiner-stage", default="l1_gps_combiner")
    ap.add_argument("--sys", default="G", help="constellation tag for the ephemeris (G/E/C)")
    ap.add_argument("--code-length", type=float, default=1023.0, help="chips per code period")
    ap.add_argument("--t-code", type=float, default=1e-3, help="code period (s)")
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6)
    ap.add_argument("--min-snr", type=float, default=10.0, help="search detection SNR floor")
    ap.add_argument("--min-amp-snr", type=float, default=8.0, help="combiner amp_snr floor")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    args = ap.parse_args()
    global _L
    _L = args.code_length

    utc0 = get("%s/%s/adcstat" % (args.url, args.airspy))["utc0_sample0"]
    eph = parse_rinex_nav(fetch_brdc())

    # SEARCH cp0 (control) -- ref_hop is in hops at 1e6 hops/s on the 20 MSPS front end.
    search = {}
    try:
        for d in get("%s/%s/get_detections" % (args.url, args.search_stage)):
            if d.get("snr", 0) < args.min_snr:
                continue
            r = reconstruct(d["code_phase_chips"], d["doppler_hz"], d["ref_hop"] / 1e6,
                            int(d["prn"]), eph, utc0, args)
            if r is not None:
                search[int(d["prn"])] = r
    except Exception as e:
        print("search: %s" % e)

    # COMBINER code_phase_chips (REC_CP) -- t_abs is the record's utc (seconds since sample 0).
    comb = {}
    for c in get("%s/%s/get_status" % (args.url, args.combiner_stage)):
        if (c.get("code_phase_chips") or 0) == 0 or (c.get("amp_snr") or 0) < args.min_amp_snr:
            continue
        r = reconstruct(c["code_phase_chips"], c.get("doppler_hz") or 0.0, c["utc"],
                        int(c["prn"]), eph, utc0, args)
        if r is not None:
            comb[int(c["prn"])] = r

    rS, medS = center(search)
    rC, medC = center(comb)
    print("=== code-phase observable check (%s, %s) ===" % (args.band, args.sys))
    print("SEARCH cp0 (CONTROL -- should be < ~1 chip):  clock offset %.1f chips" % medS)
    for p in sorted(rS):
        print("   PRN %2d  %+8.2f chips (%+7.0f m)" % (p, rS[p], rS[p] * C_LIGHT / CHIP))
    print("COMBINER code_phase_chips (REC_CP, DUT):      clock offset %.1f chips" % medC)
    for p in sorted(rC):
        print("   PRN %2d  %+8.2f chips (%+7.0f m)" % (p, rC[p], rC[p] * C_LIGHT / CHIP))
    rmsS, rmsC = rms(rS), rms(rC)
    print("-> SEARCH   rms %6.2f chips (%7.0f m)   [%d sats]" %
          (rmsS, rmsS * C_LIGHT / CHIP, len(rS)))
    print("-> COMBINER rms %6.2f chips (%7.0f m)   [%d sats]" %
          (rmsC, rmsC * C_LIGHT / CHIP, len(rC)))
    if rmsC == rmsC and rmsS == rmsS:  # not NaN
        verdict = "FIXED (combiner matches control)" if rmsC < 5 * max(rmsS, 0.5) \
            else "BROKEN: combiner cp is not a sample-0 currency (window->sample-0 back-ref missing)"
        print("VERDICT: %s" % verdict)


if __name__ == "__main__":
    main()
