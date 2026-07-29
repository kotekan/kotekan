#!/usr/bin/env python3
"""Score the receiver-state exports (S2 / Mechanism B, observer phase).

Reads the per-chain JSON published by `--state-file` and answers the questions that must
be settled BEFORE a covariance-weighted fuser can be written. Read-only; safe on the soak.

    diag/receiver_state_score.py --samples 120 --interval 5

THE FOUR QUESTIONS
------------------
Q1  DO THE CHAINS ON ONE DONGLE AGREE, AND IS THEIR DISAGREEMENT EXPLAINED BY THEIR OWN
    MEASURED NOISE?  Every chain on a band looks through one airspy, one LO, so each is
    measuring ONE physical number. Scored as a z: pairwise |raw_i - raw_j| over the
    combined standard error each chain implies from its own MAD. |z| ~ 1 means the spread
    is noise and a weighted mean is legitimate. |z| >> 1 means a per-chain SYSTEMATIC, and
    fusing without a covariance floor would let the loudest chain capture the estimate.
    This is scored on the RAW (pre-fusion) values -- the smoothed ones already read each
    other's files, so their agreement is partly manufactured.

Q2  DO THE CARRIER AND CODE ESTIMATORS AGREE?  `clock_bias` (Hz, from search Doppler) and
    `l-a` (ppm, from the code-rate slope) are two independent routes to the same per-dongle
    clock error, with opposite sign conventions. Their sum in ppm should be zero. It never
    has been checked. A nonzero, STABLE sum is the real LO-vs-ADC split; a wandering one is
    an estimator fault.

Q3  IS `car_trim` REDUNDANT WITH `clock_bias`?  The carrier loop's fleet-median trim is
    documented as "the chain's deterministic frac-N LO offset ... stable across restarts",
    which is what clock_bias also solves for. If trim ~ clock_bias they are two measurements
    of one number and the fuser should absorb both. If trim ~ 0 the seed already carries
    clock_bias and the trim is only the residual. The docs imply the former; measure it.

Q4  WHAT ACTUALLY MOVES?  Per-chain drift over the sampling window vs its own noise. A
    quantity that does not move faster than its noise is a CONSTANT to be persisted, not a
    state to be tracked -- and two of the four (car_trim, rxclock) are not persisted at all
    today despite being described as restart-stable.

Every output is a measurement, not a decision. Nothing here writes to the live path.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receiver_state  # noqa: E402

# MAD -> sigma for a Gaussian, and sigma -> standard error of a MEDIAN (pi/2 factor).
MAD_TO_SIGMA = 1.4826
SE_MEDIAN = math.sqrt(math.pi / 2.0)


def se_of_median(mad, n):
    """Standard error of a median estimated from n samples with the given MAD."""
    if mad is None or n is None or n < 2:
        return None
    return SE_MEDIAN * MAD_TO_SIGMA * mad / math.sqrt(n)


def collect(state_dir, samples, interval, max_age_s):
    """Sample the published state. Returns {chain: [rec, ...]} in time order."""
    hist = {}
    for i in range(samples):
        t = time.time()
        try:
            names = sorted(os.listdir(state_dir))
        except Exception as e:
            print("cannot read %s: %s" % (state_dir, e))
            return hist
        for n in names:
            if not n.endswith(".json"):
                continue
            rec = receiver_state.read_state(os.path.join(state_dir, n),
                                            max_age_s=max_age_s, t_now=t)
            if rec is None:
                continue
            h = hist.setdefault(rec["chain"], [])
            # dedupe: the broker republishes at its own cadence, not ours
            if h and h[-1].get("t") == rec.get("t"):
                continue
            h.append(rec)
        if i < samples - 1:
            time.sleep(interval)
    return hist


def _series(recs, group, field):
    out = []
    for r in recs:
        v = (r.get(group) or {}).get(field)
        if v is not None:
            out.append(float(v))
    return out


def _last(recs, group, field, default=None):
    for r in reversed(recs):
        v = (r.get(group) or {}).get(field)
        if v is not None:
            return v
    return default


def _med(v):
    return statistics.median(v) if v else None


def by_dongle(hist):
    d = {}
    for chain, recs in hist.items():
        if recs:
            d.setdefault(recs[-1].get("dongle", "?"), {})[chain] = recs
    return d


def q1_intra_dongle(dongle, chains):
    """Chains on one LO: do they agree within their own claimed noise?"""
    print("\n  Q1  intra-dongle agreement (RAW, pre-fusion)")
    rows = []
    for chain, recs in sorted(chains.items()):
        raw = _med(_series(recs, "carrier", "raw_hz"))
        mad = _med(_series(recs, "carrier", "mad_hz"))
        n = _med(_series(recs, "carrier", "n"))
        fused = _med(_series(recs, "carrier", "hz"))
        se = se_of_median(mad, n)
        rows.append((chain, raw, mad, n, se, fused))
        print("      %-18s raw %s Hz  mad %s  n %s  -> se %s   (fused %s)"
              % (chain, _f(raw, "%+9.2f"), _f(mad, "%6.2f"), _f(n, "%4.0f"),
                 _f(se, "%6.2f"), _f(fused, "%+8.2f")))
    have = [r for r in rows if r[1] is not None]
    if len(have) < 2:
        print("      (need >=2 chains with a raw estimate to score agreement)")
        return
    # pairwise z on the RAW values
    zs, spread_raw, spread_fused = [], [], []
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            a, b = have[i], have[j]
            d = a[1] - b[1]
            if a[4] and b[4]:
                z = d / math.sqrt(a[4] ** 2 + b[4] ** 2)
                zs.append(abs(z))
                print("      %-14s vs %-14s  d %+7.2f Hz   z %5.1f%s"
                      % (a[0], b[0], d, abs(z), "   <-- SYSTEMATIC" if abs(z) > 3 else ""))
    spread_raw = [r[1] for r in have]
    spread_fused = [r[5] for r in have if r[5] is not None]
    print("      spread: raw %.2f Hz over %d chains%s"
          % (max(spread_raw) - min(spread_raw), len(spread_raw),
             ("   fused %.2f Hz (partly MANUFACTURED -- the .hz files read each other)"
              % (max(spread_fused) - min(spread_fused))) if len(spread_fused) > 1 else ""))
    if zs:
        mz = statistics.median(zs)
        print("      median |z| %.1f -> %s" % (
            mz,
            "spread is consistent with each chain's own noise; a weighted mean is honest"
            if mz < 3 else
            "DISAGREEMENT EXCEEDS NOISE: per-chain systematic. A fuser needs a covariance "
            "FLOOR or the loudest chain captures the estimate."))


def q2_carrier_vs_code(dongle, chains):
    """Two independent routes to one number, opposite sign. The sum should vanish."""
    print("\n  Q2  carrier vs code (independent estimators of one clock error)")
    sums = []
    for chain, recs in sorted(chains.items()):
        fc = recs[-1].get("carrier_hz")
        raw_hz = _med(_series(recs, "carrier", "raw_hz"))
        la_ppm = _med(_series(recs, "code", "raw_ppm"))
        if fc is None or raw_hz is None or la_ppm is None:
            print("      %-18s (incomplete: carrier %s, code %s)"
                  % (chain, _f(raw_hz, "%+.2f"), _f(la_ppm, "%+.4f")))
            continue
        car_ppm = raw_hz / fc * 1e6
        s = car_ppm + la_ppm
        sums.append(s)
        # propagate both scatters into the sum
        se_c = se_of_median(_med(_series(recs, "carrier", "mad_hz")),
                            _med(_series(recs, "carrier", "n")))
        se_l = se_of_median(_med(_series(recs, "code", "mad_ppm")),
                            _med(_series(recs, "code", "n")))
        se_s = None
        if se_c is not None and se_l is not None:
            se_s = math.sqrt((se_c / fc * 1e6) ** 2 + se_l ** 2)
        print("      %-18s carrier %+8.5f ppm  code %+8.5f ppm  sum %+8.5f%s"
              % (chain, car_ppm, la_ppm, s,
                 ("  (+-%.5f, %.1f sigma)" % (se_s, abs(s) / se_s)) if se_s else ""))
    if sums:
        print("      |sum| median %.5f ppm across %d chains -- %s"
              % (statistics.median([abs(x) for x in sums]), len(sums),
                 "this is the LO-vs-ADC split if STABLE, an estimator fault if it wanders"))


def q3_trim_redundancy(dongle, chains):
    """Is the carrier loop's fleet trim rediscovering clock_bias?"""
    print("\n  Q3  car_trim vs clock_bias (is the trim a second copy of the same number?)")
    for chain, recs in sorted(chains.items()):
        trim = _med(_series(recs, "carrier_trim", "median_hz"))
        n = _med(_series(recs, "carrier_trim", "n"))
        bias = _med(_series(recs, "carrier", "hz"))
        if trim is None or bias is None:
            print("      %-18s trim %s  bias %s  (incomplete)"
                  % (chain, _f(trim, "%+.2f"), _f(bias, "%+.2f")))
            continue
        r = abs(trim / bias) if bias else float("inf")
        if r > 0.5:
            verdict = "REDUNDANT: trim ~ bias, two measurements of one number"
        elif r < 0.1:
            verdict = "residual only: the seed already carries the bias"
        else:
            verdict = "partial: the seed carries some of it"
        print("      %-18s trim %+8.2f Hz (n %s)  bias %+8.2f  |trim/bias| %.2f  -> %s"
              % (chain, trim, _f(n, "%.0f"), bias, r, verdict))


def q4_motion(dongle, chains):
    """Does anything move faster than its own noise? If not, it is a constant."""
    print("\n  Q4  motion over the window vs noise (constant, or genuinely a state?)")
    for chain, recs in sorted(chains.items()):
        if len(recs) < 3:
            continue
        span = recs[-1]["t"] - recs[0]["t"]
        for grp, fld, unit in (("carrier", "raw_hz", "Hz"),
                               ("code", "raw_ppm", "ppm"),
                               ("carrier_trim", "median_hz", "Hz")):
            v = _series(recs, grp, fld)
            if len(v) < 3:
                continue
            drift = v[-1] - v[0]
            scat = statistics.pstdev(v)
            print("      %-18s %-12s span %5.0f s  drift %+9.4f %-3s  scatter %8.4f  "
                  "-> %s" % (chain, grp + "." + fld, span, drift, unit, scat,
                             "MOVES" if scat > 0 and abs(drift) > 3 * scat
                             else "flat within noise (a CONSTANT to persist)"))


def q5_persistence(hist):
    """Which quantities survive a restart today?"""
    print("\n=== persistence audit ===")
    print("  carrier   clock_bias_ema  PERSISTED (%s.hz) + sibling-fused, warm-started" % "gps_clock_bias_*")
    print("  code      code_bias_ema   PERSISTED (%s.ppm), NO fusion of any kind" % "gps_code_bias_*")
    print("  trim      car_trim median NOT PERSISTED -- rebuilt from zero every launch,")
    print("                            though its own help calls it 'stable across restarts'")
    print("  rxclock   dr_state clk    NOT PERSISTED -- every restart re-bootstraps")
    for chain, recs in sorted(hist.items()):
        rc = recs[-1].get("rxclock") or {}
        if rc:
            print("      %-18s rxclock chips %s drift %s n %s integ_mad %s untrusted %s"
                  % (chain, _f(rc.get("chips"), "%9.2f"),
                     _f(rc.get("drift_chips_s"), "%+8.4f"), _f(rc.get("n"), "%3.0f"),
                     _f(rc.get("integ_mad_chips"), "%7.4f"),
                     _f(rc.get("untrusted"), "%2.0f")))


def _f(v, fmt):
    return (fmt % v) if v is not None else (" " * max(0, len(fmt % 0) - 1) + "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/kotekan_gps/state"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-age-s", type=float, default=30.0,
                    help="ignore state older than this (a stale LO estimate is a "
                         "different epoch's number, not a usable prior)")
    ap.add_argument("--json", default=None, help="also dump the raw samples here")
    args = ap.parse_args()

    print("sampling %s: %d sample(s) @ %.1f s" % (args.dir, args.samples, args.interval))
    hist = collect(args.dir, args.samples, args.interval, args.max_age_s)
    if not hist:
        print("\nNO STATE FOUND. Is --state-file wired in run_live.sh, and has the node "
              "been restarted since? (The export is write-only; absence is harmless.)")
        return 1
    print("collected: %s" % ", ".join("%s x%d" % (c, len(r)) for c, r in sorted(hist.items())))

    for dongle, chains in sorted(by_dongle(hist).items()):
        print("\n" + "=" * 78)
        print("DONGLE %s  (%d chain(s) -- one airspy, one LO, ONE physical number)"
              % (dongle, len(chains)))
        print("=" * 78)
        q1_intra_dongle(dongle, chains)
        q2_carrier_vs_code(dongle, chains)
        q3_trim_redundancy(dongle, chains)
        if args.samples >= 3:
            q4_motion(dongle, chains)
    q5_persistence(hist)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(hist, f)
        print("\nraw samples -> %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
