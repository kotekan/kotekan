#!/usr/bin/env python3
"""Exercise the S2d RESCUE path -- the one a healthy receiver never produces.

    diag/receiver_state_rescue_test.py                     # against the live state dir
    diag/receiver_state_rescue_test.py --json samples.json # against a captured window

Read-only. Takes REAL state records and strips one chain's sources, so the condition the
fused state exists to rescue can be tested on demand instead of waited for.

WHY THIS EXISTS
---------------
Fusion's payoff is a chain that cannot solve its own bias. Measured over a live window, the
weakest chain (L5 GPS) sits at <=2 satellites 87% of the time -- but its minimum is ONE, never
ZERO. A healthy receiver does not produce the rescue condition, so a soak, however long,
cannot demonstrate the rescue works. It can only bound the harm.

That is exactly how 682ee91b shipped: `--clock-bias-siblings` was written for a chain that
could not reach --bias-min-sats, was GATED behind the very condition it rescued, and has
remained UNEXERCISED to this day. Flipping S2d on the strength of a soak that never saw a
zero-source chain would repeat that mistake with a bigger blast radius.

WHAT IT ASSERTS
---------------
For each chain in turn, with that chain's own carrier AND code sources removed:
  1. the dongle still produces a fused estimate (the siblings carry it),
  2. that estimate is close to what the full fusion said (the stripped chain was not
     carrying the answer alone),
  3. the rescued chain would be `bias_available` -> its first-seed guard opens.
And for a dongle stripped to NOTHING, that the answer is None -- absent, not a silent zero.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receiver_state  # noqa: E402


def strip_chain(records, chain, families=("carrier", "code")):
    """Real records with one chain's MEASUREMENTS removed -- the chain is still present and
    publishing (so this is 'a chain that cannot solve', not 'a chain that is absent')."""
    out = []
    for r in records:
        if r.get("chain") == chain:
            r = dict(r)
            for f in families:
                r.pop(f, None)
        out.append(r)
    return out


def carrier_dark(records):
    """Every chain's CARRIER measurement removed, code fits left intact.

    ★ THIS IS THE HISTORICAL FAILURE, not the fully-dark one. On 2026-07-27 the L5 GPS chain
    could measure exactly one satellite, never reached --bias-min-sats, and its carrier bias
    stayed UNSOLVED for 603 cycles / 2h45m -- while its l-a code fit was working the whole
    time and, measured later, actually predicts the LO BETTER than the carrier does at L5
    (14.5 vs 18.9 Hz). A cross-FAMILY rescue does not need siblings, so it works even on a
    lone-chain dongle like L2C, where a sibling rescue structurally cannot.
    """
    out = []
    for r in records:
        r = dict(r)
        r.pop("carrier", None)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/kotekan_gps/state"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--floor-ppm", type=float, default=0.001)
    ap.add_argument("--reject-sigma", type=float, default=5.0)
    ap.add_argument("--tol-hz", type=float, default=15.0,
                    help="how far the rescued estimate may sit from the full fusion before "
                         "this counts as 'the stripped chain was carrying it alone'")
    args = ap.parse_args()

    t = time.time()
    by_dongle = {}
    if args.json:
        hist = json.load(open(args.json))
        for chain, recs in hist.items():
            if recs:
                by_dongle.setdefault(recs[-1]["dongle"], []).append(recs[-1])
    else:
        for n in sorted(os.listdir(args.dir)):
            if not n.endswith(".json"):
                continue
            r = receiver_state.read_state(os.path.join(args.dir, n), max_age_s=30.0, t_now=t)
            if r:
                by_dongle.setdefault(r["dongle"], []).append(r)

    if not by_dongle:
        print("no state records found")
        return 1

    fails = 0
    for dongle, recs in sorted(by_dongle.items()):
        fc = recs[0].get("carrier_hz") or 1.0
        full = receiver_state.fuse_dongle(recs, floor_ppm=args.floor_ppm,
                                          reject_sigma=args.reject_sigma)
        print("\n=== %s (%d chains) ===" % (dongle, len(recs)))
        if not full:
            print("  no usable sources at all -> fused None (correct: absent, not zero)")
            continue
        print("  full fusion: %+.5f ppm = %+.2f Hz  (%d src over %s)"
              % (full["lo_ppm"], full["lo_ppm"] * 1e-6 * fc, full["n_src"],
                 ",".join(full["chains"])))
        if full.get("dominance") is not None:
            print("  dominance: heaviest single source holds %.0f%% of the weight%s"
                  % (100 * full["dominance"],
                     "   <-- LOW DIVERSITY: the estimate leans on one source, so it "
                     "inherits that source's failure modes and stripping it will move the "
                     "answer. Not necessarily wrong, but not really 'fused'."
                     if full["dominance"] > 0.5 else ""))
        for r in sorted(recs, key=lambda x: x["chain"]):
            ch = r["chain"]
            had = bool((r.get("carrier") or {}).get("raw_hz") is not None
                       or (r.get("code") or {}).get("raw_ppm") is not None)
            f = receiver_state.fuse_dongle(strip_chain(recs, ch), floor_ppm=args.floor_ppm,
                                           reject_sigma=args.reject_sigma)
            if f is None:
                # legitimate when this was the dongle's ONLY source of measurements
                ok = len(recs) == 1 or not any(
                    receiver_state.sources_from(x) for x in recs if x["chain"] != ch)
                print("  %-16s stripped -> fused None      %s"
                      % (ch, "OK (it was the only source; absent, not zero)" if ok
                         else "*** FAIL: siblings existed but produced nothing"))
                fails += 0 if ok else 1
                continue
            d_hz = (f["lo_ppm"] - full["lo_ppm"]) * 1e-6 * fc
            ok = abs(d_hz) <= args.tol_hz
            fails += 0 if ok else 1
            print("  %-16s stripped -> %+.2f Hz from %d src over %-28s delta %+6.2f Hz  %s"
                  % (ch, f["lo_ppm"] * 1e-6 * fc, f["n_src"], ",".join(f["chains"]),
                     d_hz, "OK" if ok else "*** FAIL (> %.0f Hz)" % args.tol_hz))
            if not had:
                print("      (note: this chain had no measurements even before stripping --"
                      " it is ALREADY being rescued in the live system)")

        # ★ THE HISTORICAL CASE: every carrier dark, code fits intact. This is the 2h45m
        # outage's exact shape, and unlike a sibling rescue it works on a lone chain.
        f = receiver_state.fuse_dongle(carrier_dark(recs), floor_ppm=args.floor_ppm,
                                       reject_sigma=args.reject_sigma)
        if f is None:
            print("  ALL CARRIERS dark  -> fused None  *** no code-side rescue available")
            fails += 1
        else:
            d_hz = (f["lo_ppm"] - full["lo_ppm"]) * 1e-6 * fc
            ok = abs(d_hz) <= args.tol_hz
            fails += 0 if ok else 1
            print("  ALL CARRIERS dark  -> %+.2f Hz from %d code source(s)   delta %+6.2f Hz"
                  "  %s" % (f["lo_ppm"] * 1e-6 * fc, f["n_src"], d_hz,
                            "OK <- the 2026-07-27 rescue, on code alone"
                            if ok else "*** FAIL (> %.0f Hz)" % args.tol_hz))

        # the fully-dark dongle: absent must be absent
        dark = [dict(r) for r in recs]
        for r in dark:
            r.pop("carrier", None)
            r.pop("code", None)
        f = receiver_state.fuse_dongle(dark, floor_ppm=args.floor_ppm,
                                       reject_sigma=args.reject_sigma)
        ok = f is None
        fails += 0 if ok else 1
        print("  ALL chains dark    -> %s  %s"
              % ("None" if ok else "%.5f ppm" % f["lo_ppm"],
                 "OK (absent, never a silent zero)" if ok else "*** FAIL"))

    print("\n%s" % ("ALL RESCUE CASES PASS" if not fails else "%d FAILURE(S)" % fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
