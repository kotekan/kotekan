#!/usr/bin/env python3
"""One-step-ahead PREDICTION CONTEST between a chain and the state that would replace it.

This is the S2c shadow score, and the gate on S2d. Read-only; safe on the soak.

    diag/receiver_state_predict.py --json live_state.json
    diag/receiver_state_predict.py --samples 200 --interval 6      # sample live

WHY A CONTEST AND NOT AN A/B
----------------------------
The obvious shadow score -- "does chain X hold coherence better with the fused prior than
without" -- requires APPLYING the prior, which is steering, which is the thing being
deferred. A contest needs no intervention: every predictor is asked to guess a chain's own
NEXT measurement, and they are judged against what that chain actually measured. The winner
is decided by evidence rather than by argument, before anything is allowed to steer.

THE PREDICTORS
--------------
  own     the chain's own latest raw measurement -- what it already trusts
  loo     LEAVE-ONE-OUT band consensus: the OTHER chains on the same dongle, inverse-
          variance weighted by each one's own measured scatter. Leave-one-out is not
          optional: a consensus that includes chain i is partly a copy of chain i, and
          scoring it against chain i is rigged toward fusion. (Same discipline as
          test_code_bias.py's leave-one-out l-a check.)
  fused   ★ THE ONE THAT WOULD SHIP: `own` and `loo` inverse-variance COMBINED.
  code    CROSS-FAMILY: this chain's own l-a, sign-flipped and scaled to Hz. Tests whether
          the CODE side alone can serve the CARRIER side.

★★ `loo` IS A STRAWMAN AND MUST NOT BE THE VERDICT. Nobody proposes REPLACING a good
chain's estimate with only its neighbours' -- and scoring that strawman would veto a sound
design. In a synthetic check here, a sigma=0.5 chain beside two sigma=4 chains went 0.44 ->
1.99 Hz under `loo`, correctly, because a consensus of two noisy chains really is worse than
one quiet one. The deployable quantity is `fused`, which cannot be worse than either input
when the inputs are unbiased -- and which is worse than `own` exactly when a chain carries a
systematic its neighbours do not share. That is the signal a covariance floor exists to
catch, so `fused` losing on a chain is INFORMATION, not noise.

★ `code` is the one that matters most. If a chain's own l-a fit predicts its next carrier
measurement about as well as the band consensus does, then a chain with no usable carrier
residuals -- but a working code fit -- still has an LO answer. That is exactly the 2026-07-27
cold start, where L5 GPS sat UNSOLVED for 2h45m at --bias-min-sats with one satellite while
the answer was available three different ways.

READING THE RESULT
------------------
A predictor only earns the right to steer if it beats `own` on the chains it would serve.
Watch for the split: fusion should help the WEAK chains (few sats, big MAD) and be roughly
neutral on the strong ones. If it helps everywhere equally, suspect the metric; if it HURTS
the strong chains, a covariance floor is mandatory before any flip.
"""

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receiver_state  # noqa: E402
from receiver_state import se_of_median  # noqa: E402


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sample_live(state_dir, samples, interval, max_age_s):
    import time
    hist = {}
    for i in range(samples):
        t = time.time()
        try:
            names = sorted(os.listdir(state_dir))
        except Exception:
            break
        for n in names:
            if not n.endswith(".json"):
                continue
            rec = receiver_state.read_state(os.path.join(state_dir, n),
                                            max_age_s=max_age_s, t_now=t)
            if rec is None:
                continue
            h = hist.setdefault(rec["chain"], [])
            if h and h[-1].get("t") == rec.get("t"):
                continue
            h.append(rec)
        if i < samples - 1:
            time.sleep(interval)
    return hist


def _g(rec, group, field):
    v = (rec.get(group) or {}).get(field)
    return None if v is None else float(v)


def align(hist, tol_s):
    """Group records from different chains into common epochs.

    Chains publish on their own cadence, so 'simultaneous' has to be built. Records within
    tol_s of an epoch's first record join it; a chain contributes at most one record per
    epoch (its earliest match), so a fast chain cannot dominate an epoch.
    """
    flat = []
    for chain, recs in hist.items():
        for r in recs:
            flat.append((float(r["t"]), chain, r))
    flat.sort()
    epochs, cur, t0 = [], {}, None
    for t, chain, r in flat:
        if t0 is None or t - t0 > tol_s:
            if cur:
                epochs.append((t0, cur))
            cur, t0 = {}, t
        cur.setdefault(chain, r)
    if cur:
        epochs.append((t0, cur))
    return epochs


def loo_consensus(epoch, dongle, exclude):
    """Inverse-variance weighted carrier estimate from the OTHER chains on this dongle.

    Returns (value, se, n_chains) -- the se is needed to COMBINE this with the chain's own
    estimate, which is the quantity that would actually ship.
    """
    num = den = 0.0
    n_used = 0
    for chain, r in epoch.items():
        if chain == exclude or r.get("dongle") != dongle:
            continue
        raw = _g(r, "carrier", "raw_hz")
        se = se_of_median(_g(r, "carrier", "mad_hz"), _g(r, "carrier", "n"))
        if raw is None or not se:
            continue
        w = 1.0 / (se * se)
        num += raw * w
        den += w
        n_used += 1
    if den <= 0 or n_used == 0:
        return None, None, 0
    return num / den, math.sqrt(1.0 / den), n_used


def ivw(a, sa, b, sb):
    """Inverse-variance combination of two estimates. None-tolerant."""
    if a is None or not sa:
        return (b, sb) if b is not None else (None, None)
    if b is None or not sb:
        return a, sa
    wa, wb = 1.0 / (sa * sa), 1.0 / (sb * sb)
    return (a * wa + b * wb) / (wa + wb), math.sqrt(1.0 / (wa + wb))


def code_predictor(rec):
    """This chain's l-a, sign-flipped and scaled to Hz at its own carrier."""
    la = _g(rec, "code", "raw_ppm")
    fc = rec.get("carrier_hz")
    if la is None or not fc:
        return None
    return -la * 1e-6 * float(fc)


def target_of(epochs, k, chain, window):
    """Truth for the prediction at epoch k: the median of this chain's NEXT `window` raw
    measurements.

    ★ WHY NOT A SINGLE NEXT SAMPLE. The per-cycle raw median is itself noisy (measured
    live: sd 1.4 Hz on L1-GAL up to 25 Hz on L5-GPS). Scoring against one noisy sample puts
    that noise in the TARGET, so a perfect predictor still scores ~sd while `own` scores
    ~sqrt(2)*sd -- capping the measurable improvement at 1.41x no matter how good fusion
    is. Averaging the next `window` samples shrinks the target's own noise by sqrt(window)
    and lets a real difference show. Still strictly causal: the target is entirely in the
    future of every predictor being scored, so nothing leaks backwards.
    """
    vals = []
    for j in range(k + 1, min(k + 1 + window, len(epochs))):
        r = epochs[j][1].get(chain)
        if r is None:
            continue
        v = _g(r, "carrier", "raw_hz")
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return statistics.median(vals)


def contest(hist, tol_s, window=1):
    epochs = align(hist, tol_s)
    if len(epochs) < 3:
        print("need >=3 aligned epochs, got %d" % len(epochs))
        return {}
    print("aligned %d epochs (tol %.1f s) spanning %.0f s; target = median of the next %d"
          % (len(epochs), tol_s, epochs[-1][0] - epochs[0][0], window))

    errs = {}   # chain -> {predictor: [abs errors]}
    for k in range(len(epochs) - 1):
        _, e_now = epochs[k]
        for chain, r in e_now.items():
            truth = target_of(epochs, k, chain, window)
            if truth is None:
                continue
            dongle = r.get("dongle")
            d = errs.setdefault(chain, {"own": [], "loo": [], "fused": [], "code": [],
                                        "pair_of": [], "pair_fu": [], "n_loo": []})
            own = _g(r, "carrier", "raw_hz")
            se_own = se_of_median(_g(r, "carrier", "mad_hz"), _g(r, "carrier", "n"))
            if own is not None:
                d["own"].append(abs(own - truth))
            lo, se_lo, nu = loo_consensus(e_now, dongle, chain)
            if lo is not None:
                d["loo"].append(abs(lo - truth))
                d["n_loo"].append(nu)
            fu, _ = ivw(own, se_own, lo, se_lo)
            if fu is not None:
                d["fused"].append(abs(fu - truth))
            # PAIRED samples: only epochs where BOTH predictors existed. Comparing medians
            # of differently-sized sets would let a predictor look good by being absent on
            # the hard epochs.
            if own is not None and fu is not None:
                d["pair_of"].append(abs(own - truth))
                d["pair_fu"].append(abs(fu - truth))
            cp = code_predictor(r)
            if cp is not None:
                d["code"].append(abs(cp - truth))
    return errs


def report(errs):
    print("\n%-16s %6s %9s %9s %9s %9s  %s"
          % ("chain", "n", "own", "loo", "FUSED", "code", "fused vs own"))
    print("-" * 92)
    # The verdict is PER CHAIN and PAIRED. A pooled median over all chains would be
    # dominated by whichever chain is noisiest and would mix quantities of different
    # precision -- it once read "DO NOT FLIP" on a synthetic case where fusion helped two
    # chains out of three.
    wins, losses, neutral = [], [], []
    for chain in sorted(errs):
        d = errs[chain]
        med = {k: (statistics.median(d[k]) if d[k] else None)
               for k in ("own", "loo", "fused", "code")}
        verdict = "-"
        if d["pair_of"] and d["pair_fu"]:
            o, f = statistics.median(d["pair_of"]), statistics.median(d["pair_fu"])
            ratio = o / f if f > 0 else float("inf")
            if ratio > 1.05:
                verdict = "improves %.2fx" % ratio
                wins.append((chain, ratio))
            elif ratio < 0.95:
                verdict = "WORSE %.2fx  <-- systematic?" % (1.0 / ratio)
                losses.append((chain, 1.0 / ratio))
            else:
                verdict = "neutral"
                neutral.append(chain)
        print("%-16s %6d %9s %9s %9s %9s  %s"
              % (chain, len(d["own"]), _f(med["own"]), _f(med["loo"]),
                 _f(med["fused"]), _f(med["code"]), verdict))
    print("-" * 92)
    print("(median |one-step-ahead error| in Hz vs each chain's own NEXT raw measurement;\n"
          " `fused` = own and loo inverse-variance combined -- the quantity that would ship)")

    n = len(wins) + len(losses) + len(neutral)
    print("\n=> %d/%d chains improve, %d neutral, %d WORSE."
          % (len(wins), n, len(neutral), len(losses)))
    if losses:
        print("   Worse on: %s" % ", ".join("%s (%.2fx)" % (c, r) for c, r in losses))
        print("   A chain that fusion HURTS carries a systematic its neighbours do not\n"
              "   share. Do not flip it blind: either find the systematic, or give the\n"
              "   fuser a covariance FLOOR so that chain cannot be dragged by the band.")
    elif wins:
        print("   No chain is hurt. That is the evidence S2d needs -- fusion is a strict\n"
              "   improvement on this window. Confirm it holds over a full soak, and\n"
              "   expect the gain to concentrate on the WEAK chains (few sats, big MAD);\n"
              "   uniform improvement everywhere would be suspicious.")
    else:
        print("   Nothing to conclude yet -- too few paired epochs.")

    # cross-family verdict, per chain and comparative
    print("\n=> Cross-family (code side alone predicting the carrier):")
    for chain in sorted(errs):
        d = errs[chain]
        if not d["code"] or not d["own"]:
            continue
        c, o = statistics.median(d["code"]), statistics.median(d["own"])
        print("     %-16s code %7.2f Hz vs own %7.2f Hz  -> %s"
              % (chain, c, o,
                 "STANDALONE-CAPABLE (a chain with no carrier residuals still has an "
                 "LO answer)" if c <= 1.5 * o else
                 "cross-CHECK only, too weak to stand alone (%.1fx own)" % (c / o)))


def _f(v):
    return ("%10.3f" % v) if v is not None else ("%10s" % "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="samples dumped by receiver_state_score.py")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/kotekan_gps/state"))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--max-age-s", type=float, default=30.0)
    ap.add_argument("--align-tol-s", type=float, default=2.0)
    ap.add_argument("--target-window", type=int, default=5,
                    help="score against the median of the next N raw measurements rather "
                         "than a single one. The target's own noise otherwise caps the "
                         "measurable improvement at 1.41x (see target_of). 1 = the strict "
                         "one-step-ahead form.")
    args = ap.parse_args()

    hist = load_json(args.json) if args.json else sample_live(
        args.dir, args.samples, args.interval, args.max_age_s)
    if not hist:
        print("no state samples")
        return 1
    print("chains: %s" % ", ".join("%s x%d" % (c, len(r)) for c, r in sorted(hist.items())))
    errs = contest(hist, args.align_tol_s, max(1, args.target_window))
    if errs:
        report(errs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
