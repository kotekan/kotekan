#!/usr/bin/env python3
"""IS THE PER-RECORD PHASE SERIES COHERENT AT ALL? -- measured WITHOUT searching a rate.

The rate sweep (kcoh_rate_probe.py) reports an argmax, and on an intermittently corrupted
series it reports the corruption AS a rate: gps_l5 gave per-satellite "rates" of 2-13 Hz that
did not replicate, while the served fold's own sinc^2 roll-off implied an effective error of
only ~1.8 Hz. Those disagree by 5x, so the argmax is not a rate and no amount of searching
will make it one.

THE LAG CORRELATION IS RATE-BLIND, which is the whole point of using it here:

    r_m = < A_{k+m} conj(A_k) >          over pairs exactly m records apart

A constant residual rate f puts ALL of its effect in arg(r_m) = 2 pi f m t_rec and NONE in
|r_m|. So |r_m| measures how much coherence survives a lag of m records no matter what the
rate is, and there is nothing to search and nothing to bias. For a pure tone in noise |r_m| is
FLAT in m; for a wandering phase it DECAYS, and the decay length is the true coherence time.

    |r_m| flat, fold fails    -> the series is coherent; the injected RATE is the problem.
    |r_m| decays              -> the phase itself wanders; no injected rate can help, and the
                                 fix is upstream in the phase transport, not in the estimator.

⚠️ m >= 1 IS ALSO WHERE THE NOISE BIAS ISN'T. Noise is independent between records, so it
cancels in E[r_m] for m >= 1 -- unlike |A|^2 at m = 0, which carries sigma^2 and is what the
probe-anchored debias exists to remove. Normalising by |r_1| reports the SHAPE, which is the
part that answers the question.

NULL: the same curve on the noise probes. They have no carrier, so their |r_m| is the
floor a series of this length produces by chance; the satellite must sit clearly above it.

    ./kcoh_phase_series.py --chain gps_l5 --blocks 3      # on the gather host
"""
import argparse
import cmath
import json
import math
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import combdll, telem  # noqa: E402

HOP_S = 5.12e-6
HOPS_PER_REC = 2048
T_REC = HOPS_PER_REC * HOP_S
REC_PER_FRAME = 4


def lag_corr(rows, m):
    """(|r_m| / mean|A|^2, arg(r_m), n_pairs) over pairs EXACTLY m records apart.

    Keyed on the hop tag, not on list position: a dropped record must not silently become a
    shorter lag, which is how a gap turns into a fake decoherence.
    """
    by_hop = {int(h): complex(re_, im_) for h, re_, im_, _s1, _s2 in rows}
    acc, npair = 0j, 0
    pwr, npw = 0.0, 0
    for h, v in by_hop.items():
        pwr += abs(v) ** 2
        npw += 1
        w = by_hop.get(h + m * HOPS_PER_REC)
        if w is not None:
            acc += w * v.conjugate()
            npair += 1
    if not npair or not npw or pwr <= 0:
        return None, None, 0
    return abs(acc / npair) / (pwr / npw), cmath.phase(acc / npair), npair


def frame_increments(series):
    """Mean phase step j -> j+1 by POSITION WITHIN THE FRAME, amplitude-weighted.

    The direct form of what |r_m| only implies. If the carrier phase accumulates properly,
    every position shows the same step (the true residual rate). If it is re-referenced at
    each frame boundary, steps 0->1,1->2,2->3 share one value and 3->0 carries the reset --
    a sawtooth, which is period-4 in the lag correlation and looks like a RATE to any search
    that folds across frames.
    """
    acc = {j: 0j for j in range(REC_PER_FRAME)}
    cnt = {j: 0 for j in range(REC_PER_FRAME)}
    for rows in series.values():
        by_hop = {int(h): complex(re_, im_) for h, re_, im_, _a, _b in rows}
        for h, v in by_hop.items():
            w = by_hop.get(h + HOPS_PER_REC)
            if w is None:
                continue
            j = (h // HOPS_PER_REC) % REC_PER_FRAME     # position of the EARLIER record
            acc[j] += w * v.conjugate()                  # amplitude-weighted: no unwrapping
            cnt[j] += 1
    return {j: (cmath.phase(acc[j]), abs(acc[j]), cnt[j]) for j in range(REC_PER_FRAME)
            if cnt[j]}


def curve(series, lags):
    """Mean over instances of |r_m|/mean|A|^2, and the lag-1 implied rate."""
    out, rate = {}, []
    for rows in series.values():
        for m in lags:
            mag, ph, n = lag_corr(rows, m)
            if mag is not None:
                out.setdefault(m, []).append(mag)
            if m == 1 and ph is not None:
                rate.append(ph / (2 * math.pi * T_REC))
    return ({m: statistics.mean(v) for m, v in out.items() if v},
            statistics.median(rate) if rate else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()

    with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                timeout=10) as r:
        rows = json.loads(r.read().decode())
    probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
    held = [x for x in rows if not x.get("noise_probe")
            and x.get("cn0_prompt_db") is not None
            and (x.get("cn0_prompt_duty") or 0) >= 0.9]
    if not held:
        raise SystemExit("INCONCLUSIVE: nothing held at duty >= 0.9 on %s." % a.chain)
    print("chain %s: %d held sat(s) %s, probes %s"
          % (a.chain, len(held), sorted(int(x["prn"]) for x in held), sorted(probes)))

    lags = [1, 2, 4, 8, 16, 32, 64]
    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    sat_raw, probe_raw = {m: [] for m in lags}, {m: [] for m in lags}
    incs = {}
    try:
        for b in range(a.blocks):
            time.sleep(a.seconds)
            got = combdll.coh_cn0(cl, a.chain, rates={}, n_win=a.windows,
                                  probe_prns=probes, keep_series=True)
            if not got:
                print("  (block %d: no fold)" % b)
                continue
            print("\n  -- block %d --   |r_m| / mean|A|^2   (m in RECORDS, 1 = %.1f ms)"
                  % (b, 1000 * T_REC))
            print("    %-6s %-7s %s" % ("prn", "rate1", "  ".join("m=%-6d" % m
                                                                  for m in lags)))
            for x in held + [{"prn": p, "_probe": True} for p in sorted(probes)]:
                p = int(x["prn"])
                v = got.get(p)
                if not v or "series" not in v:
                    continue
                c, r1 = curve(v["series"], lags)
                if 1 not in c:
                    continue
                tag = "probe" if x.get("_probe") else "     "
                print("    %-6d %-7s %s %s"
                      % (p, ("%+.2f" % r1) if r1 is not None else "--",
                         "  ".join("%-8.3f" % c.get(m, float("nan")) for m in lags), tag))
                dst = probe_raw if x.get("_probe") else sat_raw
                for m in lags:
                    if m in c:
                        dst[m].append(c[m])
                if not x.get("_probe"):
                    fi = frame_increments(v["series"])
                    if fi:
                        incs.setdefault(p, {}).update(fi)
    finally:
        cl.stop()

    if not sat_raw[1]:
        print("\nINCONCLUSIVE: no satellite series collected.")
        return 1
    print("\n  SHAPE, |r_m| / |r_4|  (normalised on ONE FRAME, not on lag 1)")
    print("    %-10s %s" % ("", "  ".join("m=%-6d" % m for m in lags)))
    # ⚠️ NORMALISE ON m=4, NOT m=1. The first cut divided by |r_1| and reported "flat, so it
    # is the rate" -- but |r_1| is precisely the corrupted lag here, so dividing by it turned
    # a dip at m=1,2 into numbers ABOVE 1.0 at every longer lag and hid the whole finding.
    # A normaliser must not be the quantity under suspicion.
    ref = 4
    sat = {m: statistics.median(v) for m, v in sat_raw.items() if v}
    prb = {m: statistics.median(v) for m, v in probe_raw.items() if v}
    if ref not in sat:
        print("INCONCLUSIVE: no m=%d lag collected." % ref)
        return 1
    print("    %-10s %s" % ("satellites", "  ".join("%-8.3f" % (sat[m] / sat[ref])
                                                    if m in sat else "--" for m in lags)))
    print("    %-10s %s" % ("probes", "  ".join("%-8.3f" % (prb[m] / prb[ref])
                                                if m in prb and prb.get(ref) else "--"
                                                for m in lags)))
    print("    absolute |r_m|/mean|A|^2: satellites m=4 %.3f, probes m=4 %.3f"
          % (sat[ref], prb.get(ref, float("nan"))))

    print("\n  PHASE STEP BY POSITION WITHIN THE FRAME (rad, amplitude-weighted)")
    print("    %-6s %s" % ("prn", "  ".join("%d->%d    " % (j, (j + 1) % REC_PER_FRAME)
                                            for j in range(REC_PER_FRAME))))
    ramps = []
    for p, inc in sorted(incs.items()):
        steps = [inc.get(j, (float("nan"),))[0] for j in range(REC_PER_FRAME)]
        print("    %-6d %s" % (p, "  ".join("%+8.3f" % v for v in steps)))
        good = [v for j, v in enumerate(steps) if j != REC_PER_FRAME - 1
                and math.isfinite(v)]
        if len(good) == REC_PER_FRAME - 1:
            ramps.append(statistics.median(good))
    print()
    if not ramps:
        print("INCONCLUSIVE: no per-position increments.")
        return 1
    ramp = statistics.median(ramps)
    hz = ramp / (2 * math.pi * T_REC)
    within = sat.get(1, 0.0) / sat[ref] if sat.get(ref) else float("nan")
    if within < 0.75 and sat[ref] > 5.0 * prb.get(ref, 1.0):
        print("⚠️ THE PHASE RESETS EVERY FRAME. Coherence at one FRAME (m=4, %.3f) is full, "
              "while m=1 keeps only %.2f of it -- records inside a frame are less coherent "
              "with each other than records a whole frame apart, which no physical "
              "decoherence can do." % (sat[ref], within))
        print("   The steps above are a SAWTOOTH: %+.3f rad per record within the frame "
              "(= %+.1f Hz), reset at the frame boundary. That intra-frame ramp is what a "
              "rate search folding ACROSS frames reports as a bogus few-Hz 'residual rate' "
              "-- and it is why those rates never replicated." % (ramp, hz))
        print("   ⚠️ The carrier phase is being referenced to the FRAME rather than "
              "accumulating. Fix that and the fold has %.2f s of real coherence to use; "
              "until then no injected rate can work, because the series it folds is not one "
              "carrier." % (a.windows * REC_PER_FRAME * T_REC))
    elif sat[ref] <= 5.0 * prb.get(ref, 1.0):
        print("⚠️ the satellites barely beat the probes at m=%d -- no coherence to diagnose."
              % ref)
    else:
        print("✅ NO FRAME STRUCTURE: m=1 keeps %.2f of the one-frame value, so the phase is "
              "not being re-referenced per frame. Step %+.3f rad/record = %+.1f Hz is then a "
              "genuine residual rate." % (within, ramp, hz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
