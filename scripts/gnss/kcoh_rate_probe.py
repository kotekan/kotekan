#!/usr/bin/env python3
"""WHAT IS THE TRUE RESIDUAL CARRIER RATE, and is the injected one anywhere near it?

The known-rate C/N0 folds at a rate the broker injects (previous cycle's rec_rate). If that
rate is wrong by df, the fold dies in ~1/(2 df) s -- so "coherence is only 0.2 s" and "the
injected rate is off by ~2 Hz" are the SAME statement, and the way to tell them apart is to
stop trusting the injected number and measure the rate directly.

This sweeps the fold rate over a grid at a FIXED long span and reports eta(f). For a real
carrier that curve is a peak of width ~1/T at the true residual rate; for noise it is flat
with a wandering argmax.

⚠️ THIS SEARCHES, WHICH IS EXACTLY WHAT THE SERVED ESTIMATOR REFUSES TO DO. A search over
rate ALWAYS finds a maximum -- that is how the deep fold came to report 41 dB-Hz on noise
(#47/#66). So it is a DIAGNOSTIC, never an estimator, and it ships with the two checks that
make an argmax mean something:

  NULL       the same sweep on the noise probes. Their best eta is what "no signal, free to
             pick its best rate" scores, and the satellite has to beat it clearly.
  STABILITY  the sweep is repeated on DISJOINT time blocks. A real residual rate is the same
             number in every block (drifting slowly at most); noise picks a new argmax each
             time. This is the discriminator -- a single block's argmax proves nothing.

    ./kcoh_rate_probe.py --chain gal_e5a --prn 4         # on the gather host
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


def fold_eta(series, f_hz, hop_s):
    """(eta, n) for one PRN's per-instance series folded at f_hz. eta = |Abar|^2 n / mean|A|^2
    -- n when fully coherent, ~1 on noise. Incoherent across instances, as coh_cn0 does."""
    etas, ns = [], []
    for rows in series.values():
        sacc, pinc, n = 0j, 0.0, 0
        for h, re_, im_, _sre, _sim in rows:
            v = complex(re_, im_)
            sacc += v * cmath.exp(-2j * math.pi * f_hz * h * hop_s)
            pinc += abs(v) ** 2
            n += 1
        if n:
            etas.append((abs(sacc / n) ** 2) * n / (pinc / n) if pinc > 0 else 0.0)
            ns.append(n)
    if not etas:
        return None, 0
    return sum(etas) / len(etas), int(statistics.median(ns))


def sweep(series, grid, hop_s):
    return [(f, fold_eta(series, f, hop_s)[0] or 0.0) for f in grid]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=None, help="default: strongest held")
    ap.add_argument("--span-hz", type=float, default=15.0)
    ap.add_argument("--step-hz", type=float, default=0.25)
    ap.add_argument("--windows", type=int, default=32, help="fold span, windows (~1.34 s)")
    ap.add_argument("--blocks", type=int, default=4, help="disjoint time blocks (stability)")
    ap.add_argument("--seconds", type=float, default=20.0, help="collect per block")
    a = ap.parse_args()

    with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                timeout=10) as r:
        rows = json.loads(r.read().decode())
    probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
    held = [x for x in rows if not x.get("noise_probe")
            and x.get("cn0_prompt_db") is not None
            and (x.get("cn0_prompt_duty") or 0) >= 0.9]
    if not held:
        raise SystemExit("INCONCLUSIVE: nothing held at duty >= 0.9 on %s -- a rate measured "
                         "on an intermittent track is a measurement of the gaps." % a.chain)
    tgt = ([x for x in held if int(x["prn"]) == a.prn] or [None])[0] if a.prn else \
        max(held, key=lambda x: x["cn0_prompt_db"])
    if tgt is None:
        raise SystemExit("PRN %d not held at duty >= 0.9 on %s" % (a.prn, a.chain))
    prn = int(tgt["prn"])
    print("chain %s PRN %d: cn0_inc %.1f dB-Hz, duty %.2f, probes %s"
          % (a.chain, prn, tgt["cn0_prompt_db"], tgt["cn0_prompt_duty"], sorted(probes)))

    n = int(a.span_hz / a.step_hz)
    grid = [i * a.step_hz for i in range(-n, n + 1)]
    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()

    print("\n  block  broker_rate   best_f   eta_peak  eta@0   n_rec   | probe best_f  "
          "probe eta")
    peaks, probe_peaks = [], []
    try:
        for b in range(a.blocks):
            time.sleep(a.seconds)
            with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                        timeout=10) as r:
                live = {int(x["prn"]): x for x in json.loads(r.read().decode())}
            br = live.get(prn, {}).get("rec_rate_hz")
            got = combdll.coh_cn0(cl, a.chain, rates={}, n_win=a.windows,
                                  probe_prns=probes, keep_series=True)
            v = (got or {}).get(prn)
            if not v or "series" not in v:
                print("  %-6d (no fold -- no records this block)" % b)
                continue
            cur = sweep(v["series"], grid, HOP_S)
            bf, be = max(cur, key=lambda t: t[1])
            e0 = dict(cur).get(0.0, 0.0)
            _, nrec = fold_eta(v["series"], bf, HOP_S)
            # NULL: let each probe pick its own best rate too.
            pb = []
            for p in probes:
                pv = (got or {}).get(p)
                if pv and "series" in pv:
                    pb.append(max(sweep(pv["series"], grid, HOP_S), key=lambda t: t[1]))
            pbf, pbe = (max(pb, key=lambda t: t[1]) if pb else (float("nan"), float("nan")))
            peaks.append((bf, be))
            probe_peaks.append(pbe)
            print("  %-6d %-13s %+7.2f  %8.1f  %5.1f  %6d   | %+8.2f  %8.1f"
                  % (b, ("%+.3f" % br) if br is not None else "--", bf, be, e0, nrec,
                     pbf, pbe))
    finally:
        cl.stop()

    if len(peaks) < 2:
        print("\nINCONCLUSIVE: %d usable block(s); stability needs at least 2." % len(peaks))
        return 1
    fs = [p[0] for p in peaks]
    es = [p[1] for p in peaks]
    spread = max(fs) - min(fs)
    print("\n  best_f over %d blocks: %s Hz   spread %.2f Hz   median eta %.1f"
          % (len(fs), " ".join("%+.2f" % f for f in fs), spread, statistics.median(es)))
    print("  probe best eta (the null): median %.1f, max %.1f"
          % (statistics.median(probe_peaks), max(probe_peaks)))
    beats = statistics.median(es) > 3.0 * max(probe_peaks)
    t_span = a.windows * 4 * 2048 * HOP_S
    width = 1.0 / t_span
    # ⚠️ THE STABILITY THAT MATTERS IS WITHIN ONE FOLD, NOT ACROSS BLOCKS. The first cut
    # compared the across-block spread straight to the peak width and concluded "the rate is
    # changing faster than T, so a longer fold cannot work" -- from data showing eta/n = 0.83
    # at the FULL span, which is a 1.3 s fold visibly working. Blocks are ~20 s apart; a rate
    # that drifts 5.5 Hz over 80 s moves 0.09 Hz within a 1.34 s fold, far inside the 0.75 Hz
    # width. A slow DRIFT is not jitter, and only jitter would bound the fold length.
    # ⚠️ DO NOT CALL THIS A DRIFT AND DIVIDE BY THE ELAPSED TIME. The argmax scatters between
    # blocks (+1.25, -5.50, +0.50, +5.75 ...), it does not ramp, so spread/seconds is not a
    # Hz/s anything -- turning it into one produced a reassuring "0.25 Hz within a fold" that
    # was arithmetic, not evidence. The WITHIN-FOLD stability needs no inference at all: eta/n
    # at the full span IS the measurement. 0.78 of n means the rate held across those records.
    med_e = statistics.median(es)
    eta_frac = med_e / float(a.windows * 4)
    print("  peak width ~1/T = %.2f Hz for T = %.2f s" % (width, t_span))
    print("  argmax block-to-block: spread %.2f Hz = %.0f peak widths (scatter, not a ramp)"
          % (spread, spread / width))
    print()
    if not beats:
        print("⚠️ THE SATELLITE DOES NOT BEAT ITS OWN NULL. A free choice of rate scores "
              "about as well on the noise probes, so this sweep has not found a carrier -- "
              "do not read the argmax as a rate.")
        return 1
    print("✅ COHERENT AT THIS SPAN: median eta %.1f of %d records (eta/n %.2f) against a "
          "probe null of %.1f -- the carrier holds for the full %.2f s AT THE RIGHT RATE."
          % (med_e, a.windows * 4, eta_frac, max(probe_peaks), t_span))
    if eta_frac < 0.5:
        print("⚠️ but eta/n < 0.5 even at the best rate, so something OTHER than the rate is "
              "costing this fold -- do not blame the injected rate alone.")
        return 0
    print("   eta/n %.2f at T = %.2f s IS the within-fold stability measurement: the rate "
          "held across all %d records. So the SPAN is not the limit." % (eta_frac, t_span,
                                                                        a.windows * 4))
    print("   WHAT COSTS THE SERVED FOLD is the injected rate. Compare best_f against "
          "broker_rate per block above: every %.2f Hz of error is one whole peak width, and "
          "the fold falls off as sinc^2." % width)
    errs = [abs(f) for f in fs]
    print("   |best_f| over blocks: %s Hz -- the residual the fold actually wants."
          % " ".join("%.2f" % e for e in errs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
