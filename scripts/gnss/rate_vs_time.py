#!/usr/bin/env python3
"""#61: WHAT is the residual carrier rate doing in time? Steps, drift, or noise?

    scripts/gnss/rate_vs_time.py [--chain gal_e5a] [--prn N] [--seconds 40]

THE SETUP. The prompt's phase is the residual AFTER the tracker despreads against a carrier WE
command, so it advances at (f_true - f_commanded). coh_time.py measured that as a clean linear
ramp of ~1-20 Hz over 1.34 s, and removing it recovers x38 of coherence. The question this
answers is what that frequency error DOES over tens of seconds, because the shape names the
mechanism and the candidates make different predictions:

  SEED STALENESS   -> a SAWTOOTH at the seed POST cadence. The command is refreshed, the error
      resets toward zero, then grows again at dop_rate (~0.6 Hz/s for MEO). ⚠️ the broker posts
      every `interval` (2 s live), which would give only ~1.2 Hz of sag -- so if the sawtooth
      is there but the amplitude is 6-20 Hz, the staleness is NOT the 2 s post cadence and the
      seed's own reference epoch is the thing to look at.
  MODEL ERROR      -> a SLOW SMOOTH DRIFT, no cadence, tracking geometry over minutes.
  A HELD OFFSET    -> flat, non-zero, until something re-pins.
  NOISE            -> no structure; the per-window fits scatter about zero.

Method: fit the rate in short sub-windows (default 32 records = 0.34 s) stepped by half a
window, and print the series plus its autocorrelation. A sub-window is only reported when its
derotated coherence clears --min-coh, so a fit to noise is never plotted as a measurement.

⚠️ THIS IS A SHAPE MEASUREMENT, NOT A MECHANISM PROOF. Read it to choose which mechanism to
test next, not to declare one.
"""
import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402
from coh_time import HOPS_PER_SEC, _fit_rate, fleet_series, ladder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gal_e5a")
    ap.add_argument("--prn", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--sub", type=int, default=32, help="records per rate fit")
    ap.add_argument("--min-coh", type=float, default=0.45)
    a = ap.parse_args()

    need = int(a.seconds / 0.0419)  # windows of 4 records = 41.9 ms
    c = telem.TelemClient(host=a.host, port=a.port, depth=need + 32, retry_s=1.0,
                          chains={a.chain}).start()
    t0 = time.time()
    while time.time() - t0 < a.seconds + 30:
        if len(c.windows(a.chain, lag=1)) >= need:
            break
        time.sleep(1.0)
    wins = c.windows(a.chain, lag=1)[-need:]
    if len(wins) < 32:
        c.stop()
        sys.exit("only %d windows" % len(wins))

    prn = a.prn
    if prn is None:  # strongest
        best, bamp = None, -1.0
        for p in sorted({q for w in wins[-16:] for f in c.frame_set(a.chain, w).values()
                         for q in f.prns()}):
            s, _ = fleet_series(c, a.chain, wins[-16:], p)
            if s:
                m = statistics.mean([abs(v) for _, v in s])
                if m > bamp:
                    best, bamp = p, m
        prn = best
    series, per = fleet_series(c, a.chain, wins, prn)
    c.stop()
    if len(series) < 4 * a.sub:
        sys.exit("PRN %s: only %d records" % (prn, len(series)))

    hops = [h for h, _ in series]
    vals = [v for _, v in series]
    dt = (hops[1] - hops[0]) / HOPS_PER_SEC
    print("chain %s  PRN %d  %d instances  %d records  span %.1f s  sub-window %.2f s"
          % (a.chain, prn, len(per), len(vals), len(vals) * dt, a.sub * dt))
    print()

    step = max(1, a.sub // 2)
    ts, rates = [], []
    for k in range(0, len(vals) - a.sub + 1, step):
        chunk = vals[k:k + a.sub]
        r, resid, dero = _fit_rate(chunk, dt)
        coh = ladder(dero, [a.sub]).get(a.sub, 0.0)
        if coh >= a.min_coh:
            ts.append(k * dt)
            rates.append(r)
    if len(rates) < 8:
        sys.exit("only %d sub-windows cohered above %.2f -- PRN too weak for this measurement"
                 % (len(rates), a.min_coh))

    lo, hi = min(rates), max(rates)
    span = max(1e-9, hi - lo)
    print("residual rate vs time  (%d fits above coh %.2f; %.2f .. %.2f Hz)"
          % (len(rates), a.min_coh, lo, hi))
    for t, r in zip(ts, rates):
        col = int(58 * (r - lo) / span)
        print("  %6.2f s %+8.3f Hz |%s*" % (t, r, " " * col))
    print()
    print("rate: mean %+.3f  sd %.3f  peak-to-peak %.3f Hz"
          % (statistics.mean(rates), statistics.pstdev(rates), span))

    # d(rate)/dt: a seed reset shows as a large one-step jump against a small typical step.
    steps = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    if steps:
        ab = sorted(abs(s) for s in steps)
        med = ab[len(ab) // 2]
        big = [(ts[i + 1], steps[i]) for i in range(len(steps))
               if med > 0 and abs(steps[i]) > 5 * med]
        print("step between fits: median |d| %.3f Hz; %d jumps over 5x that%s"
              % (med, len(big), (" at t = " + ", ".join("%.2f" % t for t, _ in big[:8]))
                 if big else ""))
        if len(big) >= 2:
            gaps = [big[i + 1][0] - big[i][0] for i in range(len(big) - 1)]
            print("  spacing between jumps: %s s" % " ".join("%.2f" % g for g in gaps))
            print("  => compare against the broker's seed POST cadence (`interval`, 2 s live).")

    # Autocorrelation of the rate series -- a periodic sawtooth shows as a clear dip/peak.
    n = len(rates)
    mu = statistics.mean(rates)
    dv = [r - mu for r in rates]
    den = sum(x * x for x in dv)
    print()
    print("autocorrelation of the rate series (lag in fits, %.2f s each):" % (step * dt))
    for lag in range(1, min(24, n // 2)):
        num = sum(dv[i] * dv[i + lag] for i in range(n - lag))
        print("  lag %-3d (%5.2f s)  %+.3f" % (lag, lag * step * dt, num / den if den else 0.0))
    print()
    print("READING IT: a SAWTOOTH at a fixed period => a command that is refreshed and then")
    print("goes stale (look at the period, and at whether the amplitude matches dop_rate x that")
    print("period). A slow smooth DRIFT with no period => a model error. Flat and non-zero =>")
    print("a held offset. No structure => noise, and the ramp is not what it looked like.")


if __name__ == "__main__":
    sys.exit(main())
