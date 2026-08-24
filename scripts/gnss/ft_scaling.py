#!/usr/bin/env python3
"""Does the broker's hot path actually PARALLELISE? -- the measurement no digest can make.

`broker_equiv` proves the free-threaded interpreter computes the same POST stream. It says
nothing about concurrency, because a replay is one chain and deterministic: a race would
reproduce every digest and still be a race (docs/CHORD_FREE_THREADING.md section 6). This
tool asks the two questions the digests leave open, on the real estimator:

  1. SPEEDUP.  N chains' worth of `fleet_coherent` in N threads, against the same work run
     serially. Under the GIL this is ~1.0 by construction. Free-threaded it should approach
     N. This is the entire thesis of the migration, measured on our code rather than assumed
     from PEP 703.

  2. AGREEMENT. Every thread's result compared against the SERIAL result for the same input.
     fleet_coherent is supposed to be pure -- `got` in, per-PRN dict out -- so a difference
     means shared mutable state, which is exactly what the GIL has been hiding.

⚠️ THREADS MUST NOT SHARE INPUT DATA. Each thread gets its own `got` built from its own
seed. Handing every thread the same dict would make agreement trivially true (nothing to
race over) and would measure numpy's read scaling rather than ours -- a gate that cannot
fail. The distinct seeds also mean a per-thread result is only comparable to ITS OWN serial
run, which is why the serial pass is keyed by seed.

⚠️ AND IT IS NOT A SOAK. One pass per thread, seconds total. If the speedup is real it is
visible immediately; if it is not, waiting will not create it.

    venv/bin/python    scripts/gnss/ft_scaling.py --threads 5     # GIL arm (expect ~1x)
    venv-ft/bin/python scripts/gnss/ft_scaling.py --threads 5     # free-threaded arm
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "python", "scripts", "gnss"))

from solve_vec_harness import make_got, cmp_out          # noqa: E402
from gnss_broker import fleet                            # noqa: E402


def solve(got):
    fleet_now = max(h for per in got.values() for d in per.values() for h in d)
    return fleet.fleet_coherent([], 2, 8, log=None, seed=7, hop_rate_hz=195312.5,
                                source=(got, fleet_now))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=5,
                    help="how many chains to simulate (default: today's five)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes per thread, to lift the timed region above start-up noise")
    ap.add_argument("--instances", type=int, default=12, help="fleet size per chain")
    ap.add_argument("--hops", type=int, default=128, help="records in the coherent window")
    a = ap.parse_args()

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print("%s  GIL %s  threads=%d reps=%d shape=%dx%d"
          % (sys.version.split()[0], "ON" if gil else "OFF", a.threads, a.reps,
             a.instances, a.hops))

    # Distinct input per thread -- see the warning in the docstring.
    work = [make_got(seed=100 + i, n_inst=a.instances, n_hop=a.hops)
            for i in range(a.threads)]

    ser_out, t0 = [], time.perf_counter()
    for g in work:
        for _ in range(a.reps):
            r = solve(g)
        ser_out.append(r)
    serial = time.perf_counter() - t0

    par_out = [None] * a.threads
    err = [None] * a.threads

    def worker(i):
        try:
            for _ in range(a.reps):
                r = solve(work[i])
            par_out[i] = r
        except Exception as e:                       # a thread that dies must be LOUD
            import traceback
            err[i] = traceback.format_exc()

    ts = [threading.Thread(target=worker, args=(i,), name="chain%d" % i)
          for i in range(a.threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    par = time.perf_counter() - t0

    print("  serial   %6.3f s" % serial)
    print("  parallel %6.3f s   speedup %.2fx  (ideal %.1fx)"
          % (par, serial / par if par else 0.0, a.threads))

    bad = 0
    for i in range(a.threads):
        if err[i]:
            print("  *** chain%d DIED\n%s" % (i, err[i]))
            bad += 1
            continue
        diff = cmp_out(ser_out[i], par_out[i])
        if diff:
            bad += 1
            print("  *** chain%d DISAGREES with its serial run (%d field(s))" % (i, len(diff)))
            for m in diff[:4]:
                print("        %s" % m)
    print("AGREEMENT %s" % ("OK -- every thread reproduced its serial result"
                            if not bad else "FAILED on %d thread(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
