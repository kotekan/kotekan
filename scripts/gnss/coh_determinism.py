#!/usr/bin/env python3
"""#61 DECISIVE TEST: is fleet_coherent a FUNCTION of its input, or does it wander?

    scripts/gnss/coh_determinism.py [--chain gps_l5] [--trials 3]

WHY. Two same-cycle calls to fleet_coherent over the same twelve endpoints agree at r = 0.077
-- essentially not at all (measured 2026-08-14; scripts/gnss/fixtures/telem_ab/). Two
explanations survive, and they call for completely different work:

  (1) THE INPUTS DIFFERED. Each call does its own REST poll, so each got a slightly different
      set of records; the estimator is fine and the question becomes why a ~1.34 s window of
      records decorrelates from its neighbour (-> the deep fold / phase tracker, #10 #56 #40).
  (2) THE ESTIMATOR IS NON-DETERMINISTIC beyond its shuffled-null seed -- a bug to find.

This separates them by removing the only difference: poll ONCE, then run the estimator
repeatedly on the BIT-IDENTICAL `got` dict.

  [A] same input, SAME null seed      -> must be bit-identical. Anything else is (2).
  [B] same input, DIFFERENT null seed -> the shuffled null sets the FLOOR only, so deep_snr
      must be unchanged and only `floor` may move. If deep_snr moves, the reported VALUE
      depends on the null realisation, which would be its own defect.

Offline and read-only: it polls the same endpoints the broker does and calls the same function.
It changes nothing and needs no restart.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker.fleet import fleet_coherent  # noqa: E402
from gnss_broker.transport import _get, parse_endpoints  # noqa: E402

# The live combiner list, exactly as config/gnss_chains_chord.yaml carries it.
CHAINS = {
    "gps_l5": "gnss{0..1}_n2combine",
    "gal_e5a": "gnss{0..1}_e5a_n2combine",
    "bds_b2a": "gnss{0..1}_b2a_n2combine",
    "gal_e5b": "gnss{0..1}_e5b_n2combine",
    "bds_b2b": "gnss{0..1}_b2b_n2combine",
}
NODES = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]


def poll_once(endpoints, prns=None):
    """Build fleet_coherent's `got` structure -- ONE poll, reused for every trial."""
    got, fleet_now = {}, 0
    for url in endpoints:
        try:
            recs = _get("%s/get_records" % url)
        except Exception as e:
            print("  (%s unreachable: %s)" % (url, e))
            continue
        per = {}
        for r in recs or []:
            prn = int(r.get("prn", -1))
            if prn <= 0:
                continue
            d = {}
            for x in r.get("records") or []:
                try:
                    hop, re_, im_, en = int(x[0]), float(x[1]), float(x[2]), float(x[3])
                except (TypeError, ValueError, IndexError):
                    continue
                fleet_now = max(fleet_now, hop)
                if en > 0.0:
                    d[hop] = (complex(re_, im_), en)
            if d and (prns is None or prn in prns):
                per[prn] = d
        if per:
            got[url] = per
    return got, fleet_now


def run(got, fleet_now, seed):
    return fleet_coherent([], min_instances=3, min_records=16, prns=None, log=None,
                          floor_margin=3.0, seed=seed, source=(got, fleet_now))


def summarize(label, runs, key):
    """Per-PRN spread of `key` across repeated runs on identical input."""
    prns = sorted(set().union(*[set(r) for r in runs]))
    worst, rows = 0.0, []
    for prn in prns:
        vals = [r[prn][key] for r in runs if prn in r]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        rel = spread / max(1e-9, statistics.mean(vals))
        worst = max(worst, rel)
        rows.append((prn, vals, spread, rel))
    print("  %s:" % label)
    for prn, vals, spread, rel in rows:
        flag = "" if rel < 1e-9 else ("   <-- MOVED" if rel > 1e-6 else "  (tiny)")
        print("    PRN %-3d %s  spread %.4g (%.2g%%)%s"
              % (prn, " ".join("%.4f" % v for v in vals), spread, 100 * rel, flag))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default="gps_l5", choices=sorted(CHAINS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--port", type=int, default=12049)
    a = ap.parse_args()

    spec = ",".join("http://%s:%d/%s" % (n, a.port, CHAINS[a.chain]) for n in NODES)
    endpoints = parse_endpoints(spec, "http://cx19:%d" % a.port)
    print("chain %s, %d endpoints" % (a.chain, len(endpoints)))

    got, fleet_now = poll_once(endpoints)
    if not got:
        sys.exit("no records: is the fleet up?")
    n_rec = {u: {p: len(d) for p, d in per.items()} for u, per in got.items()}
    print("polled ONCE: %d instances, PRNs %s"
          % (len(got), sorted(set().union(*[set(p) for p in got.values()]))))
    print("records per instance (first PRN): %s"
          % sorted(v[min(v)] for v in n_rec.values() if v))
    print()

    # -- [A] identical input, identical seed: must be bit-identical --------------------------
    runs = [run(got, fleet_now, seed=12345) for _ in range(a.trials)]
    print("[A] same input, SAME null seed (%d trials) -- must be BIT-IDENTICAL" % a.trials)
    w_snr = summarize("deep_snr", runs, "deep_snr")
    w_flr = summarize("floor", runs, "floor")
    a_ok = w_snr == 0.0 and w_flr == 0.0
    print("    => %s\n" % ("DETERMINISTIC" if a_ok else
                           "*** NOT DETERMINISTIC -- the estimator wanders on fixed input ***"))

    # -- [B] identical input, different seed: the null sets the FLOOR, not the value ---------
    runs_b = [run(got, fleet_now, seed=1000 + i) for i in range(a.trials)]
    print("[B] same input, DIFFERENT null seed (%d trials)" % a.trials)
    w_snr_b = summarize("deep_snr  (must NOT move: the null sets the floor, not the value)",
                        runs_b, "deep_snr")
    summarize("floor     (SHOULD move: it is a different shuffled realisation)",
              runs_b, "floor")
    b_ok = w_snr_b == 0.0
    print("    => %s" % ("deep_snr independent of the null, as designed" if b_ok else
                         "*** deep_snr DEPENDS ON THE NULL SEED -- the value is contaminated "
                         "by its own floor estimate ***"))
    print()

    if a_ok and b_ok:
        print("VERDICT: fleet_coherent IS a function of its input. The r = 0.077 between two")
        print("         same-cycle calls is therefore NOT estimator noise -- the two calls were")
        print("         fed DIFFERENT RECORDS. Next question is why a ~1.34 s window of records")
        print("         decorrelates from its neighbour (#10, #56, #40), not this function.")
    else:
        print("VERDICT: the estimator is not a function of its input. Fix that before")
        print("         attributing any of the churn to the sky or the transport.")
    return 0 if (a_ok and b_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
