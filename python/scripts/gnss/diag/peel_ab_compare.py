#!/usr/bin/env python3
"""Compare the two legs of the L5 peel A/B bench.

    peel_ab_compare.py [/tmp/peel_ab]

Reads <dir>/A_peeloff_*_combiner.json and <dir>/B_peelon_*_combiner.json and prints, per
chain, every PRN that either leg saw (deep_snr > 5 or amp_snr > 5), side by side.

THE QUESTION: does enabling the peel corrupt the tracking observables the add-back is
supposed to keep exact? Live 2026-07-26 the corruption was unmissable -- deep_snr 158-296
collapsed to 2-8 on exactly the peeling satellites. On identical replayed voltage the same
mechanism must reproduce; B ~= A (within noise + the peel's <5% cost) means the add-back
holds and the live corruption came from something the bench does not reproduce.
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/peel_ab"

def load(leg, chain):
    p = os.path.join(d, "%s_%s_combiner.json" % (leg, chain))
    try:
        return {int(x["prn"]): x for x in json.load(open(p))}
    except Exception:
        return None

any_fail = False
for chain in ("gps", "gal", "bds"):
    A = load("A_peeloff", chain)
    B = load("B_peelon", chain)
    if A is None or B is None:
        print("%s: missing leg (A=%s B=%s)" % (chain, A is not None, B is not None))
        continue
    prns = sorted(p for p in set(A) | set(B)
                  if max((A.get(p, {}).get("deep_snr") or 0), (B.get(p, {}).get("deep_snr") or 0),
                         (A.get(p, {}).get("amp_snr") or 0), (B.get(p, {}).get("amp_snr") or 0)) > 5)
    if not prns:
        print("%s: no active PRNs in either leg" % chain)
        continue
    print("%s:  %-22s %-22s verdict" % (chain, "A (peel OFF)", "B (peel ON)"))
    for p in prns:
        a, b = A.get(p, {}), B.get(p, {})
        da, db = a.get("deep_snr") or 0, b.get("deep_snr") or 0
        aa, ab_ = a.get("amp_snr") or 0, b.get("amp_snr") or 0
        if da > 10 and db < max(5.0, 0.2 * da):
            v = "*** PEEL ATE IT (the live failure reproduced) ***"
            any_fail = True
        elif da > 10 and db < 0.7 * da:
            v = "degraded"
        else:
            v = "ok"
        print("  PRN%-3d dsnr %6.1f/%5.1f  asnr %6.1f/%5.1f   %s" % (p, da, db, aa, ab_, v))
print()
print("REPRODUCED: the add-back corruption is on the bench -- iterate HERE, never on the sky"
      if any_fail else
      "NOT reproduced: B tracks A -- the add-back holds on this capture/config; the live "
      "corruption needs an ingredient the bench lacks")
