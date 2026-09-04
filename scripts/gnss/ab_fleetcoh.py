#!/usr/bin/env python3
"""Paired A/B on the broker's FLEET-COH lines (task #59): gathered vs polled records.

    scripts/gnss/ab_fleetcoh.py <file of FLEET-COH lines>
    # e.g. ssh cf06 "grep -a 'broker gps_l5.*FLEET-COH' /tmp/gnss_broker.log" > ab.txt

WHY THE COMPARISON IS PAIRED AT ALL. This broker runs fleet_coherent TWICE per cycle, and on
CHORD `dll-combiners` and `n2-combiners` are the same YAML anchor -- the same twelve endpoints.
Only the first call takes the telemetry feed (--telem-coherent), so one log line carries both
arms for the SAME satellites at the SAME instant through the SAME estimator over the SAME
instances. A before/after across restarts could never resolve this: the sky churns faster
(2026-08-13, deep_snr max swung 52-197 in four minutes).

⚠️ RUN THE CONTROL TOO. With --telem-coherent OFF, BOTH arms poll REST, so the same analysis
measures the estimator's OWN reproducibility with the transport taken out of the question. Do
that before attributing any A-vs-B disagreement to the transport -- it is the difference
between "the new feed adds noise" and "the estimator was never reproducible".

The two arms are the SAME estimator over the SAME instances at the SAME instant, so the
comparison is PAIRED and the right statistic is a sign test on the pairs -- not a comparison
of the two marginal distributions, which the sky's own churn would dominate (deep_snr max has
been measured swinging 52-197 in four minutes).

⚠️ deep_snr on its own is not the quantity to judge. It is a MAX over ladder rungs and it
scales with record count when phase-limited. What discriminates a working combine from a
broken one is whether a satellite CLEARS ITS OWN SHUFFLED-NULL FLOOR -- the '*' in the log
marks one that did not. So the headline here is the clear-rate, and deep_snr is reported
beside it.
"""
import re
import statistics
import sys

LINE = re.compile(r"FLEET-COH: (.*)$")
# "PRN 3 A 58/10 (best inst 67, floor 4.4) | B 6/10* (best inst 9, floor 5.1)"
ENT = re.compile(
    r"PRN (\d+) A ([\d.]+)/(\d+)(\*?) \(best inst ([\d.]+), floor ([\d.]+)\)"
    r" \| B ([\d.]+)/(\d+)(\*?) \(best inst ([\d.]+), floor ([\d.]+)\)")

pairs = []
for raw in open(sys.argv[1]):
    m = LINE.search(raw)
    if not m:
        continue
    for e in m.group(1).split(";"):
        g = ENT.search(e)
        if not g:
            continue
        pairs.append({
            "prn": int(g.group(1)),
            "a": float(g.group(2)), "a_n": int(g.group(3)), "a_clear": g.group(4) != "*",
            "a_best": float(g.group(5)), "a_floor": float(g.group(6)),
            "b": float(g.group(7)), "b_n": int(g.group(8)), "b_clear": g.group(9) != "*",
            "b_best": float(g.group(10)), "b_floor": float(g.group(11)),
        })

if not pairs:
    sys.exit("no pairs parsed")

n = len(pairs)
print("%d paired samples over %d PRNs, %d polls"
      % (n, len(set(p["prn"] for p in pairs)),
         sum(1 for raw in open(sys.argv[1]) if "FLEET-COH" in raw)))
print()

# -- THE HEADLINE: does a satellite clear its own shuffled-null floor? --------------------
ac = sum(p["a_clear"] for p in pairs)
bc = sum(p["b_clear"] for p in pairs)
both = sum(p["a_clear"] and p["b_clear"] for p in pairs)
neither = sum((not p["a_clear"]) and (not p["b_clear"]) for p in pairs)
a_only = sum(p["a_clear"] and not p["b_clear"] for p in pairs)
b_only = sum(p["b_clear"] and not p["a_clear"] for p in pairs)
print("CLEARS ITS OWN FLOOR (⚠️ SELECTION-BIASED, see below)")
print("  A(gathered) %d/%d = %.1f%%   B(polled) %d/%d = %.1f%%"
      % (ac, n, 100.0 * ac / n, bc, n, 100.0 * bc / n))
print("  both %d   neither %d   A-only %d   B-only %d" % (both, neither, a_only, b_only))
print("  ⚠️ THE FLEET-COH LINE ONLY PRINTS A PRN WHEN AT LEAST ONE ARM CLEARED "
      "(gps_distributed_broker.py: `if not ((a and a.present) or (b and b.present)): continue`).")
print("     So the `neither` cell is STRUCTURALLY ZERO and both absolute rates above are")
print("     inflated -- neither is a detection rate. Do not quote them.")
print("     McNemar below is UNAFFECTED: it uses only the discordant cells, which the")
print("     selection leaves untouched. That is the statistic to read.")
# McNemar: only the DISCORDANT pairs carry information about a difference.
d = a_only + b_only
if d:
    # two-sided sign test on the discordant pairs
    from math import comb
    k = min(a_only, b_only)
    p_val = 2.0 * sum(comb(d, i) for i in range(k + 1)) / (2.0 ** d)
    print("  discordant %d (A-only %d vs B-only %d), sign test p = %.4g%s"
          % (d, a_only, b_only, min(p_val, 1.0),
             "  <-- SIGNIFICANT" if p_val < 0.05 else "  (no detectable difference)"))
print()

# -- deep_snr, paired ---------------------------------------------------------------------
da = [p["a"] for p in pairs]
db = [p["b"] for p in pairs]
wins_a = sum(1 for p in pairs if p["a"] > p["b"])
wins_b = sum(1 for p in pairs if p["b"] > p["a"])
print("deep_snr   A median %.1f  mean %.1f     B median %.1f  mean %.1f"
      % (statistics.median(da), statistics.mean(da),
         statistics.median(db), statistics.mean(db)))
print("  A>B in %d, B>A in %d of %d" % (wins_a, wins_b, n))
try:
    print("  correlation A vs B: %.3f" % statistics.correlation(da, db))
except Exception:
    pass
print()

# -- best SINGLE INSTANCE is the control: it is the same underlying sky in both arms -------
# If the two arms disagree only in the COMBINE, their best-instance numbers should agree far
# better than their fleet numbers. If best-instance ALSO disagrees, the arms are not even
# looking at the same records and the comparison is about window choice, not alignment.
ba = [p["a_best"] for p in pairs]
bb = [p["b_best"] for p in pairs]
print("best single instance   A median %.1f   B median %.1f" % (statistics.median(ba),
                                                                statistics.median(bb)))
try:
    print("  correlation A vs B: %.3f" % statistics.correlation(ba, bb))
except Exception:
    pass
print("  ⚠️ WEAK CONTROL, stated honestly: best_inst_snr is a MAX over ~10 instances, and a")
print("     max is a noisy statistic whose argmax moves -- so a low correlation here is")
print("     SUGGESTIVE that the two arms see effectively independent realisations, not proof.")
print("     A decisive version would feed BOTH arms from the same source and check they agree.")
print()

print("per PRN (median deep_snr, clear-rate):")
for prn in sorted(set(p["prn"] for p in pairs)):
    ps = [p for p in pairs if p["prn"] == prn]
    print("  PRN %-3d n=%-3d  A %5.1f (%3.0f%% clear)   B %5.1f (%3.0f%% clear)"
          % (prn, len(ps),
             statistics.median([p["a"] for p in ps]),
             100.0 * sum(p["a_clear"] for p in ps) / len(ps),
             statistics.median([p["b"] for p in ps]),
             100.0 * sum(p["b_clear"] for p in ps) / len(ps)))
