"""Peer-comparison audit -- driven by DECISION SITES, not by reduction calls.

    python3 scripts/gnss/peer_audit.py

Companion to docs/CHORD_PEER_COMPARISON_PURGE.md. KV's rule (2026-08-27): a peer comparison
can never tell us about a given signal, only fleet-relative properties, so none is justified
anywhere in the tracking / seeding / control loop.

⚠️ THIS IS A LEAD-FINDER, NOT A VERDICT. It flags reductions whose iterable looks per-PRN;
every hit still needs reading, and it MISSES: a bar computed in one function and used in
another, a peer statistic assembled by hand without a named reducer, and any ratio between
two satellites' quantities with no reduction at all. The C++ side is not covered at all --
those 22 sites were read by hand and are listed in the doc.

A peer comparison is: a per-satellite DECISION whose reference is a statistic over OTHER
SATELLITES. So enumerate the decisions, then name each one's reference population.
"""
import os, io, re
from collections import defaultdict

ROOTS = ["python/scripts/gnss/gnss_broker", "python/scripts/gnss"]
# a reference computed over a population, used as a bar
BAR = re.compile(r"\b(\w*(?:floor|bar|thresh|_med|median|_mad|sigma|cut|limit|gate)\w*)\b", re.I)
REDUCERS = ("median(", "mean(", "percentile(", "quantile(", "pstdev(", "stdev(",
            "_floor(", "nth_element", "max(", "min(", "sorted(")
# iterables whose ELEMENTS are distinct satellites
PEER_ITER = re.compile(
    r"(out\.values\(\)|out\.items\(\)|fleet\.values\(\)|fleet\.items\(\)"
    r"|status\.values\(\)|status\.items\(\)|seeds\.values\(\)|seeds\.items\(\)"
    r"|pred\.values\(\)|pred\.items\(\)|\brows\b|\bsats\b|c\.row\b|_ctx\.status)")
PROBE = re.compile(r"probe", re.I)
SAME = re.compile(r"\binst|\bchan|\bsrc\b|rung|\brec\b|record|\bwin\b|hist|series|sample"
                  r"|elem|\bbin\b|spec|resid|per-?prn|one sat|same sat", re.I)

rows = []
seen = set()
for root in ROOTS:
    for dp, _, fs in os.walk(root):
        if "test_" in dp: continue
        for f in sorted(fs):
            if not f.endswith(".py") or f.startswith("test_"): continue
            path = os.path.join(dp, f)
            if path in seen: continue
            seen.add(path)
            lines = io.open(path, encoding="utf-8").readlines()
            for i, ln in enumerate(lines):
                code = ln.split("#")[0]
                if not any(r in code for r in REDUCERS): continue
                if not PEER_ITER.search(code): continue
                ctx = "".join(lines[max(0, i-6):i+2])
                if PROBE.search(code) or PROBE.search(ctx):
                    kind = "probe-anchored (OK)"
                elif SAME.search(code):
                    kind = "same-sat? READ"
                else:
                    kind = "PEER"
                rows.append((kind, path, i+1, " ".join(code.split())))
by = defaultdict(list)
for k, p, l, c in rows: by[k].append((p, l, c))
for k in ("PEER", "same-sat? READ", "probe-anchored (OK)"):
    print("== %s : %d ==" % (k, len(by[k])))
    for p, l, c in by[k]:
        print("   %s:%d\n       %s" % (p.replace("python/scripts/gnss/", ""), l, c[:170]))
    print()
