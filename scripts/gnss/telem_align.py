#!/usr/bin/env python3
"""ARE THE INSTANCES ACTUALLY FIT TOGETHER? Per-instance record streams, straight from #59.

    scripts/gnss/telem_align.py [--chain gps_l5] [--windows 32] [--prn N]

TWO SEPARATE QUESTIONS, and this script keeps them apart because conflating them is how the
cross-instance story went wrong twice already (#52, #53):

  [1] ADDRESSING -- do the instances contribute the SAME RECORDS? Since the #59 transport this
      is true BY CONSTRUCTION: every frame carries an absolute window index computed from the
      F-engine sample counter, so equal `win` is the same sky with no tolerance. Reported here
      anyway, because "guaranteed by construction" is a claim and this is the measurement.

  [2] PHASE -- given the same records, do the instances' prompts line up? This is a DIFFERENT
      question and the transport does not touch it. The metric is the pairwise coherence
          |<A_i conj(A_j)>| / sqrt(<|A_i|^2><|A_j|^2>)
      whose MAGNITUDE is insensitive to the arbitrary constant phase offset each instance
      carries (different comb, different NCO history) -- exactly the offset fleet_coherent
      rotates out before summing. So:
        high |corr| => coherent, and a constant rotation is all that is needed (healthy)
        low  |corr| => scattered in a way NO rotation fixes; the fleet sum cannot work
      Prior art: the two GPUs of one node measured 0.86-0.99 at phase 0.00 +- 0.17 rad
      (2026-08-07). The open question has always been ACROSS NODES.

Read-only: it consumes the gather's broker stream and computes. It changes nothing.
"""
import argparse
import cmath
import collections
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402


def collect(host, port, chain, windows, timeout_s):
    c = telem.TelemClient(host=host, port=port, depth=max(64, windows + 8), retry_s=1.0).start()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if len(c.windows(chain, lag=1)) >= windows:
            break
        time.sleep(0.5)
    c.stop()
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--prn", type=int, default=None, help="default: every PRN with signal")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    c = collect(a.host, a.port, a.chain, a.windows, a.timeout)
    wins = c.windows(a.chain, lag=1)[-a.windows:]
    if not wins:
        sys.exit("no windows for chain %r -- is the gather up and is that chain sending?" % a.chain)
    insts = sorted({i for w in wins for i in c.frame_set(a.chain, w)})
    print("chain %s: %d windows, %d instances: %s" % (a.chain, len(wins), len(insts),
                                                      " ".join(insts)))

    # -- [1] ADDRESSING ----------------------------------------------------------------------
    hops_by_inst = collections.defaultdict(set)
    for w in wins:
        for inst, f in c.frame_set(a.chain, w).items():
            for r in range(f.n_rec):
                if f.has_record(r):
                    hops_by_inst[inst].add(f.hop(r))
    union = set().union(*hops_by_inst.values())
    common = set.intersection(*hops_by_inst.values())
    print("[1] ADDRESSING: %d hops in the union, %d shared by ALL %d instances (%.1f%%)"
          % (len(union), len(common), len(insts), 100.0 * len(common) / max(1, len(union))))
    odd = {i: len(union - h) for i, h in hops_by_inst.items() if union - h}
    print("    instances missing any hop: %s" % (odd or "none -- exact agreement"))
    if not common:
        sys.exit("no common hops: addressing is broken, phase is moot")

    prns = ([a.prn] if a.prn else
            sorted({p for w in wins for f in c.frame_set(a.chain, w).values() for p in f.prns()}))

    # -- [2] PHASE ---------------------------------------------------------------------------
    # ⚠️ |corr| HAS A FLOOR AND IT IS NOT ZERO. Two independent noise series of length N
    # correlate at ~1/sqrt(N) by chance, so a value near that says "no measurable common
    # signal" -- which is NOT the same as "the instances are scattered". One instance sees 7 of
    # ~106 channels (-11.8 dB), so its per-record prompt is noise-dominated for all but the
    # brightest satellites, and for those this measurement simply cannot answer the question.
    # Reading 0.09 as "misaligned" when it is the floor would be a confident wrong verdict, so
    # the floor is printed and every value is quoted as a MULTIPLE of it.
    floor = 1.0 / (len(common) ** 0.5)
    print("[2] PHASE across instances, on the %d shared hops" % len(common))
    print("    chance floor for |corr| = 1/sqrt(%d) = %.3f -- values at ~1x are UNMEASURABLE,"
          % (len(common), floor))
    print("    not necessarily misaligned.")
    print("    %-4s %-9s %-8s %-6s  %s"
          % ("PRN", "|A|rms", "med|corr|", "xfloor", "per-pair |corr| spread"))
    for prn in prns:
        series = {}
        for inst in insts:
            d = {}
            for w in wins:
                f = c.frame_set(a.chain, w).get(inst)
                if not f:
                    continue
                for r in range(f.n_rec):
                    if not f.has_record(r) or f.hop(r) not in common:
                        continue
                    row = f.row(r, prn)
                    if row is None:
                        continue
                    e = row[telem.REC_P_ENERGY]
                    if e > 0.0:
                        d[f.hop(r)] = complex(row[telem.REC_P_RE] / e, row[telem.REC_P_IM] / e)
            if len(d) >= max(8, len(common) // 2):
                series[inst] = d
        if len(series) < 2:
            continue
        amp = statistics.median(
            [statistics.mean([abs(v) ** 2 for v in d.values()]) ** 0.5 for d in series.values()])
        corrs, phases = [], []
        names = sorted(series)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                di, dj = series[names[i]], series[names[j]]
                hs = sorted(set(di) & set(dj))
                if len(hs) < 8:
                    continue
                num = sum(di[h] * dj[h].conjugate() for h in hs)
                den = (sum(abs(di[h]) ** 2 for h in hs) * sum(abs(dj[h]) ** 2 for h in hs)) ** 0.5
                if den > 0:
                    corrs.append(abs(num) / den)
                    phases.append(cmath.phase(num))
        if not corrs:
            continue
        corrs.sort()
        # A CONSTANT offset per instance is expected and harmless (fleet_coherent rotates it
        # out); what matters is the MAGNITUDE. Quoting the 10th/90th percentile rather than a
        # mean because one bad instance should be visible, not averaged away.
        med = statistics.median(corrs)
        mark = ("  COHERENT" if med > 4 * floor else
                ("  marginal" if med > 2 * floor else "  at floor"))
        print("    %-4d %-9.3g %-8.3f %-6.1f  min %.3f  p10 %.3f  p90 %.3f  max %.3f (%d pr)%s"
              % (prn, amp, med, med / floor, corrs[0],
                 corrs[max(0, len(corrs) // 10)], corrs[min(len(corrs) - 1, 9 * len(corrs) // 10)],
                 corrs[-1], len(corrs), mark))
    print()
    print("READING IT:")
    print("  COHERENT (>4x floor) -- the instances line up on these records and need only the")
    print("    constant rotation fleet_coherent already applies. The fleet sum is sound.")
    print("  at floor (~1x)       -- NO COMMON SIGNAL IS MEASURABLE at one instance's SNR.")
    print("    That is the expected state for most satellites (one instance = 6.7% of the")
    print("    lobe) and is NOT evidence of misalignment. It is also the regime in which a")
    print("    fitted per-instance phase can align NOISE, so treat a high fleet deep_snr on a")
    print("    PRN sitting at the floor as a claim to CHECK, not a detection (#10, #61).")


if __name__ == "__main__":
    sys.exit(main())
