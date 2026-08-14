#!/usr/bin/env python3
"""#61: is a fleet "detection" backed by measurable cross-instance coherence, or fitted?

    scripts/gnss/coh_selfref.py [--chain gps_l5] [--snapshots 5] [--windows 32]

THE SUSPICION. fleet_coherent aligns each instance onto the strongest by a phase FITTED from
the very records it then sums. Where a satellite is bright that is just bookkeeping. Where one
instance's per-record prompt is at the noise -- which is 13 of 16 PRNs, because one instance
sees 7 of ~106 channels -- a fitted phase can align NOISE, and the sum then reports a
confident number built out of nothing. That is the mechanism #10 has circled for weeks
("align 0.15 IS the no-signal value") and a live candidate for #61's churn.

THE TEST. Take ONE snapshot of records from the #59 gather and compute, FROM THE SAME BYTES:

  (a) cross-instance coherence  |<A_i conj(A_j)>| / sqrt(<|A_i|^2><|A_j|^2>), median over all
      instance pairs. FIT-FREE: no per-instance phase is estimated, so this cannot manufacture
      agreement. Its chance floor is 1/sqrt(N_records) and every value is quoted against it.
  (b) fleet_coherent's own deep_snr, its shuffled-null floor, and its `present` verdict.

Both from one snapshot, so the comparison is not across time. Then repeat over several
snapshots to see which numbers are STABLE.

WHAT WOULD FALSIFY THE SUSPICION: if every PRN that clears its null floor also shows
cross-instance coherence well above the chance floor, there is no self-reference to worry
about and the churn is elsewhere. The interesting cell is CLEARS-BUT-AT-FLOOR.

⚠️ The shuffled null is SUPPOSED to catch exactly this -- it destroys the common per-record
phase and re-runs the identical math, alignment fit included, so it measures the
aligned-noise level directly. If a PRN clears that floor while sitting at the coherence floor,
then either the null is not modelling what the estimator does, or the coherence measure is
missing signal the estimator finds. Both are worth knowing; neither is assumed here.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402
from gnss_broker.fleet import fleet_coherent  # noqa: E402


def pair_coherence(client, chain, wins, prn, insts):
    """Median |corr| over instance pairs, and the record count it was computed on."""
    series = {}
    for inst in insts:
        d = {}
        for w in wins:
            f = client.frame_set(chain, w).get(inst)
            if not f:
                continue
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                row = f.row(r, prn)
                if row is None:
                    continue
                e = row[telem.REC_P_ENERGY]
                if e > 0.0:
                    d[f.hop(r)] = complex(row[telem.REC_P_RE] / e, row[telem.REC_P_IM] / e)
        if len(d) >= 16:
            series[inst] = d
    if len(series) < 2:
        return None, 0
    names = sorted(series)
    corrs, nrec = [], 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            di, dj = series[names[i]], series[names[j]]
            hs = sorted(set(di) & set(dj))
            if len(hs) < 16:
                continue
            nrec = max(nrec, len(hs))
            num = sum(di[h] * dj[h].conjugate() for h in hs)
            den = (sum(abs(di[h]) ** 2 for h in hs)
                   * sum(abs(dj[h]) ** 2 for h in hs)) ** 0.5
            if den > 0:
                corrs.append(abs(num) / den)
    return (statistics.median(corrs) if corrs else None), nrec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--snapshots", type=int, default=5)
    ap.add_argument("--gap", type=float, default=8.0, help="seconds between snapshots")
    a = ap.parse_args()

    c = telem.TelemClient(host=a.host, port=a.port, depth=a.windows + 24, retry_s=1.0).start()
    t0 = time.time()
    while time.time() - t0 < 60 and len(c.windows(a.chain, lag=1)) < a.windows:
        time.sleep(0.5)

    rows = {}   # prn -> list of (corr_xfloor, deep_snr, cleared)
    for s in range(a.snapshots):
        wins = c.windows(a.chain, lag=1)[-a.windows:]
        if not wins:
            break
        insts = sorted({i for w in wins for i in c.frame_set(a.chain, w)})
        got, fleet_now = c.coherent_source(a.chain, prns=None, n_win=a.windows, lag=1)
        if not got:
            break
        fc = fleet_coherent([], min_instances=3, min_records=16, prns=None, log=None,
                            floor_margin=3.0, seed=1234 + s, source=(got, fleet_now))
        prns = sorted(fc)
        for prn in prns:
            corr, nrec = pair_coherence(c, a.chain, wins, prn, insts)
            if corr is None or nrec == 0:
                continue
            floor = 1.0 / (nrec ** 0.5)
            rows.setdefault(prn, []).append(
                (corr / floor, fc[prn]["deep_snr"], bool(fc[prn].get("present"))))
        if s + 1 < a.snapshots:
            time.sleep(a.gap)
    c.stop()

    if not rows:
        sys.exit("no data -- is the gather up and this chain sending?")

    print("chain %s, %d snapshots, %d instances, %d records each"
          % (a.chain, a.snapshots, len(insts), a.windows * 4))
    print()
    print("%-4s %-9s %-9s %-7s %-6s %s"
          % ("PRN", "corr/floor", "deep_snr", "CV(snr)", "clears", "verdict"))
    suspects = []
    for prn in sorted(rows):
        v = rows[prn]
        xf = statistics.median([x[0] for x in v])
        sn = [x[1] for x in v]
        cl = sum(x[2] for x in v)
        cv = (statistics.pstdev(sn) / statistics.mean(sn)) if statistics.mean(sn) > 0 else 0.0
        # THE CELL THAT MATTERS: clears the null floor while showing NO measurable
        # cross-instance coherence. That is a detection with nothing fit-free behind it.
        if cl > 0 and xf < 2.0:
            verdict = "*** CLEARS BUT AT COHERENCE FLOOR ***"
            suspects.append(prn)
        elif xf >= 4.0:
            verdict = "coherent, backed"
        elif cl == 0:
            verdict = "never clears (consistent)"
        else:
            verdict = "marginal"
        print("%-4d %-9.1f %-9.1f %-7.2f %-6s %s"
              % (prn, xf, statistics.median(sn), cv, "%d/%d" % (cl, len(v)), verdict))

    print()
    if suspects:
        print("SUSPECTS: PRN %s clear the shuffled-null floor while their cross-instance"
              % ", ".join(str(p) for p in suspects))
        print("coherence sits at the 1/sqrt(N) chance level. Either the null is not modelling")
        print("what the estimator does, or the coherence measure misses what it finds --")
        print("BOTH are worth resolving before any of these numbers reaches a beam map.")
    else:
        print("NO SUSPECTS: every PRN that clears its null floor also shows cross-instance")
        print("coherence above the chance level. The alignment self-reference is NOT")
        print("manufacturing detections here, and #61's churn must be looked for elsewhere.")


if __name__ == "__main__":
    sys.exit(main())
