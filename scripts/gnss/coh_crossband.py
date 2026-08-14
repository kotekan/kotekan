#!/usr/bin/env python3
"""#61/#33 DECISIVE: is the fold-killing carrier residual PHYSICAL or INSTRUMENTAL?

    scripts/gnss/coh_crossband.py [--pair gal] [--windows 32]

THE RESIDUAL. The fleet-summed prompt loses all cross-record coherence to a linear phase rate
of ~5-20 Hz; removing it restores coh_frac from ~0.02 to ~0.85, a x40-70 recovery. So the deep
fold is not fighting noise, it is fighting one unmodelled number per satellite per chain.

THE QUESTION. Two families of cause, and they demand completely different fixes:
  PHYSICAL -- a line-of-sight range-rate error, or a receiver clock frequency error. BOTH are
      proportional to the carrier: residual_Hz = f_carrier * (v/c)  or  f_carrier * (df/f).
      So the SAME satellite seen on TWO carriers must show the SAME SIGN and the ratio
          f_E5b / f_E5a = 1207.14 / 1176.45 = 1.02609
  INSTRUMENTAL -- something in the commanded Doppler / NCO / seed path, which is authored PER
      CHAIN. Then the two bands are independent and their ratio is arbitrary.

Galileo and BeiDou each broadcast BOTH sidebands from ONE modulator on ONE spacecraft, so a PRN
present on both chains is the same ray at the same instant. That is exactly the leverage the
second band was added for.

⚠️ THE COMPARISON MUST BE SIMULTANEOUS. Measuring the two bands even ~40 s apart is worthless
if the residual is seed-driven: the seed refreshes on a ~10 s cadence, so a stale-seed term
resets between the two measurements and any difference is unattributable. This reads BOTH
chains from the SAME absolute window indices of the #59 gather -- identical sky, to the record.
That is only possible because the transport keys on an absolute window; it could not have been
done off the REST polls.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402
from coh_time import HOPS_PER_SEC, _fit_rate, fleet_series, ladder  # noqa: E402

PAIRS = {
    # name: (lower chain, f_lo, upper chain, f_hi)
    "gal": ("gal_e5a", 1176.45e6, "gal_e5b", 1207.14e6),
    "bds": ("bds_b2a", 1176.45e6, "bds_b2b", 1207.14e6),
}


def band_rate(client, chain, wins, prn):
    s, per = fleet_series(client, chain, wins, prn)
    if len(s) < 32:
        return None
    v = [x for _, x in s]
    hp = [h for h, _ in s]
    dt = (hp[1] - hp[0]) / HOPS_PER_SEC
    rate, resid, dero = _fit_rate(v, dt)
    L = len(v)
    return {"rate": rate, "resid": resid, "n": L, "inst": len(per),
            "coh0": ladder(v, [L]).get(L, 0.0), "coh1": ladder(dero, [L]).get(L, 0.0),
            "amp": statistics.mean([abs(x) for x in v])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--pair", default="gal", choices=sorted(PAIRS))
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--min-coh", type=float, default=0.5,
                    help="derotated coherence a band must reach for its rate to be believed")
    a = ap.parse_args()

    lo, f_lo, hi, f_hi = PAIRS[a.pair]
    expect = f_hi / f_lo
    c = telem.TelemClient(host=a.host, port=a.port, depth=a.windows + 24, retry_s=1.0).start()
    t0 = time.time()
    while time.time() - t0 < 60:
        if (len(c.windows(lo, lag=1)) >= a.windows and len(c.windows(hi, lag=1)) >= a.windows):
            break
        time.sleep(0.5)

    # SAME ABSOLUTE WINDOWS ON BOTH CHAINS -- identical sky, to the record.
    common = sorted(set(c.windows(lo, lag=1)) & set(c.windows(hi, lag=1)))[-a.windows:]
    if len(common) < 8:
        c.stop()
        sys.exit("only %d shared windows between %s and %s" % (len(common), lo, hi))
    prns_lo = {p for w in common for f in c.frame_set(lo, w).values() for p in f.prns()}
    prns_hi = {p for w in common for f in c.frame_set(hi, w).values() for p in f.prns()}
    prns = sorted(prns_lo & prns_hi)
    print("%s / %s  --  %d SHARED windows (%d..%d), %d PRNs on both bands"
          % (lo, hi, len(common), common[0], common[-1], len(prns)))
    print("PHYSICAL prediction: rate_hi / rate_lo = %.5f, SAME SIGN" % expect)
    print()
    print("%-4s %-20s %-20s %-9s %s"
          % ("PRN", "%s rate/coh" % lo, "%s rate/coh" % hi, "ratio", "verdict"))

    rows = []
    for prn in prns:
        rl = band_rate(c, lo, common, prn)
        rh = band_rate(c, hi, common, prn)
        if not rl or not rh:
            continue
        if rl["coh1"] < a.min_coh or rh["coh1"] < a.min_coh:
            continue
        ratio = rh["rate"] / rl["rate"] if abs(rl["rate"]) > 1e-9 else float("nan")
        ok = abs(ratio - expect) < 0.15 * expect
        rows.append((prn, rl, rh, ratio, ok))
        print("%-4d %+8.3f Hz / %.3f    %+8.3f Hz / %.3f    %-9.3f %s"
              % (prn, rl["rate"], rl["coh1"], rh["rate"], rh["coh1"], ratio,
                 "PHYSICAL (matches)" if ok else "NOT physical"))
    c.stop()

    print()
    if not rows:
        print("No PRN cohered on BOTH bands this snapshot (need derotated coh > %.2f on each)."
              % a.min_coh)
        print("Re-run: the set of well-tracked satellites turns over in minutes.")
        return 0
    good = sum(1 for r in rows if r[4])
    print("%d/%d PRNs match the physical prediction." % (good, len(rows)))
    if good == len(rows):
        print("=> The residual scales with the CARRIER: it is a range-rate or a receiver clock")
        print("   term, i.e. PHYSICAL, and the vector state (#33) is modelling the right thing.")
    elif good == 0:
        print("=> The residual does NOT scale with the carrier. Same satellite, same ray, same")
        print("   records -- so it cannot be a range rate and cannot be a receiver clock.")
        print("   It is authored PER CHAIN: the commanded Doppler / NCO / seed path. That")
        print("   reframes #33: rrate has been estimating an instrumental term, which is why")
        print("   it reads m/s where physics allows sub-cm/s.")
    else:
        print("=> MIXED. Some satellites scale with the carrier and some do not, so there are")
        print("   likely BOTH a physical term and a per-chain one. Separate them before")
        print("   modelling either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
