#!/usr/bin/env python3
"""WHAT DOES THE WITHIN-INSTANCE CHANNEL SUM ACTUALLY COST? Measure it, do not argue it.

    scripts/gnss/comb_tau.py [--chain gps_l5] [--windows 32]

THE CLAIM TO TEST. GnssGpuRecordAssemble sums each PRN's covering channels COHERENTLY into one
prompt. That is only lossless if the channels are in phase -- and they are not: an instrumental
delay tau puts a ramp -2*pi*f*tau across them. One instance spans ~1.37 MHz (7 channels of
195.3 kHz on a stride-16 comb), and at the cable-scale tau of ~416 ns that is
    2*pi * 1.37e6 * 416e-9 = 3.6 rad
of spread ACROSS ONE INSTANCE'S OWN CHANNELS. If that is real, the tracker's channel sum is not
merely an information loss (the frequency axis, which is what #52 needed) -- it is actively
DESTRUCTIVE, throwing away signal the fleet already paid for.

WHAT IS COMPUTED, per (instance, PRN), from the #59 comb:
  * the per-channel mean prompt over the window, and arg() vs freq_id
  * a least-squares delay tau from that ramp, in ns and in chips
  * |sum of the channels| BLIND (what the tracker does today)  versus
    |sum after derotating by the fitted ramp| (what a broker with the comb can do)
    -> the ratio IS the loss, in dB, that purging the sum recovers.

⚠️ A FIT ACROSS A SPARSE COMB WRAPS. The columns are 16 PFB bins = 3.125 MHz apart, so a delay
beyond ~160 ns already turns the phase more than pi between neighbours and the slope aliases.

⚠️⚠️ AND A 2-PARAMETER FIT TO 6 NOISY POINTS RECOVERS AMPLITUDE FROM NOTHING. I first wrote
that "a wrong tau cannot make the sum bigger" -- that is FALSE, and the data said so immediately
(several pairs came out NEGATIVE, which is only possible because a constrained linear fit can
scatter phases that were already aligned). It cuts the other way too: fitting a slope to noise
reliably makes the derotated sum LARGER. So every gain is quoted against a NULL that applies the
identical fit to the same amplitudes with their phases PERMUTED across channels. Only the excess
over that null is a recovery; the null itself is what the estimator manufactures from nothing.
"""
import argparse
import cmath
import math
import random
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402

CHAN_HZ = 195312.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--min-amp", type=float, default=0.0)
    a = ap.parse_args()

    c = telem.TelemClient(host=a.host, port=a.port, depth=a.windows + 24, retry_s=1.0,
                          chains={a.chain}).start()
    t0 = time.time()
    while time.time() - t0 < 60 and len(c.windows(a.chain, lag=1)) < a.windows:
        time.sleep(0.5)
    wins = c.windows(a.chain, lag=1)[-a.windows:]
    if not wins:
        c.stop()
        sys.exit("no windows for %r" % a.chain)

    # Per (inst, prn, chan): the coherent mean prompt over the window. Averaging over records
    # first is legitimate here -- the residual carrier ramp is COMMON to all channels of an
    # instance, so it scales every column by the same factor and cannot create or hide a ramp
    # ACROSS frequency, which is the only thing being measured.
    acc = {}
    for w in wins:
        for inst, f in c.frame_set(a.chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                for prn in f.prns():
                    for fid, A, e in f.comb(r, prn):
                        k = (inst, prn)
                        d = acc.setdefault(k, {})
                        s, n = d.get(fid, (0j, 0))
                        d[fid] = (s + A * e, n + e)
    c.stop()

    print("chain %s, %d windows (%d records)\n" % (a.chain, len(wins), 4 * len(wins)))
    print("%-9s %-4s %-3s %-11s %-11s %-8s %-8s %-9s %-8s"
          % ("inst", "PRN", "nch", "|blind sum|", "|derot sum|", "gain dB", "null dB",
             "EXCESS dB", "tau ns"))
    rows = []
    for (inst, prn), d in sorted(acc.items()):
        if len(d) < 4:
            continue
        fids = sorted(d)
        vals = [d[f][0] / d[f][1] for f in fids]          # energy-weighted mean per channel
        wts = [d[f][1] for f in fids]
        amp = sum(abs(v) * w for v, w in zip(vals, wts)) / max(1e-30, sum(wts))
        if amp < a.min_amp:
            continue
        # BLIND coherent sum -- exactly what the tracker's cross-channel sum does today.
        blind = abs(sum(v * w for v, w in zip(vals, wts)))
        # DEROTATED: fit a linear phase vs frequency (a delay) and remove it. Unwrapped along
        # the sorted comb; see the header note on aliasing.
        ph, prev, off = [], None, 0.0
        for v in vals:
            q = cmath.phase(v)
            if prev is not None:
                while q + off - prev > math.pi:
                    off -= 2 * math.pi
                while q + off - prev < -math.pi:
                    off += 2 * math.pi
            prev = q + off
            ph.append(prev)
        xs = [(f - fids[0]) * CHAN_HZ for f in fids]
        mx, my = statistics.mean(xs), statistics.mean(ph)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = (sum((xs[i] - mx) * (ph[i] - my) for i in range(len(xs))) / sxx) if sxx else 0.0
        derot = abs(sum(v * w * cmath.exp(-1j * (my + slope * (xs[i] - mx)))
                        for i, (v, w) in enumerate(zip(vals, wts))))
        gain = 20.0 * math.log10(derot / blind) if blind > 0 else 0.0
        tau_ns = -slope / (2 * math.pi) * 1e9
        # THE NULL: the identical fit, same amplitudes, phases PERMUTED across channels. What it
        # recovers is what 2 parameters extract from 6 noisy points with no ramp present.
        rng = random.Random(hash((inst, prn)) & 0xffff)
        nulls = []
        for _ in range(96):
            perm = list(vals)
            rng.shuffle(perm)
            nph, nprev, noff = [], None, 0.0
            for v in perm:
                q = cmath.phase(v)
                if nprev is not None:
                    while q + noff - nprev > math.pi:
                        noff -= 2 * math.pi
                    while q + noff - nprev < -math.pi:
                        noff += 2 * math.pi
                nprev = q + noff
                nph.append(nprev)
            nmy = statistics.mean(nph)
            nsl = (sum((xs[i] - mx) * (nph[i] - nmy) for i in range(len(xs))) / sxx) if sxx else 0.
            nb = abs(sum(v * w for v, w in zip(perm, wts)))
            nd = abs(sum(perm[i] * wts[i] * cmath.exp(-1j * (nmy + nsl * (xs[i] - mx)))
                         for i in range(len(perm))))
            if nb > 0:
                nulls.append(20.0 * math.log10(nd / nb))
        null = statistics.median(nulls) if nulls else 0.0
        excess = gain - null
        rows.append((inst, prn, len(fids), blind, derot, gain, null, excess, tau_ns))
        print("%-9s %-4d %-3d %-11.4g %-11.4g %-+8.2f %-+8.2f %-+9.2f %-8.1f"
              % (inst, prn, len(fids), blind, derot, gain, null, excess, tau_ns))

    if not rows:
        sys.exit("\nno (instance, PRN) had >= 4 comb columns")
    print()
    print("median gain %+.2f dB, median null %+.2f dB, MEDIAN EXCESS %+.2f dB over %d pairs"
          % (statistics.median([r[5] for r in rows]), statistics.median([r[6] for r in rows]),
             statistics.median([r[7] for r in rows]), len(rows)))
    strong = sorted(rows, key=lambda r: -r[3])[:3]
    print("the three STRONGEST pairs (where the phases are actually measurable):")
    for r in strong:
        print("  %-9s PRN %-3d |A| %-10.4g gain %+.2f null %+.2f EXCESS %+.2f  tau %+.1f ns"
              % (r[0], r[1], r[3], r[5], r[6], r[7], r[8]))
    print()
    print("READING IT: EXCESS near 0 means the channels were ALREADY in phase and the tracker's")
    print("sum cost only the frequency AXIS -- still worth purging (the fleet delay fit needs")
    print("it), but it was not destroying signal. A large positive EXCESS on the strong pairs")
    print("would mean the sum was throwing away signal the array had already paid for.")


if __name__ == "__main__":
    sys.exit(main())
