#!/usr/bin/env python3
"""Is the per-record phase WHITE, or is it a SECONDARY-CODE SIGN FLIP? (#61 / #63)

    scripts/gnss/phase_flip.py [--chain gps_l5] [--prn N] [--windows 32]

WHY THIS EXISTS. coh_time.py says the fleet-summed prompt is incoherent record to record on
every satellite -- coh_frac 0.03-0.17 at L=128 against a ~0.14 null, residual 2.5-10 rad after
a linear rate is removed. That is normally read as "the phase is white, nothing can recover
it". But there is a second thing that produces exactly that number and IS fully recoverable:
the PILOT SECONDARY CODE.

GPS L5-Q carries Neuman-Hoffman 20 (20 ms), Galileo E5a-Q a 100-chip CS, BeiDou B2a-P a 100-
chip one. A record is 10.4857 ms, so consecutive records straddle secondary chips and pick up
a +1 or -1 that the despread does not remove. A pseudo-random +-1 per record makes arg(A) jump
by pi about half the time, which:
  * puts the structure function at ~pi/sqrt(3) = 1.81 rad at LAG 1 and keeps it flat,
  * puts coh_frac at the incoherent null for every L,
  * and is completely destroyed by a rate search, because it is not a ramp.
Indistinguishable from white noise by any of the statistics coh_time.py computes.

THE DISCRIMINATOR IS THE SHAPE OF THE DISTRIBUTION, not its width. White phase gives
d(phi) UNIFORM on [-pi, pi). A sign flip gives it BIMODAL: a peak at 0 (no flip) and a peak at
+-pi (flip), with the width of each peak set by the actual phase noise. So:

    fraction of |d(phi)| within +-pi/4 of 0 or of pi   -- near 1 for a sign flip
                                                       -- near 0.5 for uniform (that is the
                                                          area of those two bands)

and the amplitude test that goes with it: a sign flip leaves |A| UNTOUCHED, so |A| stays
steady while the phase jumps. Real decoherence takes the amplitude with it.

⚠️ THIS DOES NOT PROVE WHICH CODE. It distinguishes "flipping" from "wandering". If it says
flipping, the fix is to wipe the known secondary before folding -- which is what the tracker's
combiner already does with its overlay wipe and what a broker-side fold does NOT get for free,
because the comb ships the prompt BEFORE that wipe.
"""
import argparse
import cmath
import math
import os
import statistics
import sys

K = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402


def fleet_series(client, chain, wins, prn):
    """{hop: summed complex prompt} -- instances aligned by ONE constant phase each.

    The same MRC-weighted, one-constant-per-instance sum fleet_coherent forms, so the series
    tested here is the one the fold actually consumes.
    """
    per = {}   # inst -> {hop: (A, E)}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                cmb = f.comb(r, prn)
                if not cmb:
                    continue
                g = sum(A * e for _fid, A, e in cmb)
                tot = sum(e for _fid, _A, e in cmb)
                if tot > 0:
                    per.setdefault(inst, {})[f.hop(r)] = (g / tot, tot)
    if len(per) < 2:
        return {}
    ref = max(per, key=lambda i: len(per[i]))
    out = {}
    for inst, series in per.items():
        shared = set(series) & set(per[ref])
        if not shared:
            continue
        rot = sum(series[h][0] * per[ref][h][0].conjugate() for h in shared)
        rot = (abs(rot) / rot) if rot != 0 else 1.0 + 0j   # e^{-i arg}, unit modulus
        for h, (A, E) in series.items():
            g, tot = out.get(h, (0j, 0.0))
            out[h] = (g + A * E * rot, tot + E)
    return {h: g / t for h, (g, t) in out.items() if t > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--prn", type=int, default=None, help="default: every PRN")
    a = ap.parse_args()

    cl = telem.TelemClient(a.host, a.port, depth=max(64, a.windows * 2), chains=[a.chain])
    cl.start()
    import time
    t0 = time.time()
    while time.time() - t0 < 60 and len(cl.windows(a.chain, lag=1)) < a.windows:
        time.sleep(1.0)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]
    if len(wins) < 2:
        raise SystemExit("no windows for %r: %s" % (a.chain, cl.stats()))

    prns = [a.prn] if a.prn else sorted(
        {p for w in wins for f in cl.frame_set(a.chain, w).values() for p in f.prns()})
    print("chain %s  %d windows\n" % (a.chain, len(wins)))
    print("PRN   n    |A| cv    near0   nearPi   BAND     uniform=0.50   verdict")
    for prn in prns:
        s = fleet_series(cl, a.chain, wins, prn)
        if len(s) < 16:
            continue
        hops = sorted(s)
        d = []
        amps = []
        for h0, h1 in zip(hops, hops[1:]):
            a0, a1 = s[h0], s[h1]
            if a0 == 0 or a1 == 0:
                continue
            d.append(cmath.phase(a1 * a0.conjugate()))
            amps.append(abs(a0))
        if len(d) < 16:
            continue
        q = math.pi / 4.0
        near0 = sum(1 for x in d if abs(x) <= q) / len(d)
        nearpi = sum(1 for x in d if abs(abs(x) - math.pi) <= q) / len(d)
        band = near0 + nearpi
        cv = statistics.pstdev(amps) / statistics.fmean(amps) if amps else 0.0
        # A uniform phase puts exactly half its mass in those two quarter-pi-wide bands, so
        # 0.50 is the null and anything well above it is structure, not noise.
        verdict = ("SIGN FLIP (bimodal)" if band >= 0.70 else
                   "uniform -- genuinely white" if band <= 0.58 else "ambiguous")
        print("%-4d %4d  %.3f     %.3f   %.3f    %.3f    %s"
              % (prn, len(d), cv, near0, nearpi, band, verdict))
    print("\nnear0 = |dphi| within pi/4 of 0; nearPi = within pi/4 of +-pi; BAND = their sum.")
    print("A UNIFORM phase puts 0.50 of its mass in those bands BY AREA -- that is the null.")
    print("BAND >> 0.5 with a LOW |A| cv means the amplitude is steady while the phase flips:")
    print("an unwiped secondary code, recoverable. BAND ~ 0.5 means genuinely white, and no")
    print("rate search or phase tracker will help.")
    cl.stop()


if __name__ == "__main__":
    main()
