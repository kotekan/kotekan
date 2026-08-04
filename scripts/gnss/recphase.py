#!/usr/bin/env python3
"""What is the per-record prompt phase DOING? -- the question the rung dump forced.

The ladder says the coherent sum reads exactly Rayleigh (~1) on every rung, while the incoherent
moments say the per-record signal is +10..15 dB. Those are only compatible if the per-record
COMPLEX prompt is being scrambled between records, and there are three ways that happens, with
three different fixes:

  SIGN FLIPS      x[k] = +-s: the phase takes two values 180 deg apart. Sum -> ~0, |x| untouched.
                  Cause: an overlay/nav sign the replica is not removing (NH20 misaligned, or a
                  data bit where a pilot was assumed). The squaring test SEES this: x^2 kills a
                  +-1 and leaves a coherent sum, so |sum x^2| / sum|x|^2 -> 1.
  LINEAR RAMP     x[k] = s.exp(j.2pi.df.k.dt): residual Doppler. Sum -> ~0 once df exceeds
                  1/(L.dt). Visible as a PEAK when the records are Fourier-transformed, and the
                  peak's position IS df -- which the tracker could then just correct.
  RANDOM WALK     genuine phase noise (the 8.18.2 story). No structure in either test; the
                  lag-1 correlation is high and decays smoothly with lag.

All three read identically in |sum x|. This separates them:

  * squaring test   |sum x^2| / sum |x|^2      ~1 => sign flips (or a stable phase)
  * lag-1 coherence |sum x[k+1] conj(x[k])| / sum |x|^2   ~1 => phase is smooth record to record
  * FFT over records: peak height and offset => residual Doppler, in Hz
  * phase histogram: two clumps 180 apart => signs; flat => random; a rotating ramp => Doppler

Deliberately runs the same four tests on a PRN's records and on a HOP-SHUFFLED copy of the same
records, because every one of these statistics has a nonzero H0 value and quoting it without the
null is how a statistic gets believed (8.18.5).

usage:  recphase.py [--endpoint URL] [--prn N] [--top 5]
"""
import argparse
import cmath
import json
import math
import random
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://cx19:12049/gnss0_combine")
ap.add_argument("--prn", type=int, default=0)
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--rec-dt", type=float, default=0.0, help="record period s (0 = from status)")
a = ap.parse_args()


def get(what):
    return json.load(urllib.request.urlopen(a.endpoint + what, timeout=15))


st = {int(e["prn"]): e for e in get("/get_status") if e.get("prn")}
recs = {int(e["prn"]): e for e in get("/get_records") if e.get("prn")}
if not recs:
    print("no records exported")
    sys.exit(1)

rec_dt = a.rec_dt
if not rec_dt:
    for e in st.values():
        if (e.get("deep_records") or 0) > 1 and (e.get("coherence_s") or 0) > 0:
            rec_dt = e["coherence_s"] / e["deep_records"]
            break
rec_dt = rec_dt or 0.010486

prns = ([a.prn] if a.prn else
        sorted(recs, key=lambda p: -(st.get(p, {}).get("amp_snr") or 0))[:a.top])


def tests(xs, hops):
    """All four tests, indexed by ABSOLUTE HOP rather than by position in the ring.

    The ring has gaps (a PRN the broker drops for a window contributes no record), so position is
    not time: a 9-record gap counted as one step turns a clean ramp into a scrambled one and puts
    the DFT peak at the wrong frequency, or nowhere. hop is the F-engine's own counter, so
    (hop - hop0) * dt_hop is exact time regardless of what is missing.
    """
    n = len(xs)
    pw = sum(abs(x) ** 2 for x in xs)
    if n < 8 or pw <= 0:
        return None
    dt_hop = rec_dt / 2048.0                 # a record is 2048 hops (fft_len), CHORD-wide
    t = [(h - hops[0]) * dt_hop for h in hops]
    coh = abs(sum(xs)) / math.sqrt(pw * n)               # plain coherent fraction
    sq = abs(sum(x * x for x in xs)) / pw                # SIGN-IMMUNE coherent fraction
    # lag-1 in TIME: only consecutive-in-hop pairs, so a gap does not masquerade as a long lag.
    num, den = 0j, 0.0
    for k in range(n - 1):
        if hops[k + 1] - hops[k] == 2048:
            num += xs[k + 1] * xs[k].conjugate()
            den += abs(xs[k + 1]) * abs(xs[k])
    lag = abs(num) / den if den > 0 else float("nan")
    # Residual Doppler by DFT on the true timebase. Nyquist is set by the RECORD rate; a ramp
    # faster than that aliases, so the reported value is modulo 1/rec_dt.
    best, bf = 0.0, 0.0
    M = 4096
    for i in range(M):
        f = (i / M - 0.5) / rec_dt
        s = abs(sum(x * cmath.exp(-2j * math.pi * f * tk) for x, tk in zip(xs, t)))
        if s > best:
            best, bf = s, f
    return dict(n=n, coh=coh, sq=sq, lag=lag, dop=bf,
                dop_frac=best / math.sqrt(pw * n))


print("record period %.6f s (%.1f Hz), endpoint %s\n" % (rec_dt, 1 / rec_dt, a.endpoint))
print("%-5s %5s %8s %8s %8s %10s %8s   %s"
      % ("prn", "n", "coh", "sq(+-)", "lag1", "dop_hz", "dop_coh", "amp_snr"))
for p in prns:
    rows = recs[p].get("records") or []
    xs = [complex(r[1], r[2]) for r in rows]
    hops = [int(r[0]) for r in rows]
    # Uniform sampling is assumed by the DFT; drop the test if the ring has gaps.
    d = sorted(set(hops[i + 1] - hops[i] for i in range(len(hops) - 1)))
    t = tests(xs, hops)
    if not t:
        continue
    print("%-5d %5d %8.3f %8.3f %8.3f %10.3f %8.3f   %.1f%s"
          % (p, t["n"], t["coh"], t["sq"], t["lag"], t["dop"], t["dop_frac"],
             st.get(p, {}).get("amp_snr") or 0.0,
             "" if len(d) == 1 else "  (GAPPY: hop steps %s)" % d[:4]))

# H0: the same records, shuffled. Order is the only thing every test above uses, so shuffling
# destroys every temporal structure while preserving n and every amplitude exactly.
print("\nNULL (same amplitudes, order shuffled -- what these statistics read on nothing):")
rnd = random.Random(7)
for p in prns[:3]:
    rows = list(recs[p].get("records") or [])
    hops = [int(r[0]) for r in rows]
    xs = [complex(r[1], r[2]) for r in rows]
    rnd.shuffle(xs)              # amplitudes keep their values, lose their times
    t = tests(xs, hops)
    if t:
        print("%-5d %5d %8.3f %8.3f %8.3f %10.3f %8.3f"
              % (p, t["n"], t["coh"], t["sq"], t["lag"], t["dop"], t["dop_frac"]))

# Phase structure of the strongest, printed rather than summarised: 2 clumps 180 apart = signs,
# a slow rotation = Doppler, flat = noise. 16 bins over the full circle.
p0 = prns[0]
xs = [complex(r[1], r[2]) for r in (recs[p0].get("records") or [])]
if xs:
    ph = [cmath.phase(x) for x in xs]
    hist = [0] * 16
    for v in ph:
        hist[int((v + math.pi) / (2 * math.pi) * 16) % 16] += 1
    print("\nPRN %d phase histogram (16 bins, -180..180):" % p0)
    print("  " + " ".join("%2d" % h for h in hist))
    print("  first 24 phases (deg): " + " ".join("%4.0f" % math.degrees(v) for v in ph[:24]))
