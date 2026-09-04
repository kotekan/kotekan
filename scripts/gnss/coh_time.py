#!/usr/bin/env python3
"""#61: the fleet sum is coherent ACROSS INSTANCES -- is it coherent ACROSS RECORDS?

    scripts/gnss/coh_time.py [--chain gps_l5] [--prn N] [--windows 32]

THE DISTINCTION THIS EXISTS TO DRAW, because conflating the two is what left the fold's
instability unexplained. telem_align.py measures CROSS-INSTANCE coherence: do the ten
instances agree with each other AT EACH RECORD? Measured 2026-08-14: yes, 0.97 on a bright
satellite. The deep fold needs something ORTHOGONAL to that -- CROSS-RECORD coherence: does the
fleet-summed prompt hold its phase FROM ONE RECORD TO THE NEXT? Ten instances can agree
perfectly at every record (they see the same sky) while the sky phase wanders record to record,
and the coherent sum then collapses. Prior art says exactly that shape: the per-record phase
was 0.984 coherent across instances but only ~0.57 autocorrelated at lag 1.

So a satellite can be "demonstrably coherent" by the instance measure and still have its fleet
deep_snr swing 60% between snapshots -- which is #61's remaining question.

WHAT IS COMPUTED, all from one snapshot of the #59 gather:
  * the fleet-summed per-record series S(t) = sum_i E_i A_i(t), each instance rotated by ONE
    constant phase (arg<A_i conj(A_ref)>) -- the same MRC-weighted sum fleet_coherent forms.
  * coh_frac(L) = <|sum over L consecutive|> / <sum of |.| over L> for a ladder of L. Flat near
    1 = coherent; falling as 1/sqrt(L) = incoherent.
  * A PHASE-SHUFFLED NULL for the same ladder. ⚠️ The null PERMUTES PHASES and keeps the
    amplitudes, so it has the real amplitude distribution and no phase structure -- a null that
    also resampled amplitudes would answer a different question.
  * the phase structure function D(lag) = rms wrapped(phi(t+lag)-phi(t)). FLAT => white
    per-record phase noise; RISING AS sqrt(lag) => a random walk; RISING LINEARLY => an
    unmodelled RATE.
  * a linear rate fit, and the ladder recomputed after derotating it -- i.e. how much of the
    loss a rate search can recover, which is what the C++ fold already tries to do.

Read-only; it consumes the gather stream and computes.
"""
import argparse
import cmath
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "python", "scripts", "gnss"))

from gnss_broker import telem  # noqa: E402

HOPS_PER_SEC = 195312.5


def fleet_series(client, chain, wins, prn):
    """[(hop, S)] -- the MRC fleet sum per record, instances rotated by one constant each."""
    per = {}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                row = f.row(r, prn)
                if row is None:
                    continue
                e = row[telem.REC_P_ENERGY]
                if e > 0.0:
                    per.setdefault(inst, {})[f.hop(r)] = (
                        complex(row[telem.REC_P_RE], row[telem.REC_P_IM]), e)
    if len(per) < 2:
        return [], {}
    # Reference = the instance with the most total energy; rotate every other onto it by ONE
    # constant phase over the whole window. That constant is legitimate (different combs,
    # different NCO history) and is exactly what fleet_coherent removes before summing.
    ref = max(per, key=lambda i: sum(e for _, e in per[i].values()))
    hops = sorted(set.intersection(*[set(d) for d in per.values()]))
    rot = {}
    for inst, d in per.items():
        acc = sum(d[h][0] * per[ref][h][0].conjugate() for h in hops)
        rot[inst] = cmath.exp(-1j * cmath.phase(acc)) if acc != 0 else 1.0
    out = []
    for h in hops:
        s = sum(per[i][h][0] * rot[i] for i in per)
        out.append((h, s))
    return out, per


def _fit_rate(vals, dt):
    """Least-squares linear phase rate on the UNWRAPPED phase -> (Hz, residual rms, derotated).

    Unwrapping is the only delicate part: it assumes the phase moves less than pi per record,
    i.e. the rate is inside +-47.7 Hz at CHORD's record cadence. Beyond that this aliases, and
    it would alias QUIETLY -- so the caller must judge the fit by whether derotating actually
    restores coherence, never by the fitted number alone.
    """
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
    n = len(ph)
    tx = [i * dt for i in range(n)]
    mx, my = statistics.mean(tx), statistics.mean(ph)
    sxx = sum((t - mx) ** 2 for t in tx)
    slope = (sum((tx[i] - mx) * (ph[i] - my) for i in range(n)) / sxx) if sxx else 0.0
    resid = [ph[i] - (my + slope * (tx[i] - mx)) for i in range(n)]
    dero = [vals[i] * cmath.exp(-1j * (my + slope * (tx[i] - mx))) for i in range(n)]
    return slope / (2 * math.pi), statistics.pstdev(resid), dero


def ladder(vals, lengths):
    """coh_frac(L): mean over non-overlapping chunks of |sum| / sum|.|."""
    res = {}
    for L in lengths:
        if L > len(vals):
            continue
        fr = []
        for k in range(0, len(vals) - L + 1, L):
            chunk = vals[k:k + L]
            den = sum(abs(v) for v in chunk)
            if den > 0:
                fr.append(abs(sum(chunk)) / den)
        if fr:
            res[L] = statistics.mean(fr)
    return res


def shuffled_null(vals, lengths, trials, rng):
    """Same amplitudes, phases PERMUTED -- the honest 'no phase structure' reference."""
    amps = [abs(v) for v in vals]
    phs = [cmath.phase(v) for v in vals]
    acc = {}
    for _ in range(trials):
        p = phs[:]
        rng.shuffle(p)
        sh = [a * cmath.exp(1j * q) for a, q in zip(amps, p)]
        for L, f in ladder(sh, lengths).items():
            acc.setdefault(L, []).append(f)
    return {L: statistics.mean(v) for L, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11061)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--prn", type=int, default=None, help="default: the strongest")
    ap.add_argument("--null-trials", type=int, default=64)
    ap.add_argument("--all-prns", action="store_true",
                    help="fit the rate for EVERY PRN in one snapshot. THE DECOMPOSITION: a rate "
                         "common to every satellite is a RECEIVER term (clock / f_carrier); one "
                         "that differs per satellite is PER-SATELLITE (seed staleness, "
                         "ephemeris, rrate). That is the #33 split, measured instead of assumed.")
    a = ap.parse_args()

    c = telem.TelemClient(host=a.host, port=a.port, depth=a.windows + 24, retry_s=1.0).start()
    t0 = time.time()
    while time.time() - t0 < 60 and len(c.windows(a.chain, lag=1)) < a.windows:
        time.sleep(0.5)
    wins = c.windows(a.chain, lag=1)[-a.windows:]
    if not wins:
        c.stop()
        sys.exit("no windows for %r" % a.chain)

    prns = sorted({p for w in wins for f in c.frame_set(a.chain, w).values() for p in f.prns()})

    if a.all_prns:
        print("chain %s, %d windows -- one snapshot, every PRN\n" % (a.chain, len(wins)))
        print("%-4s %-5s %-6s %-10s %-9s %-9s %-9s"
              % ("PRN", "inst", "n_rec", "|A|rms", "rate_Hz", "resid_rad", "cohL/coh0"))
        got = []
        for p in prns:
            s, per = fleet_series(c, a.chain, wins, p)
            if len(s) < 32:
                continue
            v = [x for _, x in s]
            hp = [h for h, _ in s]
            dt_ = (hp[1] - hp[0]) / HOPS_PER_SEC
            rate, resid, dero = _fit_rate(v, dt_)
            L = len(v)
            c0 = ladder(v, [L]).get(L, 0.0)
            c1 = ladder(dero, [L]).get(L, 0.0)
            amp = statistics.mean([abs(x) for x in v])
            print("%-4d %-5d %-6d %-10.3g %-+9.3f %-9.3f %-9s"
                  % (p, len(per), len(v), amp, rate, resid,
                     ("%.3f/%.3f" % (c1, c0))))
            got.append((p, rate, resid, c1, amp))
        c.stop()
        strong = [g for g in got if g[3] > 0.5]   # rate is only meaningful where it cohered
        # ⚠️ FOUR SATELLITES MINIMUM BEFORE ANY VERDICT. With two, "common" and "per-satellite"
        # are not distinguishable by any spread test -- two numbers always have a spread and it
        # always has a ratio to their mean. Saying "COMMON across satellites" off n=2 would be
        # exactly the kind of confident wrong verdict this project keeps paying for.
        if 0 < len(strong) < 4:
            print()
            print("only %d PRN(s) cohered (%s) -- NOT ENOUGH to separate a receiver-common rate"
                  % (len(strong), ", ".join(str(g[0]) for g in strong)))
            print("from a per-satellite one. Rates: %s Hz"
                  % " ".join("%+.2f" % g[1] for g in strong))
            print("Accumulate more snapshots, or run the other chains, before concluding.")
        if len(strong) >= 4:
            rates = [g[1] for g in strong]
            print()
            print("PRNs whose derotated coherence exceeds 0.5 (i.e. the rate is REAL): %s"
                  % ", ".join(str(g[0]) for g in strong))
            print("  their rates: %s Hz" % " ".join("%+.2f" % r for r in rates))
            print("  spread %.2f Hz about a mean of %+.2f Hz"
                  % (max(rates) - min(rates), statistics.mean(rates)))
            print()
            if max(rates) - min(rates) < 0.2 * max(1e-9, abs(statistics.mean(rates))):
                print("  => COMMON across satellites: a RECEIVER term (clock / f_carrier).")
            else:
                print("  => DIFFERS per satellite: a PER-SATELLITE term. A common receiver")
                print("     clock error cannot produce different rates on different rays, so")
                print("     this is seed staleness / ephemeris / rrate territory (#33).")
        return 0

    if a.prn:
        prns = [a.prn]
    else:  # pick the strongest by fleet amplitude
        best, bamp = None, -1.0
        for p in prns:
            s, _ = fleet_series(c, a.chain, wins, p)
            if s:
                m = statistics.mean([abs(v) for _, v in s])
                if m > bamp:
                    best, bamp = p, m
        prns = [best]
    prn = prns[0]

    series, per = fleet_series(c, a.chain, wins, prn)
    c.stop()
    if len(series) < 32:
        sys.exit("PRN %s: only %d common records" % (prn, len(series)))

    hops = [h for h, _ in series]
    vals = [v for _, v in series]
    dt = (hops[1] - hops[0]) / HOPS_PER_SEC
    print("chain %s  PRN %d  %d instances  %d records  dt %.4f s  span %.2f s"
          % (a.chain, prn, len(per), len(vals), dt, len(vals) * dt))
    print()

    Ls = [L for L in (1, 2, 4, 8, 16, 32, 64, 128) if L <= len(vals)]
    rng = random.Random(20260814)
    real = ladder(vals, Ls)
    null = shuffled_null(vals, Ls, a.null_trials, rng)

    # -- linear rate fit on the UNWRAPPED phase ---------------------------------------------
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
    n = len(ph)
    tx = [i * dt for i in range(n)]
    mx, my = statistics.mean(tx), statistics.mean(ph)
    sxy = sum((tx[i] - mx) * (ph[i] - my) for i in range(n))
    sxx = sum((t - mx) ** 2 for t in tx)
    slope = sxy / sxx if sxx else 0.0
    resid = [ph[i] - (my + slope * (tx[i] - mx)) for i in range(n)]
    derot = [vals[i] * cmath.exp(-1j * (my + slope * (tx[i] - mx))) for i in range(n)]
    real_d = ladder(derot, Ls)

    print("COHERENCE vs RECORDS SUMMED  (1.0 = perfect, ~1/sqrt(L) = none)")
    print("  %-5s %-9s %-9s %-9s %s" % ("L", "measured", "derotated", "null", "verdict"))
    for L in Ls:
        r, d, nu = real.get(L), real_d.get(L), null.get(L)
        if r is None:
            continue
        mark = ""
        if L > 1 and nu:
            if r > 2.0 * nu:
                mark = "coherent"
            elif r > 1.3 * nu:
                mark = "partial"
            else:
                mark = "AT THE INCOHERENT NULL"
        print("  %-5d %-9.3f %-9.3f %-9.3f %s" % (L, r, d if d else 0.0, nu if nu else 0.0, mark))
    print()
    print("linear rate over the window: %+.3f Hz  (residual phase rms %.3f rad)"
          % (slope / (2 * math.pi), statistics.pstdev(resid)))
    gain = (real_d.get(max(Ls), 0.0) / real.get(max(Ls), 1e-9)) if real.get(max(Ls)) else 0.0
    print("removing it changes coh_frac at L=%d by x%.2f" % (max(Ls), gain))
    print()

    # -- structure function: WHAT KIND of phase error is it? --------------------------------
    print("PHASE STRUCTURE FUNCTION  D(lag) = rms wrapped(dphi), radians")
    print("  flat => WHITE per-record noise | ~sqrt(lag) => RANDOM WALK | ~lag => a RATE")
    for lag in [l for l in (1, 2, 4, 8, 16, 32, 64) if l < n]:
        d = []
        for i in range(n - lag):
            q = ph[i + lag] - ph[i]
            d.append(q)
        # about the mean, so a pure rate contributes nothing to the spread
        mu = statistics.mean(d)
        rms = statistics.pstdev(d)
        print("    lag %-3d (%.3f s)  D %.3f  (mean %+.3f)" % (lag, lag * dt, rms, mu))
    print()
    print("READING IT: if D(lag) is already ~1.5-2 rad at lag 1 and stays flat, the phase is")
    print("WHITE record to record -- no rate search and no phase tracker can recover it, and")
    print("the coherent sum can never beat sqrt(N). If D grows with lag, the error is a walk")
    print("or a rate and IS recoverable -- which is what the fold's rate search exists for.")


if __name__ == "__main__":
    sys.exit(main())
