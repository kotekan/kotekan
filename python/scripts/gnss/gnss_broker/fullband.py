"""THE FULL-BAND COMBINE: every instance's channels, as one comb, in the broker.

Step 1 of purging the within-instance cross-channel sum (KV, 2026-08-14: "all derived
quantities should happen in the broker, never a tracker, and they should use the full-band
combine we just built"). This runs ALONGSIDE the existing path and is judged against it; the
tracker's sum is deleted only once this is shown better on sky.

WHAT THE TRACKER'S SUM COSTS, measured before building this (scripts/gnss/comb_tau.py, 65
instance/PRN pairs, against a phase-shuffled null): median excess +0.63 dB, but the two
strongest pairs -- the SAME satellite at two independent instances -- gave +6.87 and +6.41 dB
with a consistent ~60 ns delay. A per-node delay is exactly what different cable runs predict.
So on the pairs where it matters, summing an instance's channels blind throws away ~6.5 dB the
array had already paid for, and everywhere it destroys the frequency axis a delay lives on.

THE MODEL. For instance i, channel c (sky frequency f_c), the measured prompt phase is
    arg(A_{i,c}) = k_i - 2*pi*f_c*tau + (sky)
  * k_i is a genuinely ARBITRARY per-instance constant and cannot be derived. The tracker's NCO
    phase is an accumulator zeroed on acquisition and stepped on re-pin, so its origin is that
    instance's own history. One constant per instance is irreducible -- what changes here is
    that it is no longer ALSO absorbing the channel structure.
  * tau is ONE number for the whole fleet: an instrumental delay. Derived, not fitted per
    instance.

⚠️ SO THIS IS NOT "12 PARAMETERS DOWN TO 2". I said that earlier and it was wrong: the
per-instance constant is irreducible. The win is that ~60 channels get COHERENTLY combined
instead of 12 blind sub-sums, each of which had already lost up to 6.5 dB before the fleet ever
saw it.

HOW tau IS FOUND, and why the objective is what it is:
    S(tau) = SUM_i | SUM_c  A_{i,c} * E_{i,c} * exp(+2i*pi*f_c*tau) |
Taking the MAGNITUDE per instance before summing over instances makes S(tau) blind to every
k_i -- so tau is determined by the channel structure alone, with no need to know or fit the
per-instance constants first. A 1-D search, and it uses every instance's channels at once, so
the lever arm is the full ~20 MHz rather than one instance's 1.37 MHz.

⚠️ AMBIGUITY. Within one instance the columns are 16 PFB bins = 3.125 MHz apart, so tau is
unambiguous only within |tau| < 1/(2*3.125e6) = 160 ns. Measured residual delays are 5-124 ns,
comfortably inside -- the ~416 ns cable delay is already absorbed by the code phase the
despread runs at. The search is bounded there and says so rather than wrapping silently.

⚠️ AND IT IS SCORED AGAINST A NULL. A delay search over noisy channels recovers amplitude from
nothing (measured: a median +1.16 dB from a 2-parameter fit to 6 points). The same search is
run on channel-PERMUTED data -- same amplitudes, no ramp -- and only the excess counts. Where
the excess does not clear the null, this falls back to the blind sum, so it can never do worse
than the path it replaces.
"""
import cmath
import math
import random
import statistics

# Unambiguous half-window set by an instance's own comb spacing (16 bins x 195312.5 Hz).
TAU_MAX_S = 1.0 / (2.0 * 16 * 195312.5)   # 160 ns
CHAN_HZ = 195312.5


def _per_channel(client, chain, wins, prn, lag=1):
    """{inst: {freq_id: (coherent mean prompt, total energy, {hop: (A, E)})}}."""
    out = {}
    for w in wins:
        for inst, f in client.frame_set(chain, w).items():
            for r in range(f.n_rec):
                if not f.has_record(r):
                    continue
                cmb = f.comb(r, prn)
                if not cmb:
                    continue
                hop = f.hop(r)
                d = out.setdefault(inst, {})
                for fid, A, e in cmb:
                    s, tot, per = d.get(fid, (0j, 0.0, {}))
                    per[hop] = (A, e)
                    d[fid] = (s + A * e, tot + e, per)
    return out


def _score(per_inst, tau):
    """S(tau) = sum over instances of |sum over channels of mean*exp(+2i pi f tau)|.

    Magnitude per instance FIRST: that is what makes the score independent of the arbitrary
    per-instance constant, so tau comes from the channel structure alone.
    """
    s = 0.0
    for _inst, chans in per_inst.items():
        acc = 0j
        for fid, (mean, _tot, _per) in chans.items():
            acc += mean * cmath.exp(2j * math.pi * fid * CHAN_HZ * tau)
        s += abs(acc)
    return s


def fit_tau_per_instance(per_inst, step_ns=1.0, rng=None, null_trials=32):
    """{inst: (tau, gain, null, excess)} -- ⚠️ RETRACTED AS A MODEL. DIAGNOSTIC ONLY.

    ⚠️⚠️ A PER-NODE DELAY IS PHYSICALLY IMPOSSIBLE HERE AND I SHOULD NOT HAVE PROPOSED ONE.
    Every channel comes from ONE PFB operating on the same raw samples; the split to nodes is
    `freq_id mod 8`, i.e. a routing decision taken AFTER the signal path. A node is where a
    channel happened to land. No physical parameter can localise to one (KV, 2026-08-14 --
    the same mistake, made before, which last time turned out to be a bug).

    WHAT ACTUALLY HAPPENED: the physically correct global fit (fit_tau over the whole fleet)
    DECLINED on 13/13 PRNs, which was the right answer -- there is no measurable delay. Rather
    than accept that, I rescued a marginal per-instance result (5 of 70 pairs clearing a 1 dB
    bar AT 1.00-1.17 dB, i.e. consistent with chance) by inventing a per-node mechanism, and
    read "two instances agreed within 13 ns" as corroboration when a 13 ns coincidence in a
    320 ns search window is a ~4% event.

    Kept ONLY as a diagnostic: scatter in these per-instance values is a measure of the fit's
    own noise, and if it ever becomes SMALL and CONSISTENT that is evidence of a real common
    delay that the global fit should then find with more power, not less. It must never be used
    to align anything.
    """
    out = {}
    rng = rng or random.Random(20260814)
    for inst, chans in per_inst.items():
        one = {inst: chans}
        out[inst] = fit_tau(one, step_ns=step_ns, rng=rng, null_trials=null_trials)
    return out


def fit_tau_joint(per_prn, step_ns=0.5, rng=None, null_trials=32):
    """ONE tau for the whole array, fitted across every instance AND every satellite.

    THE PHYSICALLY CORRECT MODEL, and the strongest form of it. tau is a property of the signal
    path ahead of the PFB, so it is common to every channel and every satellite; fitting it
    jointly multiplies the lever arm by the number of satellites and is the only version of this
    measurement with real power.

    per_prn: {prn: {inst: {fid: (mean, tot, records)}}}. The score sums |.| per (satellite,
    instance) BEFORE summing, so every arbitrary per-instance constant AND every per-satellite
    sky phase drops out -- leaving only the ramp across frequency.
    """
    def score(tau, store):
        s = 0.0
        for _prn, per_inst in store.items():
            s += _score(per_inst, tau)
        return s

    if not per_prn:
        return 0.0, 0.0, 0.0, 0.0
    taus = [(-TAU_MAX_S + i * step_ns * 1e-9)
            for i in range(int(2 * TAU_MAX_S / (step_ns * 1e-9)) + 1)]
    blind = score(0.0, per_prn)
    if blind <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    best_t, best_s = 0.0, blind
    for t in taus:
        v = score(t, per_prn)
        if v > best_s:
            best_t, best_s = t, v
    gain = 20.0 * math.log10(best_s / blind)

    rng = rng or random.Random(20260814)
    nulls = []
    for _ in range(null_trials):
        shuf = {}
        for prn, per_inst in per_prn.items():
            s2 = {}
            for inst, chans in per_inst.items():
                fids = list(chans)
                vals = [chans[f] for f in fids]
                rng.shuffle(vals)
                s2[inst] = {f: v for f, v in zip(fids, vals)}
            shuf[prn] = s2
        nb = score(0.0, shuf)
        if nb <= 0:
            continue
        nulls.append(20.0 * math.log10(max(score(t, shuf) for t in taus) / nb))
    null = statistics.median(nulls) if nulls else 0.0
    return best_t, gain, null, gain - null


def fit_tau(per_inst, step_ns=1.0, rng=None, null_trials=32):
    """(tau_s, gain_dB, null_dB, excess_dB). excess <= 0 means: do not use it.

    Searches ONE tau across whatever instances are passed. Called with a single instance by
    fit_tau_per_instance (the live-validated model); called with the whole fleet it tests the
    common-delay hypothesis, which the sky rejected -- see that function.
    """
    if not per_inst:
        return 0.0, 0.0, 0.0, 0.0
    taus = [(-TAU_MAX_S + i * step_ns * 1e-9)
            for i in range(int(2 * TAU_MAX_S / (step_ns * 1e-9)) + 1)]
    blind = _score(per_inst, 0.0)
    if blind <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    best_t, best_s = 0.0, blind
    for t in taus:
        v = _score(per_inst, t)
        if v > best_s:
            best_t, best_s = t, v
    gain = 20.0 * math.log10(best_s / blind)

    # THE NULL: same amplitudes, channel LABELS permuted within each instance, so any real ramp
    # is destroyed while the search space and the noise are unchanged.
    rng = rng or random.Random(20260814)
    nulls = []
    for _ in range(null_trials):
        shuf = {}
        for inst, chans in per_inst.items():
            fids = list(chans)
            vals = [chans[f] for f in fids]
            rng.shuffle(vals)
            shuf[inst] = {f: v for f, v in zip(fids, vals)}
        nb = _score(shuf, 0.0)
        if nb <= 0:
            continue
        ns = max(_score(shuf, t) for t in taus)
        nulls.append(20.0 * math.log10(ns / nb))
    null = statistics.median(nulls) if nulls else 0.0
    return best_t, gain, null, gain - null


def instance_series(per_inst, tau, hops):
    """{inst: {hop: (A_aligned, E)}} -- channels coherently combined at this tau.

    This is the drop-in replacement for the tracker's cross-channel sum: same shape, same
    meaning, but the channels are phase-aligned by a DERIVED delay instead of added blind.
    tau = 0 reproduces the tracker's sum exactly, which is what makes the fallback honest.
    """
    out = {}
    for inst, chans in per_inst.items():
        acc = {}
        for fid, (_mean, _tot, per) in chans.items():
            rot = cmath.exp(2j * math.pi * fid * CHAN_HZ * tau)
            for hop, (A, e) in per.items():
                if hop not in hops:
                    continue
                g, tot = acc.get(hop, (0j, 0.0))
                acc[hop] = (g + A * e * rot, tot + e)
        # Back to the (A, E) convention the rest of the broker uses: A = G/E.
        out[inst] = {h: (g / tot, tot) for h, (g, tot) in acc.items() if tot > 0.0}
    return out


def fullband_source(client, chain, prns=None, n_win=32, lag=1, min_excess_db=1.0, log=None):
    """(got, fleet_now, info) -- fleet_coherent's input, built from the COMB.

    Same `got` shape fleet_coherent already consumes ({inst: {prn: {hop: (A, E)}}}), so the
    estimator downstream is untouched and the comparison against the polled path stays a
    measurement rather than an argument. THE DIFFERENCE IS WHERE THE CHANNELS WERE COMBINED:
    here, in the broker, from the un-summed comb -- which is the whole of task #63. It is the
    drop-in replacement for TelemClient.coherent_source(), which reads the tracker's SUMMED
    prompt slots and therefore cannot be the input to a deep fold that outlives the sum.

    `min_excess_db` is the guard on the delay. A delay search recovers amplitude from noise,
    so tau is applied only when the excess over the channel-permuted null clears this bar;
    otherwise tau = 0, which reproduces the blind sum exactly. On today's array the joint fit
    does not clear it (+0.867 dB against a 0.409 dB null over 434 channel-series), so this
    ships the blind sum and the tau machinery is dormant, kept for arrays with real spectral
    tilt (KV).
    """
    wins = client.windows(chain, lag=lag)
    if not wins:
        return {}, 0, {}
    wins = wins[-int(n_win):]
    want = None if prns is None else set(int(p) for p in prns)
    all_prns = sorted({p for w in wins for f in client.frame_set(chain, w).values()
                       for p in f.prns()})
    # ONE DELAY FOR THE WHOLE ARRAY, fitted across every instance AND every satellite at once.
    # ⚠️ THIS WAS PER-INSTANCE AND THAT WAS WRONG (retracted 2026-08-14, commit bb9d8739a; the
    # correction stands in fit_tau_per_instance's docstring and [[chord-nothing-is-per-node]]).
    # Every channel comes from ONE PFB on the same raw samples and the split to nodes is
    # `freq_id mod 8` -- routing taken AFTER the signal path -- so a delay cannot localise to
    # an instance. The joint fit is also the STRONGEST form of the measurement, which is what
    # makes its verdict worth acting on: over 434 channel-series it found +0.867 dB against a
    # 0.409 dB null, i.e. NO DETECTION, and this therefore falls through to tau = 0 on today's
    # array. tau = 0 reproduces the blind sum exactly, so that fallback is bit-identical to
    # what the tracker ships and this path can never be worse than the one it replaces.
    per_prn = {}
    for prn in all_prns:
        if want is not None and prn not in want:
            continue
        pc = _per_channel(client, chain, wins, prn)
        if len(pc) >= 2:
            per_prn[prn] = pc
    if not per_prn:
        return {}, 0, {}
    tau, gain, null, excess = fit_tau_joint(per_prn)
    used = tau if excess >= min_excess_db else 0.0
    got, fleet_now, info = {}, 0, {}
    for prn, per_inst in per_prn.items():
        hops = set()
        for chans in per_inst.values():
            for _fid, (_m, _t, per) in chans.items():
                hops |= set(per)
        fleet_now = max(fleet_now, max(hops) if hops else 0)
        for inst, d in instance_series(per_inst, used, hops).items():
            if d:
                got.setdefault(inst, {})[prn] = d
        info[prn] = {"tau_ns": used * 1e9, "gain_db": gain, "null_db": null,
                     "excess_db": excess, "applied": 1 if used else 0,
                     "n_inst": len(per_inst),
                     "n_chan": sum(len(c) for c in per_inst.values())}
    if log:
        log("FULLBAND %s: joint tau %+.1f ns, gain %+.2f dB vs null %+.2f (excess %+.2f dB) "
            "over %d satellites -- %s"
            % (chain, tau * 1e9, gain, null, excess, len(per_prn),
               "APPLIED" if used else "not applied, blind sum (bit-identical to the tracker's)"))
    return got, fleet_now, info
