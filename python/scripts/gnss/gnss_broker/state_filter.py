"""The receiver-state filter (task #33, docs/CHORD_JOINT_TRACKING.md section 3).

Built INCREMENTALLY, one state behind one gate at a time -- the #30 lesson is that a
plausible estimator change deployed whole can make a working instrument worse in ways its
own diagnostics excuse. First state to close the loop: the per-satellite slow bias b_sat,
because P1 measured it directly (E5a PRN 3 +0.081 +- 0.016 chips at 5.2 sigma, PRN 23
-0.090 at 6.6 sigma, B2a 36 -0.087 -- while GPS's search-anchored sats hold zero to
+-0.03), it is physical (iono at 1176 MHz is 0.1-0.3 chips, plus tropo and BRDC error),
and it is exactly what the #30 slew institutionalizes today by holding seeds at a model
that does not know it.

The clock/rate/carrier states arrive in SHADOW MODE next: updated and logged beside the
EMAs they will replace, consumed by nothing until the comparison says so.
"""
import math


class SatBiasFilter:
    """Per-satellite slow code-phase bias (chips), estimated from the fleet slope-fit tau.

    tau measures sky-minus-replica directly (P1's sign convention, proven by the GPS
    zero-check and the E/L anti-correlation), so once seeds CONSUME b_sat the measured tau
    becomes the filter's residual: convergence drives tau -> 0 while b_sat holds the bias.
    That closure is also the health metric -- if b_sat wanders instead of holding values
    near P1's open-loop means, the loop is eating its own error signal and must be opened.

    THE THREE GATES, each from a measured P1 caveat rather than taste:
      * PRESENCE (caller-side): weak-sat tau is self-reference-biased toward zero (the
        phi_i solve at tau=0 pre-aligns that point), so a weak sat's "tau ~ 0" is NOT a
        measurement. The caller only feeds tau for sats passing its lock metric, and this
        class additionally requires min_inst instances -- a 4-instance fit during a rolling
        restart read 0.9 chips of scatter where 12 instances read 0.27.
      * INNOVATION: lobe captures land near +-3.27 chips (measured ~0.3% of strong
        windows, NOT flagged by peak/floor) and real biases are < ~0.3, so anything with
        |tau - b| > innovation_max is rejected as an outlier, counted, never averaged in.
      * AGE: iono decorrelates over tens of minutes, so a bias not measured for max_age_s
        stops being APPLIED (get() returns 0) while the value is retained for logging --
        a stale correction silently applied is exactly the un-inspectable state this
        filter exists to remove.

    Slow by design: gain 0.02/update at ~2 s cadence is a ~100 s time constant, far above
    the measurement noise (steady-state sigma_b ~ 0.04 chips at sigma_win 0.4) and far
    below the iono timescale. Clamped because no physical bias at this band is a chip.
    """

    def __init__(self, gain=0.02, clamp=1.0, innovation_max=1.0, max_age_s=600.0,
                 min_inst=6):
        self.gain = float(gain)
        self.clamp = float(clamp)
        self.innovation_max = float(innovation_max)
        self.max_age_s = float(max_age_s)
        self.min_inst = int(min_inst)
        self._b = {}        # prn -> chips
        self._t = {}        # prn -> last accepted-measurement time
        self._n = {}        # prn -> accepted count
        self.rejected = 0   # innovation-gate rejections (lobes), for the log

    def update(self, prn, tau_chips, n_inst, t_now):
        """Feed one presence-gated tau. Returns True if accepted."""
        if n_inst < self.min_inst or not math.isfinite(tau_chips):
            return False
        b = self._b.get(prn, 0.0)
        if abs(tau_chips - b) > self.innovation_max:
            self.rejected += 1
            return False
        b += self.gain * (tau_chips - b)
        self._b[prn] = max(-self.clamp, min(self.clamp, b))
        self._t[prn] = t_now
        self._n[prn] = self._n.get(prn, 0) + 1
        return True

    def get(self, prn, t_now):
        """The bias to APPLY (chips): 0 unless measured recently enough to trust."""
        t = self._t.get(prn)
        if t is None or (t_now - t) > self.max_age_s:
            return 0.0
        return self._b.get(prn, 0.0)

    def summary(self, t_now):
        """One log line's worth: live biases, stale ones marked, rejection count."""
        parts = []
        for prn in sorted(self._b):
            stale = (t_now - self._t.get(prn, 0.0)) > self.max_age_s
            parts.append("%d:%+.3f%s(n%d)" % (prn, self._b[prn],
                                              "*" if stale else "", self._n.get(prn, 0)))
        return "%s rej=%d" % (" ".join(parts) if parts else "-", self.rejected)


class JointReceiverState:
    """P2a: ONE joint solve over [clk, clk_rate, b_sat[i]] (CHORD_JOINT_TRACKING section 3a).

    THE REVISION THIS IMPLEMENTS. SatBiasFilter above estimates b_sat with no clock state,
    which is sound only while the clock is somebody else's solved constant. It is not: the
    BRDC model is per-satellite wrong by +-3-7 chips (docs 11.22), the existing clock is
    the circular MEDIAN of exactly those wrong residuals, and b_sat and clk are degenerate
    within a single epoch -- N sats, N+2 unknowns, N measurements. The ONLY thing that
    separates them is that the clock moves fast and the biases do not, i.e. process noise,
    i.e. a filter. Estimating biases first (SatBiasFilter) means estimating them with no
    clock removed, which is today's failure mode. Hence: one solve, never staged.

    THE MEASUREMENT is the same scalar for every source, which is what makes constellations
    poolable at all:

        y_i = (where the replica actually is, physically) - (pure model, no clock, no bias)

    For a MODEL-PRIMARY sat (E5a/B2a) that is `dr_seed_phys(seed) + dll_trim - cp_predicted`
    plus the fleet slope-fit tau (sky-minus-replica, P1's proven sign). For a SEARCH-ANCHORED
    sat (GPS) it is the dead-reckon integrity residual BEFORE the median is removed -- the
    quantity the broker has logged every cycle for a month while treating its per-sat spread
    as an error rather than as the b_sat measurement it always was. Both satisfy

        y_i = clk + b_i + noise

    so the whole receiver -- three constellations, two seeding disciplines -- contributes to
    one clock.

    WHY THE STATE STAYS SMALL WHERE IT COUNTS. b_sat[i] is per-satellite, so the state grows
    with the fleet; the vector-tracking win survives because what carries a fading satellite
    is BANDWIDTH, not parameter count. clk/clk_rate are fast and shared; each b_i is bounded
    and moves on minute timescales (q_b defaults to ~0.1 chips per 60 s), contributing no
    fast dynamics. A satellite that loses SNR keeps its last b_i frozen and its replica
    keeps moving with the shared clock -- that is the discriminating test in section 3a, and
    `coast_error` exercises it in the selftest.

    THE GAUGE. Adding c to clk and subtracting c from every b_i predicts identical
    measurements: the common mode is structurally unobservable and would random-walk. It is
    pinned by a mean(b) = 0 pseudo-measurement over the ACTIVE sats each cycle, which is not
    an arbitrary choice -- it makes clk "the fleet-mean offset" and b_i "this sat's deviation
    from it", exactly the convention the circular-median clock already implies, so the
    shadow comparison is like-for-like.

    Modulo: every innovation is wrapped into (-L/2, L/2]. The state itself is unwrapped, so
    clk_rate sees a continuous clock.

    ⚠️ ROBUSTNESS IS NOT OPTIONAL HERE, AND IT WAS LEARNED THE EXPENSIVE WAY. This state
    replaces a CIRCULAR MEDIAN, which is outlier-proof by construction; the gauge above is a
    MEAN, which is not. Deployed on 2026-08-09 with no input quality gate, innov_max=None
    and a birth path that bypassed gating, it took ONE weak-sat detection -- the broker's own
    comments document snr<60 giving ~2000-chip residuals -- to walk clk_rate to +0.074 ppm
    in 60 s. On sky it reached -0.028 ppm and fed every unlocked seed 17 chips/min of
    fictitious drift. THE LESSON: when replacing a robust estimator with an efficient one,
    the robustness has to be re-supplied explicitly, or it is simply gone.

    KNOWN AND INTENDED -- THE GAUGE LEAKS 1/N. One satellite whose true bias moves by d
    shifts the fleet mean by d/N, so the gauge moves clk by d/N and every other bias by
    -d/N (measured in the selftest: d = 2.0 chips over 6 sats -> dclk = 0.30). This is not
    an error, it is what "clk is the fleet mean" MEANS, and it is the same convention the
    circular-median clock already had. It does argue for keeping N healthy: with three
    constellations feeding one clock the leak is ~4%, and it is another reason not to let
    a chain solve its own clock in isolation.
    """

    def __init__(self, code_len=10230.0, sigma_clk0=200.0, sigma_rate0=1.0,
                 sigma_b0=10.0, q_clk=1e-3, q_rate=1e-4, q_b=0.013,
                 gauge_sigma=0.1, max_age_s=900.0, innov_max=200.0, innov_nsigma=6.0,
                 reject_escape=5, escape_spread=5.0,
                 birth_max=12.0, gauge_max_b=60.0, escape_max_step=100.0, rate_max=1.0,
                 ref_band=None, sigma_tau0=20.0, q_tau=1e-5,
                 sigma_fcar0=50.0, q_fcar=0.01,
                 sigma_rr0=2.0, q_rr=0.02, rr_gauge_sigma=0.5, f_ref_hz=None,
                 clk0=0.0, escape_min_sats=3, p_floor=0.04):
        import numpy as np
        self._np = np
        self.L = float(code_len)
        self.sigma_clk0 = float(sigma_clk0)
        self.sigma_rate0 = float(sigma_rate0)
        self.sigma_b0 = float(sigma_b0)
        # process noise, per sqrt(second)
        self.q_clk = float(q_clk)      # clock white walk beyond what clk_rate explains
        # clk_rate random walk -- the l-a EMA's job. MEASURED CHOICE, not taste: at 6 sats
        # and sigma 0.3 chips, q_rate 2e-3 gives a rate estimate with sd 0.0035 chips/s
        # (3.4e-4 ppm) while 1e-4 gives 0.00035 and still tracks a 0.01 chips/s ramp with
        # no measurable lag (verified over 3000 s). The estimate is unbiased at every
        # setting, so this trades ONLY noise against agility -- and the thing being tracked
        # is a GPS-disciplined oscillator, whose rate does not wander at milli-chips/s per
        # root-second. Anything looser just re-imports the l-a EMA's +-0.007 ppm scatter,
        # which is what this state exists to remove.
        self.q_rate = float(q_rate)
        self.q_b = float(q_b)          # per-sat bias walk: SLOW is the whole point
        self.gauge_sigma = float(gauge_sigma)
        self.max_age_s = float(max_age_s)
        # ROBUSTNESS, ALL THREE FROM ONE INCIDENT (2026-08-09, see the class note below).
        self.innov_max = innov_max     # absolute garbage ceiling (chips), any state
        self.innov_nsigma = innov_nsigma  # and the UNCERTAINTY-SCALED gate that does the work
        self.reject_escape = reject_escape  # consecutive rejections before we believe reality
        self._rej_run = {}             # key -> consecutive rejections
        # ...and the innovations THEMSELVES, because the escape hatch tests whether the run
        # agrees with itself, not merely that it is long (2026-08-10; see update()).
        self._rej_hist = {}            # key -> recent rejected innovations (rolling)
        self.escape_spread = float(escape_spread)
        # An escape may believe a run that AGREES; it may not believe one that is
        # physically impossible. Same separation argument as the clock solve's MAD
        # bound: real per-sat offsets cluster inside ~+-10 chips while a wrap alias is
        # thousands, so anything from 50 to 500 separates them.
        self.escape_max_step = float(escape_max_step)
        # ...UNLESS THE EVIDENCE IS EXTRAORDINARY. escape_max_step is a flat ceiling, and a
        # flat ceiling cannot express "I would believe this if enough independent things
        # said it". The per-satellite consistency test above is weaker than it looks: it
        # asks whether ONE satellite agrees with ITSELF, and _rej_hist is keyed per
        # satellite, so a single sat feeding coherent nonsense clears it alone. That is
        # precisely how PRN 25 -- transited out of the beam, code phase uniform over 10230
        # chips -- talked the filter into a -1396 chip step on 2026-08-10 with nothing to
        # contradict it.
        #
        # Independent satellites are the axis that test is missing. Their code phases share
        # exactly one thing, the receiver clock, so a step that MANY of them report at once
        # is a clock event almost by definition, while noise has no reason to agree across
        # satellites at all. So a step beyond escape_max_step may still be believed, but
        # only with corroboration: escape_min_sats DISTINCT satellites, each with its own
        # self-consistent run, all implying the same step to within escape_spread.
        #
        # This is the "extraordinary claims" rule with the count done on the right noun.
        # Thirty rejections from nine satellites in thirty seconds are not thirty
        # independent samples -- they are nine satellites and a correlation time -- so the
        # evidence is counted in SATELLITES, never in samples.
        self.escape_min_sats = int(escape_min_sats)
        self._rej_band = {}            # key -> band of its rejected run, for the corroboration
        self.rate_max = float(rate_max)   # chips/s; 1.0 = 0.1 ppm = 2500x the GPSDO truth
        self.diverged = False   # latched: an escape was refused as non-physical
        self.escapes = 0
        # birth_max default 50 -> 12 (2026-08-12): the 00:08 zombie's garbage biases
        # (+-20-46 chips) all slid UNDER 50. Real biases measure <= ~6 (G28 -5.7 the
        # largest ever seen healthy); 12 is 2x margin. Note the gate is open during
        # bootstrap BY DESIGN (P00 >= 100), so this alone cannot stop garbage-era
        # births -- that is the feed warmup's job (broker --joint-feed-warmup-s).
        self.birth_max = birth_max     # refuse to BIRTH a sat implausibly far from clk
        self.gauge_max_b = gauge_max_b # a wild bias does not get a vote in the gauge
        # P FLOOR (2026-08-12, the zombie's terminal symptom). A flood of self-consistent
        # measurements crushed P[0,0] to ~1e-7: gain ~0, sigma printed 0.000, rej=0 -- a
        # zero-gain state cannot MOVE and cannot REJECT, so no escape hatch can ever
        # rescue it. No amount of self-agreeing evidence may claim the clock (or a bias)
        # better than the floor; healthy runs measure sigma 0.05-0.08, so 0.04 caps
        # confidence just below the best honest reading without touching normal operation.
        self.p_floor = float(p_floor)
        # Events worth an operator's attention (escapes, incoherent runs, gauge fallbacks).
        # The filter has no logger of its own, so it queues one-line notes and the broker
        # drains them. The single most damaging event of 2026-08-10 fired SILENTLY.
        self.notes = []
        self.gauge_fallbacks = 0
        # -- tau_band (P3): the per-BAND group delay, a DECLARED STATE ------------------
        # y_i = clk + b_i + tau_band(beta) + noise. Without a tau state the per-band delay
        # has nowhere to live, so it lands in b_sat -- and b_sat is per SATELLITE, so a sat
        # seen in two bands gets two irreconcilable biases and the filter splits the
        # difference. That is not a modelling nicety: it is why the second band needed a
        # hand-wired cross-band clock BOOTSTRAP to track at all (task #34). Here the offset
        # is estimated instead of borrowed.
        #
        # GAUGE: the reference band has NO ROW and is pinned at tau = 0 by construction, so
        # `clk` means "the clock at the reference band" and every other tau is a DIFFERENTIAL
        # delay -- which is the physically meaningful, measurable quantity. Without that pin
        # clk and the taus are exactly degenerate (add c to clk, subtract c from every tau).
        #
        # OBSERVABILITY: tau_beta separates from b_sat ONLY through satellites seen in BOTH
        # bands. One ray, two carriers -- which is precisely what an AltBOC sideband pair
        # gives for free, and precisely what the disjoint E5a/E5b PRN lists destroyed before
        # they were made an identity. With no dual-band satellite the two are degenerate and
        # this filter will say so rather than invent a split (see `tau_observability`).
        #
        # q_tau is TINY: cable and filter delays drift on hours, not seconds (section 3a's
        # timescale table). It is a state that costs almost no degrees of freedom, which is
        # the whole DOF argument for putting it here rather than fitting it per epoch.
        self.ref_band = ref_band       # None -> the first band seen becomes the reference
        self.sigma_tau0 = float(sigma_tau0)
        self.q_tau = float(q_tau)
        self._band_idx = {}            # band -> row in x (the reference band has none)
        self._band_seen = {}           # band -> last time a measurement carried it
        self._dual = {}                # sat key -> set of bands it has been measured in
        # -- f_carrier (P2, 2026-08-10): the RECEIVER-WIDE carrier frequency offset -------
        # Declared as the 3rd element of x in CHORD_JOINT_TRACKING.md section 1 and never
        # built until now. It is created LAZILY, on the first carrier measurement, so a
        # chain that never feeds one keeps the exact state layout (and transcript digests)
        # it had before.
        #
        # WHAT IT IS, AND WHAT IT IS NOT. This is ONE offset shared by every satellite --
        # the right shape for an LO error. The 0.478 Hz seed jitter measured on GPS
        # (section 3d) is PER-SATELLITE, injected by the search -> clock-solve -> seed path,
        # and this state cannot absorb that: a common state fitted to per-sat noise just
        # averages it. Fixing GPS's seed is a separate, larger win and needs no filter.
        # Do not read a healthy f_carrier as evidence that the seed problem is gone.
        #
        # THE MEASUREMENT. Feed `deep_rate_hz`, NOT `carrier_hz_resid`. carrier_hz_resid is
        # documented SIGNAL-FREE at CHORD SNR (0.519 Hz on signal, 0.492 on noise), so a
        # filter fed by it produces a confident, meaningless estimate -- the failure mode of
        # a self-referential phase estimator, which shows up in the VARIANCE, not the mean.
        # deep_rate_hz is already characterised (--carrier-source rate, default since
        # 2026-08-04): peak/median 17.9-22.0 on signal vs 2.8-6.1 on noise, precision ~0.2 Hz
        # by split-half on strong sats, with the gate --carrier-rate-min-q 10.0. Pass that
        # gate's q as the quality cut and ~0.2 Hz as sigma_hz.
        self.sigma_fcar0 = float(sigma_fcar0)
        self.q_fcar = float(q_fcar)    # Hz per sqrt(s); a GPSDO-disciplined LO, so SLOW
        self._fcar_idx = None          # row in x, or None until the first carrier update
        self.n_fcar = 0                # accepted carrier measurements
        self.fcar_rejected = 0

        # -- rrate_sat: the PER-SATELLITE ORBITAL DOPPLER ERROR (task #33 P3, 2026-08-14) --
        # ⚠️ THIS IS NOT A PER-SATELLITE f_carrier, AND THE DISTINCTION IS PHYSICAL. An LO or
        # clock frequency error is ONE number for the whole receiver -- it cannot vary
        # satellite to satellite, and f_carrier above is correctly receiver-wide. What varies
        # per satellite is the ORBIT: a range-rate model error for that satellite, which only
        # becomes Hz when multiplied by a carrier. So this state is held in m/s, where it is
        # BAND-INDEPENDENT and physically meaningful, not in Hz at some reference band.
        #
        #     y_(i,b) [Hz] = -(f_b/c) * rrate_i   +   (f_b/f_ref) * f_carrier   +   noise
        #                     \_ orbit, per sat _/     \_ LO, receiver-wide _/
        #
        # The two are degenerate WITHIN one band -- both scale with f_b -- and are separated
        # only by the same thing that separates clk from b_sat: one is common across
        # satellites and the other is not. Hence one solve, never staged.
        #
        # WHY m/s IS THE RIGHT UNIT AND Hz IS NOT. The same satellite's E5a and E5b Doppler
        # errors are not two numbers, they are ONE range-rate error seen through two carriers
        # (1207.14/1176.45 = 1.0261 apart). Holding the state in m/s makes both bands
        # measurements OF THE SAME ROW; holding it in Hz-at-a-reference-band would work
        # arithmetically but invites exactly the per-band duplication this task exists to
        # remove -- today gal_e5a and gal_e5b fit this quantity independently and never
        # compare, because dop_hist is a local in a per-chain main().
        #
        # THE GAUGE, same argument as b_sat: adding c to f_carrier and subtracting the
        # equivalent from every rrate_i predicts identical measurements, so the common mode is
        # unobservable and would random-walk. mean(rrate) = 0 over active satellites pins it,
        # making f_carrier "the receiver-wide term" and rrate_i "this orbit's deviation".
        #
        # ⚠️ NOT YET COUPLED TO b_sat, DELIBERATELY -- see the note on update_rrate(). b_sat
        # (chips) and rrate (m/s) are the POSITION and VELOCITY of one per-satellite range
        # error and a proper vector state would link them; that link is the carrier-aided code
        # loop and it is the obvious next step. It is not taken here because a per-satellite
        # RATE state is precisely what diverged on mass birth (the filter read an OFFSET as a
        # RATE, 12.5 ppm), and that failure is worth re-earning deliberately rather than by
        # accident.
        # The carrier the receiver-wide f_carrier is EXPRESSED AT. Latched on the first
        # carrier measurement if not given, exactly as ref_band is for tau -- so a
        # single-band deployment never has to state it, and a second band arriving later
        # cannot silently redefine what f_carrier means.
        self.f_ref_hz = float(f_ref_hz) if f_ref_hz else None
        self.sigma_rr0 = float(sigma_rr0)
        self.q_rr = float(q_rr)        # m/s per sqrt(s); orbit error, so SLOW
        # Moderate on purpose, but NOT because a stiffer one breaks it -- that was my first
        # diagnosis and it is WRONG. The freeze measured while building this (one satellite
        # rejected 298 times of 300, innov 17.6 Hz against a 3.1 Hz bar) came from the BIRTH
        # PATH and the missing escape, not the gauge: with those fixed, a 0.02 gauge converges
        # to 2e-11. Recorded because I changed three things at once and credited the wrong one;
        # test_a_stiff_gauge_is_NOT_what_froze_it pins the correction so it stays corrected.
        self.rr_gauge_sigma = float(rr_gauge_sigma)
        self._rr_idx = {}              # key -> row in x
        self._rr_t_seen = {}           # key -> last carrier measurement time
        self._rr_reject_run = {}       # key -> consecutive rejections (escape)
        self.rr_escape_runs = 8
        self.n_rrate = 0
        self.rrate_rejected = 0
        self.x = np.zeros(2)           # [clk chips, clk_rate chips/s]
        # WARM START (2026-08-11). Starting at 0 when the receiver clock is ~151 chips is
        # not a neutral prior, it is a DEADLOCK: every measurement then arrives ~151 chips
        # out, which is beyond escape_max_step (100), so the escape hatch refuses the
        # correction as "a wrap alias, not a clock" and the state rejects 100% of updates
        # forever, sitting at clk ~ -1 and sliding. Observed on sky 2026-08-11 15:25 UTC:
        # 34 rejections agreeing to 0.22 chips, all implying +151.7, all refused.
        #
        # The bound is not wrong in KIND -- the clock IS physically bounded, so a
        # thousand-chip step really is noise -- it was simply set below the very quantity
        # it guards. Whoever creates this state already has a perfectly good clock estimate
        # (the legacy dead-reckon solve); handing it over costs nothing and removes the
        # cold-start case entirely, so the escape machinery below is left to handle only
        # the hard case it was written for: a state poisoned in flight.
        #
        # Exactly the lesson of #28, which fixed this same shape for the dead-reckon clock:
        # a filter that re-bootstraps from 0 on every restart is one restart away from
        # measuring nothing. clk0=0.0 keeps the old behaviour for callers that have no
        # estimate to offer.
        self.x[0] = float(clk0)
        self.P = np.diag([sigma_clk0 ** 2, sigma_rate0 ** 2])
        self._idx = {}                 # key -> row in x
        self._t_seen = {}              # key -> last accepted measurement time
        self._n = {}                   # key -> accepted count
        self._t = None                 # state epoch
        self.rejected = 0
        self.n_updates = 0

    # -- structure ---------------------------------------------------------------
    def _add(self, key, b0, t_now):
        np = self._np
        i = self.x.size
        self.x = np.append(self.x, float(b0))
        P = np.zeros((i + 1, i + 1))
        P[:i, :i] = self.P
        P[i, i] = self.sigma_b0 ** 2
        self.P = P
        self._idx[key] = i
        self._t_seen[key] = t_now
        self._n[key] = 0
        self._membership_changed()
        return i

    def _add_band(self, band, t_now):
        """Give a band its own tau row. The FIRST band seen becomes the reference and gets
        no row at all -- that is the gauge, not an optimisation."""
        np = self._np
        if self.ref_band is None:
            self.ref_band = band
        if band == self.ref_band or band in self._band_idx:
            self._band_seen[band] = t_now
            return self._band_idx.get(band)
        i = self.x.size
        self.x = np.append(self.x, 0.0)
        P = np.zeros((i + 1, i + 1))
        P[:i, :i] = self.P
        P[i, i] = self.sigma_tau0 ** 2
        self.P = P
        self._band_idx[band] = i
        self._band_seen[band] = t_now
        return i

    @staticmethod
    def _key_str(key):
        """(tag, prn) -> 'G25'. Keys are chain-tagged tuples; notes want the sat."""
        try:
            return "%s%d" % (key[0][0].upper(), int(key[1]))
        except Exception:
            return str(key)

    def _note(self, msg):
        """Queue a one-line operator note; the broker drains `notes` each cycle. Bounded, so
        a pathological chain cannot grow this without limit."""
        self.notes.append(msg)
        if len(self.notes) > 200:
            del self.notes[:-200]

    def drain_notes(self):
        out, self.notes = self.notes, []
        return out

    def _drop(self, keys):
        """Remove stale sats. Their rows leave the state entirely -- a sat that has not
        been measured in max_age_s must not keep voting in the gauge, which would drag the
        clock toward a bias nothing is refreshing."""
        if not keys:
            return
        np = self._np
        rows = sorted(self._idx[k] for k in keys)
        keep = [i for i in range(self.x.size) if i not in set(rows)]
        self.x = self.x[keep]
        self.P = self.P[np.ix_(keep, keep)]
        for k in keys:
            del self._idx[k], self._t_seen[k], self._n[k]
            self._rej_run.pop(k, None)
            self._rej_hist.pop(k, None)
            self._rej_band.pop(k, None)
        # Reindex from the SURVIVING ROW ORDER, not from a hardcoded base of 2. The old
        # form assumed every row past clk_rate was a satellite; tau_band rows share that
        # space now, and a base-2 renumber would silently alias a band onto a satellite.
        remap = {old_row: new_row for new_row, old_row in enumerate(keep)}
        self._idx = {k: remap[i] for k, i in self._idx.items() if i in remap}
        self._band_idx = {b: remap[i] for b, i in self._band_idx.items() if i in remap}
        # f_carrier rides the same remap. Missing this is how a receiver-wide state silently
        # aliases onto a satellite's bias row the first time a sat ages out.
        if self._fcar_idx is not None:
            self._fcar_idx = remap.get(self._fcar_idx)
        # rrate rides the same remap. A per-satellite row that is not remapped points at
        # ANOTHER satellite's state after a drop -- silent, and wrong in the worst way.
        self._rr_idx = {k: remap[i] for k, i in self._rr_idx.items() if remap.get(i) is not None}
        self._rr_t_seen = {k: v for k, v in self._rr_t_seen.items() if k in self._rr_idx}
        for k in keys:
            self._dual.pop(k, None)
        self._membership_changed()

    def _membership_changed(self):
        """A satellite joining or leaving changes WHICH sats the mean(b)=0 gauge averages
        over, so the gauge legitimately steps clk (~1 chip at 6 sats -- measured). That step
        is a GAUGE artefact, not clock motion, and must not be differentiated into clk_rate:
        untreated it spikes the rate ~0.005 chips/s per event, which is 12x the true value.
        Break the correlation and widen the clock so the step is absorbed as offset."""
        self.P[0, 1] = self.P[1, 0] = 0.0
        self.P[0, 0] += 4.0

    # -- filter ------------------------------------------------------------------
    def predict(self, t_now):
        if self._t is None:
            self._t = t_now
            return
        dt = t_now - self._t
        if dt <= 0.0:
            return
        self._t = t_now
        np = self._np
        n = self.x.size
        F = np.eye(n)
        F[0, 1] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T
        # Q: clock gets the integrated-rate terms (nearly-constant-velocity), biases a
        # plain random walk. Off-diagonal clk/rate coupling matters -- without it the
        # filter believes a rate error leaves the clock untouched and under-weights fresh
        # clock measurements after a gap.
        qr = self.q_rate ** 2
        self.P[0, 0] += self.q_clk ** 2 * dt + qr * dt ** 3 / 3.0
        self.P[0, 1] += qr * dt ** 2 / 2.0
        self.P[1, 0] += qr * dt ** 2 / 2.0
        self.P[1, 1] += qr * dt
        if n > 2:
            brows = set(self._band_idx.values())
            # The f_carrier row is NOT a satellite bias and must not collect q_b -- it is
            # in Hz, not chips, and q_b would inflate it by ~1000x its own process noise.
            other = set(brows)
            if self._fcar_idx is not None:
                other.add(self._fcar_idx)
            # rrate rows are m/s, not chips -- q_b would inflate them by ~1000x their own
            # process noise, the same trap the f_carrier line above avoids.
            other |= set(self._rr_idx.values())
            srows = [i for i in range(2, n) if i not in other]
            if srows:
                idx = np.array(srows)
                self.P[idx, idx] += self.q_b ** 2 * dt
            for i in brows:
                # hours-timescale state: q_tau is ~100x below q_b on purpose
                self.P[i, i] += self.q_tau ** 2 * dt
            if self._fcar_idx is not None:
                self.P[self._fcar_idx, self._fcar_idx] += self.q_fcar ** 2 * dt
            for i in self._rr_idx.values():
                # orbit error: moves with geometry, so faster than the GPSDO-disciplined LO
                # but far slower than a chip bias
                self.P[i, i] += self.q_rr ** 2 * dt
        # P floor -- see __init__. Applied to clk and the satellite-bias rows (NOT tau,
        # NOT f_carrier: different units and timescales own their own priors). The floor
        # goes on the DIAGONAL only; correlations keep whatever structure they earned.
        if self.p_floor > 0.0:
            f2 = self.p_floor ** 2
            if self.P[0, 0] < f2:
                self.P[0, 0] = f2
            if n > 2:
                other = set(self._band_idx.values())
                if self._fcar_idx is not None:
                    other.add(self._fcar_idx)
                other |= set(self._rr_idx.values())
                for i in range(2, n):
                    if i not in other and self.P[i, i] < f2:
                        self.P[i, i] = f2

    def _scalar_update(self, H, y, R):
        """One scalar measurement, innovation already wrapped by the caller."""
        np = self._np
        PH = self.P @ H
        S = float(H @ PH + R)
        if S <= 0.0:
            return 0.0
        K = PH / S
        self.x = self.x + K * y
        # Joseph form: this filter runs for days at 2 s cadence with sats entering and
        # leaving, and the simple (I-KH)P loses symmetry/positivity over that many updates.
        n = self.x.size
        A = np.eye(n) - np.outer(K, H)
        self.P = A @ self.P @ A.T + np.outer(K, K) * R
        self.P = 0.5 * (self.P + self.P.T)
        return y / math.sqrt(S)

    def wrap(self, v):
        return ((v + self.L / 2.0) % self.L) - self.L / 2.0

    def update(self, key, y_chips, sigma_chips, t_now, band=None):
        """Feed one y_i = clk + b_i observation. Returns the normalized innovation, or
        None if rejected/deferred. A NEW satellite is BORN AT ITS MEASUREMENT (b0 = y - clk)
        rather than walked in from zero at loop gain: the model error it must absorb is
        chips, and starting at zero would drag the clock for minutes (section 3a)."""
        np = self._np
        if not (math.isfinite(y_chips) and sigma_chips > 0.0):
            return None
        self.predict(t_now)
        # Resolve this measurement's band BEFORE the birth gate: a satellite first seen in a
        # non-reference band is offset by tau, and judging its plausibility against the bare
        # clock would reject exactly the newcomers a second band brings in.
        _tau_row = self._add_band(band, t_now) if band is not None else None
        _tau_val = float(self.x[_tau_row]) if _tau_row is not None else 0.0
        if band is not None:
            self._dual.setdefault(key, set()).add(band)
        born = key not in self._idx
        if born:
            # BIRTH THROUGH THE COVARIANCE, not by hand. The obvious version -- set
            # b0 = y - clk and return -- puts the whole offset into the bias, and since a
            # newborn row has ZERO cross-covariance with clk, the mean(b)=0 gauge then
            # shears that offset straight back out of the biases without ever depositing it
            # in the clock. Deployed 2026-08-09 and caught on the first shadow line: six
            # sats born at once, biases spread +-4.7 chips, and `clk +0.000 +- 200` -- the
            # clock's absolute value simply lost. It recovered over ~400 cycles by grinding
            # a 151-chip innovation down, which is both slow and a wrap hazard: at a clock
            # near L/2 that innovation folds and the filter converges to an ALIAS.
            # Adding the row at b0 = 0 with its prior sigma and running a normal update
            # instead lets the gains do it: sigma_clk0 >> sigma_b0 means the first
            # satellite's innovation lands ~entirely in clk (the clock is what is unknown),
            # while later ones -- facing an already-determined clock -- land in their own
            # bias. Same intent, correct mechanism, no special case.
            # BIRTH WINDOW. Birth is the one path with no innovation to gate on, which is
            # exactly how the 2026-08-09 incident got in. Once the clock is determined,
            # refuse a newborn whose measurement is implausibly far from it: a real bias is
            # chips, a weak-sat detection is hundreds. While the clock is still unknown
            # (large P00) everything is accepted, because that is the bootstrap.
            if (self.birth_max is not None and len(self._idx) >= 2
                    and self.P[0, 0] < 100.0
                    and abs(self.wrap(y_chips - self.x[0] - _tau_val)) > self.birth_max):
                self.rejected += 1
                return None
            self._add(key, 0.0, t_now)
        i = self._idx[key]
        H = np.zeros(self.x.size)
        H[0] = 1.0
        H[i] = 1.0
        # y_i = clk + b_i + tau_band. The reference band has NO row (tau == 0 by the gauge),
        # so this term simply does not appear for it -- which is what makes `clk` mean "the
        # clock at the reference band" rather than an unpinned average of both.
        if _tau_row is not None:
            H[_tau_row] = 1.0
        innov = self.wrap(y_chips - float(H @ self.x))
        # NOT on the birth cycle: a newborn sits at b=0, so its innovation is the WHOLE
        # measurement (~the clock, 151 chips here) and any physical gate would reject it --
        # which is precisely what happened when innov_max was first given a finite value:
        # every satellite was born and then immediately refused, and the filter read zero
        # forever. Birth is vetted by birth_max above; the innovation gate is for sats that
        # already have a state to be inconsistent with.
        # THE GATE IS NORMALIZED, not absolute. A fixed chip bound cannot be both tight
        # enough to catch garbage on a converged state and loose enough to let a legitimate
        # innovation through after a gap -- set to 30 chips it evicted the ENTIRE state
        # after a 900 s outage, because the clock's extrapolated uncertainty had genuinely
        # grown past the bound and every satellite was refused, then aged out. Scaling by
        # sqrt(S) is the filter saying "I no longer know where the clock is", which is the
        # correct response to an outage. The absolute innov_max stays as a garbage ceiling.
        if not born:
            PH_ = self.P @ H
            S_ = float(H @ PH_ + sigma_chips ** 2)
            z_ = abs(innov) / math.sqrt(S_) if S_ > 0 else 0.0
            _bad = z_ > self.innov_nsigma or (self.innov_max is not None
                                              and abs(innov) > self.innov_max)
            if _bad:
                # ESCAPE HATCH. A gate with no way out is a deadlock: a genuine step (a
                # clock event, a re-anchor) is rejected, so the state never follows it, so
                # every subsequent measurement is rejected too -- forever. After
                # reject_escape consecutive rejections, believe the world rather than the
                # state, and inflate the covariance so the correction is taken quickly.
                #
                # PERSISTENCE IS NOT ENOUGH, and assuming it was cost the whole state on
                # 2026-08-10. The original test counted consecutive rejections and never
                # asked whether the rejected measurements agreed with EACH OTHER. PRN 25
                # transited out of the primary beam (elevation RISING 45->65 deg, deep_snr
                # 30->2.4, coh_frac 0.94->0.19) and its code phase became uniform noise over
                # the 10230-chip code. Five uniform draws are perfectly "persistent" and
                # mutually meaningless: the hatch believed a -1396-chip innovation, which
                # corrupted clk, which made every OTHER sat's innovation large, which
                # escaped them too, until the gauge had no inliers left and switched itself
                # off (see gauge()). The observable clk + b_i stayed correct throughout,
                # so nothing downstream noticed for 50 minutes.
                #
                # A REAL move is CONSISTENT: the state is not moving (every measurement is
                # being rejected), so a genuine re-anchor gives the same innovation every
                # time. Noise does not. Requiring the run to agree with itself within
                # escape_spread is a far stronger discriminator than any magnitude bound --
                # five uniform draws landing within 5 chips of each other has probability
                # ~(2*5/10230)^4 ~ 1e-8 -- and because it is strong, it can be trusted with
                # LARGE innovations, which keeps the deadlock escape this hatch exists for.
                # That is why there is deliberately no absolute ceiling here: a consistent
                # 1400-chip move is a real re-anchor and must be followed; an inconsistent
                # one is noise at any magnitude.
                # CONSISTENCY IS TESTED ON THE MEASUREMENT y, NOT ON THE INNOVATION, and the
                # difference is not pedantry. An innovation is y minus a MOVING state: while
                # a run of rejections is in progress the gauge and the process model keep
                # shifting x, so a satellite reporting the same thing every time produces
                # innovations that drift apart. Measured: clean sigma-0.3 measurements gave
                # an innovation spread of 20 chips purely because the state was sliding
                # underneath them, which would block the hatch exactly when a real re-anchor
                # needs it. y is what the satellite actually said; two honest measurements of
                # a static truth agree no matter what the filter did in between.
                hist = self._rej_hist.setdefault(key, [])
                hist.append(y_chips)
                self._rej_band[key] = band   # so _corroborators can remove its tau
                if len(hist) > self.reject_escape:
                    del hist[:-self.reject_escape]
                self._rej_run[key] = self._rej_run.get(key, 0) + 1
                spread = None
                if len(hist) >= self.reject_escape:
                    # circular spread about the run's first sample, so a cluster straddling
                    # +-L/2 is not read as maximally scattered
                    spread = max(abs(self.wrap(v - hist[0])) for v in hist)
                if spread is None or spread > self.escape_spread:
                    self.rejected += 1
                    if spread is not None:
                        # The run is long but INCOHERENT: this satellite is feeding noise,
                        # which is a tracking failure and not a state failure. Say so --
                        # silence here is what let the 2026-08-10 event pass unremarked.
                        self._note("REJECT-RUN %s: %d consecutive, spread %.1f chips "
                                   "(> escape_spread %.1f) -- incoherent, NOT believed; "
                                   "this sat is feeding noise"
                                   % (self._key_str(key), self._rej_run[key], spread,
                                      self.escape_spread))
                    return None
                # HOW BIG A STEP IS THE RUN ASKING US TO BELIEVE? The consistency test above
                # only asks whether the rejected measurements agree with EACH OTHER. They
                # can agree perfectly and still imply a step that is not physical: on a
                # 10230-chip code, a jump of thousands is a WRAP ALIAS, not a clock. Once
                # believed it is applied at near-unit gain (P[0,0] += 25 against R = 0.09),
                # so the state moves by essentially the whole innovation -- and then every
                # subsequent measurement disagrees, producing more runs, more escapes, and a
                # free-running state.
                #
                # MEASURED 2026-08-10 22:22-22:45 UTC: escapes believing -2766, +4521 and
                # +2669 chip steps drove clk to +169517 chips at a rate of +127.66 chips/s
                # (12.5 ppm, against a GPS-disciplined truth of 4e-5 ppm), b_sat to +-300
                # chips, and 1050 rejections against 280 updates -- the filter deaf and
                # free-running, 90 s after a clean start.
                if abs(innov) > self.escape_max_step:
                    # EXTRAORDINARY CLAIM -> EXTRAORDINARY EVIDENCE. Count how many OTHER
                    # satellites are independently asking for the SAME step right now.
                    # Every implied step is recomputed here, at this instant, against the
                    # current state -- so they are all referred to one epoch and the
                    # "innovations drift while the state slides" problem that forced the
                    # spread test onto y does not arise for the comparison BETWEEN sats.
                    _corr = self._corroborators(key, innov)
                    if _corr + 1 < self.escape_min_sats:
                        self.rejected += 1
                        self.diverged = True
                        self._note("ESCAPE REFUSED %s: %d rejections agreeing to %.2f chips "
                                   "but implying a %+.1f chip step (bound %.0f on a %.0f-chip "
                                   "code) and only %d/%d satellites agree -- one satellite's "
                                   "self-consistency is not evidence about the CLOCK. The "
                                   "STATE may still be wrong; corroboration, or a warm start, "
                                   "is the way out -- not a jump on one sat's word"
                                   % (self._key_str(key), self._rej_run[key], spread, innov,
                                      self.escape_max_step, self.L, _corr + 1,
                                      self.escape_min_sats))
                        return None
                    self._note("ESCAPE CORROBORATED %s: %+.1f chip step exceeds the %.0f "
                               "bound, but %d satellites independently agree to within %.1f "
                               "chips -- that is a clock event, not one sat's noise"
                               % (self._key_str(key), innov, self.escape_max_step,
                                  _corr + 1, self.escape_spread))
                self._note("ESCAPE %s: %d consecutive rejections agreeing to %.2f chips "
                           "-- believing a %+.1f chip step and inflating P"
                           % (self._key_str(key), self._rej_run[key], spread, innov))
                self.escapes += 1
                self.P[0, 0] += 25.0
                self.P[i, i] += 25.0
                self.P[0, 1] = self.P[1, 0] = 0.0
        self._rej_hist.pop(key, None)
        self._rej_run.pop(key, None)
        self._rej_band.pop(key, None)
        z = self._scalar_update(H, innov, sigma_chips ** 2)
        self._t_seen[key] = t_now
        self._n[key] = self._n.get(key, 0) + 1
        self.n_updates += 1
        return z

    def gauge(self):
        """Pin the unobservable common mode: mean(b over active sats) = 0."""
        np = self._np
        # A wild bias gets no vote. The gauge is a MEAN, and the estimator it replaces was a
        # circular MEDIAN -- robust by construction. Carrying the mean over without carrying
        # the robustness is what let one bad detection move the clock (see the class note).
        #
        # INLIERS ARE RELATIVE TO THE MEDIAN, NOT TO ZERO, and that is the whole robustness
        # argument rather than a detail. Measuring |b_i| against zero silently assumes the
        # biases are ALREADY centred -- i.e. it assumes the very thing the gauge exists to
        # enforce. When that assumption failed on 2026-08-10 (every bias driven to ~+97 by a
        # corrupted clk) the inlier set came back EMPTY, `len(use) < 2` returned early, and
        # the gauge quietly stopped running for 50 minutes. A gauge that switches itself off
        # exactly when the state most needs pinning is worse than none, because the common
        # mode is then unobservable and clk drifts freely while clk + b_i stays right --
        # invisible to every consumer that reads the sum.
        #
        # The median is always an inlier of itself, so this set is never empty; with a
        # single wild sat it excludes exactly that sat, as before; and with the state
        # uniformly displaced it includes everything and pulls the whole comb back. That
        # last case is self-healing, which the zero-centred form could not be.
        idx = list(self._idx.values())
        if len(idx) < 2:
            return
        b = [float(self.x[i]) for i in idx]
        med = float(np.median(b))
        use = [i for i in idx
               if self.gauge_max_b is None or abs(float(self.x[i]) - med) <= self.gauge_max_b]
        if len(use) < 2:
            # Cannot happen with the median centre (the median's own neighbours are always
            # within range), but if it ever does, that is an alarm and not a reason to skip:
            # pin on everything rather than leave the common mode floating.
            self.gauge_fallbacks += 1
            self._note("GAUGE: only %d inlier(s) of %d about median %+.2f -- pinning on ALL"
                       % (len(use), len(idx), med))
            use = idx
        H = np.zeros(self.x.size)
        for i in use:
            H[i] = 1.0 / len(use)
        self._scalar_update(H, -float(H @ self.x), self.gauge_sigma ** 2)

    def _corroborators(self, key, innov):
        """How many OTHER satellites have a self-consistent rejection run implying the SAME
        step as `innov`, right now.

        The discriminator the per-satellite test cannot provide. Independent satellites
        share exactly one term -- the receiver clock -- so agreement ACROSS them is evidence
        about the clock, while a single satellite agreeing with itself is evidence only that
        its own measurement is repeatable (uniform noise over the code is repeatable in
        exactly the wrong way, which is what cost the state on 2026-08-10).

        Only fully-formed runs vote: a satellite must have reject_escape samples and pass
        the same escape_spread coherence test as the caller. Its own band's tau is removed,
        so a second band's group delay is not mistaken for disagreement.
        """
        n = 0
        for k2, h2 in self._rej_hist.items():
            if k2 == key or len(h2) < self.reject_escape:
                continue
            if max(abs(self.wrap(v - h2[0])) for v in h2) > self.escape_spread:
                continue          # that sat is feeding noise; it gets no vote
            innov2 = self.wrap(h2[-1] - self.predicted(k2)
                               - self.tau(self._rej_band.get(k2)))
            if abs(self.wrap(innov2 - innov)) <= self.escape_spread:
                n += 1
        return n

    def cycle(self, measurements, t_now):
        """predict -> updates -> gauge -> expire, in the one order that is correct.

        `measurements` is an iterable of (key, y_chips, sigma_chips). Returns the number
        accepted."""
        self.predict(t_now)
        n_ok = 0
        for m in measurements:
            # (key, y, sigma) or (key, y, sigma, band) -- the 3-tuple form is the
            # single-band caller and keeps working unchanged.
            key, y, sig = m[0], m[1], m[2]
            band = m[3] if len(m) > 3 else None
            if self.update(key, y, sig, t_now, band=band) is not None:
                n_ok += 1
        self.gauge()
        # PHYSICAL RATE CEILING (2026-08-10 divergence). The reference is GPS-disciplined:
        # measured truth ~4e-4 chips/s, the noisiest legitimate estimator scatters +-0.07.
        # rate_max = 1.0 chips/s (0.1 ppm) is 2500x the truth, so nothing real is clamped --
        # same separation argument as the broker's --dr-max-drift-chips-s, which is the
        # bound that stopped this incident's 12.5 ppm from ever reaching a seed. The state
        # must refuse to CARRY an impossible rate, not merely refuse to publish it: rate
        # multiplies dt in every predict, so a wild value corrupts clk within seconds and
        # then every innovation wraps. Clamp AND widen P[1,1] so the rate re-learns from
        # measurements instead of trusting the clamped value.
        if abs(float(self.x[1])) > self.rate_max:
            _was = float(self.x[1])
            self.x[1] = self.rate_max if _was > 0 else -self.rate_max
            self.P[1, 1] += _was ** 2
            self.P[0, 1] = self.P[1, 0] = 0.0
            self.diverged = True
            self._note("RATE CLAMPED %+.3f -> %+.1f chips/s (bound %.1f = 0.1 ppm, "
                       "2500x the GPSDO truth): a common-mode step is an OFFSET, never a "
                       "frequency -- the state was diverging" % (_was, float(self.x[1]),
                                                                 self.rate_max))
        self._drop([k for k, t in self._t_seen.items()
                    if (t_now - t) > self.max_age_s])
        return n_ok

    # -- readout -----------------------------------------------------------------
    def _add_fcar(self):
        """Give f_carrier its own row, lazily. Chains that never feed a carrier measurement
        keep the exact state layout they had before this existed."""
        np = self._np
        i = self.x.size
        self.x = np.append(self.x, 0.0)
        P = np.zeros((i + 1, i + 1))
        P[:i, :i] = self.P
        P[i, i] = self.sigma_fcar0 ** 2
        self.P = P
        self._fcar_idx = i
        return i

    def update_carrier(self, y_hz, t_now, sigma_hz=1.0):
        """One carrier-frequency measurement, in Hz, common to every satellite.

        SEPARATE FROM update() ON PURPOSE. The code measurement is modular (chips on a
        10230-chip code, wrapped) and carries a per-sat bias; this one is neither -- Hz do
        not wrap and there is no per-sat carrier state to share the residual with. Folding
        them into one path would mean wrapping a frequency, which is silently wrong rather
        than loudly wrong.

        Returns the accepted innovation (Hz), or None if the measurement was gated out.
        """
        self.predict(t_now)
        if self._fcar_idx is None:
            # BIRTH ON THE FIRST MEASUREMENT rather than walking in from 0 at loop gain --
            # the same rule the sat biases use. sigma_fcar0 is wide enough that this lands
            # inside its own uncertainty on cycle one.
            self._add_fcar()
            self.x[self._fcar_idx] = float(y_hz)
            self.n_fcar = 1
            return 0.0
        np = self._np
        H = np.zeros(self.x.size)
        H[self._fcar_idx] = 1.0
        innov = float(y_hz) - float(H @ self.x)
        var = float(H @ self.P @ H) + float(sigma_hz) ** 2
        # Same two-part gate as the code path: an uncertainty-scaled test that does the work,
        # under an absolute ceiling so a wild value cannot ride a wide P through it.
        if var > 0.0 and abs(innov) > self.innov_nsigma * var ** 0.5:
            self.fcar_rejected += 1
            self._note("f_carrier REJECT: innov %+.3f Hz vs %.1f-sigma bar %.3f"
                       % (innov, self.innov_nsigma, self.innov_nsigma * var ** 0.5))
            return None
        self._scalar_update(H, innov, float(sigma_hz) ** 2)
        self.n_fcar += 1
        return innov

    C_LIGHT = 299792458.0

    def _add_rrate(self, key, rr0, t_now):
        """Give a satellite its own range-rate-error row, lazily."""
        np = self._np
        i = self.x.size
        self.x = np.append(self.x, float(rr0))
        P = np.zeros((i + 1, i + 1))
        P[:i, :i] = self.P
        P[i, i] = self.sigma_rr0 ** 2
        self.P = P
        self._rr_idx[key] = i
        self._rr_t_seen[key] = t_now
        return i

    def update_rrate(self, key, y_hz, t_now, f_band_hz, sigma_hz=0.2):
        """One carrier-frequency measurement for ONE satellite, from ONE band.

        `y_hz` is that satellite's measured carrier residual in this band -- feed
        deep_rate_hz, gated on deep_rate_q, for the reasons f_carrier's note gives (
        carrier_hz_resid is documented SIGNAL-FREE at CHORD SNR, so a filter fed by it
        produces a confident meaningless estimate).

            y = -(f_b/c) * rrate_i  +  (f_b/f_ref) * f_carrier  +  noise

        BOTH ROWS ARE UPDATED BY ONE MEASUREMENT, which is the whole point: the receiver-wide
        term and the per-orbit term are degenerate within a band and are separated only by
        the fact that one is shared. A second band on the SAME satellite lands on the SAME
        rrate row with a different f_b, which is how cross-band information actually combines
        instead of being fitted twice.

        Returns the accepted innovation (Hz), or None if gated out.
        """
        np = self._np
        self.predict(t_now)
        f_b = float(f_band_hz)
        if self.f_ref_hz is None:
            self.f_ref_hz = f_b
        f_ref = float(self.f_ref_hz)
        if self._fcar_idx is None:
            # No receiver-wide term yet: the first measurement cannot separate the two, so it
            # all goes to f_carrier -- the shared term is the better prior for one sample, and
            # the gauge will hand back whatever is really per-satellite as soon as a second
            # satellite reports.
            self._add_fcar()
            self.x[self._fcar_idx] = y_hz * (f_ref / f_b)
            self.n_fcar = 1
            # ...and give this satellite its row NOW, at zero. One measurement cannot split
            # the receiver-wide term from this orbit's deviation, so f_carrier takes it all
            # and the gauge hands back whatever is really per-satellite once a second
            # satellite reports. Without a row here the FIRST satellite is structurally
            # unable to disagree with the value it itself defined.
            self._add_rrate(key, 0.0, t_now)
            return 0.0
        if key not in self._rr_idx:
            # Birth on the measurement, like b_sat: initialise to the residual left AFTER the
            # current receiver-wide estimate, not to zero. Walking in from zero at loop gain
            # is how a real per-sat offset gets read as noise for its first minutes.
            resid = y_hz - (f_b / f_ref) * float(self.x[self._fcar_idx])
            self._add_rrate(key, -resid * self.C_LIGHT / f_b, t_now)
            return 0.0
        self._rr_t_seen[key] = t_now
        i = self._rr_idx[key]
        H = np.zeros(self.x.size)
        H[i] = -(f_b / self.C_LIGHT)
        H[self._fcar_idx] = f_b / f_ref
        innov = float(y_hz) - float(H @ self.x)
        var = float(H @ self.P @ H) + float(sigma_hz) ** 2
        if var > 0.0 and abs(innov) > self.innov_nsigma * var ** 0.5:
            self.rrate_rejected += 1
            self._rr_reject_run[key] = self._rr_reject_run.get(key, 0) + 1
            # ESCAPE. A gate with no way out turns a bad birth into a permanent one: the state
            # is confident, the satellite is real, and every cycle it is thrown away for
            # disagreeing. Measured while building this -- one satellite rejected 298 times
            # out of 300. After a sustained run of consistent rejections, believe the
            # SATELLITE and re-open its row rather than the estimate that has stopped
            # predicting it. Same principle as the code path's escape.
            if self._rr_reject_run[key] >= self.rr_escape_runs:
                # RE-OPEN, DO NOT ADOPT (2026-08-13, the PRN 32 walk). The code filter's
                # escape snaps x to the rejected measurement because there a sustained
                # rejection means a REAL satellite the gate wrongly excluded. On carrier
                # rows the sustained-rejection population is different: weak sats whose
                # rate picks scatter multi-Hz emit to emit -- junk the gate is RIGHT to
                # reject. Snapping adopts the junk, and the command then walks (measured:
                # -7 -> -14.5 Hz in 4 min). Re-opening P alone gets both cases right:
                # sigma growth de-commands the sat (the --rrate-cmd-max-sigma gate), and
                # a genuinely mis-converged row is pulled in by NORMAL updates the moment
                # its measurements turn consistent -- through a wide-open gate.
                i_ = self._rr_idx[key]
                self.P[i_, i_] = self.sigma_rr0 ** 2
                self._rr_reject_run[key] = 0
                self._note("rrate ESCAPE %s: %d consistent rejections -- re-opening the "
                           "row WITHOUT adopting (x held)" % (self._key_str(key),
                                                              self.rr_escape_runs))
            else:
                self._note("rrate REJECT %s: innov %+.3f Hz vs %.1f-sigma bar %.3f"
                           % (self._key_str(key), innov, self.innov_nsigma,
                              self.innov_nsigma * var ** 0.5))
            return None
        self._rr_reject_run[key] = 0
        self._scalar_update(H, innov, float(sigma_hz) ** 2)
        self.n_rrate += 1
        return innov

    def gauge_rrate(self):
        """Pin mean(rrate) = 0 over the active satellites.

        Without this, f_carrier and the rrate rows share an unobservable common mode: add c
        to the receiver-wide term, subtract the equivalent from every orbit, and every
        prediction is identical. Same construction as gauge() for b_sat, including the
        median-centred inlier test -- a wild orbit does not get a vote, and the estimator
        this replaces (a per-chain fit with no shared term at all) had no common mode to
        pin, so the robustness has to be supplied here rather than inherited.
        """
        np = self._np
        idx = list(self._rr_idx.values())
        if len(idx) < 2 or self._fcar_idx is None:
            return
        v = [float(self.x[i]) for i in idx]
        med = float(np.median(v))
        use = [i for i in idx if abs(float(self.x[i]) - med) <= max(10.0 * self.sigma_rr0, 1.0)]
        if len(use) < 2:
            use = idx
        H = np.zeros(self.x.size)
        for i in use:
            H[i] = 1.0 / len(use)
        self._scalar_update(H, -float(H @ self.x), self.rr_gauge_sigma ** 2)

    def rrate(self, key):
        """This satellite's range-rate model error, m/s. 0.0 if never measured."""
        i = self._rr_idx.get(key)
        return 0.0 if i is None else float(self.x[i])

    def rrate_sigma(self, key):
        """1-sigma, m/s. inf when unmeasured -- an unmeasured state must not read as a
        confident zero, which is how a dead feed passes for a healthy one."""
        i = self._rr_idx.get(key)
        return float("inf") if i is None else float(self.P[i, i]) ** 0.5

    def carrier_correction_hz(self, key, f_band_hz):
        """THE ONE COMMAND: total carrier correction for this satellite in this band, Hz.

        This is what a seed adds to its model Doppler -- receiver-wide term and orbit term
        together, scaled to the band being seeded. It exists so there is exactly ONE number
        leaving the estimator per (satellite, band), rather than a seed and a separate trim
        that a continuity gate then has to referee. See task #33: the reason the shared
        carrier loop was structurally inert was two controllers on one state.
        """
        f_b = float(f_band_hz)
        f_ref = float(self.f_ref_hz or f_b)
        fc = float(self.x[self._fcar_idx]) if self._fcar_idx is not None else 0.0
        return (f_b / f_ref) * fc - (f_b / self.C_LIGHT) * self.rrate(key)

    def f_carrier(self):
        """Receiver-wide carrier frequency offset (Hz). 0.0 before any measurement."""
        return float(self.x[self._fcar_idx]) if self._fcar_idx is not None else 0.0

    def f_carrier_sigma(self):
        """Its 1-sigma uncertainty (Hz). inf before any measurement -- an UNMEASURED state
        must not read as a confident zero, which is how a dead feed passes for a healthy one."""
        if self._fcar_idx is None:
            return float("inf")
        return float(self.P[self._fcar_idx, self._fcar_idx]) ** 0.5

    def tau(self, band):
        """Differential group delay of `band` against the reference band, chips. 0 for the
        reference (it has no row -- that is the gauge, not a special case)."""
        i = self._band_idx.get(band)
        return 0.0 if i is None else float(self.x[i])

    def tau_sigma(self, band):
        i = self._band_idx.get(band)
        return 0.0 if i is None else float(self._np.sqrt(max(self.P[i, i], 0.0)))

    def tau_observability(self, band):
        """How many satellites have been measured in BOTH this band and the reference.

        tau_band separates from b_sat ONLY through dual-band satellites -- one ray, two
        carriers. With zero of them the two are exactly degenerate and any tau this filter
        reports is an artefact of the priors, so a consumer must check this before believing
        the number. This is the quantity that the disjoint E5a/E5b PRN lists drove to zero
        while every individual chain looked healthy."""
        if band == self.ref_band or self.ref_band is None:
            return len(self._dual)
        return sum(1 for bands in self._dual.values()
                   if band in bands and self.ref_band in bands)

    @property
    def clk(self):
        return float(self.x[0])

    @property
    def clk_rate(self):
        return float(self.x[1])

    def bias(self, key):
        i = self._idx.get(key)
        return float(self.x[i]) if i is not None else 0.0

    def predicted(self, key):
        """clk + b_i: what a seed for this sat should add to the pure model."""
        return self.clk + self.bias(key)

    def sigma(self, key=None):
        """1-sigma on clk (key None) or on this sat's clk+b_i."""
        np = self._np
        if key is None:
            return math.sqrt(max(0.0, self.P[0, 0]))
        i = self._idx.get(key)
        if i is None:
            return math.sqrt(max(0.0, self.P[0, 0]))
        v = self.P[0, 0] + 2.0 * self.P[0, i] + self.P[i, i]
        return math.sqrt(max(0.0, v))

    def age(self, key, t_now):
        t = self._t_seen.get(key)
        return None if t is None else (t_now - t)

    def summary(self, t_now, max_sats=12):
        parts = []
        for key, i in sorted(self._idx.items(), key=lambda kv: -abs(self.x[kv[1]])):
            if len(parts) >= max_sats:
                break
            stale = (t_now - self._t_seen.get(key, 0.0)) > self.max_age_s
            parts.append("%s:%+.2f%s" % (key if isinstance(key, str) else
                                         "%s%d" % (str(key[0])[:1], key[1]),
                                         self.x[i], "*" if stale else ""))
        return ("clk %+.3f+-%.3f chips  rate %+.4f chips/s  n=%d sat(s) upd=%d rej=%d | %s"
                % (self.clk, self.sigma(), self.clk_rate, len(self._idx),
                   self.n_updates, self.rejected, " ".join(parts) if parts else "-"))
