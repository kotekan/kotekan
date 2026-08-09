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
                 reject_escape=5,
                 birth_max=50.0, gauge_max_b=60.0,
                 ref_band=None, sigma_tau0=20.0, q_tau=1e-5):
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
        self.birth_max = birth_max     # refuse to BIRTH a sat implausibly far from clk
        self.gauge_max_b = gauge_max_b # a wild bias does not get a vote in the gauge
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
        self.x = np.zeros(2)           # [clk chips, clk_rate chips/s]
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
        # Reindex from the SURVIVING ROW ORDER, not from a hardcoded base of 2. The old
        # form assumed every row past clk_rate was a satellite; tau_band rows share that
        # space now, and a base-2 renumber would silently alias a band onto a satellite.
        remap = {old_row: new_row for new_row, old_row in enumerate(keep)}
        self._idx = {k: remap[i] for k, i in self._idx.items() if i in remap}
        self._band_idx = {b: remap[i] for b, i in self._band_idx.items() if i in remap}
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
            srows = [i for i in range(2, n) if i not in brows]
            if srows:
                idx = np.array(srows)
                self.P[idx, idx] += self.q_b ** 2 * dt
            for i in brows:
                # hours-timescale state: q_tau is ~100x below q_b on purpose
                self.P[i, i] += self.q_tau ** 2 * dt

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
                # every subsequent measurement is rejected too -- forever. Garbage is
                # sporadic; a real move is PERSISTENT. After reject_escape consecutive
                # rejections for the same satellite, believe the world rather than the
                # state, and inflate the covariance so the correction is taken quickly.
                self._rej_run[key] = self._rej_run.get(key, 0) + 1
                if self._rej_run[key] < self.reject_escape:
                    self.rejected += 1
                    return None
                self.P[0, 0] += 25.0
                self.P[i, i] += 25.0
                self.P[0, 1] = self.P[1, 0] = 0.0
        self._rej_run.pop(key, None)
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
        use = [i for i in self._idx.values()
               if self.gauge_max_b is None or abs(self.x[i]) <= self.gauge_max_b]
        if len(use) < 2:
            return
        H = np.zeros(self.x.size)
        for i in use:
            H[i] = 1.0 / len(use)
        self._scalar_update(H, -float(H @ self.x), self.gauge_sigma ** 2)

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
        self._drop([k for k, t in self._t_seen.items()
                    if (t_now - t) > self.max_age_s])
        return n_ok

    # -- readout -----------------------------------------------------------------
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
