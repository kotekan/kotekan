"""The Receiver: what every chain on this telescope shares.

What belongs here is decided by physics. CHORD is one receiver: one F-engine digitising the
whole band off one GPS-disciplined reference. GPS L5 and Galileo E5a both run at
--carrier-hz 1176.45e6 because they are the same 24 MHz of sky through the same feed, cable
and clock. Anything that is a property of the instrument rather than of a signal lives here
and is estimated once:

    time anchor   utc0_sample0, the F-engine's frame 0. One number for the telescope.
    sky           the BRDC store. Multi-system (keys are (sys, prn)) and filtered per chain
                  at predict time, so one parse serves every constellation.
    code clock    per band. Cable and PFB group delay are per carrier, so a code bias
                  measured at 1176.45 MHz is not the one at 1207.14 MHz.

Contribute / consume, never overwrite. A chain that solves a quantity itself keeps its own
estimate and publishes it here; a chain that cannot (one with --detectors empty never solves
a clock) consumes a sibling's. --dr-clock-adopt takes its donor clock from here.

A single chain is therefore unaffected by construction: every consumer excludes the caller,
so a chain alone gets back only what it supplied itself (the time anchor, the BRDC store,
the joint state it feeds). That is a property of the contract, not something to re-check at
every call site.

Per-signal state is deliberately not here: seeds, per-PRN track state, the DLL and carrier
loops, the CL machinery and nav decode belong to the chain.

Chains are threads of one process (scripts/gnss/broker_multi.py). Every method that reads or
writes shared state takes the one RLock (time_anchor's fetch and brdc's build run outside it),
and a contributed _Shared is replaced, never mutated.
"""
import threading

from .transport import _log


class _Shared(object):
    """One contributed quantity: who said it, what, how well, and when."""

    __slots__ = ("value", "src", "weight", "t", "extra")

    def __init__(self, value, src, weight, t, extra=None):
        self.value, self.src, self.weight, self.t = value, src, weight, t
        self.extra = extra or {}


class Receiver(object):
    """Instrument-scope state, shared by every chain in the process."""

    def __init__(self, log=None):
        self._lock = threading.RLock()
        self._log = log or _log
        self._anchor = None  # utc0_sample0, latched once for the telescope
        self._anchor_src = None
        self._brdc = {}  # cache key -> the {"mod","eph","eph_t"} state dict
        self._code = {}  # (band, chain) -> _Shared (dimensionless l-a)
        self._dr = {}  # (band, chain) -> _Shared (chips, + drift in extra)
        self._clkmod = {}  # chain -> _Shared (the receiver clock in seconds,
        # known mod extra["epoch_s"], e.g. the NH joint
        # fit's clock mod 20 ms): what extends a
        # short-window donor clock.
        self._joint = {}  # "__receiver__" -> the one JointReceiverState
        self._joint_unit = {}  # chain -> (chip_rate_hz, code_len) of the code
        # phases it feeds into the receiver-wide joint state

    # -- the joint receiver state -----------------------------------------------------------
    def joint_receiver(self, band, code_len, **kw):
        """One joint state for the whole receiver, shared by every chain.

        Created on first use, so a process that never runs the joint solve never builds it
        (its constructor imports numpy).

        The modulus is why this takes code_len. A shared state has one wrap, and a
        measurement from a chain with a shorter code is known only mod that code, so the
        state must wrap at the shortest period any caller passes, or it would treat an
        aliased value as a real excursion. A shorter code_len shortens the modulus in place;
        joint_ambiguous() reports a clock that has grown close to the wrap. code_len is
        compared as given, in the caller's own chips.

        Same argument as the clock adoption's modulus rule: reducing to a shorter period is
        well defined, lengthening is not.
        """
        from .state_filter import JointReceiverState

        with self._lock:
            cur = self._joint.get("__receiver__")
            if cur is None:
                kw = dict(kw)
                kw["code_len"] = float(code_len)
                self._joint["__receiver__"] = JointReceiverState(**kw)
            elif float(code_len) < cur.L:
                # Shorten in place, never rebuild: a rebuild would discard every converged
                # row the moment another chain joined. Reducing a modulus loses no
                # information (JointReceiverState.shorten_modulus): same rows, same
                # covariance, clk re-wrapped.
                cur.shorten_modulus(float(code_len))
            return self._joint["__receiver__"]

    def joint_ambiguous(self):
        """True when |clk| exceeds 0.4 of the state's wrap, close enough that a measurement
        from the shortest-code chain could be aliased. A consumer should not believe clk
        while this holds; the JOINT[shadow] log line marks it AMBIGUOUS."""
        with self._lock:
            st = self._joint.get("__receiver__")
            return bool(st is not None and abs(st.clk) > 0.4 * st.L)

    def joint_declare_unit(self, chain, chip_rate_hz, code_len):
        """A chain that feeds code phases into the joint state declares the chips they are in.

        The state stores bare numbers: its clk, biases and sigma are in the chips of whoever
        fed it, known mod that feeder's code. Nothing inside the filter records which chips
        those are, so a consumer at another chip rate cannot convert without being told, and
        the unit must come from the feeders, not from the consumer's assumption. Declared on
        every feed cycle, before the feed, so a state fit to consume always has a declaration
        behind it."""
        with self._lock:
            self._joint_unit[chain] = (float(chip_rate_hz), float(code_len))

    def joint_unit(self):
        """(chip_rate_hz, code_len) of the joint state's code phases, or None when no feeder
        has declared one or the feeders disagree. One state holds one unit: mixed-rate feeds
        cannot be converted by any single factor, so a consumer must refuse them."""
        with self._lock:
            units = set(self._joint_unit.values())
            return units.pop() if len(units) == 1 else None

    # -- time anchor --------------------------------------------------------------------
    def time_anchor(self, fetch, chain):
        """The F-engine's frame-0 UTC, latched at most once per process.

        `fetch` is called only if nobody has it yet, and may raise or return a falsy value
        (the endpoint being down is the normal outage case); then nothing is latched and the
        next chain tries again.

        Never re-anchors. A changed frame 0 also invalidates the epoch every node cached at
        its own startup, so re-anchoring here would fix this process's arithmetic and leave
        the nodes wrong. The broker loop's 60 s re-read logs TIME ANCHOR CHANGED and, after
        _ANCHOR_STRIKES confirmations, exits the process so the startup path latches the new
        frame 0.
        """
        with self._lock:
            if self._anchor:
                return self._anchor
        v = fetch()  # outside the lock: this is a network call
        if not v:
            return None
        with self._lock:
            if not self._anchor:
                self._anchor, self._anchor_src = float(v), chain
                return self._anchor
            if abs(self._anchor - float(v)) > 1e-3:
                # Two chains disagreeing about frame 0 means the F-engine restarted between
                # their fetches. Neither is trustworthy; say which is in use and by whom.
                self._log(
                    "*** TIME ANCHOR DISAGREEMENT: %s has %.9f, %s fetched %.9f "
                    "(%+.3f days). The F-engine restarted mid-startup. Restart the "
                    "nodes AND this broker."
                    % (
                        self._anchor_src,
                        self._anchor,
                        chain,
                        float(v),
                        (float(v) - self._anchor) / 86400.0,
                    )
                )
            return self._anchor

    # -- sky ----------------------------------------------------------------------------
    def brdc(self, key, build):
        """The shared BRDC store for `key`, built once.

        The parse is multi-system (entries keyed (sys, prn), each consumer filtering at
        predict time), so one store serves GPS, Galileo and BeiDou together. `build` runs
        outside the lock, so two chains that miss together may both build; the first stored
        wins and both get it. The returned dict is shared and is mutated by
        sky.brdc_predict without this lock.
        """
        with self._lock:
            if key in self._brdc:
                return self._brdc[key]
        st = build()  # outside the lock: fetch + parse
        with self._lock:
            return self._brdc.setdefault(key, st)

    # -- clocks -------------------------------------------------------------------------
    def contribute_code_bias(self, chain, band, la, n_sats, t):
        """Publish the code-rate clock offset (l-a), a fractional frequency, keyed by band.
        code_bias_any_band reads it across bands: a rate carries no per-band delay."""
        if la is None:
            return
        with self._lock:
            self._code[(band, chain)] = _Shared(float(la), chain, int(n_sats or 0), t)

    def code_bias(self, band, exclude=None, max_age_s=120.0, t_now=None):
        """Another chain's (l-a) in this band, or None. The one fitted from the most
        satellites wins (the only quality signal these estimators carry), and an entry older
        than max_age_s is skipped, because a chain that stopped solving must stop being
        believed rather than fading silently."""
        return self._best(self._code, exclude, max_age_s, t_now, key2=band)

    def code_bias_any_band(self, exclude=None, max_age_s=120.0, t_now=None):
        """(l-a) from any band: the clock rate is receiver-wide even though the phase is not.

        Group delay is per carrier, which matters for the code phase: a signal at 1207.14 MHz
        arrives through a different filter chain than one at 1176.45 MHz and sits at a
        different delay (tau_band). But (l-a) is a fractional frequency (consumed as
        code_bias * chip_hz / hops_per_sec, fits.cp_rate_from_code_bias), and tau_band is a
        constant whose time derivative is zero, so the rate is a property of the receiver.

        Without this a band where no chain solves its own clock has no rate: its seeds ship
        code_phase_rate 0, the fleet DLL never closes, and the loop closes on itself (no
        clock -> no rate -> no lock -> no detections -> no measured (l-a) -> no clock).

        The phase stays per band: dr_clock keeps its band key, because adopting a code phase
        across carriers carries exactly the tau_band offset (see dr_clock_any_band).
        """
        return self._best(self._code, exclude, max_age_s, t_now, key2=None)

    def contribute_dr_clock(
        self,
        chain,
        band,
        chips,
        drift,
        t,
        code_length,
        chip_rate_hz=None,
        epoch=None,
        held=False,
        healthy=None,
    ):
        """Publish the dead-reckon receiver clock (chips, mod the code period) and its drift.

        clk and drift are the receiver's; the rest of dr_state is the chain's per-PRN
        bookkeeping and stays with the chain.

        Carried with its code length: chips are modular, and a value mod 10230 means nothing
        to a chain whose code is 1023000 long. A consumer that ignored this would adopt a
        number that is numerically fine and physically wrong.

        And with its chip rate: the clock is a time and chips are only a unit. 150.2 chips at
        10.23 Mcps is 14.7 us, which is 75.1 chips at E6's 5.115 Mcps; taken as raw chips the
        clock would read as twice the time (Receiver.clock_chips_convert).

        And with its epoch: a count the solving chain bumps each time it snaps its clock on
        purpose (bootstrap, re-roll, confirmed re-pin). A consumer bounds every step of the
        published value except one that arrives with an epoch it has not adopted, so a
        deliberate re-pin reaches it at once and an undeclared step is refused. None = not
        declared (every step bounded). `held` marks a value the solving chain is holding at
        zero rate (it did not update this cycle): valid at any instant, so a consumer takes
        it at its own now instead of extrapolating it from `t`.
        """
        if chips is None:
            return
        with self._lock:
            self._dr[(band, chain)] = _Shared(
                float(chips),
                chain,
                1,
                t,
                {
                    "drift": drift,
                    "code_length": code_length,
                    "chip_rate_hz": chip_rate_hz,
                    "epoch": epoch,
                    "held": bool(held),
                    # False = the solving chain's search solve is not established
                    # (clockhealth.dr_solve_healthy); a consumer must not bootstrap from it.
                    # None = the contributor made no claim.
                    "healthy": healthy,
                },
            )

    def dr_clock(self, band, exclude=None, max_age_s=120.0, t_now=None):
        return self._best(self._dr, exclude, max_age_s, t_now, key2=band)

    def dr_clock_any_band(self, exclude=None, max_age_s=120.0, t_now=None):
        """The dead-reckon clock phase from any band: a bootstrap prior, not a measurement.

        Unlike code_bias_any_band this one is approximate. The receiver clock is common to
        every band, but the number stored here is a code phase, so a consumer in another band
        inherits the donor's group delay: the adopted value is off by tau_band, the
        per-carrier instrumental delay difference (plus an ionospheric term, which is tiny:
        40.3*TEC/c*(1/fa^2-1/fb^2) is ~0.005 chips per 10 TECU between the E5 sidebands).

        That is the right trade. A band that solves no clock otherwise sits at its primed
        value, wrong by the whole receiver clock against a +-1 chip correlation peak; nothing
        tracks through that, so tau_band is never measured. Borrowing the phase leaves a
        tau_band-sized error inside the DLL's pull-in, and the loop's steady-state residual
        then is tau_band. The caller logs the adoption as a cross-band BOOTSTRAP, never as a
        same-band ADOPTED, so the log tells a borrowed phase from a measured one.
        """
        return self._best(self._dr, exclude, max_age_s, t_now, key2=None)

    def contribute_clock_mod_epoch(
        self, chain, seconds, epoch_s, n_sats, t, healthy=None
    ):
        """The receiver clock as a time, known mod epoch_s. Contributed by an estimator that
        resolves a longer modulus than a primary code period: the NH joint fit (--nh-joint;
        clock mod 20 ms from the strong GPS satellites' votes). Consumers use it to extend a
        precise short-window donor clock across their own longer code period."""
        with self._lock:
            self._clkmod[chain] = _Shared(
                float(seconds),
                chain,
                int(n_sats or 0),
                t,
                extra={"epoch_s": float(epoch_s), "healthy": healthy},
            )

    def clock_mod_epoch(self, min_epoch_s, exclude=None, max_age_s=3600.0, t_now=None):
        """Freshest clock-mod record whose epoch covers min_epoch_s. The long default age
        bound is deliberate: the quantity is a run constant (the F-engine axis is
        drift-free), refreshed each time the NH joint fit reaches its quorum while the
        contributing chain's search solve is established."""
        with self._lock:
            best = None
            for chain, rec in self._clkmod.items():
                if chain == exclude:
                    continue
                if rec.extra.get("epoch_s", 0.0) < float(min_epoch_s) - 1e-12:
                    continue
                if (
                    t_now is not None
                    and max_age_s is not None
                    and t_now - rec.t > max_age_s
                ):
                    continue
                if best is None or rec.t > best.t:
                    best = rec
            return best

    @staticmethod
    def clock_extend_mod(
        donor_seconds, donor_window_s, mod_seconds, mod_epoch_s, our_window_s
    ):
        """Extend a donor clock known mod its (short) window across our (longer) code
        period, using an independent measurement of the same clock mod >= our window.

        The donor supplies the precision (e.g. gps_l5's clock mod 1 ms); the mod
        measurement supplies only the ambiguity resolution, which of the donor's windows.
        Candidates are a full donor window apart, so it need be right only to half of that.
        Returns (seconds mod our_window_s, resid_s), where resid_s is the disagreement
        between the extended donor and the mod measurement: the caller must refuse the
        extension when |resid_s| is a large fraction of the donor window, because then the
        k choice itself is in doubt."""
        dm = float(mod_seconds) % float(mod_epoch_s)
        k = round((dm - float(donor_seconds)) / float(donor_window_s))
        t_full = float(donor_seconds) + k * float(donor_window_s)
        return t_full % float(our_window_s), (t_full - dm)

    # -- unit conversion (module-level contract, selftest-gated) -------------------------

    @staticmethod
    def clock_chips_convert(donor_chips, donor_rate_hz, our_rate_hz, our_code_len):
        """A donor's clock, re-expressed in our chips and reduced mod our code.

        The clock is a time; chips are a unit. selftest.py checks it with gal_e6's numbers
        (150.2 chips at 10.23 Mcps = 75.1 chips at 5.115 Mcps)."""
        return (float(donor_chips) * our_rate_hz / donor_rate_hz) % our_code_len

    @staticmethod
    def joint_clk_delta(
        joint_chips, joint_rate_hz, joint_code_len, our_chips, our_rate_hz, our_code_len
    ):
        """The correction that moves our clock onto the joint clock: (delta in our chips,
        the same delta in the joint's chips).

        Both clocks are the same receiver time in different units and known mod different
        windows: the joint mod its feeders' code (joint_code_len joint chips), ours mod our
        code. The joint is converted into our chips first, then the difference is wrapped in
        the shorter of the two windows, the only one both values are defined in. So the
        joint moves our clock only within that window and our clock keeps saying which
        period of a longer code we are in (a 20 ms L2C-CM code under a 1 ms joint window).

        The second value exists so a bound stays one physical time on every chain: a gate
        written in the joint's chips means the same nanoseconds whatever our chip rate is.
        At our rate == joint rate the factor is exactly 1.0 and the window is our code, so
        this reproduces the same-rate arithmetic bit for bit."""
        r = float(our_rate_hz) / float(joint_rate_hz)
        w = min(float(joint_code_len) * r, float(our_code_len))
        d = ((float(joint_chips) * r - float(our_chips) + w / 2.0) % w) - w / 2.0
        return d, d / r

    # -- internals ----------------------------------------------------------------------
    def _best(self, store, exclude, max_age_s, t_now, key2):
        with self._lock:
            best = None
            for k, s in store.items():
                chain = k[1] if isinstance(k, tuple) else k
                if chain == exclude:
                    continue
                if key2 is not None and k[0] != key2:
                    continue
                if t_now is not None and max_age_s and (t_now - s.t) > max_age_s:
                    continue
                if best is None or s.weight > best.weight:
                    best = s
            return best

    def summary(self):
        """One line for the periodic log -- what the instrument currently believes."""
        with self._lock:
            return (
                "receiver: anchor=%s (%s), brdc=%d store(s), code from %s, dr from %s"
                % (
                    "%.9f" % self._anchor if self._anchor else "none",
                    self._anchor_src or "-",
                    len(self._brdc),
                    sorted(c for _, c in self._code) or "-",
                    sorted(c for _, c in self._dr) or "-",
                )
            )
