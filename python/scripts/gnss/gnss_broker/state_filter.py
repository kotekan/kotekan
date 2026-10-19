"""The receiver-state filters (docs/CHORD_JOINT_TRACKING.md section 3).

SatBiasFilter: a chain-local, slow per-satellite code bias b_sat fed by the fleet slope-fit
tau. JointReceiverState: one receiver-wide Kalman state over the clock (clk, clk_rate), the
per-satellite code biases b_sat[i], the carrier frequency offset f_carrier and the
per-satellite range-rate errors rrate[i], shared by every chain thread.
"""
import math
import threading


def _locked(fn):
    """Serialize a JointReceiverState entry point on the instance lock.

    One state is shared by every chain thread (the code feed's cycle(), each chain's rate
    feeds, the gauges, predict() and the age-out _drop()), and Receiver.joint_receiver()
    locks only the lookup. x and P are resized in separate statements, so an unlocked thread
    switch between them desyncs their shapes and every later birth builds on the mismatch.
    The matrices are about 100 rows, so the lock is cheap against a 2 s cycle."""

    def _wrap(self, *a, **k):
        with self._lk:
            return fn(self, *a, **k)

    _wrap.__name__ = fn.__name__
    _wrap.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
    _wrap.__doc__ = fn.__doc__
    return _wrap


class SatBiasFilter:
    """Per-satellite slow code-phase bias (chips), estimated from the fleet slope-fit tau.

    tau is sky minus replica, and the dead-reckon seeds add b_sat, so the measured tau is
    the filter's residual: convergence drives tau to 0 while b_sat holds the bias. If b_sat
    wanders instead of holding, the loop is feeding back its own error and must be opened.

    Three gates:
      * Presence (caller side): a weak satellite's tau is biased toward zero by self-reference
        (the phi_i solve at tau = 0 pre-aligns that point), so it is not a measurement. The
        caller feeds tau only for satellites passing its lock metric, and this class also
        requires min_inst instances: fits from a few instances scatter several times more.
      * Innovation: lobe captures land near +-3.27 chips and are not flagged by peak/floor,
        while real biases are under ~0.3, so |tau - b| > innovation_max is counted as a
        rejection and never averaged in.
      * Age: iono decorrelates over tens of minutes, so a bias not measured for max_age_s
        stops being applied (get() returns 0) but is kept for logging.

    Slow by design: gain 0.02 per update at a ~2 s cadence is a ~100 s time constant, well
    above the measurement noise and well below the iono timescale. Clamped because no
    physical bias at this band reaches a chip.
    """

    def __init__(
        self, gain=0.02, clamp=1.0, innovation_max=1.0, max_age_s=600.0, min_inst=6
    ):
        self.gain = float(gain)
        self.clamp = float(clamp)
        self.innovation_max = float(innovation_max)
        self.max_age_s = float(max_age_s)
        self.min_inst = int(min_inst)
        self._b = {}  # prn -> chips
        self._t = {}  # prn -> last accepted-measurement time
        self._n = {}  # prn -> accepted count
        self.rejected = 0  # innovation-gate rejections (lobes), for the log

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
        """The bias to apply (chips): 0 unless measured within max_age_s."""
        t = self._t.get(prn)
        if t is None or (t_now - t) > self.max_age_s:
            return 0.0
        return self._b.get(prn, 0.0)

    def summary(self, t_now):
        """One log line's worth: live biases, stale ones marked, rejection count."""
        parts = []
        for prn in sorted(self._b):
            stale = (t_now - self._t.get(prn, 0.0)) > self.max_age_s
            parts.append(
                "%d:%+.3f%s(n%d)"
                % (prn, self._b[prn], "*" if stale else "", self._n.get(prn, 0))
            )
        return "%s rej=%d" % (" ".join(parts) if parts else "-", self.rejected)


class JointReceiverState:
    """One joint solve over [clk, clk_rate, b_sat[i]] (docs/CHORD_JOINT_TRACKING.md section
    3a), plus the carrier rows f_carrier and rrate[i].

    Why one solve: the BRDC model is wrong per satellite by chips, and b_sat and clk are
    degenerate within a single epoch (N satellites, N+2 unknowns, N measurements). Only their
    timescales separate them -- the clock moves fast, the biases slowly -- which is process
    noise, i.e. a filter. Estimating the biases first would estimate them with no clock
    removed.

    The measurement is the same scalar for every source:

        y_i = (where the replica is) - (pure model, no clock, no bias) = clk + b_i + noise

    The broker feeds the dead-reckon offsets of search detections (dr_joint_shadow): the
    search's code phase minus the model, before the legacy clock solve takes their circular
    median.

    clk and clk_rate are fast and shared; each b_i is bounded and moves on minute timescales
    (q_b is ~0.1 chips per 60 s). A satellite that loses SNR keeps its last b_i while its
    prediction moves with the shared clock (section 3a's test; the selftest's coast check).

    The gauge: adding c to clk and subtracting c from every b_i predicts identical
    measurements, so the common mode is unobservable and gauge() pins it (see gauge_mode).
    Under the prior gauge, a soft mean(b) = 0, one satellite whose true bias moves by d
    shifts clk by ~d/N and every other bias by ~-d/N; clk + b_i, the only observable a
    consumer reads, is unaffected.

    Every code-phase innovation is wrapped into [-L/2, L/2). The state itself is unwrapped,
    so clk_rate sees a continuous clock.

    The legacy clock is a circular median, outlier-proof by construction; this filter is
    not, so robustness is supplied explicitly: the birth window (birth_max), the normalized
    innovation gate and its corroborated escape, the rate ceiling (rate_max), the P floor
    and the deaf latch.
    """

    def __init__(
        self,
        code_len=10230.0,
        sigma_clk0=200.0,
        sigma_rate0=0.05,
        sigma_b0=10.0,
        q_clk=1e-3,
        q_rate=1e-4,
        q_b=0.013,
        rereference=False,
        gauge_sigma=0.1,
        gauge_mode="median",
        gauge_prior_sigma=2.0,
        max_age_s=900.0,
        innov_max=200.0,
        innov_nsigma=6.0,
        reject_escape=5,
        escape_spread=5.0,
        birth_max=12.0,
        gauge_max_b=60.0,
        escape_max_step=100.0,
        rate_max=1.0,
        sigma_fcar0=50.0,
        q_fcar=0.01,
        sigma_rr0=2.0,
        q_rr=0.02,
        rr_gauge_sigma=0.5,
        f_ref_hz=None,
        clk0=0.0,
        escape_min_sats=3,
        p_floor=0.04,
        birth_gate_after=30,
    ):
        import numpy as np

        self._np = np
        self._lk = (
            threading.RLock()
        )  # see _locked; re-entrant because locked methods call each other
        self.L = float(code_len)
        self.sigma_clk0 = float(sigma_clk0)
        # Prior sigma of clk_rate (chips/s). The receiver's true drift is under ~0.01 chips/s,
        # so 0.05 is generous for anything physical. It must not be as wide as rate_max, the
        # garbage ceiling: a near-free rate learns from the first innovations, predict()
        # integrates it, and clk walks away from the measurements that would correct it. The
        # prior only shapes the transient; the converged state does not depend on it.
        self.sigma_rate0 = float(sigma_rate0)
        self.sigma_b0 = float(sigma_b0)
        # process noise, per sqrt(second)
        self.q_clk = float(q_clk)  # clock random walk beyond what clk_rate explains
        # clk_rate random walk. This trades rate noise against agility only (the estimate is
        # unbiased at any value): at 6 satellites and sigma 0.3 chips, 1e-4 gives a rate sd of
        # ~0.00035 chips/s and still follows a 0.01 chips/s ramp without visible lag, where
        # 2e-3 gives 0.0035. The reference is GPS-disciplined, so its rate wanders far slower.
        self.q_rate = float(q_rate)
        # Median gauge only: on a membership change, re-reference exactly instead of
        # inflating the clock (see _membership_changed). Off by default.
        self.rereference = bool(rereference)
        self.q_b = float(q_b)  # per-sat bias walk, chips per sqrt(s); slow by design
        self.gauge_sigma = float(gauge_sigma)
        # The gauge's reference. "median" pins the population's middle bias, so b_i is
        # fleet-relative and the state steps when the population changes. "prior" gives
        # each b_i an absolute prior N(0, gauge_prior_sigma) instead (see gauge()); the
        # broker runs "prior".
        self.gauge_mode = str(gauge_mode)
        if self.gauge_mode not in ("median", "prior"):
            raise ValueError(
                "gauge_mode must be 'median' or 'prior', not %r" % gauge_mode
            )
        # b_sat is the per-satellite code bias left after the BRDC model: ephemeris residual
        # (~1-2 m), group delay / DCB mismatch (~1-3 m) and the differential part of
        # iono+tropo at 1176 MHz (up to ~10-20 m at low elevation; the common part belongs
        # to clk). One chip is 29.3 m, so |b| <~ 1 chip is the physical envelope and 2.0
        # chips is that envelope with margin.
        self.gauge_prior_sigma = float(gauge_prior_sigma)
        self.max_age_s = float(max_age_s)
        # Innovation gating (see update()).
        self.innov_max = innov_max  # absolute garbage ceiling (chips), any state
        self.innov_nsigma = (
            innov_nsigma  # the gate in units of sqrt(S); this one does the work
        )
        self.reject_escape = (
            reject_escape  # run length (consecutive rejections) the escape needs
        )
        self._rej_run = {}  # key -> consecutive rejections
        # The rejected measurements themselves: the escape requires the run to agree with
        # itself, not merely to be long.
        self._rej_hist = {}  # key -> recent rejected measurements y (rolling)
        self._rej_t = {}  # key -> time of its newest rejection
        self.escape_spread = float(escape_spread)
        # An escape may believe a run that agrees, but not one that is physically
        # impossible. Same separation as the clock solve's MAD bound: real per-sat offsets
        # cluster inside ~+-10 chips while a wrap alias is thousands, so anything from 50 to
        # 500 separates them.
        self.escape_max_step = float(escape_max_step)
        # Beyond escape_max_step a step is believed only with corroboration: escape_min_sats
        # distinct satellites, each with its own coherent run, implying the same step to
        # within escape_spread. One satellite agreeing with itself says nothing about the
        # clock (one satellite can feed a self-consistent wrong value); the receiver
        # clock is the one term independent satellites share, so agreement across them is
        # a clock event. The evidence is counted in satellites, never in samples.
        self.escape_min_sats = int(escape_min_sats)
        # A run votes for another satellite's escape only while it is current. A run is kept
        # until that satellite's next accepted update or its age-out, so a satellite that
        # stopped reporting would otherwise vote with an old y. 20 s is two full runs at the
        # broker's 2 s cycle, so a corroborator that misses a cycle still votes.
        self.escape_vote_age_s = 20.0
        self.rate_max = float(
            rate_max
        )  # chips/s; 1.0 = 0.1 ppm = 2500x the GPSDO truth
        # Deaf latch. A state wrong enough to reject every measurement never corrects, while
        # predict() keeps integrating clk_rate and walks it further away. The escape cannot
        # catch this: it fires per satellite on a coherent run, and such a state refuses
        # newcomers at birth (birth_max), which never reaches _rej_hist. After deaf_after
        # consecutive rejections with nothing accepted the state latches `deaf` and notes it;
        # the broker's _joint_state then refuses it to the code consumers. The next accepted
        # update clears it.
        self.deaf_after = 40  # consecutive rejections with zero accepts -> deaf
        self.rej_since_upd = 0  # rejections since the last accepted update
        self.deaf = False  # latched: this state has stopped accepting anything
        # Healthy biases stay within ~6 chips; 12 is 2x margin. The window is open during
        # bootstrap (see update()), so garbage from before the clock is determined is kept
        # out by the broker's joint_feed_warmup_s instead.
        self.birth_max = birth_max  # refuse to birth a sat implausibly far from clk
        self.gauge_max_b = gauge_max_b  # median gauge: inlier bound about the median
        # P floor (1-sigma, chips) on clk and the bias rows. A flood of self-consistent
        # measurements can crush P toward zero, and a zero-gain state cannot move, so no
        # escape can rescue it. Healthy clock sigma is ~0.05-0.08 chips, so 0.04 sits just
        # below the best honest value without touching normal operation.
        self.p_floor = float(p_floor)
        # Bootstrap also ends after this many accepted updates, whatever P00 says (update()).
        self.birth_gate_after = int(birth_gate_after)
        # Events worth an operator's attention (escapes, incoherent runs, the deaf latch,
        # gauge fallbacks). The filter has no logger, so it queues one-line notes and the
        # broker drains them.
        self.notes = []
        # f_carrier: the receiver-wide carrier frequency offset (Hz at f_ref_hz), one number
        # for every satellite, the shape of an LO error; per-satellite errors belong in the
        # rrate rows. Its row is created on the first carrier measurement, so a state fed no
        # carrier keeps the code-only layout.
        #
        # Feed it a carrier rate that depends on the signal (the broker's deep_rate_full_hz
        # gated on carrier_rate_min_q, the fine ADR phase rate and the KCOH fold rate), never
        # carrier_hz_resid, which reads the same on noise as on signal at CHORD SNR: a filter
        # fed by it produces a confident, meaningless estimate.
        self.sigma_fcar0 = float(sigma_fcar0)
        self.q_fcar = float(q_fcar)  # Hz per sqrt(s); a GPSDO-disciplined LO, so slow
        self._fcar_idx = None  # row in x, or None until the first carrier update

        # rrate: each satellite's range-rate model error (m/s), i.e. its orbit error. An LO
        # error is one number for the receiver (f_carrier); what varies per satellite is the
        # orbit, which becomes Hz only through a carrier. Held in m/s it is band-independent:
        # a satellite's E5a and E5b Doppler errors are one range-rate error seen through two
        # carriers (1207.14/1176.45 = 1.0261 apart), so both bands update the same row.
        #
        #     y_(i,b) [Hz] = -(f_b/c) * rrate_i   +   (f_b/f_ref) * f_carrier   +   noise
        #                     \_ orbit, per sat _/     \_ LO, receiver-wide _/
        #
        # Within one band the two terms are degenerate (both scale with f_b); like clk and
        # b_sat they are separated only because one is common across satellites, hence one
        # solve. Their common mode is unobservable and gauge_rrate() pins it.
        #
        # Not coupled to b_sat: b_sat (chips) and rrate (m/s) are the position and velocity of
        # one per-satellite range error, and a vector state would link them (a carrier-aided
        # code loop). That link is not built: a rate state fed by code offsets can read an
        # offset step as a rate (see rate_max).
        # The carrier f_carrier is expressed at. Latched from the first carrier measurement if
        # not given, so a band arriving later cannot redefine what f_carrier means.
        self.f_ref_hz = float(f_ref_hz) if f_ref_hz else None
        self.sigma_rr0 = float(sigma_rr0)
        self.q_rr = float(q_rr)  # m/s per sqrt(s); orbit error, so slow
        # Moderate by choice: stiffer and looser gauges converge too (test_rrate_state.py,
        # test_gauge_stiffness_does_not_freeze_the_rows).
        self.rr_gauge_sigma = float(rr_gauge_sigma)
        self._rr_idx = {}  # key -> row in x
        self._rr_t_seen = {}  # key -> last carrier measurement time
        self._rr_reject_run = {}  # key -> consecutive rejections (escape)
        self.rr_escape_runs = 8
        self.n_rrate = 0
        self.rrate_rejected = 0
        self.x = np.zeros(2)  # [clk chips, clk_rate chips/s]
        # Warm start: clk0 seeds clk when the state is created; the prior sigma is still
        # sigma_clk0. dr_joint_shadow passes its dead-reckon clock, but any earlier
        # joint_receiver() call creates the state without one.
        self.x[0] = float(clk0)
        self.P = np.diag([sigma_clk0 ** 2, sigma_rate0 ** 2])
        self._idx = {}  # key -> row in x
        self._t_seen = {}  # key -> last accepted measurement time
        self._n = {}  # key -> accepted count
        self._t = None  # state epoch
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

    @staticmethod
    def _key_str(key):
        """(tag, prn) -> 'G25'. Keys are chain-tagged tuples; notes want the sat."""
        try:
            return "%s%d" % (key[0][0].upper(), int(key[1]))
        except Exception:
            return str(key)

    def _reject(self, why=None):
        """Count a rejection; after deaf_after consecutive rejections with nothing accepted,
        latch `deaf` and say so in a note (see the deaf latch in __init__). The next accepted
        update clears it."""
        self.rejected += 1
        self.rej_since_upd += 1
        if (not self.deaf) and self.rej_since_upd >= self.deaf_after:
            self.deaf = True
            self._note(
                "DEAF: %d consecutive rejections with nothing accepted (clk %+.1f "
                "chips, rate %+.4f chips/s, sigma %.3f, n=%d). The state is refusing "
                "every measurement, so it cannot correct itself, and predict() keeps "
                "integrating the rate. Not fit to consume until it accepts again%s"
                % (
                    self.rej_since_upd,
                    float(self.x[0]),
                    float(self.x[1]),
                    math.sqrt(max(0.0, self.P[0, 0])),
                    len(self._idx),
                    "" if why is None else " (%s)" % why,
                )
            )

    def _note(self, msg):
        """Queue a one-line operator note; the broker drains `notes` each cycle. Bounded to
        the newest 200, so a pathological chain cannot grow it without limit."""
        self.notes.append(msg)
        if len(self.notes) > 200:
            del self.notes[:-200]

    @_locked
    def drain_notes(self):
        # Locked: the swap reads then rebinds, so an unlocked drain could lose or duplicate
        # notes.
        out, self.notes = self.notes, []
        return out

    def _remove_rows(self, rows):
        """Delete these rows from x and P and renumber every row index that survives."""
        np = self._np
        gone = set(rows)
        keep = [i for i in range(self.x.size) if i not in gone]
        self.x = self.x[keep]
        self.P = self.P[np.ix_(keep, keep)]
        # Reindex from the surviving row order: the b_sat, f_carrier and rrate rows share the
        # space past clk_rate, so a renumber from a fixed base would alias them onto each
        # other.
        remap = {old_row: new_row for new_row, old_row in enumerate(keep)}
        self._idx = {k: remap[i] for k, i in self._idx.items() if i in remap}
        if self._fcar_idx is not None:
            self._fcar_idx = remap.get(self._fcar_idx)
        self._rr_idx = {
            k: remap[i] for k, i in self._rr_idx.items() if remap.get(i) is not None
        }
        self._rr_t_seen = {
            k: v for k, v in self._rr_t_seen.items() if k in self._rr_idx
        }

    def _drop(self, keys):
        """Remove stale sats. Their rows leave the state entirely: a sat with no accepted
        measurement in max_age_s must not keep voting in the gauge, which would drag the
        clock toward a bias nothing refreshes. These are the b_sat rows; _drop_rr ages out
        the rrate rows."""
        if not keys:
            return
        rows = [self._idx[k] for k in keys]
        for k in keys:
            del self._idx[k], self._t_seen[k], self._n[k]
            self._rej_run.pop(k, None)
            self._rej_hist.pop(k, None)
            self._rej_t.pop(k, None)
        self._remove_rows(rows)
        self._membership_changed()

    def _drop_rr(self, keys):
        """Remove stale rrate rows, for the same reason as _drop: a satellite with no carrier
        measurement in max_age_s must not keep voting in gauge_rrate's pin with a value
        nothing refreshes. Its rrate_sigma then reads inf, so it is not commanded until its
        next carrier measurement births the row again."""
        if not keys:
            return
        rows = [self._rr_idx[k] for k in keys]
        for k in keys:
            self._rr_reject_run.pop(k, None)
        self._remove_rows(rows)

    def _membership_changed(self):
        """Handle the gauge step a satellite joining or leaving can cause.

        Under the median gauge a membership change can step clk (~1 chip at 6 sats). That
        step is a change of convention, not clock motion, and must not be differentiated
        into clk_rate, so each mode zeroes the clk/rate cross-covariance (the exact relabel
        only when it shifts something). Beyond that:

        prior: nothing; no satellite's prior mentions another, so nothing steps.

        median, rereference=False: also add 4 chips^2 to P[0,0] so the step is absorbed as
        offset. This treats the step as new information: the next measurement moves the
        clock hard, predict() rebuilds the clk/rate correlation at once (F[0,1] = dt), and
        the yank reaches the rate. Sound with many satellites, self-defeating with few and
        frequent churn.

        median, rereference=True: relabel exactly, shifting the common mode out of the
        biases and into clk:

            delta = median(b over the new membership);  b_i -= delta;  clk += delta

        clk + b_i, the only observable a consumer reads, is unchanged for every satellite.
        The median is over every row, which is the median gauge() pins only while every bias
        lies within gauge_max_b of it; otherwise gauge() still pins a small residual. P is
        carried through the same linear map (P -> A P A^T); no uncertainty is injected
        because none was created."""
        if self.gauge_mode == "prior":
            # Nothing to relabel; only the slow equilibrium moves (by ~b_gone/N, paced by
            # gauge_prior_sigma), and that relaxation is gauge motion, not clock motion.
            self.P[0, 1] = self.P[1, 0] = 0.0
            return
        if not self.rereference:
            self.P[0, 1] = self.P[1, 0] = 0.0
            self.P[0, 0] += 4.0
            return
        np = self._np
        idx = list(self._idx.values())
        if not idx:
            return
        # The median over every row. While every bias is an inlier it is the median gauge()
        # pins, so the next gauge() has no step to grind out through gauge_sigma; with an
        # outlier excluded there, gauge() still pins a small residual.
        _ordered = sorted(idx, key=lambda i: float(self.x[i]))
        _n = len(_ordered)
        _sel = (
            [(_ordered[_n // 2], 1.0)]
            if _n % 2
            else [(_ordered[_n // 2 - 1], 0.5), (_ordered[_n // 2], 0.5)]
        )
        delta = sum(w * float(self.x[j]) for j, w in _sel)
        if delta == 0.0:
            return
        # x' = A x, with A: clk += median(b), b_i -= median(b). An exact linear relabelling
        # (the median rows are fixed at relabel time), so P -> A P A^T is the correct
        # covariance transport.
        A = np.eye(self.x.size)
        for j, w in _sel:
            A[0, j] += w  # clk picks up the median of the biases
            for i in idx:
                A[i, j] -= w  # every bias loses it
        self.x = A @ self.x
        self.P = A @ self.P @ A.T
        # A relabelling is not clock motion, so it must not be differentiated into clk_rate.
        self.P[0, 1] = self.P[1, 0] = 0.0

    # -- filter ------------------------------------------------------------------
    @_locked
    def predict(self, t_now):
        # Geometry tripwire. Under the lock x and P cannot desync, so unequal sizes mean an
        # unlocked mutation path; fail here by name rather than with a numpy shape error in
        # a later update.
        if self.x.size != self.P.shape[0]:
            raise RuntimeError(
                "JOINT STATE DESYNC: x has %d rows, P has %d -- an "
                "unlocked mutation path corrupted the filter"
                % (self.x.size, self.P.shape[0])
            )
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
            # q_b (chips) goes on the bias rows only; f_carrier (Hz) and the rrate rows (m/s)
            # take their own process noise below.
            other = set()
            if self._fcar_idx is not None:
                other.add(self._fcar_idx)
            other |= set(self._rr_idx.values())
            srows = [i for i in range(2, n) if i not in other]
            if srows:
                idx = np.array(srows)
                self.P[idx, idx] += self.q_b ** 2 * dt
            if self._fcar_idx is not None:
                self.P[self._fcar_idx, self._fcar_idx] += self.q_fcar ** 2 * dt
            for i in self._rr_idx.values():
                self.P[i, i] += self.q_rr ** 2 * dt  # orbit error: a random walk
        # P floor (see __init__) on clk and the bias rows, not on f_carrier or rrate, whose
        # units and timescales differ. Diagonal only; the correlations are left as they are.
        if self.p_floor > 0.0:
            f2 = self.p_floor ** 2
            if self.P[0, 0] < f2:
                self.P[0, 0] = f2
            if n > 2:
                other = set()
                if self._fcar_idx is not None:
                    other.add(self._fcar_idx)
                other |= set(self._rr_idx.values())
                for i in range(2, n):
                    if i not in other and self.P[i, i] < f2:
                        self.P[i, i] = f2

    def _scalar_update(self, H, y, R):
        """One scalar measurement; y is the innovation, wrapped by the caller where it is a
        code phase. Returns the normalized innovation y/sqrt(S), or 0.0 if S <= 0."""
        np = self._np
        PH = self.P @ H
        S = float(H @ PH + R)
        if S <= 0.0:
            return 0.0
        K = PH / S
        # RATE-TEACH: note every update that moves clk_rate by more than 0.01 chips/s (~25x
        # the true drift), with the measurement's rows, innovation and gain, so the
        # measurement that taught a wild rate can be named. Log only.
        _dr = float(K[1] * y) if self.x.size > 1 else 0.0
        if abs(_dr) > 0.01:
            _hn = [i for i in range(len(H)) if H[i] != 0.0]
            self._note(
                "RATE-TEACH %+.4f chips/s (rate %+.4f -> %+.4f): innov %+.3f, "
                "K1 %+.5f, H rows %s, P01 %+.3f P11 %+.5f S %.4f"
                % (
                    _dr,
                    float(self.x[1]),
                    float(self.x[1]) + _dr,
                    y,
                    float(K[1]),
                    _hn[:6],
                    float(self.P[0, 1]),
                    float(self.P[1, 1]),
                    S,
                )
            )
        self.x = self.x + K * y
        # Joseph form: this filter runs for days at a 2 s cadence with sats entering and
        # leaving, and the simple (I-KH)P loses symmetry and positivity over that many updates.
        n = self.x.size
        A = np.eye(n) - np.outer(K, H)
        self.P = A @ self.P @ A.T + np.outer(K, K) * R
        self.P = 0.5 * (self.P + self.P.T)
        return y / math.sqrt(S)

    def wrap(self, v):
        return ((v + self.L / 2.0) % self.L) - self.L / 2.0

    @_locked
    def shorten_modulus(self, code_len):
        """Adopt a shorter code period, keeping every row.

        Receiver.joint_receiver() calls this when a chain with a shorter code joins, so the
        rows already converged survive. Reducing the modulus loses no information:
          * the biases and clk_rate are small signed offsets, not modular residues, and carry
            over untouched;
          * the covariance is unchanged: knowing a value mod a shorter period makes no
            estimate less certain;
          * only clk is a modular residue, so it alone is re-wrapped into the new period.
        Refuses to lengthen: a measurement known only mod 10230 cannot be promoted to mod
        1023000 without inventing the segment.

        Returns True if the modulus changed.
        """
        L_new = float(code_len)
        if not (L_new > 0.0) or L_new >= self.L:
            return False
        self.L = L_new
        # clk is the only modular quantity in the state. Everything else is an offset.
        self.x[0] = ((float(self.x[0]) + L_new / 2.0) % L_new) - L_new / 2.0
        return True

    @_locked
    def update(self, key, y_chips, sigma_chips, t_now):
        """Feed one y_i = clk + b_i observation. Returns the normalized innovation, or
        None if the input is unusable or rejected. A new satellite's row is added at b = 0
        with prior sigma_b0 and takes a normal update, so its first innovation lands in clk
        while the clock is unknown and in its own bias once the clock is determined; it is
        never walked in from zero at loop gain, which would drag the clock for minutes."""
        np = self._np
        if not (math.isfinite(y_chips) and sigma_chips > 0.0):
            return None
        self.predict(t_now)
        born = key not in self._idx
        if born:
            # Birth goes through the covariance, not by hand. Writing b0 = y - clk would put
            # the whole offset in a row with no cross-covariance to clk, and the gauge would
            # then shear it out of the biases without depositing it in clk, losing the
            # clock's absolute value (and near L/2 converging to a wrapped alias). With
            # sigma_clk0 >> sigma_b0 the gains split the first innovations correctly.
            # Birth window: a birth has no innovation to gate on, so once the clock is
            # determined a newborn more than birth_max chips from clk is refused (a real bias
            # is chips, a weak-sat detection hundreds). The clock counts as determined when
            # P00 < 100 or after birth_gate_after accepted updates; the second bound holds
            # even if a degenerate row keeps P00 high. With fewer than two satellites every
            # birth is accepted.
            if (
                self.birth_max is not None
                and len(self._idx) >= 2
                and (self.P[0, 0] < 100.0 or self.n_updates >= self.birth_gate_after)
                and abs(self.wrap(y_chips - self.x[0])) > self.birth_max
            ):
                self._reject("birth")
                return None
            self._add(key, 0.0, t_now)
        i = self._idx[key]
        H = np.zeros(self.x.size)
        H[0] = 1.0
        H[i] = 1.0
        innov = self.wrap(y_chips - float(H @ self.x))
        # Not on the birth cycle: a newborn sits at b = 0, so its innovation carries its whole
        # bias (and at bootstrap the whole clock), and birth_max vets it instead.
        # The gate is normalized by sqrt(S): a fixed chip bound tight enough to catch garbage
        # on a converged state would, after an outage, refuse every satellite whose clock
        # uncertainty had legitimately grown, and they would then age out. innov_max stays
        # as an absolute garbage ceiling.
        if not born:
            PH_ = self.P @ H
            S_ = float(H @ PH_ + sigma_chips ** 2)
            z_ = abs(innov) / math.sqrt(S_) if S_ > 0 else 0.0
            _bad = z_ > self.innov_nsigma or (
                self.innov_max is not None and abs(innov) > self.innov_max
            )
            if _bad:
                # Escape. A gate with no way out deadlocks on a genuine step (a clock event, a
                # re-anchor): the state never follows it, so every later measurement is
                # rejected too. After reject_escape consecutive rejections that agree with
                # each other to within escape_spread, believe them and inflate P so the step
                # is taken quickly. Agreement, not persistence, is the test: a satellite that
                # has left the beam reports code phase uniform over the code, which persists
                # but does not agree (on a 10230-chip code, four uniform draws all within 5
                # chips of a fifth: (10/10230)^4 ~ 1e-12).
                # Agreement is tested on the measurement y, not the innovation: while a run of
                # rejections lasts, the gauge and the process model keep moving x, so honest
                # repeated measurements give innovations that drift apart.
                hist = self._rej_hist.setdefault(key, [])
                hist.append(y_chips)
                self._rej_t[key] = t_now
                if len(hist) > self.reject_escape:
                    del hist[: -self.reject_escape]
                self._rej_run[key] = self._rej_run.get(key, 0) + 1
                spread = None
                if len(hist) >= self.reject_escape:
                    # circular spread about the run's first sample, so a cluster straddling
                    # +-L/2 is not read as maximally scattered
                    spread = max(abs(self.wrap(v - hist[0])) for v in hist)
                if spread is None or spread > self.escape_spread:
                    self._reject("gate" if spread is None else "incoherent-run")
                    if spread is not None:
                        # A full run that disagrees with itself: this satellite is feeding
                        # noise, a tracking failure rather than a state failure. Say so.
                        self._note(
                            "REJECT-RUN %s: %d consecutive, spread %.1f chips "
                            "(> escape_spread %.1f) -- incoherent, not believed; "
                            "this sat is feeding noise"
                            % (
                                self._key_str(key),
                                self._rej_run[key],
                                spread,
                                self.escape_spread,
                            )
                        )
                    return None
                # A coherent run can still imply an unphysical step: on a 10230-chip code a
                # jump of thousands is a wrap alias, not a clock. Once believed it is applied
                # at near-unit gain (P[0,0] += 25 against R = sigma^2, 0.09 at the broker's
                # joint_sigma of 0.3 chips), so the state moves by the whole innovation, every
                # other satellite then disagrees, and the escapes feed on each other.
                if abs(innov) > self.escape_max_step:
                    # Beyond escape_max_step, require escape_min_sats satellites (this one
                    # included) with their own coherent runs implying the same step. Each
                    # implied step is recomputed against the current state, so all are
                    # referred to one epoch.
                    _corr = self._corroborators(key, innov, t_now)
                    if _corr + 1 < self.escape_min_sats:
                        self._reject("escape-refused")
                        self._note(
                            "ESCAPE REFUSED %s: %d rejections agreeing to %.2f chips "
                            "but implying a %+.1f chip step (bound %.0f on a %.0f-chip "
                            "code) and only %d/%d satellites agree -- one satellite's "
                            "self-consistency is not evidence about the clock. The "
                            "state may still be wrong; corroboration, or a warm start, "
                            "is the way out -- not a jump on one sat's word"
                            % (
                                self._key_str(key),
                                self._rej_run[key],
                                spread,
                                innov,
                                self.escape_max_step,
                                self.L,
                                _corr + 1,
                                self.escape_min_sats,
                            )
                        )
                        return None
                    self._note(
                        "ESCAPE CORROBORATED %s: %+.1f chip step exceeds the %.0f "
                        "bound, but %d satellites independently agree to within %.1f "
                        "chips -- that is a clock event, not one sat's noise"
                        % (
                            self._key_str(key),
                            innov,
                            self.escape_max_step,
                            _corr + 1,
                            self.escape_spread,
                        )
                    )
                self._note(
                    "ESCAPE %s: %d consecutive rejections agreeing to %.2f chips "
                    "-- believing a %+.1f chip step and inflating P"
                    % (self._key_str(key), self._rej_run[key], spread, innov)
                )
                self.P[0, 0] += 25.0
                self.P[i, i] += 25.0
                self.P[0, 1] = self.P[1, 0] = 0.0
        self._rej_hist.pop(key, None)
        self._rej_t.pop(key, None)
        self._rej_run.pop(key, None)
        z = self._scalar_update(H, innov, sigma_chips ** 2)
        self._t_seen[key] = t_now
        self._n[key] = self._n.get(key, 0) + 1
        self.n_updates += 1
        self.rej_since_upd = 0  # it is listening again
        self.deaf = False
        return z

    @_locked
    def gauge(self):
        """Pin the unobservable common mode of clk and the biases (gauge_mode).

        "median" pins median(b) = 0 over the inlier satellites: the convention of the legacy
        clock, the circular median of the same residuals, so bare clk steps on a rise or set
        only when the middle element changes. "prior" (what the broker runs) gives each b_i
        an absolute prior instead; see below. clk + b_i, the only gauge-invariant
        observable, is unchanged by either."""
        np = self._np
        if self.gauge_mode == "prior":
            # Prior gauge: each b_i carries a weak absolute prior b_i ~ N(0,
            # gauge_prior_sigma), applied every cycle as a pseudo-measurement. No prior
            # mentions another satellite, so a rise or set relabels nothing and steps
            # nothing. In steady state the split settles at a soft mean(b) ~ 0, so the
            # population enters only through that slow equilibrium: a membership change moves
            # it by ~b_gone/N and the state relaxes there over many cycles.
            #
            # Robustness: beyond cap = 3 sigma, R is inflated by (|y|/cap)^2 (Huber), so the
            # pull falls as ~1/|y| and a wild bias is nearly ignored while ordinary
            # re-centring runs at full strength. A hard clamp would keep full gain on the
            # outlier and drag clk through the correlations. Unlike an inlier screen centred
            # on zero, the prior never switches itself off. The trade: a uniformly displaced
            # comb heals slowly (all its innovations are beyond cap). That is accepted
            # because clk + b_i stays correct throughout, while a dragged clk would corrupt
            # the rate and the bare-clk consumers at once.
            #
            # No freshness screen: dropping the prior for satellites without a recent
            # measurement re-centres the equilibrium when a satellite goes silent, a second
            # membership edge. The R-inflation already cuts a drifting bias's weight
            # gradually, and every satellite keeps its prior until the ordinary age-out.
            for key, i in list(self._idx.items()):
                y = -float(self.x[i])
                _R = self.gauge_prior_sigma ** 2
                _cap = 3.0 * self.gauge_prior_sigma
                if abs(y) > _cap:
                    _R *= (y / _cap) ** 2
                H = np.zeros(self.x.size)
                H[i] = 1.0
                self._scalar_update(H, y, _R)
            return
        # Median gauge. Inliers lie within gauge_max_b of the median, so a wild bias cannot
        # become the middle element of a small set. They are measured from the median, not
        # from zero: a zero-centred screen assumes the biases are already centred, and when
        # every bias is displaced it comes back empty and the gauge stops, leaving clk free
        # to drift while clk + b_i still looks right. Centred on the median, a single wild
        # sat is excluded, and a uniformly displaced comb is included whole and pulled back.
        idx = list(self._idx.values())
        if len(idx) < 2:
            return
        b = [float(self.x[i]) for i in idx]
        med = float(np.median(b))
        use = [
            i
            for i in idx
            if self.gauge_max_b is None
            or abs(float(self.x[i]) - med) <= self.gauge_max_b
        ]
        if len(use) < 2:
            # The biases are spread wider than gauge_max_b about the median. Pin on all of
            # them rather than leave the common mode floating, and say so.
            self._note(
                "GAUGE: only %d inlier(s) of %d about median %+.2f -- pinning on all"
                % (len(use), len(idx), med)
            )
            use = idx
        H = np.zeros(self.x.size)
        # H selects the median of the inliers (the middle row, or the two middle rows at half
        # weight), so H @ x is the median and the pin stays a linear measurement.
        _ordered = sorted(use, key=lambda i: float(self.x[i]))
        _n = len(_ordered)
        if _n % 2:
            H[_ordered[_n // 2]] = 1.0
        else:
            H[_ordered[_n // 2 - 1]] = 0.5
            H[_ordered[_n // 2]] = 0.5
        self._scalar_update(H, -float(H @ self.x), self.gauge_sigma ** 2)

    def _corroborators(self, key, innov, t_now):
        """How many other satellites have a coherent rejection run implying the same step
        as `innov` now.

        Independent satellites share one term, the receiver clock, so agreement across them
        is evidence about the clock, while one satellite agreeing with itself shows only that
        its measurement repeats. A run votes only if it holds reject_escape samples, its
        newest rejection is within escape_vote_age_s, and it passes the same escape_spread
        test as the caller; its implied step is its newest measurement against the current
        prediction.
        """
        n = 0
        for k2, h2 in self._rej_hist.items():
            if k2 == key or len(h2) < self.reject_escape:
                continue
            if t_now - self._rej_t.get(k2, -math.inf) > self.escape_vote_age_s:
                continue  # that sat stopped reporting; its old y is not current evidence
            if max(abs(self.wrap(v - h2[0])) for v in h2) > self.escape_spread:
                continue  # that sat is feeding noise; it gets no vote
            innov2 = self.wrap(h2[-1] - self.predicted(k2))
            if abs(self.wrap(innov2 - innov)) <= self.escape_spread:
                n += 1
        return n

    @_locked
    def cycle(self, measurements, t_now):
        """One cycle: predict, update each measurement, gauge, clamp the rate, expire.

        `measurements` is an iterable of (key, y_chips, sigma_chips). Returns the number
        accepted."""
        self.predict(t_now)
        n_ok = 0
        for key, y, sig in measurements:
            if self.update(key, y, sig, t_now) is not None:
                n_ok += 1
        self.gauge()
        # Physical rate ceiling. The reference is GPS-disciplined (true drift ~4e-4
        # chips/s), so rate_max = 1.0 chips/s (0.1 ppm) clamps nothing real; the broker
        # bounds its drift estimate the same way (dr_max_drift_chips_s). The state must not
        # carry an impossible rate, not merely refuse to publish it: the rate multiplies dt
        # in every predict, so a wild value corrupts clk within seconds. Clamp and widen
        # P[1,1] so the rate re-learns from measurements.
        if abs(float(self.x[1])) > self.rate_max:
            _was = float(self.x[1])
            self.x[1] = self.rate_max if _was > 0 else -self.rate_max
            self.P[1, 1] += _was ** 2
            self.P[0, 1] = self.P[1, 0] = 0.0
            self._note(
                "RATE CLAMPED %+.3f -> %+.1f chips/s (bound %.1f, 0.1 ppm at 10.23 Mcps, "
                "2500x the GPSDO drift): a common-mode step is an offset, never a "
                "frequency -- the state was diverging"
                % (_was, float(self.x[1]), self.rate_max)
            )
        self._drop([k for k, t in self._t_seen.items() if (t_now - t) > self.max_age_s])
        self._drop_rr(
            [k for k, t in self._rr_t_seen.items() if (t_now - t) > self.max_age_s]
        )
        return n_ok

    # -- carrier rows and readout ------------------------------------------------
    def _add_fcar(self):
        """Give f_carrier its own row, on the first carrier measurement, so a state fed no
        carrier keeps the code-only layout."""
        np = self._np
        i = self.x.size
        self.x = np.append(self.x, 0.0)
        P = np.zeros((i + 1, i + 1))
        P[:i, :i] = self.P
        P[i, i] = self.sigma_fcar0 ** 2
        self.P = P
        self._fcar_idx = i
        return i

    C_LIGHT = 299792458.0

    def _add_rrate(self, key, rr0, t_now):
        """Give a satellite its rrate row, lazily."""
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

    @_locked
    def update_rrate(self, key, y_hz, t_now, f_band_hz, sigma_hz=0.2):
        """One carrier-frequency measurement for one satellite, from one band.

        `y_hz` is that satellite's measured carrier-rate residual in this band (Hz); feed a
        rate that depends on the signal, for the reasons f_carrier's note in __init__ gives.

            y = -(f_b/c) * rrate_i  +  (f_b/f_ref) * f_carrier  +  noise

        One measurement updates both rows: the receiver-wide and per-orbit terms are
        degenerate within a band and are separated only because one is shared. A second band
        on the same satellite lands on the same rrate row with a different f_b, so the bands
        combine instead of being fitted twice.

        Returns the accepted innovation (Hz), 0.0 for a birth, or None if gated out.
        """
        np = self._np
        self.predict(t_now)
        f_b = float(f_band_hz)
        if self.f_ref_hz is None:
            self.f_ref_hz = f_b
        f_ref = float(self.f_ref_hz)
        if self._fcar_idx is None:
            # The first carrier measurement: one sample cannot split the receiver-wide term
            # from this orbit's, so f_carrier takes it all and the gauge hands back the
            # per-satellite part once a second satellite reports.
            self._add_fcar()
            self.x[self._fcar_idx] = y_hz * (f_ref / f_b)
            # The satellite gets its row now, at zero; without one it could never disagree
            # with the value it defined.
            self._add_rrate(key, 0.0, t_now)
            return 0.0
        if key not in self._rr_idx:
            # Birth on the measurement: initialise to the residual left after the current
            # receiver-wide estimate, so a real per-sat offset is not read as noise while
            # the row walks in from zero at loop gain.
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
            # Escape. A gate with no way out turns a bad birth into a permanent one. After
            # rr_escape_runs consecutive rejections (a count only; no agreement test), the
            # row is re-opened.
            if self._rr_reject_run[key] >= self.rr_escape_runs:
                # Re-open, do not adopt. On carrier rows a sustained rejection usually means
                # a weak satellite whose rate picks scatter by several Hz, which the gate is
                # right to refuse; adopting them would walk the command. Re-opening P alone
                # handles both cases: the wider sigma de-commands the satellite (the
                # broker's rrate_cmd_max_sigma gate), and a mis-converged row is pulled in by
                # normal updates through the wide gate once its measurements turn consistent.
                i_ = self._rr_idx[key]
                # Zero the row's cross-covariances too: resetting the diagonal alone can
                # leave |P_ij| > sqrt(P_ii P_jj), a non-PSD covariance whose later updates
                # drive other diagonals negative.
                self.P[i_, :] = 0.0
                self.P[:, i_] = 0.0
                self.P[i_, i_] = self.sigma_rr0 ** 2
                self._rr_reject_run[key] = 0
                self._note(
                    "rrate ESCAPE %s: %d consecutive rejections -- re-opening the "
                    "row without adopting (x held)"
                    % (self._key_str(key), self.rr_escape_runs)
                )
            else:
                self._note(
                    "rrate REJECT %s: innov %+.3f Hz vs %.1f-sigma bar %.3f"
                    % (
                        self._key_str(key),
                        innov,
                        self.innov_nsigma,
                        self.innov_nsigma * var ** 0.5,
                    )
                )
            return None
        self._rr_reject_run[key] = 0
        self._scalar_update(H, innov, float(sigma_hz) ** 2)
        self.n_rrate += 1
        return innov

    @_locked
    def gauge_rrate(self):
        """Pin mean(rrate) = 0 over the rrate rows within max(10 sigma_rr0, 1 m/s) of their
        median (all rows if fewer than two qualify). cycle() ages out a row with no carrier
        measurement in max_age_s, so a satellite that has set loses its vote.

        Without this, f_carrier and the rrate rows share an unobservable common mode:
        shifting the receiver-wide term and every orbit by equivalent amounts leaves every
        prediction unchanged. The screen is centred on the median so a wild orbit gets no
        vote.
        """
        np = self._np
        idx = list(self._rr_idx.values())
        if len(idx) < 2 or self._fcar_idx is None:
            return
        v = [float(self.x[i]) for i in idx]
        med = float(np.median(v))
        use = [
            i
            for i in idx
            if abs(float(self.x[i]) - med) <= max(10.0 * self.sigma_rr0, 1.0)
        ]
        if len(use) < 2:
            use = idx
        H = np.zeros(self.x.size)
        for i in use:
            H[i] = 1.0 / len(use)
        self._scalar_update(H, -float(H @ self.x), self.rr_gauge_sigma ** 2)

    @_locked
    def rrate(self, key):
        """This satellite's range-rate model error, m/s. 0.0 if never measured."""
        i = self._rr_idx.get(key)
        return 0.0 if i is None else float(self.x[i])

    @_locked
    def rrate_sigma(self, key):
        """1-sigma, m/s. inf when unmeasured -- an unmeasured state must not read as a
        confident zero, which is how a dead feed passes for a healthy one."""
        i = self._rr_idx.get(key)
        if i is None:
            return float("inf")
        # A negative diagonal is a numerically broken covariance, and ** 0.5 of a negative
        # float is complex in Python, which fails at the next comparison. Broken reads as
        # unmeasured: inf sends every sigma gate to its refuse branch (not v >= 0 also
        # catches NaN).
        _v = float(self.P[i, i])
        return float("inf") if not (_v >= 0.0) else _v ** 0.5

    @_locked
    def carrier_correction_hz(self, key, f_band_hz):
        """Total carrier correction for this satellite in this band (Hz).

        What a seed adds to its model Doppler: the receiver-wide term and the orbit term
        together, scaled to the band being seeded. It is the one number the estimator emits
        per (satellite, band): one command, rather than a seed plus a separate trim on the
        same state.
        """
        f_b = float(f_band_hz)
        f_ref = float(self.f_ref_hz or f_b)
        fc = float(self.x[self._fcar_idx]) if self._fcar_idx is not None else 0.0
        return (f_b / f_ref) * fc - (f_b / self.C_LIGHT) * self.rrate(key)

    @_locked
    def f_carrier(self):
        """Receiver-wide carrier frequency offset (Hz). 0.0 before any measurement."""
        return float(self.x[self._fcar_idx]) if self._fcar_idx is not None else 0.0

    @_locked
    def f_carrier_sigma(self):
        """Its 1-sigma uncertainty (Hz). inf before any measurement: an unmeasured state
        must not read as a confident zero, which is how a dead feed passes for a healthy one."""
        if self._fcar_idx is None:
            return float("inf")
        _v = float(self.P[self._fcar_idx, self._fcar_idx])
        # same negative-diagonal guard as rrate_sigma
        return float("inf") if not (_v >= 0.0) else _v ** 0.5

    @property
    def clk(self):
        return float(self.x[0])

    @property
    def clk_rate(self):
        return float(self.x[1])

    # The read accessors tolerate a key that has left the state. Callers iterate an unlocked
    # snapshot of _idx while another chain's cycle drops rows, so a missing key reads as the
    # clock-only sigma or a 0.0 bias rather than raising: a diagnostic must never kill a
    # chain thread.
    @_locked
    def bias(self, key):
        i = self._idx.get(key)
        return float(self.x[i]) if i is not None and i < len(self.x) else 0.0

    @_locked
    def predicted(self, key):
        """clk + b_i: what a seed for this sat should add to the pure model."""
        return self.clk + self.bias(key)

    @_locked
    def sigma(self, key=None):
        """1-sigma on clk (key None) or on this sat's clk+b_i."""
        np = self._np
        if key is None:
            return math.sqrt(max(0.0, self.P[0, 0]))
        i = self._idx.get(key)
        if i is None or i >= self.P.shape[0]:
            return math.sqrt(max(0.0, self.P[0, 0]))
        v = self.P[0, 0] + 2.0 * self.P[0, i] + self.P[i, i]
        return math.sqrt(max(0.0, v))

    @_locked
    def age(self, key, t_now):
        t = self._t_seen.get(key)
        return None if t is None else (t_now - t)

    @_locked
    def summary(self, t_now, max_sats=12):
        parts = []
        for key, i in sorted(self._idx.items(), key=lambda kv: -abs(self.x[kv[1]])):
            if len(parts) >= max_sats:
                break
            stale = (t_now - self._t_seen.get(key, 0.0)) > self.max_age_s
            parts.append(
                "%s:%+.2f%s"
                % (
                    key if isinstance(key, str) else "%s%d" % (str(key[0])[:1], key[1]),
                    self.x[i],
                    "*" if stale else "",
                )
            )
        return (
            "clk %+.3f+-%.3f chips  rate %+.4f chips/s  n=%d sat(s) upd=%d rej=%d | %s"
            % (
                self.clk,
                self.sigma(),
                self.clk_rate,
                len(self._idx),
                self.n_updates,
                self.rejected,
                " ".join(parts) if parts else "-",
            )
        )
