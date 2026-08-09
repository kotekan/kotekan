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
