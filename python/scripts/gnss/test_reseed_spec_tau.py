"""Task #50: the far-regime re-seed from spec_tau.

WHY IT EXISTS. #49's deep gate admits an off-peak satellite to the DLL, but far off the peak
E, P and L are all noise: q -> 1.0, disc -> 0, and the loop has no gradient to follow. spec_tau
(#32) measures the offset from the phase ramp ACROSS CHANNELS, so it does not need E/P/L to
straddle the peak.

WHAT WAS VALIDATED BEFORE THIS WAS WIRED IN, and what was NOT:
  SIGN      established. Predicted a priori (disc<0 => L>E => peak later => tau>0, hence
            anti-correlated) and confirmed on the shoulder: r -0.47..-0.62, slope -0.67
            chips per unit disc.
  MAGNITUDE NOT established. Far-regime, within-satellite (the across-satellite version is
            confounded by brightness): r -0.375 on strong fits. Hence reseed_gain < 1 -- step
            a fraction and re-measure.
These tests encode that asymmetry: they pin the DIRECTION hard and deliberately do not assert
that one step lands on the answer.
"""
import unittest


# The decision rule as implemented in gps_distributed_broker.py, isolated so the truth table is
# testable without standing up a broker. Kept deliberately in the same order as the source.
def reseed_step(fl, tau, ratio, q, present, opted_in,
                gain=0.5, max_chips=0.75, q_max=1.20, min_ratio=1.5, span=2.0):
    if not (opted_in and present and q < q_max and ratio >= min_ratio and tau is not None):
        return None, "gated"
    if abs(tau) >= span * 0.95:
        return None, "span-edge"
    return max(-max_chips, min(max_chips, gain * tau)), "applied"


class TestReseedGating(unittest.TestCase):

    def test_fires_in_the_far_regime(self):
        """E33's far-regime signature: admitted by the deep gate, no gradient, strong fit."""
        s, why = reseed_step(None, tau=1.360, ratio=3.03, q=0.19, present=True, opted_in=True)
        self.assertEqual(why, "applied")
        self.assertAlmostEqual(s, 0.68, places=6)

    def test_sign_follows_the_validated_convention(self):
        """+tau = the sky arrives LATER than the replica => the code phase must INCREASE.
        This is the one thing the validation actually established, so pin it hard."""
        self.assertGreater(reseed_step(None, 1.0, 3.0, 0.2, True, True)[0], 0)
        self.assertLess(reseed_step(None, -1.0, 3.0, 0.2, True, True)[0], 0)

    def test_does_not_fire_where_the_DLL_can_work(self):
        """q above the bar means the taps straddle the peak -- the trim owns it, and a seed
        step would fight the loop."""
        self.assertEqual(reseed_step(None, 1.36, 3.03, q=2.8, present=True, opted_in=True)[1],
                         "gated")

    def test_does_not_fire_on_a_weak_fit(self):
        """spec_peak_ratio is a SHUFFLED-NULL significance; ~1.0 means the fold did not beat
        its own null. 88.7% of emissions are like this."""
        self.assertEqual(reseed_step(None, 1.36, 1.02, 0.2, True, True)[1], "gated")

    def test_refuses_a_fit_at_the_span_edge(self):
        """fit_spectrum_delay scans +-span_chips. At the edge the true offset is
        UNREPRESENTABLE, so the value says nothing about how far to go -- stepping by it would
        be acting on a rail. This is the failure mode that has to be refused, not clamped."""
        s, why = reseed_step(None, tau=1.98, ratio=3.0, q=0.2, present=True, opted_in=True)
        self.assertIsNone(s)
        self.assertEqual(why, "span-edge")

    def test_respects_the_49_gate_and_the_opt_in(self):
        self.assertEqual(reseed_step(None, 1.36, 3.03, 0.2, present=False, opted_in=True)[1],
                         "gated")
        self.assertEqual(reseed_step(None, 1.36, 3.03, 0.2, present=True, opted_in=False)[1],
                         "gated")

    def test_step_is_a_FRACTION_because_the_magnitude_is_unproven(self):
        """The direction is validated and the size is not. A gain of 1.0 would bet the whole
        correction on a number whose far-regime correlation is only -0.375."""
        s, _ = reseed_step(None, 1.0, 3.0, 0.2, True, True, gain=0.5)
        self.assertLess(abs(s), 1.0)

    def test_step_is_capped(self):
        s, _ = reseed_step(None, 1.9, 3.0, 0.2, True, True, gain=1.0, span=4.0)
        self.assertAlmostEqual(abs(s), 0.75, places=6)

    def test_converges_over_repeated_opportunities(self):
        """The design claim: fractional steps close a real offset over the ~5-9 strong fits
        available per excursion, without any single step being trusted."""
        err = 1.4
        for _ in range(9):
            s, why = reseed_step(None, err, 3.0, 0.2, True, True)
            if why != "applied":
                break
            err -= s
        self.assertLess(abs(err), 0.1)

    def test_gating_and_firing_differ_on_the_same_satellite(self):
        """Axis test: identical satellite, only q moved -- the rule must change its answer, or
        it is not reading q at all."""
        far = reseed_step(None, 1.36, 3.03, q=0.19, present=True, opted_in=True)[1]
        near = reseed_step(None, 1.36, 3.03, q=2.80, present=True, opted_in=True)[1]
        self.assertNotEqual(far, near)


if __name__ == "__main__":
    unittest.main(verbosity=2)
