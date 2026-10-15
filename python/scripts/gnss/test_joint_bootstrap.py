#!/usr/bin/env python3
"""Warm start and corroborated escape in JointReceiverState.

escape_max_step, a flat ceiling on the step an escape may take, stops one satellite whose
rejected measurements agree with each other but sit far from the clock from talking the filter
into a huge step. But the receiver clock it guards can exceed the ceiling, and then a state
converged at the wrong value rejects every correct measurement.

The ceiling is right in kind (the clock is bounded) but must yield to enough evidence. So the
state warm-starts from a known clock (clk0), and corroboration lifts the ceiling, counted in
satellites: independent satellites share exactly one term, the receiver clock, so agreement
across them is evidence about the clock; a satellite agreeing with itself is not.

Run: python3 test_joint_bootstrap.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.state_filter import JointReceiverState  # noqa: E402

L = 10230.0
TRUE_CLK = 151.7  # a receiver clock above escape_max_step (100), chips


def feed(js, sats, clk, t0=0.0, n=12, dt=2.0, noise=None):
    """n cycles of `sats` all reporting y = clk + their own bias."""
    for c in range(n):
        t = t0 + c * dt
        mm = [
            ((s, p), (clk + (noise(s, p, c) if noise else 0.0)) % L, 0.3)
            for s, p in sats
        ]
        js.cycle(mm, t)
    return js


class TestWarmStart(unittest.TestCase):
    def test_a_converged_state_at_the_WRONG_value_deadlocks(self):
        """Without corroboration, a state converged at the wrong value deadlocks.

        A cold state does not: sigma_clk0 is 200 chips, so a 151-chip innovation is 0.76
        sigma and passes the 6-sigma gate. The deadlock needs P to be small already, a
        state converged somewhere wrong. Correct measurements then arrive far outside the
        tight gate, every one is rejected, and the escape that exists to rescue this is
        refused because 151 > escape_max_step (100). escape_min_sats is set out of reach so
        the deadlock stays reproducible.
        """
        js = JointReceiverState(code_len=L, escape_min_sats=99)
        sats = [("G", p) for p in (10, 20, 23, 27)]
        feed(js, sats, 0.0, n=25)  # converge at the wrong value
        self.assertLess(
            js.sigma(), 5.0, "state did not converge tightly enough to deadlock"
        )
        rej0 = js.rejected
        feed(js, sats, TRUE_CLK, t0=200.0, n=25)  # now tell it the truth
        self.assertLess(
            abs(js.clk),
            20.0,
            "expected DEADLOCK at the wrong value, got clk %.1f" % js.clk,
        )
        self.assertGreater(js.rejected - rej0, 40, "expected mass rejection")

    def test_corroboration_breaks_the_deadlock(self):
        """The evidence rule gets it out. Same setup with the quorum reachable: four
        satellites agreeing on the same +151.7 step is a clock event, so it is followed."""
        js = JointReceiverState(code_len=L, escape_min_sats=3)
        sats = [("G", p) for p in (10, 20, 23, 27)]
        feed(js, sats, 0.0, n=25)
        feed(js, sats, TRUE_CLK, t0=200.0, n=40)
        self.assertAlmostEqual(
            js.clk, TRUE_CLK, None, "deadlock not broken (clk %.1f)" % js.clk, 25.0
        )

    def test_warm_start_converges(self):
        """Handed a nearby clock as clk0, a fresh state converges with no rejections."""
        js = JointReceiverState(code_len=L, clk0=148.0)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK)
        self.assertAlmostEqual(js.clk, TRUE_CLK, delta=1.0)
        self.assertEqual(js.rejected, 0)

    def test_warm_start_is_only_a_prior(self):
        """A wrong warm start must be overridden by the sky, not believed."""
        js = JointReceiverState(code_len=L, clk0=120.0)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK, n=40)
        self.assertAlmostEqual(js.clk, TRUE_CLK, delta=1.0)

    def test_default_is_unchanged(self):
        self.assertEqual(JointReceiverState(code_len=L).clk, 0.0)


class TestCorroboratedEscape(unittest.TestCase):
    """The evidence rule. A step past escape_max_step needs distinct satellites."""

    def _settled(self, **kw):
        js = JointReceiverState(code_len=L, clk0=TRUE_CLK, **kw)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK, n=10)
        return js

    def test_one_satellite_cannot_move_the_clock(self):
        """One satellite, perfectly self-consistent, asking for a huge step: agreeing with
        itself must not be enough."""
        js = self._settled()
        clk_before = js.clk
        for c in range(20):
            js.cycle([(("G", 25), (TRUE_CLK - 1396.0) % L, 0.3)], 100.0 + 2.0 * c)
        self.assertAlmostEqual(
            js.clk,
            clk_before,
            None,
            "a single satellite moved the clock by %.1f chips" % (js.clk - clk_before),
            5.0,
        )

    def test_a_quorum_is_believed(self):
        """A real re-anchor, seen by the whole fleet, must still be followed."""
        js = self._settled()
        new = TRUE_CLK + 400.0
        for c in range(30):
            js.cycle(
                [(("G", p), new % L, 0.3) for p in (10, 20, 23, 27)], 100.0 + 2.0 * c,
            )
        self.assertAlmostEqual(
            js.clk,
            new,
            None,
            "fleet-wide re-anchor was not followed (clk %.1f)" % js.clk,
            25.0,
        )

    def test_quorum_must_agree_on_the_SAME_step(self):
        """Three satellites each implying a different step is not corroboration."""
        js = self._settled()
        clk_before = js.clk
        for c in range(20):
            js.cycle(
                [
                    (("G", 10), (TRUE_CLK + 900.0) % L, 0.3),
                    (("G", 20), (TRUE_CLK - 1200.0) % L, 0.3),
                    (("G", 23), (TRUE_CLK + 3000.0) % L, 0.3),
                ],
                100.0 + 2.0 * c,
            )
        self.assertAlmostEqual(
            js.clk,
            clk_before,
            None,
            "disagreeing satellites were treated as corroboration",
            30.0,
        )

    def test_small_steps_still_escape_on_one_sat(self):
        """Below escape_max_step one satellite's coherent run still escapes: the
        corroboration rule must not make the ordinary case more brittle."""
        js = self._settled()
        for c in range(25):
            js.cycle([(("G", 10), (TRUE_CLK + 60.0) % L, 0.3)], 100.0 + 2.0 * c)
        self.assertGreater(
            js.bias(("G", 10)) + js.clk,
            TRUE_CLK + 20.0,
            "a modest single-sat step no longer escapes",
        )

    def test_noise_does_not_corroborate_itself(self):
        """Several satellites each feeding incoherent noise: no run passes escape_spread,
        so none of them gets a vote."""
        js = self._settled()
        clk_before = js.clk
        rng = [3000.0, -4200.0, 1500.0, -900.0, 4800.0, -2600.0]
        for c in range(24):
            js.cycle(
                [
                    (("G", p), (TRUE_CLK + rng[(c + i) % len(rng)]) % L, 0.3)
                    for i, p in enumerate((10, 20, 23))
                ],
                100.0 + 2.0 * c,
            )
        self.assertAlmostEqual(
            js.clk,
            clk_before,
            None,
            "incoherent runs were believed (clk moved %.1f)" % (js.clk - clk_before),
            30.0,
        )

    def test_corroboration_is_counted_in_satellites_not_samples(self):
        """One satellite reporting 100 times is one satellite."""
        js = self._settled()
        clk_before = js.clk
        for c in range(100):
            js.cycle([(("G", 25), (TRUE_CLK + 800.0) % L, 0.3)], 100.0 + 2.0 * c)
        self.assertAlmostEqual(
            js.clk,
            clk_before,
            None,
            "sample count substituted for satellite count",
            30.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
