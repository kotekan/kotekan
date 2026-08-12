#!/usr/bin/env python3
"""Warm start + corroborated escape (task #33 / #39).

Two failures, one week apart, from the same guard:

  2026-08-10  ONE satellite (PRN 25, transited out of the beam, code phase uniform over
              10230 chips) produced a self-consistent run of nonsense and talked the filter
              into a -1396 chip step. The fix was escape_max_step, a flat ceiling.
  2026-08-11  That ceiling (100) sat BELOW the receiver clock it guards (~151), so a
              cold-started state rejected 100% of updates forever: 34 rejections agreeing
              to 0.22 chips, all implying +151.7, all refused as "a wrap alias".

The ceiling was not wrong in kind -- the clock IS bounded -- it was unbounded evidence
against a bounded prior. So: warm-start from the legacy clock (the cold case never arises),
and let extraordinary evidence lift the ceiling, where the evidence is counted in
SATELLITES. Independent satellites share exactly one term, the receiver clock, so agreement
across them is evidence about the clock; a satellite agreeing with itself is not.

Run: python3 test_joint_bootstrap.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.state_filter import JointReceiverState   # noqa: E402

L = 10230.0
TRUE_CLK = 151.7          # the sky, as measured 2026-08-11
BAND = "1176.45MHz"


def feed(js, sats, clk, t0=0.0, n=12, dt=2.0, noise=None):
    """n cycles of `sats` all reporting y = clk + their own bias."""
    for c in range(n):
        t = t0 + c * dt
        mm = [((s, p), (clk + (noise(s, p, c) if noise else 0.0)) % L, 0.3, BAND)
              for s, p in sats]
        js.cycle(mm, t)
    return js


class TestWarmStart(unittest.TestCase):

    def test_a_converged_state_at_the_WRONG_value_deadlocks(self):
        """THE REAL MECHANISM, and it is not the cold start I first blamed.

        A cold state does NOT deadlock: sigma_clk0 is 200 chips, so a 151-chip innovation
        is 0.76 sigma and sails through the 6-sigma gate. The deadlock needs P to be SMALL
        already -- a state that has converged somewhere wrong. Then correct measurements
        arrive far outside a now-tight gate, every one is rejected, and the escape that
        exists to rescue exactly this is refused because 151 > escape_max_step (100).

        That is the on-sky failure of 2026-08-11 15:25 and it is a POISONED STATE (#39),
        not a bootstrap problem. Pinned here with escape_min_sats disabled so the old
        behaviour stays reproducible.
        """
        js = JointReceiverState(code_len=L, ref_band=BAND, escape_min_sats=99)
        sats = [("G", p) for p in (10, 20, 23, 27)]
        feed(js, sats, 0.0, n=25)                       # converge at the WRONG value
        self.assertLess(js.sigma(), 5.0, "state did not converge tightly enough to deadlock")
        rej0 = js.rejected
        feed(js, sats, TRUE_CLK, t0=200.0, n=25)        # now tell it the truth
        self.assertLess(abs(js.clk), 20.0,
                        "expected DEADLOCK at the wrong value, got clk %.1f" % js.clk)
        self.assertGreater(js.rejected - rej0, 40, "expected mass rejection")

    def test_corroboration_breaks_the_deadlock(self):
        """...and the evidence rule is what gets it out. Same setup, quorum enabled: four
        satellites agreeing on the same +151.7 step is a clock event, so it is followed."""
        js = JointReceiverState(code_len=L, ref_band=BAND, escape_min_sats=3)
        sats = [("G", p) for p in (10, 20, 23, 27)]
        feed(js, sats, 0.0, n=25)
        feed(js, sats, TRUE_CLK, t0=200.0, n=40)
        self.assertAlmostEqual(js.clk, TRUE_CLK, None,
                               "deadlock not broken (clk %.1f)" % js.clk, 25.0)

    def test_warm_start_converges(self):
        """The fix: handed the legacy clock, the same feed converges."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=148.0)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK)
        self.assertAlmostEqual(js.clk, TRUE_CLK, delta=1.0)
        self.assertEqual(js.rejected, 0)

    def test_warm_start_is_only_a_prior(self):
        """A warm start that is WRONG must be overridden by the sky, not believed."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=120.0)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK, n=40)
        self.assertAlmostEqual(js.clk, TRUE_CLK, delta=1.0)

    def test_default_is_unchanged(self):
        self.assertEqual(JointReceiverState(code_len=L).clk, 0.0)


class TestCorroboratedEscape(unittest.TestCase):
    """The evidence rule. A step past escape_max_step needs distinct satellites."""

    def _settled(self, **kw):
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=TRUE_CLK, **kw)
        feed(js, [("G", p) for p in (10, 20, 23, 27)], TRUE_CLK, n=10)
        return js

    def test_one_satellite_cannot_move_the_clock(self):
        """2026-08-10 REGRESSION TEST. One sat, perfectly self-consistent, asking for a
        huge step. It agrees with itself and that must not be enough."""
        js = self._settled()
        clk_before = js.clk
        for c in range(20):
            js.cycle([(("G", 25), (TRUE_CLK - 1396.0) % L, 0.3, BAND)], 100.0 + 2.0 * c)
        self.assertAlmostEqual(js.clk, clk_before, None,
                               "a single satellite moved the clock by %.1f chips"
                               % (js.clk - clk_before), 5.0)

    def test_a_quorum_is_believed(self):
        """The deadlock escape this hatch exists for: a REAL re-anchor, seen by the whole
        fleet, must still be followed."""
        js = self._settled()
        new = TRUE_CLK + 400.0
        for c in range(30):
            js.cycle([(("G", p), new % L, 0.3, BAND) for p in (10, 20, 23, 27)],
                     100.0 + 2.0 * c)
        self.assertAlmostEqual(js.clk, new, None,
                               "fleet-wide re-anchor was not followed (clk %.1f)" % js.clk,
                               25.0)

    def test_quorum_must_agree_on_the_SAME_step(self):
        """Three satellites all shouting, each a different number, is not corroboration."""
        js = self._settled()
        clk_before = js.clk
        for c in range(20):
            js.cycle([(("G", 10), (TRUE_CLK + 900.0) % L, 0.3, BAND),
                      (("G", 20), (TRUE_CLK - 1200.0) % L, 0.3, BAND),
                      (("G", 23), (TRUE_CLK + 3000.0) % L, 0.3, BAND)], 100.0 + 2.0 * c)
        self.assertAlmostEqual(js.clk, clk_before, None,
                               "disagreeing satellites were treated as corroboration", 30.0)

    def test_small_steps_still_escape_on_one_sat(self):
        """Below the ceiling nothing changes -- the corroboration rule must not make the
        filter MORE brittle for the ordinary case it already handled."""
        js = self._settled()
        for c in range(25):
            js.cycle([(("G", 10), (TRUE_CLK + 60.0) % L, 0.3, BAND)], 100.0 + 2.0 * c)
        self.assertGreater(js.bias(("G", 10)) + js.clk, TRUE_CLK + 20.0,
                           "a modest single-sat step no longer escapes")

    def test_noise_does_not_corroborate_itself(self):
        """Several satellites each feeding INCOHERENT noise: no run passes escape_spread,
        so none of them gets a vote."""
        js = self._settled()
        clk_before = js.clk
        rng = [3000.0, -4200.0, 1500.0, -900.0, 4800.0, -2600.0]
        for c in range(24):
            js.cycle([(("G", p), (TRUE_CLK + rng[(c + i) % len(rng)]) % L, 0.3, BAND)
                      for i, p in enumerate((10, 20, 23))], 100.0 + 2.0 * c)
        self.assertAlmostEqual(js.clk, clk_before, None,
                               "incoherent runs were believed (clk moved %.1f)"
                               % (js.clk - clk_before), 30.0)

    def test_corroboration_is_counted_in_satellites_not_samples(self):
        """The independence point. ONE satellite reporting 100 times is one satellite."""
        js = self._settled()
        clk_before = js.clk
        for c in range(100):
            js.cycle([(("G", 25), (TRUE_CLK + 800.0) % L, 0.3, BAND)], 100.0 + 2.0 * c)
        self.assertAlmostEqual(js.clk, clk_before, None,
                               "sample count substituted for satellite count", 30.0)


class TestHousekeeping(unittest.TestCase):

    def test_rej_band_does_not_leak(self):
        """_rej_band is a new dict keyed like _rej_hist; it must be cleaned up at BOTH
        pop sites or a long run accumulates dead keys forever."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=TRUE_CLK, max_age_s=600.0)
        feed(js, [("G", 10)], TRUE_CLK, n=4)   # born + converged first: birth is a
        for c in range(6):                     # separate path and never records a run
            js.cycle([(("G", 10), (TRUE_CLK + 4000.0) % L, 0.3, BAND)], 10.0 + 2.0 * c)
        self.assertTrue(js._rej_band, "expected a rejected run to record its band")
        feed(js, [("G", 10)], TRUE_CLK, t0=200.0, n=8)     # accepted again -> cleared
        self.assertNotIn(("G", 10), js._rej_band, "_rej_band leaked past an accepted update")

    def test_expiry_clears_rej_band(self):
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=TRUE_CLK, max_age_s=5.0)
        feed(js, [("G", 10), ("G", 20)], TRUE_CLK, n=4)
        for c in range(6):
            js.cycle([(("G", 10), (TRUE_CLK + 4000.0) % L, 0.3, BAND)], 100.0 + 2.0 * c)
        js.cycle([(("G", 20), TRUE_CLK % L, 0.3, BAND)], 100000.0)   # ages everything out
        self.assertNotIn(("G", 10), js._rej_band, "_rej_band survived expiry of its key")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestZombieGuards(unittest.TestCase):
    """The 2026-08-12 00:08 zombie, pinned (task #33 / #45 step 2).

    On sky: the filter was born during the #42 establishment-garbage window, absorbed
    biases of +-20-46 chips (unphysical; real ones measure <= ~6) as its geometry, then a
    flood of self-consistent measurements crushed P[0,0] to ~1e-7 -- sigma printed 0.000,
    gain ~0, rej=0 for 10+ minutes. A zero-gain state can neither MOVE nor REJECT, so no
    escape hatch could ever fire. Three guards, one per layer: the P floor keeps the
    filter capable; birth_max 50 -> 12 refuses zombie-scale births once the clock is
    known; the broker-side feed warmup (--joint-feed-warmup-s, not unit-testable here)
    keeps establishment garbage from becoming birth geometry at all.
    """

    ZB = {26: -46.05, 9: +32.23, 6: +23.85, 11: +20.27, 4: -14.52}   # the 00:08 biases
    BT = {26: -3.5, 9: +2.8, 6: +1.6, 11: -1.4, 4: +0.4}             # plausible truth

    def test_p_floor_prevents_zero_gain(self):
        """A flood of self-agreeing measurements must not buy sigma below the floor."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=150.0)
        feed(js, [("G", p) for p in self.ZB], 150.5, n=400, dt=1.0)
        self.assertGreaterEqual(js.sigma(), 0.02,
                                "sigma %.4f: the P floor failed to hold" % js.sigma())

    def test_zombie_geometry_recovers_when_truth_returns(self):
        """The capability the zombie lacked. Converge INTO the zombie geometry (garbage
        clock + garbage biases, self-consistent), then let the y-feed return to truth --
        the guarded filter must reject, escape, and recover. The on-sky zombie, at
        P ~ 1e-7, could do none of those for 10+ minutes."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=150.0)
        sats = [("G", p) for p in self.ZB]
        feed(js, sats, 163.9, n=100, dt=1.0,
             noise=lambda s, p, c: self.ZB[p])
        self.assertGreater(abs(js.wrap(js.clk - 150.5)), 5.0,
                           "fixture failed to poison the state (clk %.1f)" % js.clk)
        rej0 = js.rejected
        feed(js, sats, 150.5, t0=500.0, n=120, dt=1.0,
             noise=lambda s, p, c: self.BT[p])
        self.assertGreater(js.rejected - rej0, 5,
                           "a guarded filter must be ABLE to reject (got %d)"
                           % (js.rejected - rej0))
        self.assertLess(abs(js.wrap(js.clk - 150.5)), 3.0,
                        "did not recover: clk %.1f vs truth 150.5" % js.clk)

    def test_birth_clamp_refuses_zombie_scale_biases(self):
        """Once the clock is determined, a +46-chip newborn is refused (birth_max 12)
        and a +5.7 one -- the largest healthy bias ever measured -- is admitted."""
        js = JointReceiverState(code_len=L, ref_band=BAND, clk0=150.0)
        feed(js, [("G", p) for p in (10, 20, 23)], 150.5, n=25)
        rej0 = js.rejected
        r = js.update(("G", 99), (150.5 + 46.0) % L, 0.3, 600.0, band=BAND)
        self.assertIsNone(r, "a zombie-scale birth was admitted")
        self.assertEqual(js.rejected, rej0 + 1)
        self.assertNotIn(("G", 99), js._idx)
        r2 = js.update(("G", 28), (150.5 + 5.7) % L, 0.3, 602.0, band=BAND)
        self.assertIn(("G", 28), js._idx, "a legitimate large bias was refused at birth")
