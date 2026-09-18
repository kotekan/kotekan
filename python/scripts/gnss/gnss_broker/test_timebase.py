"""The time-base detector: an epoch shift is a SPREAD ∝ Doppler rate; a clock bias is an OFFSET.

    python3 -m unittest gnss_broker.test_timebase
"""
import random
import unittest

from gnss_broker import timebase


def sky(n, seed=1):
    """n satellites with realistic Doppler rates (Hz/s) at L5."""
    rng = random.Random(seed)
    return [rng.uniform(-0.6, 0.6) for _ in range(n)]


def pairs_for(rates, dt_s=0.0, bias_hz=0.0, noise_hz=2.0, seed=2):
    rng = random.Random(seed)
    return {i: (bias_hz + a * dt_s + rng.gauss(0.0, noise_hz), a) for i, a in enumerate(rates)}


class TimeBaseTests(unittest.TestCase):
    def setUp(self):
        timebase.VERDICT.clear()
        self.det = timebase.TimeBaseDetector()

    def test_clock_bias_is_not_an_epoch_error(self):
        rates = sky(8)
        for k in range(10):
            msg = self.det.note(1000.0 + k, pairs_for(rates, bias_hz=180.0), chain="gps_l5")
            self.assertIsNone(msg)
        self.assertFalse(timebase.VERDICT.suspect, "a 180 Hz common offset is the receiver clock")

    def test_epoch_shift_is_found_and_measured(self):
        rates = sky(8)
        msgs = []
        for k in range(5):
            m = self.det.note(1000.0 + k, pairs_for(rates, dt_s=84350.0, bias_hz=40.0), chain="gps_l5")
            if m:
                msgs.append((k, m))
        self.assertEqual(len(msgs), 1, "announced exactly once")
        self.assertEqual(msgs[0][0], self.det.persist - 1, "after `persist` consecutive cycles, not before")
        v = timebase.VERDICT
        self.assertTrue(v.suspect and v.explained)
        self.assertAlmostEqual(v.dt_s / 84350.0, 1.0, delta=0.02, msg="dt recovered to 2%%: %s" % v.dt_s)
        self.assertIn("STALE", msgs[0][1])
        self.assertIn("+23.43 h", msgs[0][1])

    def test_small_shift_below_threshold_is_quiet(self):
        rates = sky(8)
        for k in range(6):
            self.det.note(1000.0 + k, pairs_for(rates, dt_s=60.0), chain="gps_l5")
        self.assertFalse(timebase.VERDICT.suspect, "a minute of error is ~25 Hz of spread: below the bar")

    def test_unexplained_spread_is_a_different_message(self):
        rates = sky(8)
        rng = random.Random(7)
        msgs = []
        for k in range(5):
            pairs = {i: (rng.gauss(0.0, 400.0), a) for i, a in enumerate(rates)}
            m = self.det.note(1000.0 + k, pairs, chain="gps_l5")
            if m:
                msgs.append(m)
        self.assertTrue(timebase.VERDICT.suspect)
        self.assertFalse(timebase.VERDICT.explained)
        self.assertIsNone(timebase.VERDICT.dt_s)
        self.assertEqual(len(msgs), 1)
        self.assertIn("does NOT explain", msgs[0])

    def test_clears_after_quiet_cycles_and_says_so(self):
        rates = sky(8)
        for k in range(4):
            self.det.note(1000.0 + k, pairs_for(rates, dt_s=3600.0), chain="gps_l5")
        self.assertTrue(timebase.VERDICT.suspect)
        m1 = self.det.note(1010.0, pairs_for(rates), chain="gps_l5")
        self.assertIsNone(m1, "one quiet cycle is not a clear")
        self.assertTrue(timebase.VERDICT.suspect)
        m2 = self.det.note(1011.0, pairs_for(rates), chain="gps_l5")
        self.assertIn("TIME BASE CLEAR", m2)
        self.assertFalse(timebase.VERDICT.suspect)

    def test_too_few_satellites_is_not_evidence(self):
        rates = sky(3)
        for k in range(6):
            self.assertIsNone(self.det.note(1000.0 + k, pairs_for(rates, dt_s=84350.0), chain="gps_l5"))
        self.assertFalse(timebase.VERDICT.suspect, "3 satellites cannot separate a shift from scatter")

    def test_verdict_is_shared_across_chains(self):
        rates = sky(8)
        for k in range(4):
            self.det.note(1000.0 + k, pairs_for(rates, dt_s=84350.0), chain="gps_l5")
        other = timebase.TimeBaseDetector()
        self.assertTrue(other.verdict.suspect, "every chain reads the one telescope-wide verdict")


if __name__ == "__main__":
    unittest.main(verbosity=2)
