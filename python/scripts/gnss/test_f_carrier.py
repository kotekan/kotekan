#!/usr/bin/env python3
"""f_carrier, the receiver-wide carrier-frequency state of JointReceiverState.

The properties that matter are the ones whose failures are silent: a state that
aliases onto a satellite bias row after a prune, a state that collects the wrong
process noise, and an unmeasured state that reads as a confident zero.

Run: python3 test_f_carrier.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.state_filter import JointReceiverState  # noqa: E402


F_L5 = 1176.45e6


def fresh(**kw):
    return JointReceiverState(code_len=10230.0, **kw)


class TestFCarrier(unittest.TestCase):
    def test_absent_until_measured_and_not_a_confident_zero(self):
        f = fresh()
        self.assertEqual(f.x.size, 2, "the row must not exist before a measurement")
        self.assertEqual(f.f_carrier(), 0.0)
        self.assertEqual(
            f.f_carrier_sigma(),
            float("inf"),
            "an unmeasured state reporting a finite sigma reads as healthy",
        )

    def test_does_not_collect_the_satellite_process_noise(self):
        """q_b is in chips and, in this fixture, 10^4 x q_fcar. If the row is swept up with the
        sat biases, its variance grows at the wrong rate and the gate opens far too wide."""
        f = fresh(q_b=10.0, q_fcar=0.001)
        for i in range(31):
            f.update_rrate(("G", 1), 0.0, float(i), F_L5, 0.1)
        s0 = f.f_carrier_sigma()
        f.predict(1000.0)  # ~1000 s of pure coasting
        grown = f.f_carrier_sigma() ** 2 - s0 ** 2
        self.assertLess(
            grown,
            (0.001 ** 2) * 1000.0 * 4.0,
            "f_carrier grew like a satellite bias, not like an LO",
        )

    def test_survives_a_satellite_prune_without_aliasing(self):
        """Rows are renumbered when a stale sat is dropped; if _fcar_idx is not remapped it
        points at whatever slid into its slot -- a sat's bias, in chips."""
        f = fresh(max_age_s=50.0)
        # birth two sats through the public cycle path, then the carrier state
        f.cycle([(("G", 1), 10.0, 0.3), (("G", 2), 11.0, 0.3)], 0.0)
        for i in range(21):
            f.update_rrate(("G", 3), -4.0, 1.0 + i, F_L5, 0.2)
        row_before, val_before = f._fcar_idx, f.f_carrier()
        self.assertAlmostEqual(val_before, -4.0, delta=0.3)
        # age BOTH sats out: their rows leave and everything after them shifts down
        f.cycle([], 5000.0)
        self.assertNotIn(("G", 1), f._idx)
        self.assertLess(
            f._fcar_idx, row_before, "rows did not actually shift; test is blind"
        )
        self.assertAlmostEqual(
            f.f_carrier(),
            val_before,
            delta=0.5,
            msg="f_carrier aliased onto another row after the prune",
        )

    def test_no_carrier_row_without_a_carrier_feed(self):
        """A chain that never feeds a carrier gets no carrier row: the state keeps the
        code-only layout of clk, clk_rate and one b_sat per satellite."""
        f = fresh()
        f.cycle([(("G", 1), 10.0, 0.3), (("G", 2), 11.0, 0.3)], 0.0)
        f.cycle([(("G", 1), 10.2, 0.3), (("G", 2), 11.1, 0.3)], 10.0)
        self.assertIsNone(f._fcar_idx)
        self.assertEqual(f.x.size, 2 + 2, "a code-only feed grew a row it never fed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
