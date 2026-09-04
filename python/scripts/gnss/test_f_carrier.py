#!/usr/bin/env python3
"""f_carrier, the receiver-wide carrier-frequency state (task #33 P2).

The properties that matter are the ones whose failures are SILENT: a state that
aliases onto a satellite bias row after a prune, a state that collects the wrong
process noise, and an unmeasured state that reads as a confident zero.

Run: python3 test_f_carrier.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.state_filter import JointReceiverState   # noqa: E402


def fresh(**kw):
    return JointReceiverState(code_len=10230.0, **kw)


class TestFCarrier(unittest.TestCase):

    def test_absent_until_measured_and_not_a_confident_zero(self):
        f = fresh()
        self.assertEqual(f.x.size, 2, "the row must not exist before a measurement")
        self.assertEqual(f.f_carrier(), 0.0)
        self.assertEqual(f.f_carrier_sigma(), float("inf"),
                         "an unmeasured state reporting a finite sigma reads as healthy")

    def test_births_on_first_measurement_rather_than_walking_in(self):
        f = fresh()
        self.assertEqual(f.update_carrier(-3.5, 100.0, 0.5), 0.0)
        self.assertAlmostEqual(f.f_carrier(), -3.5, places=9,
                               msg="must SNAP to the first measurement")
        self.assertLess(f.f_carrier_sigma(), 1e6)

    def test_converges_and_tightens(self):
        f = fresh()
        for i in range(60):
            f.update_carrier(2.0, 100.0 + i, 0.5)
        self.assertAlmostEqual(f.f_carrier(), 2.0, delta=0.15)
        self.assertLess(f.f_carrier_sigma(), 0.5)

    def test_gate_rejects_a_wild_value(self):
        f = fresh()
        for i in range(40):
            f.update_carrier(1.0, 100.0 + i, 0.3)
        before = f.f_carrier()
        self.assertIsNone(f.update_carrier(900.0, 141.0, 0.3))
        self.assertAlmostEqual(f.f_carrier(), before, delta=0.05,
                               msg="a gated measurement must not move the state")
        self.assertEqual(f.fcar_rejected, 1)

    def test_does_not_collect_the_satellite_process_noise(self):
        """q_b is in CHIPS and ~1000x q_fcar. If the row is swept up with the sat biases,
        its variance grows at the wrong rate and the gate opens far too wide."""
        f = fresh(q_b=10.0, q_fcar=0.001)
        f.update_carrier(0.0, 0.0, 0.1)
        for i in range(30):
            f.update_carrier(0.0, float(i), 0.1)
        s0 = f.f_carrier_sigma()
        f.predict(1000.0)                      # 1000 s of pure coasting
        grown = f.f_carrier_sigma() ** 2 - s0 ** 2
        self.assertLess(grown, (0.001 ** 2) * 1000.0 * 4.0,
                        "f_carrier grew like a satellite bias, not like an LO")

    def test_survives_a_satellite_prune_without_aliasing(self):
        """THE SILENT ONE. Rows are renumbered when a stale sat is dropped; if _fcar_idx
        is not remapped it points at whatever slid into its slot -- a sat's bias, in chips."""
        f = fresh(max_age_s=50.0)
        # birth two sats through the public cycle path, then the carrier state
        f.cycle([(("G", 1), 10.0, 0.3, "l5"), (("G", 2), 11.0, 0.3, "l5")], 0.0)
        f.update_carrier(-4.0, 1.0, 0.2)
        for i in range(20):
            f.update_carrier(-4.0, 2.0 + i, 0.2)
        row_before, val_before = f._fcar_idx, f.f_carrier()
        self.assertAlmostEqual(val_before, -4.0, delta=0.3)
        # age BOTH sats out: their rows leave and everything after them shifts down
        f.cycle([], 5000.0)
        self.assertNotIn(("G", 1), f._idx)
        self.assertLess(f._fcar_idx, row_before, "rows did not actually shift; test is blind")
        self.assertAlmostEqual(f.f_carrier(), val_before, delta=0.5,
                               msg="f_carrier aliased onto another row after the prune")
        self.assertLess(f.f_carrier_sigma(), 5.0)

    def test_hz_do_not_wrap(self):
        """The code path wraps on the 10230-chip modulus. A frequency must not: 6000 Hz is
        6000 Hz, not 6000-10230."""
        f = fresh()
        f.update_carrier(0.0, 0.0, 0.5)
        f.innov_nsigma = 1e9                     # disable the gate for this one check
        for i in range(200):
            f.update_carrier(6000.0, float(i + 1), 0.5)
        self.assertGreater(f.f_carrier(), 1000.0,
                           "f_carrier wrapped like a code phase")

    def test_code_path_is_untouched_when_no_carrier_is_fed(self):
        """A chain that never feeds a carrier keeps its exact old state layout."""
        a = fresh()
        a.cycle([(("G", 1), 10.0, 0.3, "l5"), (("G", 2), 11.0, 0.3, "l5")], 0.0)
        a.cycle([(("G", 1), 10.2, 0.3, "l5"), (("G", 2), 11.1, 0.3, "l5")], 10.0)
        b = fresh()
        b.cycle([(("G", 1), 10.0, 0.3, "l5"), (("G", 2), 11.0, 0.3, "l5")], 0.0)
        b.cycle([(("G", 1), 10.2, 0.3, "l5"), (("G", 2), 11.1, 0.3, "l5")], 10.0)
        self.assertEqual(a.x.size, b.x.size)
        self.assertIsNone(a._fcar_idx)
        for i in range(a.x.size):
            self.assertAlmostEqual(a.x[i], b.x[i], places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
