"""Task #33 P3: the per-satellite ORBITAL range-rate error, shared across bands.

WHAT THIS STATE IS, because the name matters and I got it wrong first. f_carrier is
RECEIVER-WIDE -- an LO/clock frequency error is one number for the whole receiver and cannot
vary satellite to satellite (KV, 2026-08-14). What varies per satellite is the ORBIT: a
range-rate model error, in m/s, which becomes Hz only when multiplied by a carrier.

    y_(i,b) [Hz] = -(f_b/c) * rrate_i  +  (f_b/f_ref) * f_carrier  +  noise

The two are degenerate WITHIN one band -- both scale with f_b -- and are separated only by the
same thing that separates clk from b_sat: one is shared across satellites and the other is not.

⚠️ THE STATE IS IN m/s ON PURPOSE. One satellite's E5a and E5b errors are not two numbers, they
are ONE range-rate error seen through two carriers 1.0261 apart. That is what makes cross-band
information ADD rather than be fitted twice, which is the entire point of the task: today
gal_e5a and gal_e5b fit this quantity independently and never compare.
"""
import math
import unittest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.state_filter import JointReceiverState

C = 299792458.0
F_E5A = 1176.45e6
F_E5B = 1207.14e6
F_B2A = 1176.45e6          # same carrier as E5a -- the CHORD case


def y_of(rrate, fcar, f_b, f_ref=F_E5A):
    """The forward model the filter inverts."""
    return -(f_b / C) * rrate + (f_b / f_ref) * fcar


def run(state, truth, fcar, bands=(F_E5A,), n=400, t0=100.0, sigma=0.2):
    for k in range(n):
        for prn, rr in truth.items():
            for fb in bands:
                state.update_rrate(prn, y_of(rr, fcar, fb), t0 + k, fb, sigma_hz=sigma)
        state.gauge_rrate()
    return state


class TestSeparatesOrbitFromReceiver(unittest.TestCase):

    def test_recovers_both_terms(self):
        """The load-bearing case: a receiver-wide offset AND per-satellite orbit errors,
        recovered separately from measurements that only ever see their sum."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}          # mean 0, matching the gauge
        run(s, truth, fcar=0.5)
        self.assertAlmostEqual(s.f_carrier(), 0.5, delta=0.05)
        for prn, rr in truth.items():
            self.assertAlmostEqual(s.rrate(prn), rr, delta=0.05)
        self.assertEqual(s.rrate_rejected, 0)

    def test_f_carrier_stays_receiver_wide(self):
        """⚠️ THE PHYSICS KV INSISTED ON. A per-satellite orbit error must NOT be absorbed
        into f_carrier beyond the gauge's structural 1/N -- an LO error is one number for the
        whole receiver. Move ONE satellite by 6 m/s and f_carrier may shift by at most its
        share of the fleet mean, not by the whole thing."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: 0.0, 21: 0.0, 28: 0.0, 30: 0.0, 32: 0.0, 34: 0.0}
        run(s, truth, fcar=0.0, n=300)
        base = s.f_carrier()
        truth[11] = 6.0                                # one orbit goes wrong
        run(s, truth, fcar=0.0, n=300, t0=500.0)
        moved_hz = abs(s.f_carrier() - base)
        # 6 m/s over 6 sats = 1 m/s of fleet mean = (f/c)*1 = 3.92 Hz IF it leaked fully.
        # The gauge's 1/N share is that same 3.92 Hz -- so the test is that it is NOT MORE,
        # i.e. the per-sat term took the rest.
        self.assertLess(moved_hz, 1.1 * (F_E5A / C) * (6.0 / len(truth)))
        self.assertAlmostEqual(s.rrate(11) - sum(s.rrate(p) for p in truth) / len(truth),
                               6.0 - 6.0 / len(truth), delta=0.2)


class TestCrossBandIsOneRow(unittest.TestCase):

    def test_two_bands_land_on_the_same_satellite_row(self):
        """THE POINT OF THE TASK. E5a and E5b measurements of one satellite are one quantity
        seen through two carriers; they must reinforce a single row, not create two."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        run(s, truth, fcar=0.5, bands=(F_E5A, F_E5B))
        self.assertEqual(len(s._rr_idx), 3, "a second band created extra rows")
        for prn, rr in truth.items():
            self.assertAlmostEqual(s.rrate(prn), rr, delta=0.05)

    def test_a_band_only_seen_once_still_gets_the_scaling_right(self):
        """E5b barely detects anything on sky, so its satellites must inherit the shared
        solution and be commanded at THEIR carrier, not E5a's."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        run(s, truth, fcar=0.5, bands=(F_E5A,))
        got_a = s.carrier_correction_hz(11, F_E5A)
        got_b = s.carrier_correction_hz(11, F_E5B)
        self.assertAlmostEqual(got_a, y_of(truth[11], 0.5, F_E5A), delta=0.05)
        self.assertAlmostEqual(got_b, y_of(truth[11], 0.5, F_E5B), delta=0.05)
        # and the two differ by exactly the carrier ratio, which is the whole reason the
        # state is held in m/s
        self.assertAlmostEqual(got_b / got_a, F_E5B / F_E5A, places=6)

    def test_same_carrier_different_constellation_pools(self):
        """On CHORD, GPS L5 / E5a / B2a all sit at 1176.45 MHz. Different constellations at
        the SAME carrier must feed one f_carrier -- that is what makes the receiver-wide term
        well determined instead of one constellation's private median."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -2.0, 21: 2.0}
        run(s, truth, fcar=0.7, bands=(F_E5A, F_B2A))
        self.assertAlmostEqual(s.f_carrier(), 0.7, delta=0.05)


class TestTheBugsFoundWhileBuilding(unittest.TestCase):

    def test_a_stiff_gauge_is_NOT_what_froze_it(self):
        """⚠️ A CORRECTION TO MY OWN DIAGNOSIS, kept as a test so it stays corrected.

        The filter froze while being built -- one satellite rejected 298 times out of 300,
        innov 17.6 Hz against a 3.1 Hz bar -- and I blamed a gauge 100x stiffer than the
        prior, loosening it in the same edit that fixed the birth path and added the escape.
        Three changes, one attributed cause. With birth and escape correct, a 0.02 gauge
        converges to ~2e-11: the stiffness was never the problem.

        The value shipped is moderate on its own merits, not as a repair."""
        for gs in (0.02, 0.5, 2.0):
            s = JointReceiverState(code_len=10230.0, rr_gauge_sigma=gs)
            truth = {11: -3.0, 21: 2.0, 28: 1.0}
            run(s, truth, fcar=0.5, n=200)
            err = max(abs(s.rrate(p) - r) for p, r in truth.items())
            self.assertLess(err, 0.1, "gauge_sigma %.2f: max error %.3f m/s" % (gs, err))

    def test_the_first_satellite_can_still_disagree_with_itself(self):
        """The first measurement defines f_carrier because one sample cannot split the two.
        If that satellite gets no row of its own it is structurally unable to disagree with
        the value it defined, and is rejected forever after."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        run(s, truth, fcar=0.5)
        self.assertIn(11, s._rr_idx)
        self.assertAlmostEqual(s.rrate(11), -3.0, delta=0.05)

    def test_a_bad_birth_escapes_instead_of_being_permanent(self):
        """A gate with no way out turns one wrong birth into a permanent rejection."""
        s = JointReceiverState(code_len=10230.0)
        run(s, {11: 0.0, 21: 0.0}, fcar=0.0, n=200)
        i = s._rr_idx[11]
        s.x[i] = 40.0                       # displace the row far outside its own gate
        s.P[i, i] = 1e-6                    # ...and make the filter certain about it
        run(s, {11: 0.0, 21: 0.0}, fcar=0.0, n=200, t0=400.0)
        self.assertLess(abs(s.rrate(11)), 1.0,
                        "a displaced, over-confident row never recovered (rrate %.2f)"
                        % s.rrate(11))


class TestUnmeasuredReadsAsUnknown(unittest.TestCase):

    def test_sigma_is_inf_before_any_measurement(self):
        """An unmeasured state must not read as a confident zero -- that is how a dead feed
        passes for a healthy one."""
        s = JointReceiverState(code_len=10230.0)
        self.assertEqual(s.rrate_sigma(11), float("inf"))
        self.assertEqual(s.rrate(11), 0.0)
        self.assertEqual(s.carrier_correction_hz(11, F_E5A), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
