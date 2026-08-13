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


class TestClosedLoopReference(unittest.TestCase):
    """THE BROKER'S REFERENCE RULE (--rrate-command). deep_rate_hz is measured on records
    the tracker already derotated by the commanded trim, so the broker must feed
    y = residual + command_applied -- the value referenced to the BASE seed. This class is
    the closed-loop demonstration of why: get the reference wrong and the loop is not
    noisy, it is WRONG BY HALF, at equilibrium, forever, with every gate green."""

    def _run_loop(self, add_back, n=300):
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        fcar = 0.5
        s = JointReceiverState(code_len=10230.0)
        cmd = {p: 0.0 for p in truth}
        resid = {}
        for k in range(n):
            for prn, rr in truth.items():
                y_true = y_of(rr, fcar, F_E5A)
                resid[prn] = y_true - cmd[prn]              # what deep_rate_hz reports
                y_fed = resid[prn] + (cmd[prn] if add_back else 0.0)
                s.update_rrate(prn, y_fed, 100.0 + k, F_E5A)
            s.gauge_rrate()
            for prn in truth:                                # next poll's command
                cmd[prn] = s.carrier_correction_hz(prn, F_E5A)
        return truth, fcar, resid

    def test_adding_the_command_back_closes_the_loop(self):
        truth, fcar, resid = self._run_loop(add_back=True)
        for prn in truth:
            self.assertLess(abs(resid[prn]), 0.05,
                            "PRN %d standing residual %.3f Hz" % (prn, resid[prn]))

    def test_feeding_the_bare_residual_parks_at_half(self):
        """The equilibrium is exact: feed y_true - cmd while commanding the prediction and
        the fixed point is cmd = y_true/2 -- a permanent 50%% residual that no gate flags
        (measurements accepted, sigma converged, commands stable). This is what the broker
        would do without rr_cmd_applied, and why that dict exists."""
        truth, fcar, resid = self._run_loop(add_back=False)
        for prn, rr in truth.items():
            y_true = y_of(rr, fcar, F_E5A)
            if abs(y_true) < 1.0:
                continue                                     # too small to discriminate
            self.assertGreater(abs(resid[prn]), 0.3 * abs(y_true),
                               "PRN %d: bare-residual feed should park near y/2, got "
                               "resid %.3f of y %.3f" % (prn, resid[prn], y_true))


class TestUnalias(unittest.TestCase):
    """The measured aliasing of deep_rate_hz (2026-08-13 trim probe): values live in
    (-4.77, +4.77] and out-of-window truth wraps by 9.537. These pin the two probe
    observations and the bound that keeps unwrapping from running away."""
    M = 9.537

    def setUp(self):
        from gnss_broker.fits import unalias
        self.unalias = unalias

    def test_the_probe_pair(self):
        # PRN 5: predicted -7.9 after the +5 Hz step, read +1.4 -- one wrap down restores
        self.assertAlmostEqual(self.unalias(1.4, -7.9, self.M), 1.4 - self.M, places=6)
        # PRN 15: predicted -6.0, read +3.5
        self.assertAlmostEqual(self.unalias(3.5, -6.0, self.M), 3.5 - self.M, places=6)

    def test_in_window_untouched(self):
        self.assertEqual(self.unalias(-3.1, -4.3, self.M), -3.1)

    def test_wrap_bound_holds(self):
        # a prediction 5 moduli away moves the sample by at most max_wraps
        self.assertAlmostEqual(self.unalias(0.0, 5 * self.M, self.M, max_wraps=2),
                               2 * self.M, places=6)

    def test_disabled_modulus_is_identity(self):
        self.assertEqual(self.unalias(3.5, -6.0, 0.0), 3.5)


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
