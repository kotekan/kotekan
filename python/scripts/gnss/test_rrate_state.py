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


class TestClosedLoopLag(unittest.TestCase):
    """ARM 3's walk (2026-08-13 15:26): the measured residual reflects the command posted
    LAG polls earlier, while the feed adds back the LATEST command -- y_fed carries a
    (cmd_now - cmd_then) misreference that is self-reinforcing at full slew. The command
    slew bound (--rrate-cmd-slew-hz) bounds the misreference and it decays to zero at the
    fixed point. This simulates exactly that loop."""

    def _run(self, slew, n=400, lag=2, meas_noise=0.0, seed=7):
        import random
        rng = random.Random(seed)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        fcar = 0.5
        s = JointReceiverState(code_len=10230.0)
        hist = {p: [0.0] * (lag + 1) for p in truth}   # [0] = latest posted
        for k in range(n):
            for prn, rr in truth.items():
                y_true = y_of(rr, fcar, F_E5A)
                resid = y_true - hist[prn][-1]          # measured under the OLD command
                y_fed = resid + hist[prn][0]            # broker adds back the LATEST
                if meas_noise:
                    y_fed += rng.gauss(0.0, meas_noise)
                s.update_rrate(prn, y_fed, 100.0 + k, F_E5A)
            s.gauge_rrate()
            for prn in truth:
                tgt = s.carrier_correction_hz(prn, F_E5A)
                step = tgt - hist[prn][0]
                if slew > 0.0:
                    step = max(-slew, min(slew, step))
                hist[prn] = [hist[prn][0] + step] + hist[prn][:lag]
        return {p: abs(y_of(truth[p], fcar, F_E5A) - hist[p][0]) for p in truth}

    def test_slewed_loop_converges_under_lag_and_noise(self):
        # The bound is set by the filter's own steady state (q_rr holds the row loose to
        # track drift), not by the loop: ~0.4 Hz at 0.3 Hz measurement noise. The claim
        # pinned here is BOUNDED AND SMALL vs the multi-Hz walk, not noiseless perfection.
        err = self._run(slew=0.5, meas_noise=0.3)
        for prn, e in err.items():
            self.assertLess(e, 0.8, "PRN %d standing error %.2f Hz" % (prn, e))

    def test_the_bound_is_what_does_it(self):
        """Same loop, no slew bound: with realistic measurement noise the misreference
        feeds back and the loop must end up WORSE than the bounded one by a clear margin
        -- this is the arm-3 walk in miniature. (Deterministically the unbounded loop can
        look fine; the instability needs noise to express, which is exactly why arm 3
        passed a desk-check and walked on sky.)"""
        bounded = max(self._run(slew=0.5, meas_noise=0.3).values())
        free = max(self._run(slew=0.0, meas_noise=0.3).values())
        self.assertGreater(free, 2.0 * bounded,
                           "free %.2f vs bounded %.2f Hz" % (free, bounded))


class TestFullBandFields(unittest.TestCase):
    """#40: the rrate feed must read the UNCAPPED rate fields. deep_rate_hz is the FOLD's
    pick, clamped to deep_rate_max_hz -- past the cap it degrades to the best in-cap noise
    bin, and closing a loop on that walked arm 1 (~1 Hz/min). rate_residuals therefore
    takes the field names; these pin that the selection actually selects."""

    def setUp(self):
        from gnss_broker.fits import rate_residuals
        self.rr = rate_residuals

    @staticmethod
    def _status(prn, capped, full, q=20.0, hop=1000):
        return {prn: {"deep_rate_hz": capped, "deep_rate_q": q,
                      "deep_rate_full_hz": full, "deep_rate_full_q": q,
                      "amp_snr": 50.0, "pow_hop": hop}}

    def _two_polls(self, **kw):
        """The continuity gate skips a PRN's first sighting, so drive two polls."""
        ph, pv = {}, {}
        self.rr(self._status(7, -4.7, -7.9, hop=1000), 10.0, 0.0,
                prev_hop=ph, prev_val=pv, **kw)
        out, _ = self.rr(self._status(7, -4.6, -7.8, hop=3048), 10.0, 0.0,
                         prev_hop=ph, prev_val=pv, **kw)
        return out

    def test_default_reads_the_capped_field(self):
        out = self._two_polls()
        self.assertAlmostEqual(out[7], -4.6)

    def test_full_fields_read_the_uncapped_value(self):
        out = self._two_polls(rate_field="deep_rate_full_hz", q_field="deep_rate_full_q")
        self.assertAlmostEqual(out[7], -7.8)


class TestAdrFineRate(unittest.TestCase):
    """#33 PLL fine observable: res_cycles differenced over adr_records. The gates are
    structural and each one exists because its absence is a known disease: an arc break
    means unobserved whole cycles (no measurement, not zero); a frozen counter must read
    ABSENT (a dead feed passing for healthy is the chord-served-cn0 class)."""
    R = 2048.0 / 195312.5

    def setUp(self):
        from gnss_broker.fits import adr_fine_rate
        self.f = adr_fine_rate

    @staticmethod
    def _row(arc, n, res):
        return {"adr_arc": arc, "adr_records": n, "res_cycles": res}

    def test_rate_and_count(self):
        out = self.f(self._row(3, 1900, 2.5), self._row(3, 0, 0.5), self.R)
        self.assertIsNotNone(out)
        rate, n = out
        self.assertAlmostEqual(rate, 2.0 / (1900 * self.R), places=9)
        self.assertEqual(n, 1900)

    def test_arc_break_is_no_measurement(self):
        self.assertIsNone(self.f(self._row(4, 100, 0.1), self._row(3, 50, 2.0), self.R))

    def test_frozen_counter_reads_absent_not_zero(self):
        self.assertIsNone(self.f(self._row(3, 100, 0.1), self._row(3, 100, 0.1), self.R))

    def test_missing_field_reads_absent(self):
        self.assertIsNone(self.f({"adr_arc": 3, "adr_records": 100},
                                 self._row(3, 0, 0.0), self.R))


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
