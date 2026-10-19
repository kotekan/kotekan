"""The per-satellite orbital range-rate error (rrate), shared across bands.

f_carrier is receiver-wide: an LO/clock frequency error is one number for the whole receiver and
cannot vary from satellite to satellite. What varies per satellite is the orbit: a range-rate
model error, in m/s, which becomes Hz only when multiplied by a carrier.

    y_(i,b) [Hz] = -(f_b/c) * rrate_i  +  (f_b/f_ref) * f_carrier  +  noise

The two are degenerate within one band (both scale with f_b) and are separated the same way clk
is separated from b_sat: one is shared across satellites and the other is not.

The state is in m/s so that one satellite's E5a and E5b errors are one range-rate error seen
through two carriers 1.0261 apart: cross-band information adds instead of being fitted twice.
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
F_B2A = 1176.45e6  # same carrier as E5a -- the CHORD case


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
        """A receiver-wide offset and per-satellite orbit errors, recovered separately from
        measurements that only see their sum."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}  # mean 0, matching the gauge
        run(s, truth, fcar=0.5)
        self.assertAlmostEqual(s.f_carrier(), 0.5, delta=0.05)
        for prn, rr in truth.items():
            self.assertAlmostEqual(s.rrate(prn), rr, delta=0.05)
        self.assertEqual(s.rrate_rejected, 0)

    def test_f_carrier_stays_receiver_wide(self):
        """A per-satellite orbit error must not leak into f_carrier beyond the gauge's
        structural 1/N share: an LO error is one number for the whole receiver. Move one
        satellite by 6 m/s and f_carrier may shift by at most its share of the fleet mean."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: 0.0, 21: 0.0, 28: 0.0, 30: 0.0, 32: 0.0, 34: 0.0}
        run(s, truth, fcar=0.0, n=300)
        base = s.f_carrier()
        truth[11] = 6.0  # one orbit goes wrong
        run(s, truth, fcar=0.0, n=300, t0=500.0)
        moved_hz = abs(s.f_carrier() - base)
        # 6 m/s over 6 satellites moves the fleet mean by 1 m/s = (f/c)*1 = 3.92 Hz, the
        # gauge's 1/N share. f_carrier may move by that (+10%) and no more: the per-satellite
        # row takes the rest.
        self.assertLess(moved_hz, 1.1 * (F_E5A / C) * (6.0 / len(truth)))
        self.assertAlmostEqual(
            s.rrate(11) - sum(s.rrate(p) for p in truth) / len(truth),
            6.0 - 6.0 / len(truth),
            delta=0.2,
        )


class TestCrossBandIsOneRow(unittest.TestCase):
    def test_two_bands_land_on_the_same_satellite_row(self):
        """E5a and E5b measurements of one satellite are one quantity seen through two
        carriers; they must update a single row, not create two."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        run(s, truth, fcar=0.5, bands=(F_E5A, F_E5B))
        self.assertEqual(len(s._rr_idx), 3, "a second band created extra rows")
        for prn, rr in truth.items():
            self.assertAlmostEqual(s.rrate(prn), rr, delta=0.05)

    def test_a_band_only_seen_once_still_gets_the_scaling_right(self):
        """A band that contributes no measurements (E5b here) inherits the shared solution and
        gets each satellite's correction scaled to the E5b carrier, not E5a's."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        run(s, truth, fcar=0.5, bands=(F_E5A,))
        got_a = s.carrier_correction_hz(11, F_E5A)
        got_b = s.carrier_correction_hz(11, F_E5B)
        self.assertAlmostEqual(got_a, y_of(truth[11], 0.5, F_E5A), delta=0.05)
        self.assertAlmostEqual(got_b, y_of(truth[11], 0.5, F_E5B), delta=0.05)
        # the two differ by exactly the carrier ratio, which is why the state is held in m/s
        self.assertAlmostEqual(got_b / got_a, F_E5B / F_E5A, places=6)

    def test_two_feeds_on_the_same_rows_recover_f_carrier(self):
        """On CHORD, GPS L5, E5a and B2a all sit at 1176.45 MHz. Two feeds at that carrier
        on the same keys, so the same rows, must still recover f_carrier."""
        s = JointReceiverState(code_len=10230.0)
        truth = {11: -2.0, 21: 2.0}
        run(s, truth, fcar=0.7, bands=(F_E5A, F_B2A))
        self.assertAlmostEqual(s.f_carrier(), 0.7, delta=0.05)

    def test_constellations_on_one_carrier_share_f_carrier(self):
        """The broker keys rows (constellation, prn), so a Galileo satellite on E5a and a
        BeiDou satellite on B2a are distinct rows, even under the same PRN number, and both
        see the one receiver-wide f_carrier. The truths average to zero over all four rows, as
        the gauge pins, but not within a constellation, so an f_carrier pooled per
        constellation lands elsewhere."""
        s = JointReceiverState(code_len=10230.0)
        gal = {("E", 11): -3.0, ("E", 21): -1.0}
        bds = {("C", 11): 1.0, ("C", 21): 3.0}
        for k in range(400):
            for key, rr in gal.items():
                s.update_rrate(key, y_of(rr, 0.7, F_E5A), 100.0 + k, F_E5A)
            for key, rr in bds.items():
                s.update_rrate(key, y_of(rr, 0.7, F_B2A), 100.0 + k, F_B2A)
            s.gauge_rrate()
        self.assertEqual(len(s._rr_idx), 4)
        self.assertAlmostEqual(s.f_carrier(), 0.7, delta=0.05)
        for key, rr in {**gal, **bds}.items():
            self.assertAlmostEqual(s.rrate(key), rr, delta=0.05)


class TestRrateBirthAndEscape(unittest.TestCase):
    def test_gauge_stiffness_does_not_freeze_the_rows(self):
        """The rrate rows converge at gauge sigmas 0.02, 0.5 and 2.0: with a correct birth and
        escape, gauge stiffness does not freeze the filter. The default rr_gauge_sigma (0.5)
        is moderate by choice."""
        for gs in (0.02, 0.5, 2.0):
            s = JointReceiverState(code_len=10230.0, rr_gauge_sigma=gs)
            truth = {11: -3.0, 21: 2.0, 28: 1.0}
            run(s, truth, fcar=0.5, n=200)
            err = max(abs(s.rrate(p) - r) for p, r in truth.items())
            self.assertLess(
                err, 0.1, "gauge_sigma %.2f: max error %.3f m/s" % (gs, err)
            )

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
        s.x[i] = 40.0  # displace the row far outside its own gate
        s.P[i, i] = 1e-6  # ...and make the filter certain about it
        run(s, {11: 0.0, 21: 0.0}, fcar=0.0, n=200, t0=400.0)
        self.assertLess(
            abs(s.rrate(11)),
            1.0,
            "a displaced, over-confident row never recovered (rrate %.2f)"
            % s.rrate(11),
        )


class TestStaleRowsAgeOut(unittest.TestCase):
    def test_a_set_satellite_leaves_the_gauge(self):
        """A satellite with no carrier measurement in max_age_s loses its rrate row, so
        gauge_rrate pins the mean over the satellites still reporting. Kept, the stale row
        holds the share of the pin it had when it set, and f_carrier and the live rows stay
        offset by it."""
        s = JointReceiverState(code_len=10230.0)
        live = {1: -2.0, 2: 2.0}
        # Satellite 3 reports first, so its row sits before the live rows and the drop has
        # to renumber them.
        run(s, {3: 6.0, **live}, fcar=0.5, n=200)
        # Satellite 3 sets at t = 299 and cycle() expires its row max_age_s later; the pin
        # then relaxes to the two live rows at the rr_gauge_sigma pace (~1000 s to 0.05 Hz).
        for k in range(int(s.max_age_s) + 1500):
            t = 300.0 + k
            for prn, rr in live.items():
                s.update_rrate(prn, y_of(rr, 0.5, F_E5A), t, F_E5A)
            s.gauge_rrate()
            s.cycle([], t)
        self.assertNotIn(3, s._rr_idx, "a set satellite kept its rrate row")
        self.assertEqual(s.rrate_sigma(3), float("inf"))
        # clk, clk_rate, f_carrier and the two live rows: the stale row left x and P
        self.assertEqual(s.x.size, 5)
        self.assertEqual(s.P.shape, (5, 5))
        self.assertAlmostEqual(s.f_carrier(), 0.5, delta=0.05)
        for prn, rr in live.items():
            self.assertAlmostEqual(s.rrate(prn), rr, delta=0.05)
            self.assertAlmostEqual(
                s.carrier_correction_hz(prn, F_E5A), y_of(rr, 0.5, F_E5A), delta=0.05
            )


class TestClosedLoopReference(unittest.TestCase):
    """The broker's reference rule under --rrate-command with --rrate-feed-applied 1: the rate
    residual is measured on records the tracker already derotated by the commanded trim, so the
    broker feeds y = residual + command applied, referenced to the base seed. This class closes
    the loop under that rule."""

    def _run_loop(self, n=300):
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        fcar = 0.5
        s = JointReceiverState(code_len=10230.0)
        cmd = {p: 0.0 for p in truth}
        resid = {}
        for k in range(n):
            for prn, rr in truth.items():
                y_true = y_of(rr, fcar, F_E5A)
                resid[prn] = y_true - cmd[prn]  # the measured rate residual
                y_fed = resid[prn] + cmd[prn]
                s.update_rrate(prn, y_fed, 100.0 + k, F_E5A)
            s.gauge_rrate()
            for prn in truth:  # next poll's command
                cmd[prn] = s.carrier_correction_hz(prn, F_E5A)
        return truth, fcar, resid

    def test_adding_the_command_back_closes_the_loop(self):
        truth, fcar, resid = self._run_loop()
        for prn in truth:
            self.assertLess(
                abs(resid[prn]),
                0.05,
                "PRN %d standing residual %.3f Hz" % (prn, resid[prn]),
            )


class TestClosedLoopLag(unittest.TestCase):
    """Under --rrate-feed-applied 1 the measured residual reflects the command posted `lag`
    polls earlier, while the feed adds back the latest command, so y_fed carries a
    (cmd_now - cmd_then) misreference that reinforces itself at full slew. The broker's
    per-poll command slew bound (rrate_cmd_slew_hz) bounds the misreference, which decays to
    zero at the fixed point. This simulates that loop."""

    def _run(self, slew, n=400, lag=2, meas_noise=0.0, seed=7):
        import random

        rng = random.Random(seed)
        truth = {11: -3.0, 21: 2.0, 28: 1.0}
        fcar = 0.5
        s = JointReceiverState(code_len=10230.0)
        hist = {p: [0.0] * (lag + 1) for p in truth}  # [0] = latest posted
        for k in range(n):
            for prn, rr in truth.items():
                y_true = y_of(rr, fcar, F_E5A)
                resid = y_true - hist[prn][-1]  # measured under the oldest command
                y_fed = resid + hist[prn][0]  # broker adds back the latest
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
        # The floor is the filter's own steady state (q_rr holds the row loose enough to
        # track drift), not the loop's. The bound pins bounded and small against a multi-Hz
        # walk, not noiseless convergence.
        err = self._run(slew=0.5, meas_noise=0.3)
        for prn, e in err.items():
            self.assertLess(e, 0.8, "PRN %d standing error %.2f Hz" % (prn, e))

    def test_the_escape_does_not_amplify_the_free_loop(self):
        """The rrate escape re-opens P without adopting the rejected measurements. An escape
        that adopted them would snap the row to lag-misreferenced noise every rr_escape_runs
        rejections and amplify the walk. Without that amplifier the free loop converges to the
        same filter-limited floor as the slew-bounded one; the slew bound remains for
        transients."""
        bounded = max(self._run(slew=0.5, meas_noise=0.3).values())
        free = max(self._run(slew=0.0, meas_noise=0.3).values())
        self.assertLess(bounded, 0.8)
        self.assertLess(
            free,
            0.8,
            "free loop diverged (%.2f Hz): the escape amplifies the walk" % free,
        )


class TestFullBandFields(unittest.TestCase):
    """The rrate feed must read the uncapped rate fields. deep_rate_hz is the fold's pick,
    capped at the combiner's DEEP_RATE_MAX_HZ, and past the cap it reports the best in-cap
    noise bin, so a loop closed on it walks. rate_residuals takes the field names as
    arguments; these pin that the selection selects."""

    def setUp(self):
        from gnss_broker.fits import rate_residuals

        self.rr = rate_residuals

    @staticmethod
    def _status(prn, capped, full, q=20.0, hop=1000):
        return {
            prn: {
                "deep_rate_hz": capped,
                "deep_rate_q": q,
                "deep_rate_full_hz": full,
                "deep_rate_full_q": q,
                "amp_snr": 50.0,
                "pow_hop": hop,
            }
        }

    def _two_polls(self, **kw):
        """The continuity gate skips a PRN's first sighting, so drive two polls."""
        ph, pv = {}, {}
        self.rr(
            self._status(7, -4.7, -7.9, hop=1000),
            10.0,
            0.0,
            prev_hop=ph,
            prev_val=pv,
            **kw
        )
        out, _ = self.rr(
            self._status(7, -4.6, -7.8, hop=3048),
            10.0,
            0.0,
            prev_hop=ph,
            prev_val=pv,
            **kw
        )
        return out

    def test_default_reads_the_capped_field(self):
        out = self._two_polls()
        self.assertAlmostEqual(out[7], -4.6)

    def test_full_fields_read_the_uncapped_value(self):
        out = self._two_polls(
            rate_field="deep_rate_full_hz", q_field="deep_rate_full_q"
        )
        self.assertAlmostEqual(out[7], -7.8)


class TestCoarseFineHandoff(unittest.TestCase):
    """The FLL->PLL handoff (--rrate-coarse-deweight). Both feeds update the same row, so
    their relative weight decides which observable governs it: the coarse one arrives every
    poll with ~60 mHz scatter, the fine one about once a minute at ~16 mHz. At equal sigma the
    coarse feed wins on count and the row tracks its noise."""

    F = F_E5A
    TRUTH = -2.0  # m/s on one satellite

    def _row_after(self, deweight, n_poll=240, fine_every=12, seed=5):
        """Returns |rrate(11) - gauge-centred truth|, m/s.

        Measured against the gauge, not raw truth: gauge_rrate() pins mean(rrate) = 0, so
        every row sits at its truth minus the fleet mean by construction, and that offset
        (0.5 m/s here) would swamp the effect under test."""
        import random

        rng = random.Random(seed)
        s = JointReceiverState(code_len=10230.0)
        # two satellites so the gauge has a fleet; only PRN 11 gets the fine feed
        truth = {11: self.TRUTH, 21: 1.0}
        centred = self.TRUTH - sum(truth.values()) / len(truth)
        for k in range(n_poll):
            for prn, rr in truth.items():
                y = y_of(rr, 0.0, self.F)
                sig = 0.2 * (deweight if (prn == 11 and deweight > 1.0) else 1.0)
                s.update_rrate(
                    prn, y + rng.gauss(0.0, 0.06), 100.0 + k, self.F, sigma_hz=sig
                )
            if k % fine_every == 0:  # the fine feed: rarer, far more precise
                s.update_rrate(
                    11,
                    y_of(truth[11], 0.0, self.F) + rng.gauss(0.0, 0.016),
                    100.0 + k,
                    self.F,
                    sigma_hz=0.02,
                )
            s.gauge_rrate()
        return abs(s.rrate(11) - centred)

    def test_deweighted_row_follows_the_fine_observable(self):
        """With the handoff on, the row's error must sit at the fine noise scale."""
        err_hz = self._row_after(deweight=8.0) * (self.F / C)
        self.assertLess(err_hz, 0.05, "phase-governed row off by %.3f Hz" % err_hz)

    def test_the_deweight_is_what_does_it(self):
        """Same measurements at equal weights: the frequent coarse feed dominates and the row
        lands further from truth, so the fine feed alone is not enough. Averaged over seeds,
        since one realisation of a noise comparison decides nothing."""
        gov = sum(self._row_after(deweight=8.0, seed=s) for s in range(6)) / 6.0
        flat = sum(self._row_after(deweight=1.0, seed=s) for s in range(6)) / 6.0
        self.assertLess(
            gov,
            flat,
            "governed %.4f vs flat %.4f m/s -- the handoff bought nothing"
            % (gov, flat),
        )


class TestAdrFineRate(unittest.TestCase):
    """The PLL fine observable: res_cycles differenced over adr_records. Each gate is
    structural: an arc break means unobserved whole cycles (no measurement, not zero); a
    frozen counter must read as absent, never as 0 Hz, or a dead feed passes for a healthy
    one."""

    R = 2048.0 / 195312.5

    def setUp(self):
        from gnss_broker.fits import adr_fine_rate

        self.f = adr_fine_rate

    @staticmethod
    def _row(arc, n, res, trim=None):
        r = {"adr_arc": arc, "adr_records": n, "res_cycles": res}
        if trim is not None:
            r["trim_cycles"] = trim
        return r

    def test_rate_and_count(self):
        out = self.f(self._row(3, 1900, 2.5), self._row(3, 0, 0.5), self.R)
        self.assertIsNotNone(out)
        rate, n, applied = out
        self.assertAlmostEqual(rate, 2.0 / (1900 * self.R), places=9)
        self.assertEqual(n, 1900)
        self.assertIsNone(applied)  # no trim_cycles: the caller uses the posted command

    def test_applied_command_is_measured_not_assumed(self):
        """The reference comes from the tracker's own integrated trim (trim_cycles), same
        span, same stream, never from what the broker believes it posted. 40 cycles of trim
        over 1900 records is the applied command in Hz."""
        out = self.f(
            self._row(3, 1900, 2.5, trim=41.0), self._row(3, 0, 0.5, trim=1.0), self.R
        )
        rate, n, applied = out
        self.assertAlmostEqual(applied, 40.0 / (1900 * self.R), places=9)

    def test_arc_break_is_no_measurement(self):
        self.assertIsNone(self.f(self._row(4, 100, 0.1), self._row(3, 50, 2.0), self.R))

    def test_frozen_counter_reads_absent_not_zero(self):
        self.assertIsNone(
            self.f(self._row(3, 100, 0.1), self._row(3, 100, 0.1), self.R)
        )

    def test_missing_field_reads_absent(self):
        self.assertIsNone(
            self.f({"adr_arc": 3, "adr_records": 100}, self._row(3, 0, 0.0), self.R)
        )

    def test_serving_churn_span_wall_mismatch_is_no_measurement(self):
        """The served row is best-of-instance and the winner churns. Instances' arcs began
        at different times, so a cross-instance difference passes the arc gate (every
        instance is on arc 1) with a record-implied span that can be wrong many times over.
        A span that disagrees with the wall clock is not a measurement."""
        # 5660 records claim a 59.3 s span across a 5.0 s wall gap -> rejected.
        self.assertIsNone(
            self.f(
                self._row(1, 5660, 2.5, trim=10.0),
                self._row(1, 0, 0.5, trim=1.0),
                self.R,
                wall_dt=5.0,
            )
        )
        # The same difference over a consistent wall span is accepted untouched.
        out = self.f(
            self._row(1, 5660, 2.5, trim=10.0),
            self._row(1, 0, 0.5, trim=1.0),
            self.R,
            wall_dt=5660 * self.R,
        )
        self.assertIsNotNone(out)

    def test_no_wall_clock_means_no_churn_gate(self):
        """wall_dt is optional: a caller without a poll clock keeps the structural gates
        and cannot detect churn."""
        self.assertIsNotNone(
            self.f(self._row(1, 5660, 2.5), self._row(1, 0, 0.5), self.R)
        )


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
