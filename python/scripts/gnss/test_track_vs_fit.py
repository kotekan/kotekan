#!/usr/bin/env python3
"""Pinning tests for the broker's seed-phase transport in gnss_broker/fits.py:
retag_seed_doppler, seed_phase_at_ref and tracker_phase_at, and for the coast re-tag that
uses them (gnss_broker/seeding.py, stage_coast_drop).

Run: python3 test_track_vs_fit.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.fits import dr_cp0  # noqa: E402

HPS = 195312.5
CHIP = 10.23e6
CARR = 1176.45e6
SGN = 1.0
L = 10230.0
MOD = 20 * L  # seeds live mod the long code
T0 = 195000.0  # ~2.26 days of F-engine age, where t_abs multiplies any Doppler error
H0 = int(round(T0 * HPS))


class TestRetagSeedDoppler(unittest.TestCase):
    """retag_seed_doppler: a Doppler re-tag must preserve the physical phase where the
    despread is running (now), not where the seed is anchored."""

    def _phys(self, cp0, dop, t):
        return (cp0 + t * CHIP * (1.0 + SGN * dop / CARR)) % MOD

    def test_retag_at_now_preserves_the_phase_now(self):
        cp0, dopA, dopB = 4321.0, 1000.0, 1002.0
        t_now = T0 + 600.0
        before = self._phys(cp0, dopA, t_now)
        from gnss_broker.fits import retag_seed_doppler

        cp0_new = retag_seed_doppler(cp0, dopA, dopB, t_now, CHIP, CARR, SGN, MOD)
        after = self._phys(cp0_new, dopB, t_now)
        # Bar sits just above the double-precision floor, not at zero: the phys
        # reconstruction runs through t*f_chip ~ 2e12 chips where one ulp is 2.4e-4,
        # and a few ulps accumulate across retag + rebuild. 2e-3 chips is ~6 cm of code;
        # a logic error here is chips, not milli-chips.
        self.assertLess(abs(((after - before + MOD / 2) % MOD) - MOD / 2), 2e-3)

    # ---- the Doppler's own epoch ---------------------------------------------------------
    #
    # `doppler_hz` is epoch-bearing too: gnss::propagate_seed applies
    # dop_applied(t) = doppler_hz + dop_rate*(t - ref_hop), so doppler_hz must be the Doppler
    # at ref_hop. A current forecast stored against an older ref_hop gets dop_rate*age added
    # on top, and the applied Doppler runs at ~2x the true rate until the next detection
    # re-anchors ref_hop. The broker's coast re-tag (seeding.stage_coast_drop) stores the
    # forecast back-propagated to ref_hop; these tests run it on a minimal context.

    PRN, CP0, DOP0, RATE, AGE = 11, 4321.0, 245.0, -0.35, 200.0

    def _coast(self, forecast):
        """One stage_coast_drop pass over a seed anchored AGE s before now that was not
        re-detected; returns the seed."""
        from types import SimpleNamespace

        from gnss_broker import seeding
        from gnss_broker.seed import Seed
        from gnss_broker.transport import _TR

        utc0 = 1.7e9
        seed = Seed.born(
            "test",
            epoch=H0,
            code_phase_chips=self.CP0,
            doppler_hz=self.DOP0,
            ref_hop=H0,
            doppler_rate_hz_s=self.RATE,
        )
        ctx = SimpleNamespace(
            seeds={self.PRN: seed},
            probe_set=set(),
            up=None,
            dr_state=None,
            best={},
            pred={self.PRN: (forecast, self.RATE)},
            cb=SimpleNamespace(seed=0.0),
            utc0_sample0=utc0,
            code_len=L,
            status={},
            have_sig=False,
            hold=SimpleNamespace(prev={}, low_hits={}, miss={}),
            coast_polls=100,
            args=SimpleNamespace(
                almanac=True,
                dr_fengine_axis=False,
                hops_per_sec=HPS,
                chip_rate_hz=CHIP,
                carrier_hz=CARR,
                code_doppler_sign=SGN,
                drop_amplitude=1.0,
                lock_prompt_hold=1.0,
                coast_budget=60.0,
            ),
        )
        # the cycle's frozen clock (transport.axis_now reads it): now is AGE past ref_hop
        _TR._t.v = utc0 + T0 + self.AGE
        try:
            seeding.stage_coast_drop(ctx)
        finally:
            _TR._t.v = None
        return ctx.seeds[self.PRN]

    def test_stored_doppler_is_back_propagated_to_ref_hop(self):
        """The tracker must end up applying the forecast's current value at now."""
        forecast = self.DOP0 + self.RATE * self.AGE + 1.5  # news: 1.5 Hz off the seed
        seed = self._coast(forecast)
        self.assertAlmostEqual(
            seed["doppler_hz"], forecast - self.RATE * self.AGE, places=9
        )
        self.assertEqual(seed["ref_hop"], H0)

    def test_a_no_news_forecast_moves_nothing(self):
        """A forecast equal to what the seed already propagates to at now is no change: the
        re-tag compares the old Doppler at now, so a standing rate cancels and neither the
        code phase nor the stored Doppler moves. Compared at ref_hop instead, the phantom
        dop_rate*age (70 Hz) would kick cp0 by t_now*(f_chip/f_car)*70, ~10^5 chips."""
        seed = self._coast(self.DOP0 + self.RATE * self.AGE)
        self.assertEqual(seed["code_phase_chips"], self.CP0)
        self.assertEqual(seed["doppler_hz"], self.DOP0)

    def test_the_anchor_epoch_tripwire(self):
        """Translating at a 600-s-stale anchor steps the phase now by
        anchor_age * (f_chip/f_car) * ddop, ~10.4 chips for a 2 Hz re-tag: the error a caller
        makes by evaluating at the anchor instead of now."""
        cp0, dopA, ddop = 4321.0, 1000.0, 2.0
        t_anchor = T0
        t_now = T0 + 600.0
        before = self._phys(cp0, dopA, t_now)
        from gnss_broker.fits import retag_seed_doppler

        cp0_bad = retag_seed_doppler(
            cp0, dopA, dopA + ddop, t_anchor, CHIP, CARR, SGN, MOD
        )
        after_bad = self._phys(cp0_bad, dopA + ddop, t_now)
        err = abs(((after_bad - before + MOD / 2) % MOD) - MOD / 2)
        self.assertAlmostEqual(
            err,
            600.0 * CHIP / CARR * ddop,
            delta=0.01,
            msg="the anchor-epoch error law changed -- investigate",
        )
        self.assertGreater(err, 10.0)


class TestSeedPhaseTransport(unittest.TestCase):
    """seed_phase_at_ref: the broker's held phase -> the field the tracker prefers. The
    conversion is the point: unconverted, the value is one hop (~52 chips) off, a total loss
    of lock."""

    def test_offset_is_one_hop(self):
        from gnss_broker.fits import seed_phase_at_ref

        v = seed_phase_at_ref(0.0, 1894.984, CHIP, HPS, CARR, SGN, MOD, 8192)
        self.assertAlmostEqual(
            v, 52.3713, delta=1e-3, msg="expected the one-hop offset, 52.3713 chips"
        )

    def test_omitting_fft_len_costs_one_sample(self):
        from gnss_broker.fits import seed_phase_at_ref

        a = seed_phase_at_ref(0.0, 1894.984, CHIP, HPS, CARR, SGN, MOD, 8192)
        b = seed_phase_at_ref(0.0, 1894.984, CHIP, HPS, CARR, SGN, MOD, None)
        self.assertAlmostEqual(b - a, 0.0064, delta=1e-3)

    def test_round_trip_matches_the_tracker(self):
        """The invariant the tracker enforces: phase_from_arg(dr_cp0(phys)) == the shipped
        phase. Both sides here are the broker's; the C++ agreement is scripts/gnss/e2e.cpp's
        job."""
        from gnss_broker.fits import seed_phase_at_ref

        phys, dop = 4321.0, -876.0
        h = H0 + int(11.0 * HPS)
        cp0 = dr_cp0(phys, h / HPS, dop, CHIP, CARR, SGN, MOD)
        shipped = seed_phase_at_ref(phys, dop, CHIP, HPS, CARR, SGN, MOD, 8192)
        # what the tracker reconstructs from the argument (its own last-sample reference)
        per_hop = CHIP / HPS * (1.0 + SGN * dop / CARR)
        from_arg = (
            cp0
            + (h / HPS) * CHIP * (1.0 + SGN * dop / CARR)
            + per_hop * (1.0 - 1.0 / 8192)
        ) % MOD
        self.assertAlmostEqual(
            ((shipped - from_arg + MOD / 2) % MOD) - MOD / 2, 0.0, delta=2e-3
        )

    def test_a_doppler_edit_cannot_move_the_phase(self):
        """Why the phase is shipped: cp0 is meaningful only paired with its Doppler, and a
        producer that edits one without the other injects ~1700 chips/Hz at this age. The
        phase has no partner: re-tagging the Doppler leaves it where it is, bar the per-hop
        code-rate term."""
        from gnss_broker.fits import seed_phase_at_ref

        phys = 4321.0
        a = seed_phase_at_ref(phys, 1000.0, CHIP, HPS, CARR, SGN, MOD, 8192)
        b = seed_phase_at_ref(phys, 1002.0, CHIP, HPS, CARR, SGN, MOD, 8192)
        self.assertLess(
            abs(b - a),
            1e-3,
            "a 2 Hz re-tag moved the shipped phase by %.4f chips" % (b - a),
        )


class TestTrackerPhaseAt(unittest.TestCase):
    """tracker_phase_at: the seed audit must model the reference propagate_seed actually
    reads, not the one the broker happened to compute."""

    DOP = 1894.984
    FFT = 8192

    def _seed(self, phys, h, **kw):
        d = {
            "code_phase_chips": dr_cp0(phys, h / HPS, self.DOP, CHIP, CARR, SGN, MOD),
            "doppler_hz": self.DOP,
            "ref_hop": h,
            "code_phase_rate": 0.0,
            "doppler_rate_hz_s": 0.0,
        }
        d.update(kw)
        return d

    def test_both_references_agree_when_consistent(self):
        from gnss_broker.fits import tracker_phase_at, seed_phase_at_ref

        h = H0
        arg_only = self._seed(4321.0, h)
        with_phase = self._seed(
            4321.0,
            h,
            code_phase_at_ref_chips=seed_phase_at_ref(
                4321.0, self.DOP, CHIP, HPS, CARR, SGN, MOD, self.FFT
            ),
        )
        a = tracker_phase_at(arg_only, h, HPS, CHIP, CARR, SGN, MOD, self.FFT)
        b = tracker_phase_at(with_phase, h, HPS, CHIP, CARR, SGN, MOD, self.FFT)
        self.assertAlmostEqual(a, b, delta=1e-3)

    def test_the_phase_wins_when_they_disagree(self):
        """cp0 and the phase come from different broker paths and can disagree; the tracker
        reads the phase, so the audit must too."""
        from gnss_broker.fits import tracker_phase_at, seed_phase_at_ref

        h = H0
        shipped = seed_phase_at_ref(
            9999.0, self.DOP, CHIP, HPS, CARR, SGN, MOD, self.FFT
        )
        d = self._seed(4321.0, h, code_phase_at_ref_chips=shipped)
        got = tracker_phase_at(d, h, HPS, CHIP, CARR, SGN, MOD, self.FFT)
        # the shipped value is the expectation, not the broker-convention phase it was
        # built from: they differ by the hop offset, which is the point of the field
        self.assertAlmostEqual(
            ((got - shipped + MOD / 2) % MOD) - MOD / 2,
            0.0,
            delta=1e-3,
            msg="the audit followed cp0 where the tracker follows the phase",
        )
        self.assertGreater(
            abs(((got - (4321.0 + shipped - 9999.0) + MOD / 2) % MOD) - MOD / 2),
            1000.0,
            "the cp0 value must NOT be what came back",
        )

    def test_advance_matches_the_code_rate(self):
        from gnss_broker.fits import tracker_phase_at

        h = H0
        d = self._seed(4321.0, h)
        dh = int(10.0 * HPS)
        a = tracker_phase_at(d, h, HPS, CHIP, CARR, SGN, MOD, self.FFT)
        b = tracker_phase_at(d, h + dh, HPS, CHIP, CARR, SGN, MOD, self.FFT)
        want = 10.0 * (CHIP + SGN * CHIP * self.DOP / CARR)
        self.assertAlmostEqual(
            ((b - a - want + MOD / 2) % MOD) - MOD / 2, 0.0, delta=1e-2
        )

    def test_unset_rates_read_as_zero(self):
        """A seed may carry code_phase_rate or doppler_rate_hz_s as None; both reference
        branches must treat that as 0, as for an absent key."""
        from gnss_broker.fits import tracker_phase_at, seed_phase_at_ref

        h = H0
        h1 = H0 + int(10.0 * HPS)
        phase = seed_phase_at_ref(4321.0, self.DOP, CHIP, HPS, CARR, SGN, MOD, self.FFT)
        for kw in ({}, {"code_phase_at_ref_chips": phase}):
            want = tracker_phase_at(
                self._seed(4321.0, h, **kw), h1, HPS, CHIP, CARR, SGN, MOD, self.FFT
            )
            got = tracker_phase_at(
                self._seed(
                    4321.0, h, code_phase_rate=None, doppler_rate_hz_s=None, **kw
                ),
                h1,
                HPS,
                CHIP,
                CARR,
                SGN,
                MOD,
                self.FFT,
            )
            self.assertEqual(got, want)

    def test_a_pair_inconsistent_seed_is_not_a_step(self):
        """The audit's purpose: a seed whose cp0 is stale but whose phase is correct is not
        a discontinuity for the tracker, and must not be reported as one."""
        from gnss_broker.fits import tracker_phase_at, seed_phase_at_ref

        h0, h1 = H0, H0 + int(8.0 * HPS)
        phase0 = seed_phase_at_ref(
            4321.0, self.DOP, CHIP, HPS, CARR, SGN, MOD, self.FFT
        )
        adv = 8.0 * (CHIP + SGN * CHIP * self.DOP / CARR)
        s0 = self._seed(4321.0, h0, code_phase_at_ref_chips=phase0)
        # cp0 deliberately garbage (a stale argument); the phase is right
        s1 = self._seed(
            4321.0 + 7777.0, h1, code_phase_at_ref_chips=(phase0 + adv) % MOD
        )
        step = (
            (
                tracker_phase_at(s1, h1, HPS, CHIP, CARR, SGN, MOD, self.FFT)
                - tracker_phase_at(s0, h1, HPS, CHIP, CARR, SGN, MOD, self.FFT)
                + MOD / 2
            )
            % MOD
        ) - MOD / 2
        self.assertLess(
            abs(step),
            1e-2,
            "a stale cp0 leaked into the audit as a %.1f-chip step" % step,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
