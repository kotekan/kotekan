#!/usr/bin/env python3
"""A0b: the broadcast group-delay term, with the controls that make it a test rather than
a restatement of the implementation.

WHY EACH CHECK EXISTS
  * SIGN AND SCALE against a hand-worked ICD number -- the whole term is one multiply, so
    the only ways to get it wrong are the sign and the gamma, and both are silent.
  * THE CROSS-TYPE CONVERSIONS. Our BRDC carries BOTH Galileo nav types (19 F/NAV + 11
    I/NAV on 2026-08-23) and best_eph picks by freshness, so a satellite's clock reference
    can FLIP mid-run. The identity that must hold is that E1's clock is the same however
    you reach it; if it does not, a refresh steps that satellite.
  * THE NEGATIVE CONTROL. signal=None must return exactly 0.0 -- every non-broker caller
    (TEC, observables, the sky map) relies on that to stay byte-identical.
  * B2a MUST RETURN ZERO, and that is a REQUIREMENT rather than an omission: TGD_B2ap lives
    in B-CNAV2 and RINEX 3 does not carry it. Borrowing TGD2 (a different band) would be a
    plausible-looking wrong answer, which is worse than none.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_ephemeris import (group_delay_s, GAMMA_L1L5, GAMMA_E1E5A, GAMMA_E1E5B)

NS = 1e-9


def _g(tgd=0.0, tgd2=0.0, ds=0):
    return dict(sys="G", tgd=tgd, iodc=tgd2, l2_codes=ds)


def _e(bgd_a=0.0, bgd_b=0.0, fnav=True):
    return dict(sys="E", tgd=bgd_a, iodc=bgd_b, l2_codes=(0x100 if fnav else 0x200))


def _c(tgd1=0.0, tgd2=0.0):
    return dict(sys="C", tgd=tgd1, iodc=tgd2, l2_codes=0)


class TestGPS(unittest.TestCase):
    def test_l5_scales_by_gamma_and_flips_sign(self):
        """IS-GPS-705: an L5 user owes gamma_15 * TGD, subtracted from the clock. The live
        median TGD is -8.38 ns, so the correction is POSITIVE ~+15 ns -- if this comes out
        negative the seeds move the wrong way by twice the term."""
        self.assertAlmostEqual(group_delay_s(_g(tgd=-8.38 * NS), "gps_l5"),
                               +GAMMA_L1L5 * 8.38 * NS, places=15)
        self.assertAlmostEqual(GAMMA_L1L5, 1.79329, places=4)

    def test_l1_is_the_unscaled_term(self):
        self.assertAlmostEqual(group_delay_s(_g(tgd=-8.38 * NS), "gps_l1"), +8.38 * NS,
                               places=15)

    def test_iodc_is_not_mistaken_for_a_second_delay(self):
        """orb[23] is IODC for GPS, not a group delay. A value of 7.0 (seconds!) must not
        reach the clock -- the field is shared with Galileo's BGD_E5b by slot index."""
        self.assertEqual(group_delay_s(_g(tgd=0.0, tgd2=7.0), "gps_l5"), 0.0)


class TestGalileo(unittest.TestCase):
    def test_fnav_e5a_and_inav_e5b_are_the_native_cases(self):
        self.assertAlmostEqual(group_delay_s(_e(bgd_a=3.0 * NS, fnav=True), "gal_e5a"),
                               -GAMMA_E1E5A * 3.0 * NS, places=15)
        self.assertAlmostEqual(group_delay_s(_e(bgd_b=3.0 * NS, fnav=False), "gal_e5b"),
                               -GAMMA_E1E5B * 3.0 * NS, places=15)

    def test_e1_clock_is_reference_invariant(self):
        """THE IDENTITY THE CROSS-TYPE CONVERSIONS EXIST FOR. t_E1 = t_IF_a - BGD_a =
        t_IF_b - BGD_b, so with t_IF_a and t_IF_b differing by (BGD_a - BGD_b), an E1 user
        must land on the same clock from either record type. The two BROADCAST clocks
        differ by (BGD_a - BGD_b), so that offset is removed from the I/NAV side before
        comparing. This is what stops a freshness-driven F/NAV <-> I/NAV flip from
        stepping a satellite."""
        a, b = 3.03 * NS, 3.26 * NS
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e1")
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e1") - (a - b)
        self.assertAlmostEqual(from_fnav, from_inav, places=15)

    def test_e5a_agrees_across_record_types(self):
        """Same identity for the signal we actually track. Reaching E5a from an I/NAV
        record must equal reaching it from an F/NAV record, once the two records' own
        clock offset (BGD_a - BGD_b) is accounted for."""
        a, b = 3.03 * NS, 3.26 * NS
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e5a")
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e5a") - (a - b)
        self.assertAlmostEqual(from_fnav, from_inav, places=15)

    def test_e5b_agrees_across_record_types(self):
        a, b = 3.03 * NS, 3.26 * NS
        from_inav = group_delay_s(_e(a, b, fnav=False), "gal_e5b") - (a - b)
        from_fnav = group_delay_s(_e(a, b, fnav=True), "gal_e5b")
        self.assertAlmostEqual(from_inav, from_fnav, places=15)

    def test_unflagged_record_defaults_to_fnav(self):
        e = dict(sys="E", tgd=3.0 * NS, iodc=0.0, l2_codes=0)
        self.assertAlmostEqual(group_delay_s(e, "gal_e5a"), -GAMMA_E1E5A * 3.0 * NS,
                               places=15)


class TestBeiDou(unittest.TestCase):
    def test_b2a_is_zero_because_the_term_is_not_broadcast(self):
        """NOT an omission. TGD_B2ap is a B-CNAV2 parameter and RINEX 3 carries only
        TGD1 (B1I/B3I) and TGD2 (B2I/B3I). Borrowing TGD2 would apply a 1207 MHz delay to
        an 1176 MHz signal -- confidently wrong beats honestly absent."""
        self.assertEqual(group_delay_s(_c(tgd1=-4.3 * NS, tgd2=-4.3 * NS), "bds_b2a"), 0.0)

    def test_b2b_uses_tgd2(self):
        self.assertAlmostEqual(group_delay_s(_c(tgd2=-4.3 * NS), "bds_b2b"), +4.3 * NS,
                               places=15)


class TestNegativeControls(unittest.TestCase):
    def test_no_signal_is_exactly_zero(self):
        """Every non-broker caller passes nothing and must be byte-identical to before."""
        for e in (_g(tgd=-8.4 * NS), _e(3 * NS, 3 * NS), _c(-4.3 * NS, -4.3 * NS)):
            self.assertEqual(group_delay_s(e, None), 0.0)
            self.assertEqual(group_delay_s(e, ""), 0.0)

    def test_unknown_signal_is_zero_not_a_guess(self):
        self.assertEqual(group_delay_s(_g(tgd=-8.4 * NS), "gps_l2c"), 0.0)
        self.assertEqual(group_delay_s(_e(3 * NS), "gal_e6"), 0.0)

    def test_zero_tgd_is_zero_correction(self):
        self.assertEqual(group_delay_s(_g(tgd=0.0), "gps_l5"), 0.0)

    def test_magnitude_stays_in_the_measured_band(self):
        """A guard against a units slip: the live BRDC spans -45..+47 ns, so no correction
        may exceed ~100 ns (0.9 chips). A ppm/seconds mix-up would blow straight past it."""
        for e, sig in ((_g(tgd=-45 * NS), "gps_l5"), (_e(-16 * NS, 4 * NS), "gal_e5a"),
                       (_e(-16 * NS, 4 * NS, fnav=False), "gal_e5b"),
                       (_c(tgd2=47 * NS), "bds_b2b")):
            self.assertLess(abs(group_delay_s(e, sig)), 100 * NS,
                            "%s correction implausibly large -- units?" % sig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
