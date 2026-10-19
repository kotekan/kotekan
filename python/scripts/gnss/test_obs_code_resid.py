#!/usr/bin/env python3
"""The observables row's code residual: the tracker's range form (receiver clock left in) while
it is fresh, else the search's clock-free dr_integ, and code_resid_clk says which.

CMC pairs code_resid with a carrier residual that carries the receiver clock, so a row flagged
clock-in whose residual is the clock-free dr_integ mixes the two forms.

Run: python3 test_obs_code_resid.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_observables import code_resid, INTEG_MAX_AGE_S, C_LIGHT  # noqa: E402

CHIP_HZ = 10.23e6


class TestCodeResid(unittest.TestCase):
    def test_fresh_tracker_residual_is_the_range_form(self):
        m, src, clk = code_resid(
            {"trk_range_s": 1e-7, "trk_resid_age_s": 2.0, "dr_integ_chips": 0.3},
            CHIP_HZ,
        )
        self.assertEqual((src, clk), ("trk", True))
        self.assertAlmostEqual(m, 1e-7 * C_LIGHT)

    def test_stale_tracker_falls_back_clock_free(self):
        m, src, clk = code_resid(
            {
                "trk_range_s": 1e-7,
                "trk_resid_age_s": INTEG_MAX_AGE_S + 110.0,
                "dr_integ_chips": 0.3,
                "dr_integ_age_s": 5.0,
            },
            CHIP_HZ,
        )
        self.assertEqual((src, clk), ("dr_integ", False))
        self.assertAlmostEqual(m, 0.3 * C_LIGHT / CHIP_HZ)

    def test_no_residual_is_not_clock_in(self):
        self.assertEqual(
            code_resid(
                {"trk_range_s": 1e-7, "trk_resid_age_s": INTEG_MAX_AGE_S + 110.0},
                CHIP_HZ,
            ),
            (None, None, False),
        )
        self.assertEqual(code_resid({}, CHIP_HZ), (None, None, False))


if __name__ == "__main__":
    unittest.main()
