"""--dr-nav-symbols: the absolute symbol index a dead-reckoned seed carries for streamed data
symbols (nav_symbol_fields), and its move with the trim (stage_push_seeds)."""

import types
import unittest
from fractions import Fraction

from gnss_broker import deadreckon as dr

C = 299792458.0


def _ctx(constellation, code_len, chip_rate_hz, flag=True):
    c = types.SimpleNamespace()
    c.args = types.SimpleNamespace(
        dr_nav_symbols=flag,
        dr_constellation=constellation,
        hops_per_sec=1.0,
        chip_rate_hz=chip_rate_hz,
    )
    c.dr_eph_mod = types.SimpleNamespace(gpst_of_utc=float)
    c.utc0_sample0 = 1.4e9  # GPST of sample 0, seconds from the GPS epoch
    c.drp = types.SimpleNamespace(t_eph_age=0.0)
    c.code_len = code_len
    return c


class NavSymbolFields(unittest.TestCase):
    def test_l2c_index_is_gpst_over_20_ms(self):
        c = _ctx("G", 10230, 511500.0)
        # ref_hop 100 at 1 hop/s: receive at GPST 1.4e9 + 100; 0.07 s of flight, a 25 us clock
        v = dict(range_m=0.07 * C, range_rate_mps=0.0, sat_clk_s=25e-6)
        f = dr.nav_symbol_fields(c, v, 100, 0.0)
        t_sv = Fraction(1400000100) - Fraction(0.07 * C) / Fraction(C) + Fraction(25e-6)
        per = Fraction(10230, 511500)
        self.assertEqual(f["nav_frame_at_ref"], int(t_sv // per))
        self.assertAlmostEqual(
            f["nav_phase_at_ref"], float((t_sv % per) / per * 10230), places=6
        )
        self.assertEqual(f["nav_frame_chips"], 10230)

    def test_b2b_index_is_bdt_over_1_ms(self):
        c = _ctx("C", 10230, 10.23e6)
        v = dict(range_m=0.08 * C, range_rate_mps=0.0, sat_clk_s=-4e-4)
        f = dr.nav_symbol_fields(c, v, 100, 0.0)
        t_sv = (
            Fraction(1400000100)
            - Fraction(0.08 * C) / Fraction(C)
            + Fraction(-4e-4)
            - 14
        )
        self.assertEqual(f["nav_frame_at_ref"], int(t_sv // Fraction(1, 1000)))

    def test_off_without_the_flag(self):
        c = _ctx("G", 10230, 511500.0, flag=False)
        v = dict(range_m=2e7, range_rate_mps=0.0, sat_clk_s=0.0)
        self.assertIsNone(dr.nav_symbol_fields(c, v, 100, 0.0))

    def test_shipped_phase_moves_the_index_across_a_wrap(self):
        c = _ctx("G", 10230, 511500.0)
        # the model's phase 3 chips into a period; the shipped one 5 chips before its start
        v = dict(range_m=0.0, range_rate_mps=0.0, sat_clk_s=3 / 511500.0)
        base = dr.nav_symbol_fields(c, v, 100, 0.0)
        moved = dr.nav_symbol_fields(c, v, 100, 0.0, shipped_phase_chips=10225.0)
        self.assertEqual(moved["nav_frame_at_ref"], base["nav_frame_at_ref"] - 1)


if __name__ == "__main__":
    unittest.main()
