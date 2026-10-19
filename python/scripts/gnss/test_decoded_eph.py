#!/usr/bin/env python3
"""decoded_eph, the dead-reckoning fallback while no BRDC is loaded, against predict_all on the
same broadcast records.

What each check guards:
  * The range rate. predict_one must rotate the velocity into the reception frame with the
    position, so range_rate_mps is d(range_m)/dt (test_range_rate holds predict_all to the
    same bar, with the reasoning).
  * The clock. sat_clk_s must be the broadcast clock predict_all computes from the same record:
    the polynomial plus the relativistic term F*e*sqrtA*sin(E), up to ~12 ns for G05 (e 0.005)
    and ~400 ns for E14 (e 0.17). Without the term, cold code seeds during a BRDC outage are
    off by that much.

The records are G05 (LNAV) and E14's F/NAV record from BRDC00WRD_R day 292, renamed into the
decoders' fields and propagated with the decoders' own sv_position functions.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoded_eph
import gnss_ephemeris as ge
from galileo_fnav import sv_position_fnav
from gps_nav_decode import sv_position_lnav
from test_range_rate import ALT, G05, TOL, parse, toe_utc, worst_rate_error

E14_FNAV = (
    "E14 2026 10 19 03 40 00 5.381254944950e-03 2.114575181622e-10-3.469446951954e-18\n"
    "     3.900000000000e+01 1.075937500000e+02 4.996279543669e-09-2.639468690797e+00\n"
    "     2.909451723099e-06 1.689912962029e-01 1.136772334576e-05 5.289389413834e+03\n"
    "     9.960000000000e+04-2.693384885788e-06-2.077118580669e+00-1.875683665276e-06\n"
    "     8.509023055765e-01 1.146562500000e+02-3.058877600587e+00-6.784925476523e-09\n"
    "    -5.250218692682e-11 2.580000000000e+02 2.441000000000e+03                   \n"
    "    -1.000000000000e+00 1.600000000000e+01-3.492459654808e-09 0.000000000000e+00\n"
    "     1.003400000000e+05                                                         \n"
)
LAT, LON = 49.32075144444, -119.62081125


def decoded(rec, toe, toc, idot):
    """A RINEX record renamed into a decoder's ephemeris dict (times in seconds of week)."""
    week0 = rec["toe_gpst"] - rec["toe_sow"]
    return {
        "sqrtA": rec["sqrta"],
        "e": rec["ecc"],
        "M0": rec["m0"],
        "dn": rec["dn"],
        "omega": rec["omega"],
        "i0": rec["i0"],
        idot: rec["idot"],
        "OMEGA0": rec["omega0"],
        "OMEGA_dot": rec["omegadot"],
        "Cuc": rec["cuc"],
        "Cus": rec["cus"],
        "Crc": rec["crc"],
        "Crs": rec["crs"],
        "Cic": rec["cic"],
        "Cis": rec["cis"],
        "af0": rec["af0"],
        "af1": rec["af1"],
        "af2": rec["af2"],
        toe: rec["toe_sow"],
        toc: rec["toc_gpst"] - week0,
    }


def g05():
    eph = parse(G05)
    rec = eph[("G", 5)][0]
    dec = decoded(rec, "toe", "toc", "idot")
    return eph, rec, ("G", 5, "GPS_L1_LNAV", dec, sv_position_lnav, rec["toe_gpst"])


def e14():
    eph = parse(E14_FNAV)
    rec = eph[("E", 14)][0]
    dec = decoded(rec, "t0e", "t0c", "iDot")
    return eph, rec, ("E", 14, "GAL_E5AI_FNAV", dec, sv_position_fnav, rec["toe_gpst"])


def predict_decoded(ent, t, lat, lon):
    out = decoded_eph.predict_from_decoders([ent], lat, lon, ALT, t, mask_deg=-90.0)
    return out.get((ent[0], ent[1]))


class TestDecodedRangeRate(unittest.TestCase):
    def test_range_rate_is_the_derivative_of_range(self):
        _, rec, ent = g05()

        def predict(t, lat, lon):
            return predict_decoded(ent, t, lat, lon)

        worst, n = worst_rate_error(predict, toe_utc(rec))
        self.assertGreater(n, 50, "too few near-zero-rate samples to test")
        self.assertLess(worst, TOL, "range rate off d(range)/dt by %.4f m/s" % worst)


class TestDecodedClock(unittest.TestCase):
    def _check(self, eph, rec, ent):
        key = (ent[0], ent[1])
        for dt in range(-10800, 10801, 600):
            t = toe_utc(rec) + dt
            want = ge.predict_all(
                eph, LAT, LON, ALT, t, mask_deg=-90.0, health_gate=False
            )[key]["sat_clk_s"]
            got = predict_decoded(ent, t, LAT, LON)["sat_clk_s"]
            self.assertLess(
                abs(got - want),
                1e-9,
                "%s%02d at toe%+d s: decoded clock off the BRDC path's by %.1f ns"
                % (key[0], key[1], dt, (got - want) * 1e9),
            )

    def test_gps_lnav_clock_matches_the_brdc_path(self):
        self._check(*g05())

    def test_eccentric_galileo_fnav_clock_matches_the_brdc_path(self):
        """E14 is the strong case: its relativistic term swings by ~800 ns over an orbit."""
        self._check(*e14())


class TestToeFold(unittest.TestCase):
    """toe_gpst_near: the GPST reference the toe-age gate measures from. A BeiDou t_oe is BDT
    seconds of week, 14 s behind GPST."""

    WK = 2440 * 604800.0  # a GPST week start

    def test_beidou_toe_is_bdt(self):
        got = decoded_eph.toe_gpst_near("C", 345600.0, self.WK + 345700.0)
        self.assertEqual(got - self.WK, 345614.0)

    def test_gps_and_galileo_toes_are_gpst(self):
        for gsys in ("G", "E"):
            got = decoded_eph.toe_gpst_near(gsys, 345600.0, self.WK + 345700.0)
            self.assertEqual(got - self.WK, 345600.0)

    def test_nearest_week(self):
        # a toe late in the week, read just after the rollover, is last week's
        got = decoded_eph.toe_gpst_near("G", 604000.0, self.WK + 100.0)
        self.assertEqual(got, self.WK - 800.0)
        # a BDT toe 10 s before the BDT rollover is 4 s into the next GPST week
        got = decoded_eph.toe_gpst_near("C", 604790.0, self.WK + 604700.0)
        self.assertEqual(got, self.WK + 604804.0)

    def test_the_broker_folds_through_it(self):
        """gps_distributed_broker's _decoded_entries builds toe_gpst with toe_gpst_near and
        keeps no fold of its own."""
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gps_distributed_broker.py"
        )
        with open(path) as f:
            tree = ast.parse(f.read(), path)
        fn = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_decoded_entries"
        ][0]
        src = ast.unparse(fn)
        self.assertIn("_decfb.toe_gpst_near(", src)
        self.assertNotIn("302400", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
