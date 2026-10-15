#!/usr/bin/env python3
"""The GLONASS propagator, pinned against an orbit it did not come from.

What each check guards:
  * The parse. An R record is four lines, not eight, and the state vector is in km; a wrong
    line count or unit would still produce a dict. The pin is one real record from
    BRDC00WRD_R day 284, with its epoch in UTC and its channel number k.
  * The integration, against GFZ0MGXRAP_20262840000 (the MGEX rapid orbit, ITRF, centre of
    mass), Lagrange-interpolated (9 points of the 5 min grid, mm-accurate) to the record's tb
    and tb +- 600 s. The propagator sits a few metres off the SP3 (under the 10 m bound): the
    record's own offset (phase centre vs centre of mass, plus the broadcast error). That
    offset must not grow over the 10 min, which is what tells an integrator from a drift.
  * The sky. predict_all must place the satellite where the SP3 does: R02 near the top of a
    78.9 deg pass over the station, with az/el taken from the SP3 position in an independent
    ENU computation.
  * The range contract. A record is meant for +-15 min about tb; further is allowed and
    warns, so a stale sky is visible rather than silently wrong.
"""
import math
import os
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge

HEADER = (
    "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
    "                                                            END OF HEADER\n"
)
R02_1215 = (
    "R02 2026 10 11 12 15 00 1.171696931120E-05 0.000000000000E+00 4.320000000000E+04\n"
    "     1.865880078130E+04 4.694385528560E-01 3.725290298460E-09 0.000000000000E+00\n"
    "     1.701093505860E+04 1.790657043460E-01-1.862645149230E-09-4.000000000000E+00\n"
    "     3.359197753910E+03-3.564683914180E+00-1.862645149230E-09 0.000000000000E+00\n"
)
R02_2215 = (
    "R02 2026 10 11 22 15 00 1.171696931124E-05 0.000000000000E+00 7.920000000000E+04\n"
    "    -1.193832763672E+04-7.385950088501E-01-1.862645149231E-09 0.000000000000E+00\n"
    "    -1.427417285156E+04-2.272885322571E+00 0.000000000000E+00-4.000000000000E+00\n"
    "     1.744660546875E+04-2.377678871155E+00-2.793967723846E-09 0.000000000000E+00\n"
)
TB_1215 = datetime(2026, 10, 11, 12, 15, tzinfo=timezone.utc)
# GFZ0MGXRAP_20262840000_01D_05M_ORB.SP3, interpolated to tb - 600 s, tb, tb + 600 s (m).
SP3_1215 = {
    -600.0: (18321121.951, 16832115.368, 5480339.411),
    0.0: (18658803.659, 17010935.041, 3359198.934),
    600.0: (18879713.274, 17050134.835, 1208904.110),
}
# R02 at the SP3 epoch 22:05:00 GPST = 22:04:42 UTC, az/el from the SP3 position at the
# station (49.32075144444, -119.62081125, 545.0).
LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
PASS_UTC = datetime(2026, 10, 11, 22, 4, 42, tzinfo=timezone.utc)
PASS_AZ, PASS_EL = 262.9597, 78.9016


def _parse(text):
    with tempfile.NamedTemporaryFile("w", suffix=".rnx", delete=False) as f:
        f.write(HEADER + text)
    try:
        return ge.parse_rinex_nav(f.name)
    finally:
        os.unlink(f.name)


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class TestParse(unittest.TestCase):
    def test_record_fields(self):
        eph = _parse(R02_1215)
        self.assertEqual(list(eph), [("R", 2)])
        e = eph[("R", 2)][0]
        self.assertEqual(e["k"], -4)
        self.assertEqual(e["health"], 0.0)
        self.assertAlmostEqual(e["pos"][0], 18658800.7813, places=3)
        self.assertAlmostEqual(e["vel"][2], -3564.68391418, places=6)
        self.assertAlmostEqual(e["acc"][0], 3.72529029846e-06, places=15)
        self.assertAlmostEqual(e["af0"], 1.171696931120e-05, places=17)
        # the epoch is UTC: 18 s of leap seconds between it and the GPST the sky runs on
        self.assertEqual(e["toe_gpst"], ge.gpst_of_utc(TB_1215))
        self.assertEqual(e["toc_gpst"], e["toe_gpst"])


class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.e = _parse(R02_1215)[("R", 2)][0]
        self.tb = self.e["toe_gpst"]

    def test_against_sp3_at_plus_minus_10_min(self):
        d = {}
        for dt, ref in SP3_1215.items():
            pos, _vel = ge.glonass_state(self.e, self.tb + dt)
            d[dt] = _dist(pos, ref)
            self.assertLess(d[dt], 10.0, "dt=%+.0f: %.2f m off the SP3" % (dt, d[dt]))
        # the offset is the record's, not the integrator's: it must not grow over 10 min
        self.assertLess(abs(d[600.0] - d[0.0]), 1.0)
        self.assertLess(abs(d[-600.0] - d[0.0]), 1.0)

    def test_tb_returns_the_record(self):
        pos, vel = ge.glonass_state(self.e, self.tb)
        self.assertEqual(pos, self.e["pos"])
        self.assertEqual(vel, self.e["vel"])

    def test_velocity_is_the_integrated_state(self):
        """sat_pos_clk hands back the integrated velocity, consistent with the position's
        own time derivative to the mm/s the Doppler hints need."""
        t = self.tb + 300.0
        _pos, vel, clk = ge.sat_pos_clk(self.e, t)
        pp = ge._pos_only(self.e, t + 0.5)
        pm = ge._pos_only(self.e, t - 0.5)
        for k in range(3):
            self.assertAlmostEqual(vel[k], pp[k] - pm[k], places=3)
        self.assertAlmostEqual(clk, self.e["af0"], places=17)

    def test_beyond_15_min_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ge.glonass_state(self.e, self.tb + 901.0)
        self.assertEqual(len(w), 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ge.glonass_state(self.e, self.tb + 900.0)
        self.assertEqual(len(w), 0)


class TestSky(unittest.TestCase):
    def test_known_pass_over_the_station(self):
        eph = _parse(R02_2215)
        pa = ge.predict_all(eph, LAT, LON, ALT, PASS_UTC, mask_deg=-90.0)
        v = pa[("R", 2)]
        self.assertAlmostEqual(v["el"], PASS_EL, delta=0.01)
        self.assertAlmostEqual(v["az"], PASS_AZ, delta=0.02)
        self.assertAlmostEqual(v["toe_age_s"], -618.0)
        self.assertGreater(v["el"], 75.0)


if __name__ == "__main__":
    unittest.main()
