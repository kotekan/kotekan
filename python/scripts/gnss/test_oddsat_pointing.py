#!/usr/bin/env python3
"""gnss_oddsat_pointing: TLE-derived R and C records, read back through gnss_ephemeris.

* parse_rinex_nav reads every record, with the GLONASS channel and BDT toe written.
* sat_pos_clk on a record follows SGP4 within 100 m over +-5 min of its epoch (GLONASS: the
  ICD integrator from the state; BeiDou: the Kepler model from the osculating elements),
  with no clock beyond the relativistic term.
* Every record reads back with the pointing clock sigma, which predict_all passes on.
* A satellite with a decoded ephemeris is left out.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from sgp4.api import Satrec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_ephemeris as ge  # noqa: E402
import gnss_oddsat_pointing as op  # noqa: E402

TLES = {
    32275: (
        "1 32275U 07052A   26331.21719277 -.00000027  00000+0  00000+0 0  9992",
        "2 32275  65.4791 310.4333 0005516 246.7309 113.2065  2.13104031148532",
    ),
    40748: (
        "1 40748U 15037A   26331.01896694 -.00000046  00000+0  00000+0 0  9996",
        "2 40748  54.7243 296.4217 0011883 272.6970  87.1520  1.86232040 77154",
    ),
}
NOW = 1795805000.0  # 2026-11-27 UTC, near the TLE epochs


class TestPointing(unittest.TestCase):
    def setUp(self):
        self._tle = op.tle
        op.tle = lambda norad, now=None: TLES[norad]
        sats = op.parse_sats("R25:32275:8,R28:32275:7,C58:40748")
        self.text, self.held = op.build(sats, NOW, 600.0)
        self.d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.d.name, "p.rnx")
        op.write(self.path, self.text)
        self.eph = ge.parse_rinex_nav(self.path)

    def tearDown(self):
        op.tle = self._tle
        self.d.cleanup()

    def test_records(self):
        n = len(op.epochs(NOW, 600.0))
        self.assertEqual(self.held, [("R", 25), ("R", 28), ("C", 58)])
        for key in (("R", 25), ("R", 28), ("C", 58)):
            self.assertEqual(len(self.eph[key]), n, key)
        self.assertEqual({e["k"] for e in self.eph[("R", 25)]}, {8})
        self.assertEqual({e["k"] for e in self.eph[("R", 28)]}, {7})
        g = ge.gpst_of_utc(op.epochs(NOW, 600.0)[0])
        self.assertAlmostEqual(self.eph[("C", 58)][0]["toe_gpst"], g, places=3)
        self.assertAlmostEqual(self.eph[("R", 25)][0]["toe_gpst"], g, places=3)

    def test_follows_sgp4(self):
        for key, norad in ((("R", 25), 32275), (("C", 58), 40748)):
            sat = Satrec.twoline2rv(*TLES[norad])
            for e in self.eph[key][::7]:
                for dt in (-300.0, 0.0, 300.0):
                    t_utc = e["toe_gpst"] + dt - (ge.gpst_of_utc(0.0) - 0.0)
                    pos, _, clk = ge.sat_pos_clk(e, e["toe_gpst"] + dt)
                    want, _ = op.ecef_state(sat, t_utc)
                    self.assertLess(
                        np.linalg.norm(np.array(pos) - want), 100.0, (key, dt)
                    )
                    self.assertLess(abs(clk), 1e-8)  # the relativistic term only

    def test_clock_unknown(self):
        for recs in self.eph.values():
            self.assertEqual(
                {e["clk_sigma_s"] for e in recs}, {ge.POINTING_CLK_SIGMA_S}
            )
        t_utc = self.eph[("C", 58)][20]["toe_gpst"] - ge.gpst_of_utc(0.0)
        pred = ge.predict_all(self.eph, 45.0, -75.0, 100.0, t_utc, mask_deg=-90.0)
        self.assertEqual(pred[("C", 58)]["clk_sigma_s"], ge.POINTING_CLK_SIGMA_S)

    def test_decoded_left_out(self):
        sats = op.parse_sats("R25:32275:8,C58:40748")
        _, held = op.build(sats, NOW, 600.0, skip={("C", 58)})
        self.assertEqual(held, [("R", 25)])


if __name__ == "__main__":
    unittest.main()
