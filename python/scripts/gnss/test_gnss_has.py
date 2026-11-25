#!/usr/bin/env python3
"""gnss_has: HAS corrections to code residuals measured against the broadcast model.

* A message's time of hour lands in the hour it was recovered in, or the one before.
* The orbit correction's radial, in-track and cross-track axes are orthonormal, radial
  outward and in-track along the velocity.
* With the broker's own issue, a clock correction and a code bias move the residual by
  themselves, and a radial correction by its projection on the line of sight.
* The clock message must be of the orbit message's set; an older orbit set with its own
  clock message is used when the newest has none.
* An IOD seen days ago is not the issue HAS names.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_ephemeris as ge  # noqa: E402
import gnss_has  # noqa: E402

T0 = 1795560000.0  # a UTC instant on 2026-11-24
G0 = ge.gpst_of_utc(T0)


def _record(iod, toe):
    """A Galileo I/NAV Kepler record (circular, MEO)."""
    return {
        "sys": "E", "prn": 4, "iode": iod, "toe_gpst": toe, "toc_gpst": toe, "toe_sow": toe % 604800,
        "week": int(toe // 604800), "sqrta": 5440.6, "ecc": 0.0, "i0": 0.97, "omega0": 1.0,
        "omega": 0.0, "m0": 0.5, "dn": 0.0, "omegadot": 0.0, "idot": 0.0, "cuc": 0.0, "cus": 0.0,
        "crc": 0.0, "crs": 0.0, "cic": 0.0, "cis": 0.0, "af0": 1e-4, "af1": 0.0, "af2": 0.0,
        "tgd": 0.0, "iodc": 0.0, "l2_codes": 0x205, "health": 0,
    }  # fmt: skip


def _msgs(orb=None, dclk=0.0, cb=0.0, iod=85, sets=((4, 1),), clock_sets=None):
    """corrections() as has_predictor gives them, the times at T0."""
    toh = G0 % 3600.0
    orb = orb or {"radial_m": 0.0, "in_track_m": 0.0, "cross_track_m": 0.0}
    out = {"orbit": [], "clock": [], "bias": []}
    for mask, iod_set in sets:
        out["orbit"].append(
            {
                "toh": toh,
                "t_rx": T0,
                "mask_id": mask,
                "iod_set": iod_set,
                "orbit_validity_s": 300,
                "sats": {"E04": {"orbit": dict(orb, iod=iod)}},
            }
        )
        out["bias"].append(
            {
                "toh": toh,
                "t_rx": T0,
                "mask_id": mask,
                "iod_set": iod_set,
                "code_bias_validity_s": 300,
                "sats": {"E04": {"code_bias_m": {"E6-C": cb}}},
            }
        )
    for mask, iod_set in clock_sets or sets:
        out["clock"].append(
            {
                "toh": toh,
                "t_rx": T0,
                "mask_id": mask,
                "iod_set": iod_set,
                "clock_validity_s": 60,
                "sats": {"E04": {"clock_m": dclk}},
            }
        )
    return out  # fmt: skip


def _corr(has, recs):
    lat, lon, alt = 45.0, -75.0, 100.0
    rx = np.asarray(ge._ecef_of_llh(lat, lon, alt))
    m = {"sys": "E", "prn": 4, "group": "gal_e6", "t": T0, "resid_m": 0.0}
    return gnss_has.correction_m(m, has, {("E", 4): recs}, [], rx)


class TestHas(unittest.TestCase):
    def test_message_time(self):
        self.assertEqual(gnss_has.message_gpst(G0 % 3600.0, T0), G0)
        self.assertEqual(
            gnss_has.message_gpst((G0 + 600.0) % 3600.0, T0), G0 + 600.0 - 3600.0
        )

    def test_axes(self):
        pos, vel = np.array([2.6e7, 1e6, 3e5]), np.array([-100.0, 3000.0, 1500.0])
        x = [gnss_has.orbit_offset_m(pos, vel, {"radial_m": r, "in_track_m": a, "cross_track_m": c})
             for r, a, c in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]  # fmt: skip
        self.assertTrue(np.allclose(np.array(x) @ np.array(x).T, np.eye(3)))
        self.assertGreater(x[0] @ pos, 0.99 * np.linalg.norm(pos))
        self.assertAlmostEqual(x[1] @ vel, np.linalg.norm(vel))

    def test_terms_move_the_residual(self):
        recs = [_record(85, G0 - 600.0)]
        self.assertAlmostEqual(_corr(_msgs(), recs), 0.0, places=6)
        self.assertAlmostEqual(_corr(_msgs(dclk=0.3), recs), 0.3, places=6)
        self.assertAlmostEqual(_corr(_msgs(cb=1.2), recs), 1.2, places=6)
        d = _corr(
            _msgs(orb={"radial_m": 1.0, "in_track_m": 0.0, "cross_track_m": 0.0}), recs
        )
        self.assertLess(d, -0.9)  # the satellite moved out: the HAS range is longer

    def test_sets_pair(self):
        recs = [_record(85, G0 - 600.0)]
        self.assertIsNone(_corr(_msgs(dclk=0.3, clock_sets=((4, 2),)), recs))
        has = _msgs(dclk=0.3, sets=((4, 2), (4, 1)), clock_sets=((4, 1),))
        self.assertAlmostEqual(_corr(has, recs), 0.3, places=6)

    def test_old_iod_is_not_the_issue(self):
        recs = [_record(88, G0 - 2 * 86400.0), _record(85, G0 - 600.0)]
        self.assertIsNone(_corr(_msgs(iod=88), recs))


if __name__ == "__main__":
    unittest.main()
