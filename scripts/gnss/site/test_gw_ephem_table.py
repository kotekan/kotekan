#!/usr/bin/env python3
"""gw_ephem_table.py (TLE -> table) and gnss_ephemeris's GuoWang records (table -> predict_all),
against skyfield's SGP4 directly, on DIGUI-79's pass of 2026-11-02 18:00Z (beam-dump window 4)."""

import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gw_ephem_table as gw  # noqa: E402

import gnss_ephemeris as ge  # noqa: E402  (on the path via gw_ephem_table)
import guowang_passes as gp  # noqa: E402

TLE = (
    "HULIANWANG DIGUI-79",
    "1 65311U 25187G   26306.12552535 -.00000115  00000+0  19775-4 0  9994",
    "2 65311  50.0042  71.4246 0000980 153.3883 206.7040 13.29324674 58048",
)
T_PASS = 1793642447.0  # 2026-11-02 18:00:47Z, DIGUI-79's closest approach (0.64 deg)


def skyfield():
    try:
        from skyfield.api import EarthSatellite, load, wgs84
    except ImportError:
        sys.exit(77)
    ts = load.timescale(builtin=True)
    sat = EarthSatellite(TLE[1], TLE[2], TLE[0], ts)
    site = wgs84.latlon(gp.SITE["lat"], gp.SITE["lon"], elevation_m=gp.SITE["h"])
    return ts, sat, site


class TestTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts, cls.sat, cls.site = skyfield()
        cls.records, cls.unknown = gw.build(
            [cls.sat], cls.ts, T_PASS - 900.0, T_PASS + 900.0, {"DIGUI-79": 2027}
        )

    def test_one_segment_around_the_pass(self):
        self.assertEqual(len(self.records), 1)
        r = self.records[0]
        self.assertEqual((r["prn"], r["name"], r["catnum"]), (2028, "DIGUI-79", 65311))
        self.assertLess(r["min_sep_deg"], 1.0)
        self.assertLess(r["t0_gpst"], ge.gpst_of_utc(T_PASS))
        self.assertLess(ge.gpst_of_utc(T_PASS), r["t1_gpst"])
        self.assertEqual(self.unknown, [])

    def test_unknown_satellites_are_listed_not_tabled(self):
        records, unknown = gw.build(
            [self.sat], self.ts, T_PASS - 900.0, T_PASS + 900.0, {}
        )
        self.assertEqual(records, [])
        self.assertEqual([u["name"] for u in unknown], ["DIGUI-79"])

    def test_interpolation_is_sgp4_off_the_grid(self):
        from skyfield.framelib import itrs

        r = self.records[0]
        worst_p = worst_v = 0.0
        t = r["t0_gpst"] + 3.7
        while t < r["t1_gpst"]:
            p, v = ge._gw_state(dict(r, sys="W"), t)
            unix = t - ge.GPS_UTC_LEAP + ge.GPS_EPOCH.timestamp()
            pp, vv = self.sat.at(gp.ts_unix(self.ts, unix)).frame_xyz_and_velocity(itrs)
            worst_p = max(worst_p, math.dist(p, pp.m))
            worst_v = max(worst_v, math.dist(v, vv.m_per_s))
            t += 7.3
        self.assertLess(worst_p, 0.5)  # m
        self.assertLess(worst_v, 0.02)  # m/s: 0.08 Hz at 1202 MHz

    def test_predict_all_against_skyfield(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gw.json")
            with open(path, "w") as fh:
                json.dump(dict(records=self.records), fh)
            eph = ge.load_gw_table(path)
        self.assertEqual(list(eph), [("W", 2028)])
        for dt in (-60.0, -17.3, 0.0, 41.9):
            unix = T_PASS + dt
            got = ge.predict_all(
                eph, gp.SITE["lat"], gp.SITE["lon"], gp.SITE["h"], unix
            )
            g = got[("W", 2028)]
            # predict_all evaluates the satellite at the transmit time (light time ~4 ms, which
            # moves the range rate by ~0.15 m/s at this pass's acceleration): so does this
            # reference. Sagnac (~2 m) is the remaining difference.
            tx = unix - g["range_m"] / ge.C_LIGHT
            topo = (self.sat - self.site).at(gp.ts_unix(self.ts, tx))
            alt, az, dist, _, _, rr = topo.frame_latlon_and_rates(self.site)
            self.assertAlmostEqual(g["el"], alt.degrees, delta=0.01)
            self.assertAlmostEqual(g["range_m"], dist.m, delta=5.0)
            self.assertAlmostEqual(g["range_rate_mps"], rr.m_per_s, delta=0.03)
            self.assertEqual(g["sat_clk_s"], 0.0)
        # outside the segment the satellite is not predicted
        far = ge.predict_all(
            eph, gp.SITE["lat"], gp.SITE["lon"], gp.SITE["h"], T_PASS + 3600.0
        )
        self.assertNotIn(("W", 2028), far)

    def test_segments_merge_and_pad(self):
        unix = list(range(0, 100, 10))
        sep = [30, 20, 10, 20, 30, 30, 20, 30, 30, 30]
        segs = gw.segments(unix, sep, 25.0, 5.0)
        self.assertEqual(
            [(s[0], s[1], s[2], s[3]) for s in segs],
            [(5, 35, 10, 20), (55, 65, 20, 60)],
        )
        self.assertEqual(len(gw.segments(unix, sep, 25.0, 20.0)), 1)


if __name__ == "__main__":
    unittest.main()
