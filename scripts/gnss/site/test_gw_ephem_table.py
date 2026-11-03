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
            [cls.sat],
            cls.ts,
            T_PASS - 900.0,
            T_PASS + 900.0,
            {"DIGUI-79": (2027, -16.578e-6)},
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
            self.assertEqual(g["sat_clk_s"], 16.578e-6)  # -theta
        # outside the segment the satellite is not predicted
        far = ge.predict_all(
            eph, gp.SITE["lat"], gp.SITE["lon"], gp.SITE["h"], T_PASS + 3600.0
        )
        self.assertNotIn(("W", 2028), far)

    def test_a_shared_prn_has_one_satellite_at_a_time(self):
        """A second satellite on DIGUI-79's code, on the same track 120 s later: each keeps
        the half of the overlap where it is closer to the boresight."""
        a = self.records[0]
        b = dict(a, name="DIGUI-X", catnum=1)
        for k in ("t0_gpst", "t1_gpst", "toe_gpst"):
            b[k] = a[k] + 120.0
        other = dict(a, prn=7, g2_phase=6, name="DIGUI-Y", catnum=2)
        out, cut = gw.one_per_prn([b, a, other])
        self.assertEqual(cut, ["DIGUI-79", "DIGUI-X"])
        self.assertIn(other, out)
        first, second = [r for r in out if r["prn"] == 2028]
        self.assertEqual((first["name"], second["name"]), ("DIGUI-79", "DIGUI-X"))
        self.assertEqual(first["t0_gpst"], a["t0_gpst"])
        self.assertEqual(second["t1_gpst"], b["t1_gpst"])
        # the handover is half way between the two closest approaches, one step uncovered
        self.assertEqual(second["t0_gpst"] - first["t1_gpst"], a["dt"])
        self.assertLess(abs(first["t1_gpst"] - (a["toe_gpst"] + 60.0)), 20.0)
        # a trimmed record interpolates as the full one did
        t = second["t0_gpst"] + 23.0
        p, v = ge._gw_state(dict(second, sys="W"), t)
        pb, vb = ge._gw_state(dict(b, sys="W"), t)
        self.assertLess(math.dist(p, pb), 1e-6)
        self.assertLess(math.dist(v, vb), 1e-9)

    def test_segments_merge_and_pad(self):
        unix = list(range(0, 100, 10))
        sep = [30, 20, 10, 20, 30, 30, 20, 30, 30, 30]
        segs = gw.segments(unix, sep, 25.0, 5.0)
        self.assertEqual(
            [(s[0], s[1], s[2], s[3]) for s in segs],
            [(5, 35, 10, 20), (55, 65, 20, 60)],
        )
        self.assertEqual(len(gw.segments(unix, sep, 25.0, 20.0)), 1)


class TestDatabase(unittest.TestCase):
    def test_every_satellite_has_a_phase_and_an_epoch(self):
        db, unknown = gw.load_db()
        self.assertGreaterEqual(len(db), 13)
        self.assertEqual(db["DIGUI-79"][0], 2027)
        self.assertAlmostEqual(db["DIGUI-79"][1], -16.55e-6, delta=0.2e-6)
        self.assertTrue(all(0 <= ph < 2047 for ph, _ in db.values()))
        self.assertTrue(all(th is not None and abs(th) < 1e-3 for _, th in db.values()))
        self.assertAlmostEqual(unknown, -9.0e-6, delta=0.5e-6)

    def test_an_unmeasured_epoch_takes_the_median(self):
        ts, sat, _ = skyfield()
        records, _ = gw.build(
            [sat],
            ts,
            T_PASS - 900.0,
            T_PASS + 900.0,
            {"DIGUI-79": (2027, None)},
            theta_unknown_s=-9.01e-6,
        )
        self.assertEqual(records[0]["clk_s"], 9.01e-6)
        self.assertFalse(records[0]["theta_measured"])


if __name__ == "__main__":
    unittest.main()
