#!/usr/bin/env python3
"""gnss_oddsat_rinex: a decoded GLONASS ephemeris through RINEX 3.04 and back.

* gnss_ephemeris.parse_rinex_nav reads the record with the decoded state, clock, health and age
  to the printed precision, the frequency channel given, and tb as its epoch.
* The frame time is t_k as seconds of the UTC week.
* Only the satellites named, one file per UTC day; an unchanged pass rewrites nothing.
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_ephemeris as ge  # noqa: E402
import gnss_oddsat_rinex as ox  # noqa: E402

# COSMOS 2433 on slot 25, decoded 2026-11-27 (tb 19:15 UTC, t_k 22:06 Moscow).
EPH = {
    "sys": "R", "prn": 25, "af0": -0.00012090709060430527, "af1": -9.094947017729282e-13,
    "frame_time": 79560.0, "pos": [7919516.6015625, 7056113.28125, 23204215.8203125],
    "vel": [-2249.0158081054688, 2246.273994445801, 83.63628387451172],
    "acc": [-1.862645149230957e-06, 3.725290298461914e-06, -9.313225746154785e-07],
    "health": 1.0, "k": None, "age": 0.0, "toc_gpst": 1479842118.0, "toe_gpst": 1479842118.0,
    "tb": 89, "slot": 25, "dtau": -5.587935447692871e-09, "mod": 1,
}  # fmt: skip


class TestRinex(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        other = dict(EPH, prn=24)
        rows = [
            {
                "sys": "R",
                "prn": 25,
                "signal": "GLO_L2OF",
                "toe_field": "toe_gpst",
                "eph": EPH,
            },
            {
                "sys": "R",
                "prn": 24,
                "signal": "GLO_L2OF",
                "toe_field": "toe_gpst",
                "eph": other,
            },
        ]
        with open(os.path.join(self.d.name, "GLO_L2OF.json"), "w") as f:
            json.dump({"schema": 1, "eph": rows}, f)
        self.sats = ox.parse_sats("R25:8")
        self.now = ox._utc(EPH["toe_gpst"]) + 600.0
        self.outbox = os.path.join(self.d.name, "out")

    def tearDown(self):
        self.d.cleanup()

    def test_round_trip(self):
        files = ox.day_files(ox.records([self.d.name], self.sats), self.sats, self.now)
        self.assertEqual(list(files), ["gx10_decoded_2026331.rnx"])
        self.assertEqual(
            ox.write_changed(self.outbox, files), ["gx10_decoded_2026331.rnx"]
        )
        eph = ge.parse_rinex_nav(os.path.join(self.outbox, "gx10_decoded_2026331.rnx"))
        self.assertEqual(list(eph), [("R", 25)])
        (e,) = eph[("R", 25)]
        self.assertAlmostEqual(e["toe_gpst"], EPH["toe_gpst"], places=3)
        for key in ("pos", "vel", "acc"):
            for a, b in zip(e[key], EPH[key]):
                self.assertAlmostEqual(a, b, delta=abs(b) * 1e-11 + 1e-12, msg=key)
        self.assertAlmostEqual(e["af0"], EPH["af0"], delta=1e-16)
        self.assertAlmostEqual(e["af1"], EPH["af1"], delta=1e-24)
        self.assertEqual((e["k"], e["health"], e["age"]), (8, 1.0, 0.0))
        # t_k 22:06 Moscow = 19:06 UTC on Friday 11-27: day 5 of the UTC week
        self.assertEqual(e["frame_time"], 5 * 86400.0 + 19 * 3600.0 + 6 * 60.0)

    def test_unchanged_not_rewritten(self):
        files = ox.day_files(ox.records([self.d.name], self.sats), self.sats, self.now)
        ox.write_changed(self.outbox, files)
        again = ox.day_files(
            ox.records([self.d.name], self.sats), self.sats, self.now + 60
        )
        self.assertEqual(ox.write_changed(self.outbox, again), [])
        # a later copy of the same issue (t_k one frame on) is not a new record
        later = {(k[0], k[1], k[2]): dict(e, frame_time=e["frame_time"] + 30.0)
                 for k, e in ox.records([self.d.name], self.sats).items()}  # fmt: skip
        self.assertEqual(
            ox.write_changed(self.outbox, ox.day_files(later, self.sats, self.now)), []
        )
        newer = dict(EPH, toe_gpst=EPH["toe_gpst"] + 1800.0)
        recs = dict(later, **{("R", 25, newer["toe_gpst"]): newer})
        self.assertEqual(
            ox.write_changed(self.outbox, ox.day_files(recs, self.sats, self.now)),
            ["gx10_decoded_2026331.rnx"],
        )


if __name__ == "__main__":
    unittest.main()
