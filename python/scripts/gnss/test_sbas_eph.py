#!/usr/bin/env python3
"""SBAS (S) records in a RINEX 3 mixed-nav file: parsed, keyed by PRN (S31 -> 131), propagated as
the MT9 state vector, and seen from DRAO where the WAAS GEOs stand (#190 coverage).

Why each check exists: the key (a two-digit RINEX number is not a PRN); the geometry (a WAAS GEO
is geostationary at its assigned longitude, so a wrong unit or axis shows as a wrong radius or
longitude); the propagation (one record's state vector must land on the next record's position
to a few metres over the 4-minute cadence, else the acceleration term or the epoch is wrong);
and the age bound (a GEO record is good for tens of minutes, not the Keplerian default).
"""
import math
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge  # noqa: E402

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "gnss", "fixtures",
    "brdc_sbas_20261018.rnx",
)


class TestSBAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = ge.parse_rinex_nav([FIX])

    def test_keyed_by_prn_and_geostationary(self):
        """S31/S33/S35 -> PRNs 131/133/135 (WAAS: Eutelsat 117 West B, SES-15, Galaxy 30) at
        geostationary radius and their assigned longitudes, 117 W for PRN 131."""
        keys = sorted(k for k in self.recs if k[0] == "S")
        self.assertEqual([k[1] for k in keys], [131, 133, 135])
        e = self.recs[("S", 131)][0]
        pos, vel, clk = ge.sat_pos_clk(e, e["toe_gpst"])
        r = math.sqrt(sum(x * x for x in pos))
        lon = math.degrees(math.atan2(pos[1], pos[0]))
        self.assertAlmostEqual(r / 1e3, 42166.0, delta=30.0)
        self.assertAlmostEqual(lon, -117.0, delta=0.1)
        self.assertLess(abs(pos[2]), 20e3)  # on the equator to 20 km
        self.assertLess(math.sqrt(sum(v * v for v in vel)), 2.0)  # geostationary: < 2 m/s
        self.assertLess(abs(clk), 1e-6)

    def test_state_vector_propagates_onto_the_next_record(self):
        recs = self.recs[("S", 131)]
        self.assertGreater(len(recs), 5)
        worst = 0.0
        for a, b in zip(recs, recs[1:]):
            p, _, _ = ge.sat_pos_clk(a, b["toe_gpst"])
            worst = max(worst, math.sqrt(sum((p[i] - b["pos"][i]) ** 2 for i in range(3))))
        self.assertLess(worst, 5.0)  # metres, over the ~4 min cadence

    def test_seen_from_drao_due_south_at_33_deg(self):
        t = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
        pa = ge.predict_all(self.recs, 49.32, -119.62, 528.0, t, mask_deg=-90.0, health_gate=False)
        self.assertIn(("S", 131), pa)
        v = pa[("S", 131)]
        self.assertAlmostEqual(v["el"], 33.4, delta=0.5)
        self.assertAlmostEqual(v["az"], 176.5, delta=1.0)
        self.assertLess(abs(v["range_rate_mps"]), 1.0)

    def test_record_units_metres_or_km(self):
        """The rapid BRDC writes the GAGAN records (S27, S28) in metres, not RINEX's km: read as
        metres when the radius says so, dropped when it is no orbit at all."""
        e_km = self.recs[("S", 131)][0]
        af = (e_km["af0"], e_km["af1"], e_km["frame_time"])
        km = [c * 1e-3 for c in e_km["pos"]]
        v = [c * 1e-3 for c in e_km["vel"]]
        a = [c * 1e-3 for c in e_km["acc"]]
        orb_km = [km[0], v[0], a[0], 0.0, km[1], v[1], a[1], 2.0, km[2], v[2], a[2], 5.0]
        orb_m = [c * 1e3 if i % 4 != 3 else c for i, c in enumerate(orb_km)]
        e_m = ge._sbas_record(31, af, orb_m, e_km["toe_gpst"])
        self.assertIsNotNone(e_m)
        for i in range(3):
            self.assertAlmostEqual(e_m["pos"][i], e_km["pos"][i], places=3)
            self.assertAlmostEqual(e_m["vel"][i], e_km["vel"][i], places=6)
        self.assertIsNone(ge._sbas_record(31, af, [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0], e_km["toe_gpst"]))
        self.assertIsNone(ge._sbas_record(31, af, [7.0e4, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0], e_km["toe_gpst"]))

    def test_age_bound(self):
        """A record older than SBAS_MAX_AGE_S is not used for a clock-grade prediction."""
        recs = self.recs[("S", 131)]
        t_late = recs[-1]["toe_gpst"] + ge.SBAS_MAX_AGE_S + 60.0
        self.assertIsNone(ge.best_eph(recs, t_late, ge.SBAS_MAX_AGE_S))
        self.assertIsNotNone(ge.best_eph(recs, recs[-1]["toe_gpst"] + 600.0, ge.SBAS_MAX_AGE_S))


if __name__ == "__main__":
    unittest.main(verbosity=2)
