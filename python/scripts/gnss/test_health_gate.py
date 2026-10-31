#!/usr/bin/env python3
"""predict_all's health gate and HEALTH_GATE_EXEMPT, on the IGS BRDC records of 2026-10-31 00:00-00:20Z
for E14 (health 130 / 16, transmitting) and E15 (healthy).

Why each check exists: E14 passed 2 deg from boresight that night with no chain tracking it, because
the gate dropped it from the chains' sky; the exemption must keep it (and the fixture must place it
where the array saw it, or the exemption is about another satellite); and the gate must still drop
an unhealthy satellite that is not on the list.
"""
import copy
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge  # noqa: E402

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "scripts",
    "gnss",
    "fixtures",
    "brdc_gal_e14_20261031.rnx",
)
DRAO = (49.32075144444, -119.62081125, 545.0)
T_PASS = 1793407620.0  # 2026-10-31T00:47:00Z
BORE = (180.0, 62.7)  # az, el: the Tau A pointing


def sep_deg(az, el, az0, el0):
    a, e, a0, e0 = (math.radians(x) for x in (az, el, az0, el0))
    c = math.sin(e) * math.sin(e0) + math.cos(e) * math.cos(e0) * math.cos(a - a0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class TestHealthGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eph = ge.parse_rinex_nav([FIX])

    def test_fixture(self):
        self.assertIn(("E", 14), self.eph)
        self.assertIn(("E", 15), self.eph)
        self.assertTrue(all(e["health"] != 0 for e in self.eph[("E", 14)]))
        self.assertTrue(all(e["health"] == 0 for e in self.eph[("E", 15)]))

    def test_exempt_satellite_passes_the_gate(self):
        pa = ge.predict_all(self.eph, *DRAO, T_PASS, mask_deg=0.0)
        self.assertIn(("E", 14), pa)
        v = pa[("E", 14)]
        self.assertLess(sep_deg(v["az"], v["el"], *BORE), 2.5)
        saved = dict(ge.HEALTH_GATE_EXEMPT)
        try:
            ge.HEALTH_GATE_EXEMPT.clear()
            self.assertNotIn(
                ("E", 14), ge.predict_all(self.eph, *DRAO, T_PASS, mask_deg=0.0)
            )
        finally:
            ge.HEALTH_GATE_EXEMPT.update(saved)

    def test_unlisted_unhealthy_satellite_is_dropped(self):
        eph = copy.deepcopy(self.eph)
        for e in eph[("E", 15)]:
            e["health"] = 130.0
        pa = ge.predict_all(eph, *DRAO, T_PASS, mask_deg=-90.0)
        self.assertNotIn(("E", 15), pa)
        self.assertIn(("E", 14), pa)
        pa = ge.predict_all(eph, *DRAO, T_PASS, mask_deg=-90.0, health_gate=False)
        self.assertIn(("E", 15), pa)


if __name__ == "__main__":
    unittest.main()
