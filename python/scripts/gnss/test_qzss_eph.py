#!/usr/bin/env python3
"""QZSS records through parse_rinex_nav and predict_all, on the IGS BRDC records of 2026-11-25
10:00 and 11:00Z for J02 (QZS-2, health 1) and J09 (QZS-7, health 55).

Why each check exists: the qzss_l5 and qzss_l2c chains seed from this sky with no search, so the
records must be keyed by the PRN their code tables use (J09 = 201), J09 must sit where the array
saw its L5 signal that night (az 243, el 13-16 from DRAO), and the health gate must read the bit
of the chain's signal alone: every QZSS word is non-zero (J02's 1 is the L1C/B bit of a
satellite that sends L1C/A), and J09 has the L5 bit set while it transmits, so only the
exemption keeps it.
"""
import copy
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
    "brdc_qzss_20261125.rnx",
)
DRAO = (49.32075144444, -119.62081125, 545.0)
T = 1795602600.0  # 2026-11-25T10:30:00Z
QZS2, QZS7 = ("J", 194), ("J", 201)


class TestQzssEph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eph = ge.parse_rinex_nav([FIX])

    def test_records_are_keyed_by_prn(self):
        self.assertEqual(sorted(self.eph), [QZS2, QZS7])
        for key in (QZS2, QZS7):
            self.assertEqual(len(self.eph[key]), 2)
            for e in self.eph[key]:
                self.assertEqual((e["sys"], e["prn"]), key)
                # QZSST = GPST and the week is the GPS week: toe is the record's epoch
                self.assertEqual(e["toe_gpst"], e["toc_gpst"])
        self.assertEqual([e["health"] for e in self.eph[QZS2]], [1.0, 1.0])
        self.assertEqual([e["health"] for e in self.eph[QZS7]], [55.0, 55.0])

    def test_j09_is_where_the_array_saw_it(self):
        pa = ge.predict_all(self.eph, *DRAO, T, mask_deg=0.0)
        v = pa[QZS7]
        self.assertAlmostEqual(v["az"], 243.65, delta=0.1)
        self.assertAlmostEqual(v["el"], 15.95, delta=0.1)
        # near the geostationary range at this elevation
        self.assertAlmostEqual(v["range_m"], 40.055e6, delta=0.05e6)

    def test_gate_reads_the_l5_bit(self):
        pa = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0)
        self.assertIn(QZS2, pa)  # health 1: L5 clear
        eph = copy.deepcopy(self.eph)
        for e in eph[QZS2]:
            e["health"] = 1.0 + ge.QZSS_HEALTH_L5
        self.assertNotIn(QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0))
        self.assertIn(
            QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0, health_gate=False)
        )
        for e in eph[QZS2]:
            e["health"] = 59.0  # every bit but L5
        self.assertIn(QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0))

    def test_an_l2c_gate_reads_the_l2c_bit(self):
        for e in self.eph[QZS2]:
            self.assertEqual(int(e["health"]) & ge.QZSS_HEALTH_L2, 0)
        pa = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l2c")
        self.assertIn(QZS2, pa)
        eph = copy.deepcopy(self.eph)
        for e in eph[QZS2]:
            e["health"] = 1.0 + ge.QZSS_HEALTH_L2
        self.assertNotIn(
            QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l2c")
        )
        for sig in ("qzss_l5", None):
            self.assertIn(
                QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0, signal=sig)
            )
        for e in eph[QZS2]:
            e["health"] = 1.0 + ge.QZSS_HEALTH_L5
        self.assertIn(
            QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l2c")
        )
        self.assertNotIn(
            QZS2, ge.predict_all(eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l5")
        )

    def test_j09_passes_by_the_exemption_only(self):
        self.assertTrue(
            all(int(e["health"]) & ge.QZSS_HEALTH_L5 for e in self.eph[QZS7])
        )
        self.assertIn(QZS7, ge.predict_all(self.eph, *DRAO, T, mask_deg=0.0))
        saved = dict(ge.HEALTH_GATE_EXEMPT)
        try:
            ge.HEALTH_GATE_EXEMPT.clear()
            pa = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0)
            self.assertNotIn(QZS7, pa)
            self.assertIn(QZS2, pa)
        finally:
            ge.HEALTH_GATE_EXEMPT.update(saved)

    def test_l5_clock_takes_the_broadcast_tgd(self):
        bare = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0)
        l5 = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l5")
        for key in (QZS2, QZS7):
            tgd = ge.best_eph(self.eph[key], ge.gpst_of_utc(T))["tgd"]
            self.assertNotEqual(tgd, 0.0)
            self.assertAlmostEqual(l5[key]["tgd_s"], -tgd, delta=1e-15)
            self.assertAlmostEqual(
                l5[key]["sat_clk_s"] - bare[key]["sat_clk_s"], -tgd, delta=1e-13
            )

    def test_l2c_clock_takes_gamma_tgd(self):
        # IS-QZSS-PNT 5.8.1: ISC_L2C = (1 - gamma_12) TGD on the satellites that send L2C.
        bare = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0)
        l2c = ge.predict_all(self.eph, *DRAO, T, mask_deg=-90.0, signal="qzss_l2c")
        tgd = ge.best_eph(self.eph[QZS2], ge.gpst_of_utc(T))["tgd"]
        want = -ge.GAMMA_L1L2 * tgd
        self.assertAlmostEqual(l2c[QZS2]["tgd_s"], want, delta=1e-15)
        self.assertAlmostEqual(
            l2c[QZS2]["sat_clk_s"] - bare[QZS2]["sat_clk_s"], want, delta=1e-13
        )


if __name__ == "__main__":
    unittest.main()
