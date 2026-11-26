#!/usr/bin/env python3
"""gnss_sbas: WAAS (DFMC MT32) corrections to code residuals measured against the broadcast model.

* corrections() takes each GPS slot's newest MT32 across the SBAS satellites' payload entries.
* With the broker's own issue, a clock correction moves the residual by itself, and a position
  correction by its projection on the line of sight; the rates act from t_D.
* The issue is the one whose LNAV IODC the IODN names; a correction past MAX_AGE_S is not used.
* Corrector.apply names the satellites it has corrections for, and those waiting for an issue
  the BRDC does not hold yet, which it leaves out.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_ephemeris as ge  # noqa: E402
import gnss_has  # noqa: E402
import gnss_sbas  # noqa: E402
import nav_fields  # noqa: E402

T0 = 1795560000.0  # a UTC instant on 2026-11-24
G0 = ge.gpst_of_utc(T0)
RX = np.asarray(ge._ecef_of_llh(45.0, -75.0, 100.0))


def _record(iodc, toe):
    """A GPS LNAV Kepler record (circular, MEO), overhead-ish of the test point at T0."""
    return {
        "sys": "G", "prn": 5, "iode": iodc % 256, "iodc": iodc, "toe_gpst": toe, "toc_gpst": toe,
        "toe_sow": toe % 604800, "week": int(toe // 604800), "sqrta": 5153.6, "ecc": 0.0,
        "i0": 0.96, "omega0": 1.0, "omega": 0.0, "m0": 0.5, "dn": 0.0, "omegadot": 0.0,
        "idot": 0.0, "cuc": 0.0, "cus": 0.0, "crc": 0.0, "crs": 0.0, "cic": 0.0, "cis": 0.0,
        "af0": 1e-4, "af1": 0.0, "af2": 0.0, "tgd": 0.0, "health": 0,
    }  # fmt: skip


def _mt32(iodn=300, dr=(0.0, 0.0, 0.0), db=0.0, rates=(0.0, 0.0, 0.0, 0.0), t_d=None):
    """An MT32 as nav_fields reads it."""
    t_d = (G0 % 86400.0) // 16 * 16 if t_d is None else t_d
    vals = (5, iodn) + tuple(dr) + (db,) + tuple(rates) + (t_d,)
    return dict(zip(gnss_sbas.FIELDS, vals), t=T0, geo=135)


class TestSbas(unittest.TestCase):
    def setUp(self):
        self.eph = {("G", 5): [_record(300, G0 - 600.0)]}
        self.m = {"sys": "G", "prn": 5, "group": "gps_l5", "t": T0, "resid_m": 0.0}

    def test_clock(self):
        d = gnss_sbas.correction_m(self.m, _mt32(db=1.5), self.eph, RX)
        self.assertAlmostEqual(d, 1.5, places=6)

    def test_position_on_line_of_sight(self):
        _, _, u, _, _ = gnss_has._geometry(
            gnss_has._brdc_state(self.eph[("G", 5)][0]), RX, G0, "G"
        )
        dr = np.array([3.0, -2.0, 1.0])
        d = gnss_sbas.correction_m(self.m, _mt32(dr=dr), self.eph, RX)
        self.assertAlmostEqual(d, -float(u @ dr), places=5)

    def test_rate_from_t_d(self):
        t_d = (G0 % 86400.0) // 16 * 16 - 32.0
        d = gnss_sbas.correction_m(
            self.m, _mt32(rates=(0, 0, 0, 0.01), t_d=t_d), self.eph, RX
        )
        self.assertAlmostEqual(d, 0.01 * (G0 % 86400.0 - t_d), places=5)

    def test_issue_and_age(self):
        self.assertIsNone(gnss_sbas.correction_m(self.m, _mt32(iodn=301), self.eph, RX))
        old = dict(_mt32(), t=T0 - gnss_sbas.MAX_AGE_S - 1.0)
        self.assertIsNone(gnss_sbas.correction_m(self.m, old, self.eph, RX))

    def test_waiting(self):
        c = gnss_sbas.Corrector("unused")
        c.eph = lambda: self.eph
        corr = {5: _mt32(iodn=300), 7: _mt32(iodn=999)}
        meas = [self.m, dict(self.m, prn=7), dict(self.m, prn=9)]
        self.eph[("G", 7)] = [_record(300, G0 - 600.0)]
        orig = gnss_sbas.corrections
        gnss_sbas.corrections = lambda glob: corr
        try:
            out = c.apply(meas, 45.0, -75.0, 100.0)
        finally:
            gnss_sbas.corrections = orig
        self.assertEqual([m["prn"] for m in out], [5])
        self.assertEqual((c.named, c.waiting), ({"G05", "G07"}, {"G07"}))

    def test_corrections_newest_slot(self):
        def entry(t, slot, iodn):
            bits = [0] * 250
            for start, width, v in ((11, 8, slot), (19, 10, iodn), (107, 13, 4000)):
                for k in range(width):
                    bits[start + k] = (v >> (width - 1 - k)) & 1
            title, rows = nav_fields.parse("SBAS_L5_DFMC", "type", 32, bits)
            return [None, t, title, rows]

        doc = {"signals": {"SBAS_L5_DFMC": {"prns": {
            "133": {"msgs": {"type 32 slot 5": entry(100.0, 5, 11)}},
            "135": {"msgs": {"type 32 slot 5": entry(200.0, 5, 22),
                             "type 32 slot 7": entry(150.0, 7, 33)}},
        }}}}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "nav_payload_d1.json"), "w") as f:
                json.dump(doc, f)
            c = gnss_sbas.corrections(os.path.join(d, "nav_payload_*.json"))
        self.assertEqual(sorted(c), [5, 7])
        self.assertEqual((c[5]["IODN"], c[5]["geo"], c[5]["t_D"]), (22, 135, 64000))
        self.assertEqual(c[7]["IODN"], 33)


if __name__ == "__main__":
    unittest.main()
