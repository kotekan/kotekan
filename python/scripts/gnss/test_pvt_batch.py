#!/usr/bin/env python3
"""gnss_pvt.batch_solve: a stationary receiver's position over a window from a sparse, fast
constellation.

  * Two low satellites sweeping across the sky for 10 min recover a known offset and the
    clock's offset and rate, within their reported errors, with no epoch holding 4 satellites.
  * A gross outlier is rejected.
  * A satellite whose clock is off by a constant through its pass is excluded, and the others
    give the position; good satellites are not excluded.
  * Under 120 s or one satellite: no answer.
  * rows_from_obs de-wraps a residual across a code-period boundary.
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_pvt  # noqa: E402

LAT, LON, ALT = 45.0, -75.0, 100.0  # a test point


def _pass(prn, az0, az1, el_peak, t0, dur, dx, c0, c1, rng, noise=3.0, step=10.0):
    rows = []
    for t in np.arange(t0, t0 + dur, step):
        f = (t - t0) / dur
        az = az0 + (az1 - az0) * f
        el = max(5.0, el_peak * math.sin(math.pi * f))
        los = gnss_pvt._los_enu(az, el)
        res = -los @ dx + c0 + c1 * (t - 300.0) + noise * rng.standard_normal()
        rows.append(
            {"t": float(t), "prn": prn, "az": az, "el": el, "resid_m": float(res)}
        )
    return rows


class TestBatch(unittest.TestCase):
    def test_two_passes(self):
        rng = np.random.default_rng(4)
        dx = np.array([3.0, -2.0, 5.0])
        rows = _pass(1, 200, 20, 70, 0, 600, dx, 1200.0, 0.05, rng)
        rows += _pass(2, 120, 300, 40, 0, 600, dx, 1200.0, 0.05, rng)
        r = gnss_pvt.batch_solve(rows, LAT, LON, ALT, min_el_deg=10.0)
        self.assertIsNotNone(r)
        for k, v in (("d_e", 3.0), ("d_n", -2.0), ("d_u", 5.0)):
            self.assertLess(abs(r[k] - v), 3 * r["sigma_" + k[-1]] + 0.5, (k, r[k], r))
        self.assertAlmostEqual(r["clock_rate_m_s"], 0.05, delta=0.02)
        self.assertEqual(r["n_sats"], 2)

    def test_outlier(self):
        rng = np.random.default_rng(5)
        dx = np.array([1.0, 1.0, 1.0])
        rows = _pass(1, 200, 20, 70, 0, 600, dx, 0.0, 0.0, rng)
        rows += _pass(2, 120, 300, 40, 0, 600, dx, 0.0, 0.0, rng)
        rows[10]["resid_m"] += 5000.0
        r = gnss_pvt.batch_solve(rows, LAT, LON, ALT)
        self.assertGreaterEqual(r["n_rejected"], 1)
        self.assertLess(abs(r["d_u"] - 1.0), 3 * r["sigma_u"] + 0.5)

    def _four(self, rng, dx, bias):
        rows = _pass(1, 200, 20, 70, 0, 600, dx, 0.0, 0.0, rng)
        rows += _pass(2, 120, 300, 40, 0, 600, dx, 0.0, 0.0, rng)
        rows += _pass(3, 330, 160, 30, 100, 400, dx, 0.0, 0.0, rng)
        bad = _pass(4, 60, 250, 55, 50, 500, dx, 0.0, 0.0, rng)
        for r in bad:
            r["resid_m"] += bias
        return rows + bad

    def test_biased_satellite(self):
        dx = np.array([2.0, -3.0, 4.0])
        for seed, bias in ((7, 1280.0), (8, -440.0)):
            r = gnss_pvt.batch_solve(
                self._four(np.random.default_rng(seed), dx, bias), LAT, LON, ALT
            )
            self.assertEqual(list(r["excluded"]), [4], (bias, r["excluded"]))
            json.dumps(r["excluded"])
            self.assertAlmostEqual(r["excluded"][4], bias, delta=30.0)
            self.assertEqual(r["n_sats"], 3)
            for k, v in zip(("d_e", "d_n", "d_u"), dx):
                self.assertLess(
                    abs(r[k] - v), 3 * r["sigma_" + k[-1]] + 0.5, (k, r[k], r)
                )

    def test_no_false_exclusion(self):
        for seed in range(9, 15):
            rows = self._four(
                np.random.default_rng(seed), np.array([1.0, 1.0, 1.0]), 0.0
            )
            r = gnss_pvt.batch_solve(rows, LAT, LON, ALT)
            self.assertEqual(r["excluded"], {}, seed)
            self.assertEqual(r["n_sats"], 4)

    def test_refusals(self):
        rng = np.random.default_rng(6)
        dx = np.zeros(3)
        short = _pass(1, 200, 20, 70, 0, 100, dx, 0, 0, rng) + _pass(
            2, 120, 300, 40, 0, 100, dx, 0, 0, rng
        )
        self.assertIsNone(gnss_pvt.batch_solve(short, LAT, LON, ALT))
        one = _pass(1, 200, 20, 70, 0, 600, dx, 0, 0, rng)
        self.assertIsNone(gnss_pvt.batch_solve(one, LAT, LON, ALT))

    def test_rows_dewrap(self):
        L = 1e-3 * gnss_pvt.C_LIGHT
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gw_1202_x.jsonl")
            with open(p, "w") as f:
                for i, res in enumerate((L / 2 - 10.0, L / 2 + 10.0 - L, L / 2 - 5.0)):
                    f.write(
                        json.dumps(
                            {
                                "t": 1000.0 + 20 * i,
                                "sys": "W",
                                "prn": 7,
                                "band": "gw_1202",
                                "az": 10.0,
                                "el": 40.0,
                                "code_resid_m": res,
                                "code_len": 2046,
                                "chip_rate_hz": 2.046e6,
                            }
                        )
                        + "\n"
                    )
            rows = gnss_pvt.rows_from_obs(
                [os.path.join(d, "*.jsonl")], 900.0, 1100.0, "W"
            )
        self.assertEqual([round(r["resid_m"] - L / 2) for r in rows], [-10, 10, -5])


if __name__ == "__main__":
    unittest.main()
