#!/usr/bin/env python3
"""Tests for inst_phase_table on a synthetic calibration.

    /home/kvand/gnss/venv/bin/python3 test_inst_phase_table.py

The table holds each input's instrumental phase with the geometry toward the calibrator taken
off, so carrying a vector between two channels by it gives the vector the instrument would have
measured there. Turned to another F-engine start it adds 2 pi f n Ts per input, and it refuses
a start whose steps are not measured.
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fengine_steps  # noqa: E402
import inst_phase_table as ipt  # noqa: E402

N = 8
K = np.arange(4096, 4096 + 600)  # 1100.0-1217.0 MHz
F = (K + 1536) * ipt.CH_HZ


def synthetic_cal(rng):
    """Instrumental phase = a delay per input plus a slow ripple, times the geometric phase toward
    the calibrator at transit (dec 22 deg)."""
    pos = rng.uniform(-30, 30, (N, 3))
    pos[:, 2] = 0.0
    dec = math.radians(22.0)
    geo = pos @ ipt.enu(0.0, dec) / ipt.C
    tau = rng.uniform(-3e-9, 3e-9, N)
    ripple = np.radians(8.0) * np.sin(
        2 * np.pi * F[:, None] / 40e6 + rng.uniform(0, 6.3, N)
    )
    inst = 2 * np.pi * F[:, None] * tau[None, :] + ripple
    G = np.exp(1j * (inst + 2 * np.pi * np.outer(F, geo)))
    cal = {
        "G": G.astype(np.complex64),
        "ok_g": np.ones(G.shape, bool),
        "kept": np.ones((len(K), 2), bool),
        "k": K,
        "pos": pos,
        "dec": dec,
        "ha_on": 0.0,
        "t_tr": 1795600000.0,
    }
    return cal, inst


class TestTable(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(246)
        self.cal, self.inst = synthetic_cal(self.rng)
        self.d = tempfile.mkdtemp()
        self.frame0 = os.path.join(self.d, "frame0.json")
        with open(self.frame0, "w") as fh:
            json.dump([{"frame0_int": 1795500000}, {"frame0_int": 1795700000}], fh)
        self.tab = ipt.build(self.cal, self.frame0, 1100.0, 1217.0)

    def test_knots_and_start(self):
        self.assertEqual(self.tab["made_frame0"], 1795500000)
        df = np.diff(self.tab["fids"]) * ipt.CH_HZ / 1e6
        self.assertTrue((df >= ipt.KNOT_MHZ).all())

    def test_carry_gives_the_instrument_at_the_target(self):
        for src, dst in ((100, 160), (300, 120), (50, 550)):
            v = np.exp(1j * self.inst[src])
            got = ipt.carry(self.tab, v, int(K[src] + 1536), int(K[dst] + 1536))
            err = np.angle(got * np.exp(-1j * self.inst[dst]))
            err -= np.angle(np.mean(np.exp(1j * err)))
            self.assertLess(np.degrees(np.abs(err)).max(), 1.5)

    def test_turned_adds_the_steps(self):
        steps = [0, 4, -3, 0, 0, 1, 0, 0]
        table = {"adc_sample_ns": 0.3125, "n_inputs": N, "rows": []}
        fengine_steps.add_start(table, 1795500000, "a")
        fengine_steps.add_start(table, 1795700000, "b")
        with self.assertRaises(ValueError):
            ipt.turned(self.tab, table, 1795700000)
        table["rows"][-1]["steps"] = steps
        with self.assertRaises(ValueError):  # the half_step is needed too
            ipt.turned(self.tab, table, 1795700000)
        table["rows"][-1]["half_step"] = 2
        out = ipt.turned(self.tab, table, 1795700000)
        self.assertEqual(
            (out["made_frame0"], out["turned_from"]), (1795700000, 1795500000)
        )
        fid = out["fids"][10]
        want = [s + (2 if i >= N // 2 else 0) for i, s in enumerate(steps)]
        d = ipt.phase_at(out, fid) - ipt.phase_at(self.tab, fid)
        turn = 2 * np.pi * fid * ipt.CH_HZ * np.array(want) * 0.3125e-9
        self.assertLess(np.abs(np.angle(np.exp(1j * (d - turn)))).max(), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
