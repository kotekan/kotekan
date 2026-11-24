"""gnss_fix: the Doppler and code stages converge from no prior and from a far one.

    python3 -m unittest test_gnss_fix

Synthetic sky: satellites on circular 26,560 km orbits (sat_pos_clk patched), range rates from
their velocities, and code ranges known only mod T per chain with a clock per chain.
"""

import math
import unittest

import numpy as np

import gnss_fix as F
import gnss_pvt

R_ORBIT = 26560e3
W_ORBIT = 2 * math.pi / 43082.0
TRUTH = np.array(gnss_pvt._llh_to_ecef(43.0, -79.0, 200.0))


def fake_sat_pos_clk(e, t):
    """A circular orbit: phase e['ph'] in a plane inclined e['inc'] with node e['node']."""
    a = e["ph"] + W_ORBIT * t
    ci, si = math.cos(e["inc"]), math.sin(e["inc"])
    cn, sn = math.cos(e["node"]), math.sin(e["node"])
    x, y, z = math.cos(a), math.sin(a) * ci, math.sin(a) * si
    dx, dy, dz = -math.sin(a), math.cos(a) * ci, math.cos(a) * si
    p = R_ORBIT * np.array([x * cn - y * sn, x * sn + y * cn, z])
    v = R_ORBIT * W_ORBIT * np.array([dx * cn - dy * sn, dx * sn + dy * cn, dz])
    return p, v, 0.0


def sky(t):
    """The records above 10 deg from TRUTH at t: six planes of ten."""
    up = TRUTH / np.linalg.norm(TRUTH)
    out = []
    for k in range(60):
        e = {
            "sys": "G",
            "node": 2 * math.pi * (k // 10) / 6,
            "ph": 2 * math.pi * (k % 10) / 10 + 0.3 * (k // 10),
            "inc": math.radians(55),
        }
        p, _, _ = fake_sat_pos_clk(e, t)
        d = p - TRUTH
        if (d / np.linalg.norm(d)) @ up > math.sin(math.radians(10)):
            out.append(e)
    return out


class TestFix(unittest.TestCase):
    def setUp(self):
        self._real = F.ge.sat_pos_clk
        F.ge.sat_pos_clk = fake_sat_pos_clk
        F.gnss_atmos.tropo_delay_m = lambda el, lat, h: 0.0

    def tearDown(self):
        F.ge.sat_pos_clk = self._real

    def measurements(self, t0=1.4e9):
        obs, meas = [], []
        clocks = {"a": 3.2e-4, "b": -7.7e-3}
        periods = {"a": 0.02, "b": 0.1}
        for i, e in enumerate(sky(t0)):
            for dt in range(0, 60, 5):
                obs.append((e, t0 - dt, F.geometry(e, TRUTH, t0 - dt)[1] + 0.05))
            ch = "a" if i % 2 else "b"
            rng = F.geometry(e, TRUTH, t0)[0]
            meas.append(
                (
                    ch,
                    e,
                    t0,
                    (rng / F.C - clocks[ch]) % periods[ch],
                    periods[ch],
                )
            )
        return obs, meas

    def test_no_prior(self):
        obs, meas = self.measurements()
        self.assertGreaterEqual(len(meas), 6)
        u = np.mean([F.ge.sat_pos_clk(e, t)[0] for e, t, _ in obs], axis=0)
        start = F.R_EARTH * u / np.linalg.norm(u)
        rx, beta, rms, _, _ = F.doppler_fix(obs, start)
        self.assertLess(np.linalg.norm(rx - TRUTH), 1.0)
        self.assertAlmostEqual(beta, 0.05, places=3)
        fix = F.code_fix(meas, rx)
        self.assertLess(np.linalg.norm(fix[0] - TRUTH), 0.01)

    def test_far_prior(self):
        _, meas = self.measurements()
        enu = gnss_pvt._enu_axes(43.0, -79.0)
        prior = TRUTH + enu.T @ np.array([2.1e6, -2.1e6, 0.0])
        fix = F.code_fix(meas, prior, sigma0_m=3.5e6)
        self.assertLess(np.linalg.norm(fix[0] - TRUTH), 0.01)

    def test_too_few(self):
        _, meas = self.measurements()
        self.assertIsNone(F.code_fix(meas[:4], TRUTH))


if __name__ == "__main__":
    unittest.main()
