"""band_consensus on constructed fleets: one group, and two groups pinned apart.

    python test_elem_shared_ref.py
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elem_shared_ref as E  # noqa: E402

N_HALF = 48


def fleet(phases_deg, seed=1):
    """{(node, stage): get_elem_cal reply} for b2b instances whose shared models share one shape,
    each rotated by its own global phase (the pin), with a little noise."""
    rng = np.random.default_rng(seed)
    shape = rng.normal(size=N_HALF) + 1j * rng.normal(size=N_HALF)
    res = {}
    for i, p in enumerate(phases_deg):
        g = []
        for _ in (0, 1):
            v = shape * np.exp(1j * np.radians(p))
            v = v + 0.05 * (rng.normal(size=N_HALF) + 1j * rng.normal(size=N_HALF))
            g += [[float(x.real), float(x.imag)] for x in v]
        res[("cx%02d" % i, "gnss0_b2b_n2assemble")] = {
            "shared_warm": True,
            "g_shared": g,
        }
    return res


class TestBandConsensus(unittest.TestCase):
    def test_one_group(self):
        e = E.band_consensus(fleet([-5, 0, 3, 8, -2, 4]), "b2b", 30.0, 0.5)
        self.assertEqual(e["n"], [6, 6])
        self.assertEqual(e["excluded"], [{}, {}])

    def test_two_groups_keep_the_larger(self):
        # The 2026-10-20 b2b pol-1 split, eleven instances: six near -34 deg and five near
        # +42 deg. The plain mean lies between the groups, more than 30 deg from every instance.
        phases = [-29, -31, -30, -35, -37, -39, 37, 35, 38, 40, 61]
        e = E.band_consensus(fleet(phases), "b2b", 30.0, 0.5)
        self.assertEqual(e["n"], [6, 6])
        self.assertEqual(
            sorted(e["excluded"][0]), ["cx06/0", "cx07/0", "cx08/0", "cx09/0", "cx10/0"]
        )
        self.assertGreater(e["R"][0], 0.98)


if __name__ == "__main__":
    unittest.main()
