"""#70/#87: the q stall verdict must fire on a real collapse and stay quiet otherwise.

The regression it encodes is 2026-08-18: gal_e5b's q duty fell 0.44 -> 0.19 after the
RFI outage railed its C++ trims, and sat there for 3.5 h with nothing saying so.

Run: python3 -m unittest test_q_stall
"""
import unittest

from gnss_broker.fits import q_stall_verdict

W, FRAC, MINBEST = 600.0, 0.6, 0.25


def run(duties, cadence=13.0):
    """Feed a duty series one cycle at a time; return (best, [verdicts])."""
    hist, best, fired = [], None, []
    for i, d in enumerate(duties):
        t = i * cadence
        hist.append((t, d))
        best, v = q_stall_verdict(hist, t, W, FRAC, MINBEST, best)
        if v:
            fired.append((i, v))
    return best, fired


class TestQStall(unittest.TestCase):
    def test_fires_on_the_e5b_collapse(self):
        # 60 cycles healthy at 0.44, then the collapse to 0.19 (#87's actual numbers)
        best, fired = run([0.44] * 60 + [0.19] * 60)
        self.assertAlmostEqual(best, 0.44, places=6)
        self.assertTrue(fired, "the guard never fired on a 0.44 -> 0.19 collapse")
        cur, bst, frac = fired[0][1]
        self.assertLess(cur, 0.25)
        self.assertAlmostEqual(bst, 0.44, places=6)
        self.assertLess(frac, 0.6)

    def test_quiet_on_a_healthy_chain(self):
        # normal transit-driven wobble around 0.8 must never fire
        duties = [0.8, 0.72, 0.85, 0.78, 0.69, 0.83, 0.75, 0.81, 0.7, 0.79] * 12
        _, fired = run(duties)
        self.assertEqual(fired, [], "fired on ordinary variance -- it will be ignored")

    def test_exempts_a_structurally_low_chain(self):
        # bds_b2b class: best 0.20 < min_best, so even a collapse to 0 stays silent
        _, fired = run([0.20] * 60 + [0.02] * 60)
        self.assertEqual(fired, [])

    def test_baseline_only_rises(self):
        # a chain that degrades cannot lower its own bar and go quiet again
        best, fired = run([0.9] * 60 + [0.3] * 200)
        self.assertAlmostEqual(best, 0.9, places=6)
        self.assertTrue(len(fired) > 50, "the guard went quiet as the degradation persisted")

    def test_waits_for_a_full_window(self):
        _, fired = run([0.9, 0.1])          # 2 samples: no verdict either way
        self.assertEqual(fired, [])


if __name__ == "__main__":
    unittest.main()
