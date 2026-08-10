#!/usr/bin/env python3
"""The joint filter diverges: P collapses and the state free-runs (task #33).

OBSERVED ON SKY 2026-08-10 22:22-22:45 UTC, within 90 s of a clean start:
    22:22:56  clk +149.330 +-1.826  rate   +0.0000 c/s  upd=4   rej=0
    22:23:28  clk +526.922 +-0.000  rate  +13.4549 c/s  upd=22  rej=11
    22:24:00  clk +7743.5  +-0.000  rate +126.1682 c/s  upd=28  rej=41
    22:45:06  clk +169517  +-0.000  rate +127.6563 c/s  upd=280 rej=1050
b_sat reached +-300 chips against a design range of +-3-7, and 79% of measurements
were rejected -- the signature of a covariance that has collapsed to zero, after
which the Kalman gain is zero and no measurement can correct anything.

THE MECHANISM UNDER TEST: gauge() pins mean(b)=0 by applying a scalar pseudo-
measurement with sigma=gauge_sigma EVERY cycle, on the SAME linear combination.
A real measurement carries new information; this one does not -- it is a
statement about coordinates, not about the sky. Re-applying it forever drives P
in that direction to sigma/sqrt(n), and since clk is exactly degenerate with
mean(b), the cross-covariance carries the collapse into clk.

Run: python3 test_gauge_divergence.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker.state_filter import JointReceiverState   # noqa: E402

TRUE_CLK = 150.0
BIASES = {("G", 1): 3.0, ("G", 3): -2.0, ("G", 8): 5.0,
          ("G", 26): -4.0, ("G", 28): 1.0, ("G", 32): -3.0}


def run(n_cycles, dt=2.0, noise=0.0, **kw):
    """Feed a PERFECTLY STATIC receiver: constant clock, constant biases, no drift.
    Any rate the filter reports is manufactured; any variance collapse is its own."""
    f = JointReceiverState(code_len=10230.0, **kw)
    seq = [0.13, -0.07, 0.21, -0.19, 0.05, -0.11, 0.17, -0.03]   # deterministic
    for c in range(n_cycles):
        ms = []
        for j, (k, b) in enumerate(sorted(BIASES.items())):
            e = noise * seq[(c + j) % len(seq)]
            ms.append((k, TRUE_CLK + b + e, 0.3, "l5"))
        f.cycle(ms, c * dt)
    return f


class TestGaugeDivergence(unittest.TestCase):

    def test_static_receiver_stays_static(self):
        """THE HEADLINE. Nothing moves, so clk must stay at 150 and rate at ~0."""
        f = run(400, noise=0.2)
        self.assertAlmostEqual(f.clk, TRUE_CLK, delta=5.0,
                               msg="clk ran away from a static receiver: %.1f" % f.clk)
        self.assertLess(abs(f.clk_rate), 0.01,
                        "manufactured a rate of %+.4f chips/s from a static clock"
                        % f.clk_rate)

    def test_covariance_does_not_collapse(self):
        """P must stay finite. Once sigma(clk) is 0 the gain is 0 and the filter is deaf --
        which is what 1050 rejections against 280 updates looks like from outside."""
        f = run(400, noise=0.2)
        s_clk = float(f.P[0, 0]) ** 0.5
        self.assertGreater(s_clk, 1e-4,
                           "sigma(clk) collapsed to %.2e -- the filter can no longer be "
                           "corrected by any measurement" % s_clk)

    def test_measurements_keep_being_accepted(self):
        """A healthy filter accepts clean measurements indefinitely."""
        f = run(400, noise=0.2)
        total = f.n_updates + f.rejected
        self.assertGreater(total, 0)
        self.assertLess(f.rejected / float(total), 0.10,
                        "rejected %d of %d clean measurements" % (f.rejected, total))

    def test_biases_stay_near_truth(self):
        """b_sat is physical (iono/tropo/ephemeris): chips, not hundreds of chips."""
        f = run(400, noise=0.2)
        for k in BIASES:
            self.assertLess(abs(f.bias(k)), 30.0,
                            "b_sat %s reached %+.1f chips" % (str(k), f.bias(k)))

    def test_observable_is_preserved_by_the_gauge(self):
        """clk + b_i is what the sky measures. The gauge fixes COORDINATES and must leave
        that sum untouched -- if it moves the sum, it is inventing information."""
        f = run(30, noise=0.0)
        before = {k: f.clk + f.bias(k) for k in BIASES}
        f.gauge()
        for k in BIASES:
            self.assertAlmostEqual(f.clk + f.bias(k), before[k], places=6,
                                   msg="gauge moved the observable for %s" % str(k))



class TestEscapeStepBound(unittest.TestCase):
    """The escape hatch must not believe a WRAP ALIAS.

    The consistency test asks whether the rejected measurements agree with each other.
    They can agree perfectly and still imply a step that is not physical -- on a
    10230-chip code, thousands of chips is an alias. Believed, it is applied at
    near-unit gain (P[0,0] += 25 against R = 0.09) and the state moves by the whole
    innovation, which is what drove clk to +169517 on sky.
    """

    def _drive(self, step, **kw):
        f = JointReceiverState(code_len=10230.0, **kw)
        for c in range(20):                       # settle on truth
            f.cycle([(k, TRUE_CLK + b, 0.3, "l5") for k, b in sorted(BIASES.items())],
                    c * 2.0)
        settled = f.clk
        # now ONE satellite reports a consistent, far-displaced value many times over
        for c in range(20, 40):
            ms = [(k, TRUE_CLK + b, 0.3, "l5") for k, b in sorted(BIASES.items())
                  if k != ("G", 1)]
            ms.append((("G", 1), TRUE_CLK + step, 0.3, "l5"))
            f.cycle(ms, c * 2.0)
        return f, settled

    def test_wrap_scale_step_is_refused(self):
        f, settled = self._drive(4521.9)
        self.assertLess(abs(f.clk - settled), 50.0,
                        "clk moved %.1f chips on a wrap-alias escape" % (f.clk - settled))
        self.assertTrue(f.diverged, "a refused escape must latch the diverged flag")
        self.assertTrue(any("ESCAPE REFUSED" in n for n in f.notes),
                        "the refusal must be announced, not silent")

    def test_a_physical_step_is_still_believed(self):
        """The bound must not disable the escape -- that was its own incident."""
        f, settled = self._drive(20.0, escape_max_step=100.0)
        self.assertTrue(f.escapes > 0 or abs(f.clk - settled) > 0.5,
                        "a plausible step must still be able to move the state")

    def test_rate_stays_physical_through_the_event(self):
        f, _ = self._drive(4521.9)
        self.assertLess(abs(f.clk_rate), 1.0,
                        "clk_rate ran to %+.3f chips/s through a refused escape"
                        % f.clk_rate)
if __name__ == "__main__":
    unittest.main(verbosity=2)
