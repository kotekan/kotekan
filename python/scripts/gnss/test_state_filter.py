#!/usr/bin/env python3
"""Tests for JointReceiverState's escape and median gauge against a satellite feeding noise.

A satellite that transits out of the primary beam, not only one that sets, can stay tracked
while its output is uniform noise over the 10230-chip code. An escape that believed any
persistent run of rejections would take a ~-1400-chip innovation from it; that corrupts clk,
every other satellite's innovation becomes large and escapes too, and a gauge whose inlier set
is centred on zero then finds no inliers and stops pinning. clk can then sit far from the truth
with nothing downstream noticing, because the observable clk + b_i stays correct throughout.

Both halves are tested: the escape must not believe incoherent runs, and the gauge must never
stop pinning.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker.state_filter import JointReceiverState  # noqa: E402

L = 10230.0
TRUE_CLK = 148.0


def _healthy(sigma=0.3, n_sat=6, cycles=60, seed=1):
    """A converged 6-satellite state, the way the sky delivers one."""
    import random

    rng = random.Random(seed)
    js = JointReceiverState(code_len=L)
    # six satellites with biases of the size a converged state holds
    biases = dict(
        zip(
            [("gps", p) for p in (6, 11, 21, 24, 25, 28)][:n_sat],
            (-3.8, -0.9, -1.2, +4.4, +1.2, +0.3),
        )
    )
    t = 1000.0
    for _ in range(cycles):
        t += 30.0
        js.cycle(
            [
                (k, TRUE_CLK + b + rng.gauss(0.0, sigma), sigma)
                for k, b in biases.items()
            ],
            t,
        )
    return js, biases, t, rng


class TestEscapeHatch(unittest.TestCase):
    def test_healthy_state_converges(self):
        """Baseline: the filter converges to the true clock and biases."""
        js, biases, _, _ = _healthy()
        self.assertAlmostEqual(js.x[0], TRUE_CLK, delta=0.5)
        for k, b in biases.items():
            self.assertAlmostEqual(js.bias(k), b, delta=0.5)

    def test_noise_satellite_does_not_destroy_the_state(self):
        """One satellite feeds uniform noise over the code; the state must survive it."""
        js, biases, t, rng = _healthy()
        bad = ("gps", 25)
        for _ in range(40):  # 20 minutes of a beam-exited satellite
            t += 30.0
            meas = [
                (k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3)
                for k, b in biases.items()
                if k != bad
            ]
            meas.append((bad, rng.uniform(0.0, L), 0.3))  # uniform noise over the code
            js.cycle(meas, t)
        self.assertAlmostEqual(
            js.x[0], TRUE_CLK, delta=1.0, msg="clk was dragged by a noise satellite"
        )
        for k, b in biases.items():
            if k == bad:
                continue
            self.assertAlmostEqual(
                js.bias(k),
                b,
                delta=1.0,
                msg="bias %s corrupted by a noise satellite" % (k,),
            )

    def test_genuine_step_is_still_followed(self):
        """The escape must keep doing its job: a consistent step is a real re-anchor and has
        to be believed, or the gate is a deadlock. This is the case the escape exists for."""
        js, biases, t, rng = _healthy()
        moved = ("gps", 24)
        step = 40.0  # far outside any normalized gate
        for _ in range(30):
            t += 30.0
            js.cycle(
                [
                    (
                        k,
                        TRUE_CLK
                        + b
                        + (step if k == moved else 0.0)
                        + rng.gauss(0.0, 0.3),
                        0.3,
                    )
                    for k, b in biases.items()
                ],
                t,
            )
        # Median gauge: one satellite moving by d does not move the median, so that
        # satellite keeps the whole step and clk stays put, where a mean gauge would leak
        # d/N into every consumer's clock. Both numbers are pinned.
        self.assertAlmostEqual(
            js.bias(moved),
            biases[moved] + step,
            delta=1.5,
            msg="a real, repeatable step was never followed",
        )
        self.assertAlmostEqual(
            js.x[0],
            TRUE_CLK,
            delta=1.0,
            msg="one satellite's step leaked into the median clock",
        )
        # ...and the only thing any consumer can actually observe is untouched.
        self.assertAlmostEqual(
            js.x[0] + js.bias(moved),
            TRUE_CLK + biases[moved] + step,
            delta=1.0,
            msg="the observable clk + b was not preserved",
        )

    def test_escape_is_logged(self):
        """An escape moves the state by a whole step; it must always leave an operator
        note."""
        js, biases, t, rng = _healthy()
        moved = ("gps", 24)
        for _ in range(12):
            t += 30.0
            js.cycle(
                [
                    (
                        k,
                        TRUE_CLK
                        + b
                        + (40.0 if k == moved else 0.0)
                        + rng.gauss(0.0, 0.3),
                        0.3,
                    )
                    for k, b in biases.items()
                ],
                t,
            )
        notes = js.drain_notes()
        self.assertTrue(
            any("ESCAPE" in n for n in notes),
            "an escape happened with no operator note: %r" % (notes,),
        )
        self.assertTrue(any("-- believing a" in n for n in notes), notes)

    def test_incoherent_run_is_reported(self):
        """A satellite feeding noise is a tracking failure; the filter should say so rather
        than silently rejecting forever."""
        js, biases, t, rng = _healthy()
        bad = ("gps", 25)
        for _ in range(20):
            t += 30.0
            meas = [
                (k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3)
                for k, b in biases.items()
                if k != bad
            ]
            meas.append((bad, rng.uniform(0.0, L), 0.3))
            js.cycle(meas, t)
        self.assertTrue(
            any("REJECT-RUN" in n for n in js.drain_notes()),
            "a noise satellite produced no diagnostic",
        )


class TestGauge(unittest.TestCase):
    def test_gauge_does_not_switch_itself_off_when_uniformly_displaced(self):
        """A uniformly displaced comb must still be pinned, tested narrowly at gauge().

        This deliberately does not simulate forward from a hand-edited state: writing x
        without a consistent P describes a situation the filter is never in, and the
        trajectory would measure the edit rather than the gauge. The failure is precise (an
        empty inlier set, and gauge() returning without pinning anything), so that is what is
        asserted, on a state with every bias ~ +97 against a clk 97 chips low."""
        js, biases, _, _ = _healthy()
        for i in js._idx.values():
            js.x[i] += 97.0
        js.x[0] -= 97.0

        # A zero-centred inlier rule, stated explicitly: measured against zero, every
        # satellite is an outlier and the set is empty, so this state exercises the failure.
        old_use = [i for i in js._idx.values() if abs(float(js.x[i])) <= js.gauge_max_b]
        self.assertEqual(
            len(old_use),
            0,
            "the zero-centred inlier rule would not have emptied here, so this "
            "state does not reproduce the incident",
        )

        mean_before = sum(js.bias(k) for k in biases) / len(biases)
        js.drain_notes()
        js.gauge()
        mean_after = sum(js.bias(k) for k in biases) / len(biases)
        self.assertLess(
            abs(mean_after),
            abs(mean_before) - 1.0,
            "gauge() did not pin the common mode at all (it switched itself off)",
        )
        self.assertFalse(
            any(n.startswith("GAUGE:") for n in js.drain_notes()),
            "the median-centred set should be full here, needing no fallback",
        )

    def test_gauge_still_excludes_a_single_wild_bias(self):
        """The median-centred inlier set still excludes a single wild bias (gauge_max_b)."""
        js, biases, t, rng = _healthy()
        wild = ("gps", 25)
        js.x[js._idx[wild]] = -1400.0
        clk_before = float(js.x[0])
        for _ in range(10):
            t += 30.0
            js.cycle(
                [
                    (k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3)
                    for k, b in biases.items()
                    if k != wild
                ],
                t,
            )
        self.assertAlmostEqual(
            js.x[0],
            clk_before,
            delta=1.0,
            msg="a single wild bias got a vote in the gauge",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
