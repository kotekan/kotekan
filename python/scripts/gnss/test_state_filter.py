#!/usr/bin/env python3
"""Regression tests for JointReceiverState, built around the 2026-08-10 incident.

THE INCIDENT, because these tests only make sense against it. The GPS-only joint state ran
healthy for eight hours (clk 148-151 chips, sigma 0.05-0.16, mean(b) ~ 0.00, 1.2% rejection).
At 10:22 PRN 25 transited out of the primary beam -- elevation RISING 45->65 deg, so not a
set -- and its tracker output became uniform noise over the 10230-chip code (deep_snr 30->2.4,
coh_frac 0.94->0.19). The filter's escape hatch counted five consecutive rejections, concluded
"a real move is PERSISTENT", and believed a -1396-chip innovation. That corrupted clk, which
made every other satellite's innovation large, which escaped them too, until no bias was
within gauge_max_b of zero and the gauge switched itself off. clk read +50.8 against a true
+148 for fifty minutes -- and NOTHING downstream noticed, because the observable combination
clk + b_i stayed correct the whole time.

Both halves are tested: the hatch must not believe incoherent runs, and the gauge must never
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
    # the six sats and their biases as the state actually held them at 10:21, one minute
    # before the incident
    biases = dict(zip([("gps", p) for p in (6, 11, 21, 24, 25, 28)][:n_sat],
                      (-3.8, -0.9, -1.2, +4.4, +1.2, +0.3)))
    t = 1000.0
    for _ in range(cycles):
        t += 30.0
        js.cycle([(k, TRUE_CLK + b + rng.gauss(0.0, sigma), sigma, None)
                  for k, b in biases.items()], t)
    return js, biases, t, rng


class TestEscapeHatch(unittest.TestCase):
    def test_healthy_state_converges(self):
        """Baseline: the filter reproduces the state we actually ran overnight."""
        js, biases, _, _ = _healthy()
        self.assertAlmostEqual(js.x[0], TRUE_CLK, delta=0.5)
        for k, b in biases.items():
            self.assertAlmostEqual(js.bias(k), b, delta=0.5)

    def test_noise_satellite_does_not_destroy_the_state(self):
        """THE INCIDENT. One satellite feeds uniform noise over the code; the state must
        survive it. Before the fix this drove clk from +148 to -1242 in a single cycle."""
        js, biases, t, rng = _healthy()
        bad = ("gps", 25)
        for _ in range(40):                      # 20 minutes of a beam-exited satellite
            t += 30.0
            meas = [(k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3, None)
                    for k, b in biases.items() if k != bad]
            meas.append((bad, rng.uniform(0.0, L), 0.3, None))   # uniform noise, as measured
            js.cycle(meas, t)
        self.assertAlmostEqual(js.x[0], TRUE_CLK, delta=1.0,
                               msg="clk was dragged by a noise satellite")
        for k, b in biases.items():
            if k == bad:
                continue
            self.assertAlmostEqual(js.bias(k), b, delta=1.0,
                                   msg="bias %s corrupted by a noise satellite" % (k,))

    def test_genuine_step_is_still_followed(self):
        """The hatch must keep doing its job: a CONSISTENT step is a real re-anchor and has
        to be believed, or the gate is a deadlock. This is the case the hatch exists for."""
        js, biases, t, rng = _healthy()
        moved = ("gps", 24)
        step = 40.0                              # far outside any normalized gate
        for _ in range(30):
            t += 30.0
            js.cycle([(k, TRUE_CLK + b + (step if k == moved else 0.0) + rng.gauss(0.0, 0.3),
                       0.3, None)
                      for k, b in biases.items()], t)
        # The gauge splits it, and that IS the convention rather than an error: one sat
        # moving by d shifts the fleet mean by d/N, so clk takes d/N and the sat keeps the
        # rest. Asserting the full step here would be asserting that clk is NOT the fleet
        # mean. Pinning both numbers documents the convention.
        n = len(biases)
        self.assertAlmostEqual(js.bias(moved), biases[moved] + step * (n - 1) / n, delta=1.5,
                               msg="a real, repeatable step was never followed")
        self.assertAlmostEqual(js.x[0], TRUE_CLK + step / n, delta=1.0)
        # ...and the only thing any consumer can actually observe is untouched.
        self.assertAlmostEqual(js.x[0] + js.bias(moved), TRUE_CLK + biases[moved] + step,
                               delta=1.0, msg="the observable clk + b was not preserved")

    def test_escape_is_logged(self):
        """The most damaging event of 2026-08-10 fired silently. It must not be able to."""
        js, biases, t, rng = _healthy()
        moved = ("gps", 24)
        for _ in range(12):
            t += 30.0
            js.cycle([(k, TRUE_CLK + b + (40.0 if k == moved else 0.0) + rng.gauss(0.0, 0.3),
                       0.3, None) for k, b in biases.items()], t)
        notes = js.drain_notes()
        self.assertTrue(any("ESCAPE" in n for n in notes),
                        "an escape happened with no operator note: %r" % (notes,))
        self.assertGreater(js.escapes, 0)

    def test_incoherent_run_is_reported(self):
        """A satellite feeding noise is a TRACKING failure; the filter should say so rather
        than silently rejecting forever."""
        js, biases, t, rng = _healthy()
        bad = ("gps", 25)
        for _ in range(20):
            t += 30.0
            meas = [(k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3, None)
                    for k, b in biases.items() if k != bad]
            meas.append((bad, rng.uniform(0.0, L), 0.3, None))
            js.cycle(meas, t)
        self.assertTrue(any("REJECT-RUN" in n for n in js.drain_notes()),
                        "a noise satellite produced no diagnostic")


class TestGauge(unittest.TestCase):
    def test_gauge_does_not_switch_itself_off_when_uniformly_displaced(self):
        """The second half of the incident, tested narrowly and at the point it broke.

        This deliberately does NOT simulate forward from a hand-edited state: writing x
        without writing a consistent P describes a situation the filter was never in, and the
        resulting trajectory measures the surgery rather than the fix. The defect was
        precise -- the INLIER SET came back empty and gauge() returned without pinning
        anything -- so that is what is asserted, on exactly the state the sky produced at
        11:12 (every bias ~ +97 against a clk 97 chips low)."""
        js, biases, _, _ = _healthy()
        for i in js._idx.values():
            js.x[i] += 97.0
        js.x[0] -= 97.0

        # The OLD rule, stated explicitly so the regression is documented rather than
        # implied: measured against ZERO, every satellite is an outlier and the set is empty.
        old_use = [i for i in js._idx.values() if abs(float(js.x[i])) <= js.gauge_max_b]
        self.assertEqual(len(old_use), 0,
                         "the zero-centred inlier rule would not have emptied here, so this "
                         "state does not reproduce the incident")

        mean_before = sum(js.bias(k) for k in biases) / len(biases)
        js.gauge()
        mean_after = sum(js.bias(k) for k in biases) / len(biases)
        self.assertLess(abs(mean_after), abs(mean_before) - 1.0,
                        "gauge() did not pin the common mode at all (it switched itself off)")
        self.assertEqual(js.gauge_fallbacks, 0,
                         "the median-centred set should be full here, needing no fallback")

    def test_gauge_still_excludes_a_single_wild_bias(self):
        """...without losing the property it was given gauge_max_b for."""
        js, biases, t, rng = _healthy()
        wild = ("gps", 25)
        js.x[js._idx[wild]] = -1400.0
        clk_before = float(js.x[0])
        for _ in range(10):
            t += 30.0
            js.cycle([(k, TRUE_CLK + b + rng.gauss(0.0, 0.3), 0.3, None)
                      for k, b in biases.items() if k != wild], t)
        self.assertAlmostEqual(js.x[0], clk_before, delta=1.0,
                               msg="a single wild bias got a vote in the gauge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
