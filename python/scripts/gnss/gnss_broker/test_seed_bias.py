"""#105: the seed-side clock-freq bias must not carry the hint EMA's quantization wander.

The defect this pins: seeds and search hints shared ClockBias.value, so the a=0.05 EMA's
+-10 Hz wander (median of 62.5 Hz bin-quantized detection sawtooths at ~5-sat counts) was
commanded into every replica's code rate, integrating ~1 chip off-peak fleet-wide every
~5 minutes (the q-crash bursts). Under --seed-bias-source=slow the seed rides its own
long-memory EMA of the same raw stream; under the default it mirrors the hint EMA exactly.

#141: under 'zero' the seed is 0.0 through every event that moves the others -- first solve,
crawl, stale re-solve -- while the hint side (value/ema) keeps solving, because the search still
needs it. A snap commanded one few-satellite median into every L5 seed at once.
"""
import ast
import math
import os
import unittest

from gnss_broker.clockbias import ClockBias

HERE = os.path.dirname(os.path.abspath(__file__))


def run_solves(cb, raws, source, alpha=0.005, bias_alpha=0.05):
    """Drive the class the way almanac.py does: snap-capture, hint EMA, then seed."""
    out = []
    for raw in raws:
        snapped = cb.ema is None or cb.stale
        if snapped:
            cb.ema = raw
            cb.stale = False
        else:
            cb.ema += bias_alpha * (raw - cb.ema)
        cb.value = cb.ema
        out.append(cb.update_seed(raw, source, alpha, snapped))
    return out


class TestSeedBias(unittest.TestCase):
    def test_default_mirrors_hint_ema(self):
        """source='ema' is the pre-#105 behaviour byte-for-byte: seed == value always."""
        cb = ClockBias()
        raws = [10.0, -20.0, 35.0, -5.0, 0.0, 12.5]
        run_solves(cb, raws, "ema")
        self.assertEqual(cb.seed, cb.value)

    def test_slow_rejects_the_wander_the_hint_ema_passes(self):
        """A +-10 Hz sinusoidal wander at the measured ~5 min period (30 solves at ~10 s)
        must reach the hint EMA (that's #105) and NOT the seed."""
        cb = ClockBias()
        true_bias = 2.0
        raws = [true_bias + 10.0 * math.sin(2 * math.pi * i / 30.0) for i in range(600)]
        seeds = run_solves(cb, raws, "slow")
        settled = seeds[300:]
        hint_swing = max(
            abs(cb.value - true_bias), 4.0
        )  # the hint EMA demonstrably wobbles
        seed_swing = max(abs(s - true_bias) for s in settled)
        self.assertLess(
            seed_swing, 1.0, "seed still carries the wander: %.2f Hz" % seed_swing
        )
        self.assertGreater(hint_swing, 3.0)

    def test_slow_follows_thermal_drift(self):
        """Hour-scale GPSDO drift (the reason a static cal was rejected) is followed:
        a 20 Hz ramp over 3600 solves lags by < 2 Hz at the end."""
        cb = ClockBias()
        raws = [i * (20.0 / 3600.0) for i in range(3600)]
        seeds = run_solves(cb, raws, "slow")
        self.assertLess(abs(seeds[-1] - raws[-1]), 2.0)

    def test_snap_on_first_solve_and_stale_resolve(self):
        """A measurement gap outranks the slow memory, exactly as it does the fast one."""
        cb = ClockBias()
        run_solves(cb, [7.0], "slow")
        self.assertEqual(cb.seed, 7.0)  # first solve snaps
        run_solves(cb, [8.0] * 5, "slow")
        self.assertLess(abs(cb.seed - 7.0), 0.1)  # then crawls
        cb.stale = True  # gap: the GPSDO may have walked
        run_solves(cb, [-40.0], "slow")
        self.assertEqual(cb.seed, -40.0)  # stale re-solve snaps

    def test_seed_numeric_before_first_solve(self):
        """Consumers add cb.seed to predictions from cycle 1 -- it starts 0.0 like value."""
        cb = ClockBias()
        self.assertEqual(cb.seed, 0.0)

    def test_zero_never_moves_the_seed(self):
        """'zero': first solve, crawl, and stale re-solve all leave the seed at exactly 0.0
        (the measured snaps were +16, -23, +11, -12 Hz), and every return value agrees."""
        cb = ClockBias()
        seeds = run_solves(cb, [16.0], "zero")  # first solve: 'slow' snaps here
        self.assertIs(type(cb.seed), float)
        self.assertEqual(seeds, [0.0])
        self.assertEqual(cb.seed, 0.0)
        seeds += run_solves(cb, [-23.0, 11.0, -12.0, 40.0], "zero")  # crawl
        cb.stale = True  # gap: 'slow' snaps again here
        seeds += run_solves(cb, [-23.0], "zero")
        self.assertEqual(seeds, [0.0] * 6)
        self.assertEqual(cb.seed, 0.0)

    def test_zero_keeps_the_hint_side_solving(self):
        """The search hints still ride the solve: value/ema follow the raw medians under
        'zero' exactly as they do under 'slow' (same snap, same a=0.05 crawl)."""
        raws = [16.0, 3.0, -5.0, 8.0]
        z, s = ClockBias(), ClockBias()
        run_solves(z, raws, "zero")
        run_solves(s, raws, "slow")
        self.assertEqual((z.value, z.ema), (s.value, s.ema))
        self.assertNotEqual(z.value, 0.0)
        z.stale = s.stale = True
        run_solves(z, [-40.0], "zero")
        run_solves(s, [-40.0], "slow")
        self.assertEqual((z.value, z.ema), (-40.0, -40.0))
        self.assertEqual((z.value, z.ema), (s.value, s.ema))
        self.assertEqual(z.seed, 0.0)

    def test_zero_ignores_a_warm_start(self):
        """A warm start writes `ema` (and `cal`), never `seed`: the seed
        is 0.0 before the first solve and stays there after it."""
        cb = ClockBias()
        cb.ema = cb.cal = -17.9  # what the warm start does
        self.assertEqual(cb.seed, 0.0)
        run_solves(cb, [-15.0, -16.0], "zero")
        self.assertEqual(cb.seed, 0.0)
        self.assertNotEqual(cb.ema, -17.9)

    def test_ema_and_slow_unchanged_by_zero(self):
        """Adding 'zero' changes nothing for the existing sources: 'ema' is the hint EMA
        (warm-started or not), 'slow' snaps on first solve and on stale re-solve and crawls at
        alpha in between -- checked against the closed forms, not against the code."""
        cb = ClockBias()
        run_solves(cb, [5.0, 9.0], "ema")
        self.assertEqual(cb.seed, 5.0 + 0.05 * (9.0 - 5.0))
        cb = ClockBias()
        run_solves(cb, [5.0, 9.0], "slow")
        self.assertEqual(cb.seed, 5.0 + 0.005 * (9.0 - 5.0))
        cb.stale = True
        run_solves(cb, [-3.0], "slow")
        self.assertEqual(cb.seed, -3.0)


def _cb_attrs(path):
    """(reads, writes) of `<...>.cb.<attr>` / `_cb.<attr>` in one source file."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    reads, writes = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        is_cb = (isinstance(base, ast.Attribute) and base.attr == "cb") or (
            isinstance(base, ast.Name) and base.id in ("cb", "_cb")
        )
        if is_cb:
            (writes if isinstance(node.ctx, ast.Store) else reads).add(node.attr)
    return reads, writes


class TestSeedBiasWiring(unittest.TestCase):
    """'zero' is only as good as the claim that update_seed is the seed's ONLY writer and
    that the seed builders read `seed`, never the hint bias. Both are structural."""

    SOURCES = [
        os.path.join(HERE, n)
        for n in sorted(os.listdir(HERE))
        if n.endswith(".py")
        and not n.startswith("test_")
        and n != "selftest.py"
        and n != "clockbias.py"
    ] + [os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")]

    def test_nothing_but_update_seed_writes_the_seed(self):
        bad = [os.path.basename(p) for p in self.SOURCES if "seed" in _cb_attrs(p)[1]]
        self.assertEqual(bad, [], "cb.seed assigned outside ClockBias: %s" % bad)

    def test_seed_builders_never_read_the_hint_bias(self):
        for name in ("seeding.py", "deadreckon.py"):
            reads, _ = _cb_attrs(os.path.join(HERE, name))
            self.assertIn("seed", reads, name)
            self.assertNotIn(
                "value",
                reads,
                "%s reads cb.value: the hint bias would reach a seed" % name,
            )

    def test_first_seed_guard_is_mode_independent(self):
        """The guard withholds first seeds until a bias exists, in EVERY seed-bias mode. Under
        'zero' the seed no longer needs the bias, but the guard also keeps a chain's first
        seeds off its first cycle, which runs before dead-reckoning has stamped the cycle's
        clock (drp.now_w is None): exempting 'zero' let gps_l5 seed on cycle 1 and die in
        _nh_joint_consensus on None - None."""
        with open(os.path.join(HERE, "seeding.py")) as f:
            tree = ast.parse(f.read())
        tests = [
            ast.unparse(n.test)
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "cb.available" in ast.unparse(n.test)
        ]
        self.assertEqual(len(tests), 1, tests)
        self.assertNotIn("seed_bias_source", tests[0])


if __name__ == "__main__":
    unittest.main()
