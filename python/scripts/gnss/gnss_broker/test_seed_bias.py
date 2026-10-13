"""#105: the seed-side clock-freq bias must not carry the hint EMA's quantization wander.

The defect this pins: seeds and search hints shared ClockBias.value, so the a=0.05 EMA's
+-10 Hz wander (median of 62.5 Hz bin-quantized detection sawtooths at ~5-sat counts) was
commanded into every replica's code rate, integrating ~1 chip off-peak fleet-wide every
~5 minutes (the q-crash bursts). Under the default the seed mirrors the hint EMA exactly.

#141: under 'zero' the seed is 0.0 through every event that moves the others -- first solve,
crawl, stale re-solve -- while the hint side (value/ema) keeps solving, because the search still
needs it. A snap commanded one few-satellite median into every L5 seed at once.
"""
import ast
import os
import unittest

from gnss_broker.clockbias import ClockBias

HERE = os.path.dirname(os.path.abspath(__file__))


def run_solves(cb, raws, source, bias_alpha=0.05):
    """Drive the class the way almanac.py does: snap-capture, hint EMA, then seed."""
    out = []
    for raw in raws:
        if cb.ema is None or cb.stale:
            cb.ema = raw
            cb.stale = False
        else:
            cb.ema += bias_alpha * (raw - cb.ema)
        cb.value = cb.ema
        out.append(cb.update_seed(source))
    return out


class TestSeedBias(unittest.TestCase):
    def test_default_mirrors_hint_ema(self):
        """source='ema' is the pre-#105 behaviour byte-for-byte: seed == value always."""
        cb = ClockBias()
        raws = [10.0, -20.0, 35.0, -5.0, 0.0, 12.5]
        run_solves(cb, raws, "ema")
        self.assertEqual(cb.seed, cb.value)

    def test_seed_numeric_before_first_solve(self):
        """Consumers add cb.seed to predictions from cycle 1 -- it starts 0.0 like value."""
        cb = ClockBias()
        self.assertEqual(cb.seed, 0.0)

    def test_zero_never_moves_the_seed(self):
        """'zero': first solve, crawl, and stale re-solve all leave the seed at exactly 0.0
        (the measured snaps were +16, -23, +11, -12 Hz), and every return value agrees."""
        cb = ClockBias()
        seeds = run_solves(cb, [16.0], "zero")  # first solve
        self.assertIs(type(cb.seed), float)
        self.assertEqual(seeds, [0.0])
        self.assertEqual(cb.seed, 0.0)
        seeds += run_solves(cb, [-23.0, 11.0, -12.0, 40.0], "zero")  # crawl
        cb.stale = True  # gap: the hint side snaps again here
        seeds += run_solves(cb, [-23.0], "zero")
        self.assertEqual(seeds, [0.0] * 6)
        self.assertEqual(cb.seed, 0.0)

    def test_zero_keeps_the_hint_side_solving(self):
        """The search hints still ride the solve: value/ema follow the raw medians under
        'zero' exactly as they do under 'ema' (same snap, same a=0.05 crawl)."""
        raws = [16.0, 3.0, -5.0, 8.0]
        z, s = ClockBias(), ClockBias()
        run_solves(z, raws, "zero")
        run_solves(s, raws, "ema")
        self.assertEqual((z.value, z.ema), (s.value, s.ema))
        self.assertNotEqual(z.value, 0.0)
        z.stale = s.stale = True
        run_solves(z, [-40.0], "zero")
        run_solves(s, [-40.0], "ema")
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

    def test_ema_unchanged_by_zero(self):
        """Adding 'zero' changes nothing for 'ema': it is the hint EMA (warm-started or
        not) -- checked against the closed form, not against the code."""
        cb = ClockBias()
        run_solves(cb, [5.0, 9.0], "ema")
        self.assertEqual(cb.seed, 5.0 + 0.05 * (9.0 - 5.0))


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
