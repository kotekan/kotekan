"""#105: the seed-side clock-freq bias must not carry the hint EMA's quantization wander.

The defect this pins: seeds and search hints shared ClockBias.value, so the a=0.05 EMA's
+-10 Hz wander (median of 62.5 Hz bin-quantized detection sawtooths at ~5-sat counts) was
commanded into every replica's code rate, integrating ~1 chip off-peak fleet-wide every
~5 minutes (the q-crash bursts). Under --seed-bias-source=slow the seed rides its own
long-memory EMA of the same raw stream; under the default it mirrors the hint EMA exactly.
"""
import math
import unittest

from gnss_broker.clockbias import ClockBias


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
        hint_swing = max(abs(cb.value - true_bias), 4.0)  # the hint EMA demonstrably wobbles
        seed_swing = max(abs(s - true_bias) for s in settled)
        self.assertLess(seed_swing, 1.0,
                        "seed still carries the wander: %.2f Hz" % seed_swing)
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
        self.assertEqual(cb.seed, 7.0)          # first solve snaps
        run_solves(cb, [8.0] * 5, "slow")
        self.assertLess(abs(cb.seed - 7.0), 0.1)  # then crawls
        cb.stale = True                          # gap: the GPSDO may have walked
        run_solves(cb, [-40.0], "slow")
        self.assertEqual(cb.seed, -40.0)         # stale re-solve snaps

    def test_seed_numeric_before_first_solve(self):
        """Consumers add cb.seed to predictions from cycle 1 -- it starts 0.0 like value."""
        cb = ClockBias()
        self.assertEqual(cb.seed, 0.0)


if __name__ == "__main__":
    unittest.main()
