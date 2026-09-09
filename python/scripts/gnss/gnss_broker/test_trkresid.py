"""The tracker code observable (trkresid): the lift is exact, the float32 slot-1 lever is real
and averages away, and the residual returns the injected per-satellite offset.

    python3 -m unittest gnss_broker.test_trkresid
"""
import random
import struct
import unittest
from fractions import Fraction

from gnss_broker import trkresid as tr

HPS = Fraction(390625, 2)          # 195312.5 hops/s
CHIP, CARR, L = 10.23e6, 1176.45e6, 10230
HOP = 86466125824                  # ~5.1 days of F-engine uptime
M_PER_CHIP = 299792458.0 / CHIP


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def arg_for(phase, dop, hop=HOP):
    """The argument a record would carry to place the replica at `phase` at `hop` (exact)."""
    t_abs = Fraction(hop) / HPS
    ramp = t_abs * int(CHIP) * (1 + Fraction(dop).limit_denominator(10 ** 9) / int(CARR))
    return float((Fraction(phase) - ramp) % L)


class TestLift(unittest.TestCase):
    def test_inverts_the_argument_exactly(self):
        for dop in (0.0, -2182.937744140625, 1734.5):
            for want in (0.0, 5000.25, 10229.9):
                got = tr.phys_chips(arg_for(want, dop), dop, HOP, HPS, CHIP, CARR, L)
                self.assertLess(abs(tr.wrap(got - want, L)), 2e-3)

    def test_float32_doppler_is_a_tenth_of_a_chip_lever(self):
        """slot 1 is float32: one ulp at 2 kHz (1.2e-4 Hz) moves the lift by ~0.1-0.5 chips at
        this uptime. That is the record-level scatter this module averages away."""
        dop = -2182.9377
        a = tr.phys_chips(arg_for(1000.0, dop), dop, HOP, HPS, CHIP, CARR, L)
        b = tr.phys_chips(arg_for(1000.0, dop), f32(dop), HOP, HPS, CHIP, CARR, L)
        lever = abs(tr.wrap(b - a, L)) / abs(f32(dop) - dop)
        self.assertGreater(lever, 1000.0)      # chips per Hz
        self.assertLess(abs(tr.wrap(b - a, L)), 1.0)


class TestResidual(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.clk = 151.39
        self.bias = {1: 0.17, 3: -0.02, 4: 0.08}
        self.pd = {("G", p): {"el": 40.0} for p in self.bias}

    def model(self, v, t_abs):
        # some smooth physical phase: nominal ramp is NOT here (cp_predicted excludes it too,
        # in the sense that both sides carry the same t*f_chip); any function works.
        return (1234.5 + 17.3 * t_abs) % L

    def records(self, prn, n, quantize=True):
        out = []
        for i in range(n):
            hop = HOP + 2048 * i
            t_abs = hop / float(HPS)
            phase = (self.model(None, t_abs) + self.clk + self.bias[prn]) % L
            dop = -2182.9377 + 0.0656 * (t_abs - HOP / float(HPS))
            arg = arg_for(phase, dop, hop)
            out.append((hop, f32(arg) if quantize else arg, f32(dop) if quantize else dop))
        return out

    def test_returns_the_injected_bias(self):
        recs = {p: self.records(p, 96) for p in self.bias}
        res = tr.residuals(recs, self.pd, "G", self.model, self.clk, 0.0,
                           HOP / float(HPS), HPS, CHIP, CARR, L)
        for p, b in self.bias.items():
            self.assertLess(abs(res[p]["chips"] - b), 0.03, (p, res[p]))
            self.assertEqual(res[p]["n"], 96)

    def test_exact_records_return_the_bias_exactly(self):
        recs = {1: self.records(1, 4, quantize=False)}
        res = tr.residuals(recs, self.pd, "G", self.model, self.clk, 0.0,
                           HOP / float(HPS), HPS, CHIP, CARR, L)
        self.assertLess(abs(res[1]["chips"] - 0.17), 2e-3)
        self.assertLess(res[1]["sd"], 2e-3)

    def test_quantization_scatter_is_reported_and_averages_down(self):
        one = tr.residuals({1: self.records(1, 1)}, self.pd, "G", self.model, self.clk, 0.0,
                           HOP / float(HPS), HPS, CHIP, CARR, L)[1]
        many = tr.residuals({1: self.records(1, 200)}, self.pd, "G", self.model, self.clk, 0.0,
                            HOP / float(HPS), HPS, CHIP, CARR, L)[1]
        self.assertGreater(many["sd"], 0.02)          # the float32 lever is visible per record
        self.assertLess(abs(many["chips"] - 0.17), abs(one["chips"] - 0.17) + 0.05)
        self.assertLess(abs(many["chips"] - 0.17), 0.03)

    def test_no_clock_no_residual(self):
        self.assertEqual(tr.residuals({1: self.records(1, 4)}, self.pd, "G", self.model, None,
                                      0.0, 0.0, HPS, CHIP, CARR, L), {})

    def test_drift_normalises_to_now(self):
        """A record 2 s old under a 0.01 chips/s drift is 0.02 chips from `now`; the residual
        must be referenced to the clock's epoch, as the integrity residual is."""
        recs = {1: self.records(1, 4, quantize=False)}
        t_now = HOP / float(HPS) + 2.0
        a = tr.residuals(recs, self.pd, "G", self.model, self.clk, 0.0, t_now, HPS, CHIP, CARR, L)[1]
        b = tr.residuals(recs, self.pd, "G", self.model, self.clk, 0.01, t_now, HPS, CHIP, CARR, L)[1]
        self.assertAlmostEqual(b["chips"] - a["chips"], 0.02, places=3)


if __name__ == "__main__":
    unittest.main()
