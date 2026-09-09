"""The code-residual physics in gnss_observables: what makes it right, and what breaks it.

Every assertion here failed at some point on 2026-09-09 while the live PVT self-survey was
being brought back up, so each one is a bug that actually happened rather than a property
someone thought worth asserting.
"""
import types
import unittest
from fractions import Fraction

import gnss_observables as go

A = types.SimpleNamespace(code_doppler_sign=1.0, carrier_hz=1176.45e6, chip_rate_hz=10.23e6,
                          sample_rate_hz=3.2e9, samples_per_hop=16384, code_length=10230)
HOP = 85404375040          # a live gps_l5 hop: ~5 days of F-engine uptime
M_PER_CHIP = 299792458.0 / 10.23e6


def wrap(d):
    return (d + A.code_length / 2) % A.code_length - A.code_length / 2


def argument_for(phase_chips, dop, hop=HOP, comb_mult=1):
    """The argument the generator would need to put the replica at `phase_chips` at `hop`."""
    scale = 1.0 + A.code_doppler_sign * dop / A.carrier_hz
    ramp = (Fraction(hop * A.samples_per_hop) * Fraction(A.chip_rate_hz)
            * Fraction(scale).limit_denominator(10 ** 15) / Fraction(int(A.sample_rate_hz)))
    return float((Fraction(phase_chips) - ramp) / comb_mult)


class TestPhysChips(unittest.TestCase):
    def test_inverts_the_generators_argument(self):
        """code_phase_chips is an ARGUMENT back-referenced to sample 0; lifting it with the
        SAME Doppler must return the physical phase exactly (chord-cp-currency)."""
        for dop in (0.0, 185.33224487304688, -2245.232666015625):
            for want in (0.0, 1234.5, 10229.75):
                got = go._phys_chips(argument_for(want, dop), 1, HOP, 0.0, dop, A)
                self.assertLess(abs(wrap(got - want)), 1e-3,
                                "dop %s want %s got %s" % (dop, want, got))

    def test_comb_mult_is_applied(self):
        """L2C's replica is comb_mult 2: ignoring it puts the phase at half the code."""
        arg = argument_for(4000.0, 0.0, comb_mult=2)
        self.assertLess(abs(wrap(go._phys_chips(arg, 2, HOP, 0.0, 0.0, A) - 4000.0)), 1e-3)
        self.assertGreater(abs(wrap(go._phys_chips(arg, 1, HOP, 0.0, 0.0, A) - 4000.0)), 100.0)

    def test_the_doppler_lever_is_why_applied_not_reported(self):
        """~5095 chips per Hz at this uptime. The reported and applied Dopplers differ by
        ~0.1 Hz, so using the wrong one is not a degraded residual -- it is kilometres."""
        base = go._phys_chips(0.0, 1, HOP, 0.0, 185.332, A)
        off = go._phys_chips(0.0, 1, HOP, 0.0, 185.432, A)   # 0.1 Hz
        self.assertGreater(abs(wrap(off - base)) * M_PER_CHIP, 5000.0)

    def test_hop_choice_cancels(self):
        """Both sides of the residual are evaluated at the SAME hop, so which hop field is
        used (fleet_hop vs pow_hop, 12288 apart) cancels to first order."""
        a1 = go._phys_chips(0.0, 1, HOP, 0.0, 185.332, A)
        a2 = go._phys_chips(0.0, 1, HOP + 12288, 0.0, 185.332, A)
        b1 = go._phys_chips(0.0, 1, HOP, 0.0, 185.432, A)
        b2 = go._phys_chips(0.0, 1, HOP + 12288, 0.0, 185.432, A)
        self.assertLess(abs(wrap((b2 - a2) - (b1 - a1))) * M_PER_CHIP, 0.1)

    def test_utc_differencing_would_lose_metres(self):
        """Why the epoch comes from the integer hop and never from (t_now - utc0): two 1.79e9
        floats subtract at 2.4e-7 s, which is 2.4 chips = 70 m of code."""
        t_abs = HOP * A.samples_per_hop / A.sample_rate_hz
        import math
        self.assertGreater(math.ulp(1.788978e9) * A.chip_rate_hz * M_PER_CHIP, 5.0)
        # the float path itself is fine once t_abs is exact -- it is the SUBTRACTION that is not
        self.assertLess(abs(wrap(go._phys_chips(0.0, 1, None, t_abs, 185.332, A)
                                 - go._phys_chips(0.0, 1, HOP, 0.0, 185.332, A))) * M_PER_CHIP, 0.1)


if __name__ == "__main__":
    unittest.main()
