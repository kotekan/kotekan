"""Task #52 Step 2: coherently sum per-channel points by a DERIVED rotation, not a fitted one.

WHAT THIS REPLACES. The old combine gave every INSTANCE a free phase, obtained by
cross-correlating it against the most energetic instance and then summing with that rotation --
twelve free parameters estimated from the data they are summed into. That aligns noise, and
when the fit fails the whole chain drops to the quadrature fallback (gps_l5 measured align
0.143 with 9/12 satellites on `quad`).

Physics gives ONE parameter: phi(f) = -2*pi*f*tau. The rotation must be applied PER CHANNEL,
because each instance's 7 channels are a comb spanning ~18.75 MHz -- at tau = 1 chip the ramp
inside ONE instance is ~11.5 rad, so no per-instance constant can represent it.

⚠️ THE LOAD-BEARING TEST IS test_shuffled_null_kills_a_noise_only_set. With a free phase per
instance the fit could always find some alignment, so a null could never fail -- the gate was
unpassable-by-construction and therefore meaningless. It only becomes a real test once the
rotation is derived.
"""
import cmath
import math
import random
import unittest

from gnss_broker.fleet import delay_combine

CHIP_HZ = 10.23e6
CHAN_HZ = 195312.5
# The live comb: 7 channels stride 16 per instance, 12 instances interleaved.
INSTANCES = ["cx%d/%d" % (n, g) for n in (19, 27, 42, 43, 44, 51) for g in (0, 1)]


def make_points(tau_chips, amp=1.0, noise=0.0, seed=7, n_chan=7, stride=16):
    """A fleet whose per-channel phases carry exactly the delay ramp the combine must undo."""
    rng = random.Random(seed)
    pts = []
    tau_s = tau_chips / CHIP_HZ
    for i, key in enumerate(INSTANCES):
        base = 5972 + i * n_chan * stride
        for c in range(n_chan):
            fid = base + c * stride
            # The sky's ramp is the CONJUGATE of what the combine applies, so a correct tau
            # cancels it exactly and the sum adds in phase.
            ph = cmath.exp(-2j * cmath.pi * fid * CHAN_HZ * tau_s)
            a = amp * ph
            if noise > 0.0:
                a += complex(rng.gauss(0, noise), rng.gauss(0, noise))
            pts.append((fid, a, 1.0, key))
    return pts


class TestDerivedRotation(unittest.TestCase):

    def test_correct_tau_sums_in_phase(self):
        pts = make_points(tau_chips=0.7)
        r = delay_combine(pts, CHIP_HZ, CHAN_HZ, tau_chips=0.7)
        self.assertAlmostEqual(r["coh_frac"], 1.0, places=6)
        self.assertEqual(r["n_inst"], 12)
        self.assertEqual(r["n_pts"], 84)          # 12 instances x 7 channels

    def test_wrong_tau_decoheres(self):
        """The whole point of deriving it: a wrong tau must COST, or the statistic is not
        measuring what it claims. A fitted per-instance phase would hide this."""
        pts = make_points(tau_chips=0.7)
        good = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.7)["coh_frac"]
        bad = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.0)["coh_frac"]
        self.assertGreater(good, 0.99)
        self.assertLess(bad, 0.5)

    def test_the_ramp_spans_wraps_within_one_instance(self):
        """Pins the geometry that makes a per-instance constant impossible: 7 channels at
        stride 16 span 96 channels = 18.75 MHz, so tau = 1 chip is ~1.8 wraps INSIDE one
        instance. If this ever shrinks, the per-instance shortcut becomes tenable again and
        this whole change should be re-argued rather than assumed."""
        span_hz = 6 * 16 * CHAN_HZ                       # first to last channel of one instance
        wraps = span_hz * (1.0 / CHIP_HZ)                # cycles at tau = 1 chip
        self.assertGreater(wraps, 1.5)
        self.assertAlmostEqual(span_hz / 1e6, 18.75, places=2)

    def test_single_instance_alone_still_needs_the_per_channel_ramp(self):
        """Even ONE instance decoheres without it -- the loss is intra-instance, before any
        cross-instance step. This is what the old channel-sum threw away."""
        pts = [p for p in make_points(tau_chips=0.7) if p[3] == INSTANCES[0]]
        good = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.7)["coh_frac"]
        bad = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.0)["coh_frac"]
        self.assertGreater(good, 0.99)
        self.assertLess(bad, 0.6)


class TestShuffledNull(unittest.TestCase):

    def test_shuffled_null_kills_a_noise_only_set(self):
        """⚠️ THE GATE THAT COULD NEVER FAIL BEFORE. Pure noise, no common ramp: the derived
        rotation has nothing to align, so the sum must not clear its own shuffled null."""
        rng = random.Random(11)
        pts = [(5972 + i * 16, complex(rng.gauss(0, 1), rng.gauss(0, 1)), 1.0,
                INSTANCES[i % 12]) for i in range(84)]
        r = delay_combine(pts, CHIP_HZ, CHAN_HZ, tau_chips=0.3, rng=random.Random(2), n_null=32)
        self.assertFalse(r["present"], "noise cleared the null: the gate is not a gate")

    def test_real_signal_clears_the_null(self):
        """...and the same gate must PASS on signal, or it is just a refusal."""
        pts = make_points(tau_chips=0.4, amp=1.0, noise=0.35)
        r = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.4, rng=random.Random(2), n_null=32)
        self.assertTrue(r["present"])
        self.assertGreater(r["snr"], 1.5)

    def test_null_shuffles_amplitudes_not_frequencies(self):
        """Shuffle, never roll: a roll IS a delay and would leave the very structure the null
        is meant to destroy. Same frequencies, same derotation, amplitudes reassigned."""
        pts = make_points(tau_chips=0.4)
        r1 = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.4, rng=random.Random(5), n_null=16)
        r2 = delay_combine(pts, CHIP_HZ, CHAN_HZ, 0.4, rng=random.Random(5), n_null=16)
        self.assertEqual(r1["floor"], r2["floor"])       # deterministic given the rng
        self.assertLess(r1["floor"], abs(r1["amplitude"]) * r1["n_pts"])

    def test_too_few_points_declines(self):
        self.assertIsNone(delay_combine([], CHIP_HZ, CHAN_HZ, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
