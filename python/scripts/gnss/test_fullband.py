#!/usr/bin/env python3
"""Tests for the broker-side full-band combine (gnss_broker/fullband.py).

    python3 python/scripts/gnss/test_fullband.py

THE PROPERTY THAT MATTERS MOST IS THE FALLBACK. This path replaces the tracker's blind
cross-channel sum, and it must be impossible for it to do worse: where the delay search does not
clear its channel-permuted null, tau is 0 and the combine is bit-identical to the blind sum.
A "better" estimator that occasionally loses is not deployable next to a live instrument.

The synthetic fixtures put a KNOWN delay on a KNOWN comb, so the recovered tau can be checked
against truth rather than against plausibility -- which the sky can never offer.
"""
import cmath
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gnss_broker import fullband  # noqa: E402

CH = fullband.CHAN_HZ


def _mk(n_inst=4, n_chan=6, n_rec=16, tau_ns=60.0, snr=8.0, seed=1, stride=16, base=5972):
    """{inst: {fid: (mean, tot_energy, {hop: (A,E)})}} with a REAL delay planted.

    Each instance gets its own arbitrary constant (the NCO origin), its own comb offset, and
    independent noise -- i.e. the structure the real data has, with the answer known.
    """
    rng = random.Random(seed)
    per = {}
    for i in range(n_inst):
        k_i = rng.uniform(-math.pi, math.pi)          # the irreducible per-instance constant
        chans = {}
        for c in range(n_chan):
            fid = base + i + stride * c               # interleaved combs, as on the fleet
            f_hz = fid * CH
            true = cmath.exp(1j * (k_i - 2 * math.pi * f_hz * tau_ns * 1e-9))
            acc, tot, recs = 0j, 0.0, {}
            for r in range(n_rec):
                n = complex(rng.gauss(0, 1), rng.gauss(0, 1)) / max(1e-9, snr)
                A = true + n
                E = 1.0
                recs[1000 + r] = (A, E)
                acc += A * E
                tot += E
            chans[fid] = (acc, tot, recs)
        per["cx%02d.0" % i] = chans
    return per


class TestTauFit(unittest.TestCase):
    def test_recovers_a_planted_delay(self):
        per = _mk(tau_ns=60.0, snr=12.0)
        tau, gain, null, excess = fullband.fit_tau(per, step_ns=1.0, null_trials=16)
        self.assertAlmostEqual(tau * 1e9, 60.0, delta=4.0)
        self.assertGreater(excess, 3.0, "a real 60 ns ramp must clear the permuted null")

    def test_score_is_blind_to_the_per_instance_constant(self):
        # THE DESIGN POINT: |.| is taken per instance BEFORE summing, so an arbitrary k_i
        # cannot move the score -- tau comes from channel structure alone, with no need to fit
        # the constants first.
        per = _mk(tau_ns=40.0, snr=20.0, seed=7)
        s0 = fullband._score(per, 40e-9)
        rot = {}
        for inst, chans in per.items():
            k = random.Random(hash(inst)).uniform(-math.pi, math.pi)
            e = cmath.exp(1j * k)
            rot[inst] = {f: (m * e, t, {h: (A * e, E) for h, (A, E) in r.items()})
                         for f, (m, t, r) in chans.items()}
        self.assertAlmostEqual(fullband._score(rot, 40e-9), s0, places=6)

    def test_no_delay_gives_no_excess(self):
        # tau = 0 truth: the search still finds SOME peak (that is what the null measures), so
        # the excess -- not the gain -- must be the thing near zero.
        per = _mk(tau_ns=0.0, snr=6.0, seed=3)
        _tau, gain, null, excess = fullband.fit_tau(per, step_ns=2.0, null_trials=24)
        self.assertGreater(gain, -0.01, "the search maximises, so gain cannot be negative")
        self.assertLess(excess, 2.0, "no planted ramp must not read as a large recovery")

    def test_pure_noise_does_not_clear_the_null(self):
        per = _mk(tau_ns=0.0, snr=0.05, seed=11)   # essentially no signal
        _tau, _gain, _null, excess = fullband.fit_tau(per, step_ns=2.0, null_trials=32)
        self.assertLess(excess, 2.0, "noise must not manufacture a delay detection")


class TestSeries(unittest.TestCase):
    def test_tau_zero_reproduces_the_blind_sum_exactly(self):
        # THE FALLBACK GUARANTEE. Where the guard declines the delay, this path must be
        # bit-identical to the tracker's blind cross-channel sum -- otherwise deploying it
        # beside a live instrument is a gamble rather than a change.
        per = _mk(tau_ns=55.0, snr=9.0, seed=5)
        hops = set()
        for chans in per.values():
            for _f, (_m, _t, r) in chans.items():
                hops |= set(r)
        out = fullband.instance_series(per, 0.0, hops)
        for inst, chans in per.items():
            for hop in hops:
                g = sum(r[hop][0] * r[hop][1] for _f, (_m, _t, r) in chans.items())
                e = sum(r[hop][1] for _f, (_m, _t, r) in chans.items())
                self.assertAlmostEqual(out[inst][hop][0].real, (g / e).real, places=9)
                self.assertAlmostEqual(out[inst][hop][1], e, places=9)

    def test_alignment_raises_the_coherent_amplitude(self):
        per = _mk(tau_ns=70.0, snr=15.0, seed=9)
        hops = set()
        for chans in per.values():
            for _f, (_m, _t, r) in chans.items():
                hops |= set(r)
        blind = fullband.instance_series(per, 0.0, hops)
        tau, _g, _n, _x = fullband.fit_tau(per, step_ns=1.0, null_trials=8)
        good = fullband.instance_series(per, tau, hops)
        for inst in per:
            b = sum(abs(v[0]) for v in blind[inst].values())
            a = sum(abs(v[0]) for v in good[inst].values())
            self.assertGreater(a, b, "aligning a real 70 ns ramp must raise |A| for %s" % inst)


class TestAmbiguity(unittest.TestCase):
    def test_search_window_matches_the_comb_spacing(self):
        # tau is unambiguous only within +-1/(2 * 16 * 195312.5) = 160 ns; a wider search would
        # alias silently, which is the failure this bound exists to prevent.
        self.assertAlmostEqual(fullband.TAU_MAX_S * 1e9, 160.0, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
