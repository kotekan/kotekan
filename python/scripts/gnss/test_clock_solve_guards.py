#!/usr/bin/env python3
"""Regression tests for the clock-solve guards (task #39).

Reproduces the 2026-08-10 19:39 UTC incident: PRN 2, which is not L5-capable, entered
the receiver-clock population as a noise/cross-correlation track and dragged the median
until the solve latched and never recovered. The offsets below are the real integrity
residuals from that run's broker log.

Run: python3 test_clock_solve_guards.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gps_distributed_broker import split_erratic_offsets   # noqa: E402

CODE_LEN = 10230.0
BOUND = 100.0
MAX_AGE = 120.0

# PRN 2's successive integrity residuals, read off /tmp/gnss_broker.log 19:47-19:55.
# Uniform over +-CODE_LEN/2 -- a real satellite cannot move 5000 chips in 30 s (that
# would be 500 km of range change).
PRN2 = [-3226.49, 2430.60, 2997.61, -4988.77, 3024.44, -4193.53,
        2897.95, 1297.45, 3231.55, 3549.15, 2304.64]
# Healthy satellites from the same lines: residuals of a few chips about a common clock.
HEALTHY = {1: [150.9, 151.2, 150.7, 151.4, 150.2, 151.0, 150.6, 151.3, 150.4, 150.8, 151.1],
           3: [152.5, 152.7, 153.1, 152.3, 152.9, 152.6, 153.0, 152.4, 152.8, 152.5, 152.9],
           8: [155.9, 155.0, 156.2, 155.4, 155.8, 155.1, 156.0, 155.5, 155.7, 155.2, 155.9],
           10: [147.0, 146.8, 147.3, 146.9, 147.1, 147.4, 146.7, 147.2, 147.0, 146.9, 147.3]}


def run_cycles(n, include_prn2=True):
    """Feed n cycles 30 s apart; return (kept_prns_per_cycle, dropped_per_cycle)."""
    hist, kept, dropped = {}, [], []
    for i in range(n):
        offs = [(prn, HEALTHY[prn][i]) for prn in sorted(HEALTHY)]
        if include_prn2:
            offs.append((2, PRN2[i] % CODE_LEN))
        k, d = split_erratic_offsets(offs, hist, i * 30.0, BOUND, MAX_AGE, CODE_LEN)
        kept.append(sorted(p for p, _ in k))
        dropped.append(sorted(p for p, _ in d))
    return kept, dropped


class TestErraticGuard(unittest.TestCase):

    def test_first_sighting_is_always_admitted(self):
        """The test needs two cycles. Refusing first sightings would stall the bootstrap."""
        kept, dropped = run_cycles(1)
        self.assertEqual(kept[0], [1, 2, 3, 8, 10])
        self.assertEqual(dropped[0], [])

    def test_noise_track_is_excluded_from_cycle_two(self):
        """THE INCIDENT: PRN 2 must be gone by the second cycle and stay gone."""
        kept, dropped = run_cycles(len(PRN2))
        for i in range(1, len(PRN2)):
            self.assertIn(2, dropped[i], "PRN 2 survived cycle %d: %r" % (i, dropped[i]))
            self.assertNotIn(2, kept[i])

    def test_healthy_satellites_are_never_excluded(self):
        """A guard that drops real satellites is worse than no guard."""
        kept, dropped = run_cycles(len(PRN2))
        for i, (k, d) in enumerate(zip(kept, dropped)):
            for prn in HEALTHY:
                self.assertIn(prn, k, "PRN %d wrongly dropped on cycle %d" % (prn, i))
            self.assertNotIn(prn, d)

    def test_one_bad_track_does_not_by_itself_break_the_median(self):
        """CALIBRATION, and it corrects the original #39 story.

        A circular median absorbs a minority. MEASURED against 4 healthy satellites:
        1-3 noise tracks leave the median at 152.6 with MAD 3-6; the break happens at
        4 of 8 -- median 383.6, MAD 233, i.e. past --dr-max-solve-mad-chips. So PRN 2
        alone did NOT latch the clock on 2026-08-10, and the guard's value is that it
        keeps the bad population BELOW half, not that it rescues a majority-noise
        solve. What drove the other satellites bad that evening is still open.
        """
        def med_mad(vals):
            ref = vals[0]
            cen = sorted(((d - ref + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2 for d in vals)
            m = cen[len(cen) // 2]
            return ((m + ref) % CODE_LEN,
                    sorted(abs(c - m) for c in cen)[len(cen) // 2])

        base = [HEALTHY[p][0] for p in sorted(HEALTHY)]
        _, mad1 = med_mad(base + [3024.44])
        self.assertLess(mad1, BOUND, "one outlier in five should NOT trip the MAD bound")
        _, mad4 = med_mad(base + [3024.44, 7301.0, 9110.0, 512.0])
        self.assertGreater(mad4, BOUND, "four of eight should trip it")

    def test_median_is_protected_once_bad_tracks_approach_half(self):
        """The regime that matters: the guard must hold the median at the true clock."""
        hist = {}
        erratic = {2: (-3226.49, 3024.44), 91: (8100.0, 1500.0),
                   92: (400.0, 6900.0), 93: (9900.0, 2750.0)}
        for cycle in (0, 1):
            offs = [(prn, HEALTHY[prn][cycle]) for prn in sorted(HEALTHY)]
            offs += [(prn, v[cycle] % CODE_LEN) for prn, v in sorted(erratic.items())]
            keep, drop = split_erratic_offsets(offs, hist, cycle * 30.0,
                                               BOUND, MAX_AGE, CODE_LEN)

        def med_mad(vals):
            ref = vals[0]
            cen = sorted(((d - ref + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2 for d in vals)
            m = cen[len(cen) // 2]
            return ((m + ref) % CODE_LEN,
                    sorted(abs(c - m) for c in cen)[len(cen) // 2])

        self.assertEqual(sorted(p for p, _ in drop), sorted(erratic))
        # MAD is what the code actually gates on, and unlike the median it is inflated
        # by noise wherever it lands -- so this is the robust statement of the property.
        _, mad_raw = med_mad([d for _, d in offs])
        med_ok, mad_ok = med_mad([d for _, d in keep])
        self.assertGreater(mad_raw, BOUND,
                           "unguarded population should trip the MAD bound -> solve REFUSED")
        self.assertLess(mad_ok, BOUND,
                        "guarded population must pass the MAD bound so the solve proceeds")
        self.assertLess(abs(med_ok - 151.4), 6.0,
                        "guarded median %.1f is not the real clock" % med_ok)

    def test_stale_history_does_not_reject(self):
        """Across a long gap the clock really can have moved; a stale compare must not fire."""
        hist = {5: (0.0, 100.0)}
        keep, drop = split_erratic_offsets([(5, 9000.0)], hist, MAX_AGE + 1.0,
                                           BOUND, MAX_AGE, CODE_LEN)
        self.assertEqual(drop, [])
        self.assertEqual([p for p, _ in keep], [5])

    def test_wrap_is_circular_not_linear(self):
        """1 chip and 10229 chips are 2 chips apart on a 10230-chip code, not 10228."""
        hist = {7: (0.0, 1.0)}
        keep, drop = split_erratic_offsets([(7, CODE_LEN - 1.0)], hist, 30.0,
                                           BOUND, MAX_AGE, CODE_LEN)
        self.assertEqual(drop, [], "wrap treated as a jump -- the modulus is missing")
        self.assertEqual([p for p, _ in keep], [7])

    def test_dropped_track_can_rejoin_when_it_settles(self):
        """History is recorded for dropped satellites too, so recovery is possible."""
        hist = {}
        split_erratic_offsets([(9, 100.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN)
        _, d1 = split_erratic_offsets([(9, 5000.0)], hist, 30.0, BOUND, MAX_AGE, CODE_LEN)
        self.assertEqual([p for p, _ in d1], [9])
        # now it stops moving: two consistent cycles and it is back
        k2, d2 = split_erratic_offsets([(9, 5001.0)], hist, 60.0, BOUND, MAX_AGE, CODE_LEN)
        self.assertEqual(d2, [])
        self.assertEqual([p for p, _ in k2], [9])




class TestDopplerLever(unittest.TestCase):
    """The guard's original premise was FALSE, and these pin the correction.

    d_i is not clk + b_i. It carries the detection's Doppler multiplied by t_i, and t_i is
    seconds since F-engine SAMPLE 0 -- 193,674 s on 2026-08-11, giving 1684 chips per Hz.
    So the 100-chip bound was a 0.059 Hz bound, and the search's own Doppler moves several
    Hz between passes. MEASURED on sky: 54 ejections in 55 min, median jump 2278 chips,
    every one mapping to an ordinary Doppler step -- and the ejected satellites were
    STRONGER than the fleet median (deep_snr 81 vs 52) with none inside 120 s of a re-seed.
    They were tracking perfectly and were thrown out of the clock solve anyway.
    """

    T_I = 193674.0          # s since sample 0, as measured
    CHIP = 10.23e6
    CARR = 1176.45e6
    LEV = T_I * CHIP / CARR   # ~1684 chips per Hz

    def lever(self, prn, dop):
        return {prn: self.T_I * self.CHIP * 1.0 * dop / self.CARR}

    def test_the_lever_is_what_we_measured(self):
        self.assertAlmostEqual(self.LEV, 1684.0, delta=5.0)
        self.assertAlmostEqual(100.0 / self.LEV, 0.059, delta=0.002,
                               msg="the 100-chip bound should be ~0.06 Hz")

    def test_a_doppler_step_no_longer_ejects_a_good_satellite(self):
        """THE BUG. A satellite whose Doppler estimate moves 1.5 Hz -- utterly ordinary --
        shifts d_i by ~2500 chips with clk and b_i both perfectly constant."""
        hist = {}
        d0, dop0 = 150.0, 1000.0
        d1 = (d0 + self.LEV * 1.5) % CODE_LEN          # same clk+b_i, Doppler moved 1.5 Hz
        split_erratic_offsets([(5, d0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN,
                              lever=self.lever(5, dop0))
        keep, drop = split_erratic_offsets([(5, d1)], hist, 30.0, BOUND, MAX_AGE, CODE_LEN,
                                           lever=self.lever(5, dop0 + 1.5))
        self.assertEqual(drop, [], "a 1.5 Hz Doppler step ejected a healthy satellite")
        self.assertEqual([p for p, _ in keep], [5])

    def test_a_real_code_jump_is_STILL_caught(self):
        """The guard must not be defanged: with the Doppler term identical, a genuine
        code discontinuity is exactly what it is there to catch (the PRN 2 incident)."""
        hist = {}
        split_erratic_offsets([(2, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN,
                              lever=self.lever(2, 1000.0))
        keep, drop = split_erratic_offsets([(2, 150.0 + 2430.0)], hist, 30.0, BOUND,
                                           MAX_AGE, CODE_LEN, lever=self.lever(2, 1000.0))
        self.assertEqual([p for p, _ in drop], [2], "a real 2430-chip jump was not caught")
        self.assertEqual(keep, [])

    def test_doppler_step_AND_code_jump_is_caught(self):
        """Removing the Doppler term must not let a real jump hide behind one."""
        hist = {}
        split_erratic_offsets([(9, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN,
                              lever=self.lever(9, 1000.0))
        d1 = (150.0 + self.LEV * 1.5 + 2000.0) % CODE_LEN
        _, drop = split_erratic_offsets([(9, d1)], hist, 30.0, BOUND, MAX_AGE, CODE_LEN,
                                        lever=self.lever(9, 1001.5))
        self.assertEqual([p for p, _ in drop], [9])
        self.assertAlmostEqual(drop[0][1], 2000.0, delta=1.0,
                               msg="the reported jump should be the CODE part alone")

    def test_old_behaviour_without_lever(self):
        """lever=None must reproduce the pre-fix behaviour exactly, so the 8 original
        tests remain a valid description of the unlevered path."""
        hist = {}
        split_erratic_offsets([(3, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN)
        _, drop = split_erratic_offsets([(3, 150.0 + 2500.0)], hist, 30.0, BOUND,
                                        MAX_AGE, CODE_LEN)
        self.assertEqual([p for p, _ in drop], [3])

    def test_history_carries_the_lever_across_a_gap(self):
        """A satellite that goes stale and returns must not be judged against a lever from
        a different Doppler epoch -- it is kept (no fresh history), and the NEW lever is
        what the next cycle differences against."""
        hist = {}
        split_erratic_offsets([(4, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN,
                              lever=self.lever(4, 1000.0))
        keep, drop = split_erratic_offsets([(4, 8000.0)], hist, MAX_AGE + 1.0, BOUND,
                                           MAX_AGE, CODE_LEN, lever=self.lever(4, 2000.0))
        self.assertEqual(drop, [], "a stale satellite must rejoin, not be judged")
        self.assertEqual(hist[4][2], self.lever(4, 2000.0)[4],
                         "history must record the NEW lever")


if __name__ == "__main__":
    unittest.main(verbosity=2)
