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




class TestDopplerEmbedCancellation(unittest.TestCase):
    """Pins the 2026-08-11 evening correction, which RETRACTED the same afternoon's
    "Doppler lever" version of this guard.

    The lever arithmetic (t_i * chip_rate / carrier ~ 1684 chips/Hz at 2.24 days of run
    age) is real, but it appears in the search's published cp0 with a MINUS sign
    (gnssSeedTransport.cpp detection_phase: cp0 = best_cp - off - drift) and in the
    broker's cp_loc with a PLUS sign, same dop, same ref_hop -- so raw d_i never carried
    it at all. MEASURED live 2026-08-11: cp_loc continuity over 1423 consecutive
    detections is median 0.27 chips with Doppler jitter of +-60 Hz pass to pass.

    The lever-subtracting guard therefore RE-INTRODUCED the term: (d - lev) stepped
    1696 chips per Hz of Doppler re-estimate, flagged every satellite every cycle, hit
    the min-sats floor and kept everything -- a silent no-op, live for ~2.5 h, and the
    "before/after" measurement used to justify it was made against that no-op.
    (Same class as the gates in [a-gate-that-cannot-fail]: it could not NOT fire.)
    """

    T_I = 195032.0          # s since sample 0, as measured 2026-08-11 20:43
    CHIP = 10.23e6
    CARR = 1176.45e6
    LEV = T_I * CHIP / CARR   # ~1696 chips per Hz

    def test_the_embed_cancels_in_cp_loc(self):
        """The whole argument in one assertion: build cp0 the way the search does,
        reconstruct cp_loc the way the broker does, step the Doppler estimate by an
        ordinary pass-to-pass jitter, and cp_loc must not move."""
        best_cp = 4321.0                       # physical phase at the snapshot start
        for dop in (1000.0, 1061.0, 875.5, 1125.0):   # +-60-125 Hz pass jitter
            t_i = self.T_I
            cp0 = (best_cp - t_i * self.CHIP * (1.0 + dop / self.CARR)) % CODE_LEN
            cp_loc = (cp0 + t_i * self.CHIP * (1.0 + dop / self.CARR)) % CODE_LEN
            err = ((cp_loc - best_cp + CODE_LEN / 2) % CODE_LEN) - CODE_LEN / 2
            self.assertLess(abs(err), 1e-3,
                            "the embed failed to cancel at dop %.1f" % dop)

    def test_an_ordinary_doppler_step_does_not_eject(self):
        """A 1.5 Hz Doppler re-estimate leaves d_i untouched (the embed cancels), so the
        plain guard keeps the satellite. This is what the lever version got wrong: it
        subtracted the lever from a quantity that never contained it."""
        hist = {}
        split_erratic_offsets([(5, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN)
        # same clk + b_i; the Doppler moved 1.5 Hz and d_i therefore did NOT move
        keep, drop = split_erratic_offsets([(5, 150.4)], hist, 30.0, BOUND,
                                           MAX_AGE, CODE_LEN)
        self.assertEqual(drop, [])
        self.assertEqual([p for p, _ in keep], [5])

    def test_the_lever_guard_would_have_flagged_everyone(self):
        """The failure mode of the retracted version, kept as a spec of what NOT to
        rebuild: subtracting a per-cycle lever from a lever-free d_i steps the tested
        quantity by ~1696 chips/Hz of ddop, over the 100-chip bound for any
        ddop > 0.059 Hz -- and pass-to-pass ddop is tens of Hz."""
        ddop = 1.5                                    # an UNREMARKABLE step
        stepped = abs(self.LEV * ddop)                # what (d - lev) moved by
        self.assertGreater(stepped, 20.0 * BOUND,
                           "1.5 Hz should overwhelm the bound 20x over")
        self.assertLess(100.0 / self.LEV, 0.06,
                        "the bound in Doppler units was ~0.059 Hz -- unmeetable")

    def test_a_real_code_jump_is_still_caught(self):
        """The PRN 2 incident remains the guard's reason to exist."""
        hist = {}
        split_erratic_offsets([(2, 150.0)], hist, 0.0, BOUND, MAX_AGE, CODE_LEN)
        keep, drop = split_erratic_offsets([(2, 150.0 + 2430.0)], hist, 30.0, BOUND,
                                           MAX_AGE, CODE_LEN)
        self.assertEqual([p for p, _ in drop], [2])
        self.assertEqual(keep, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
