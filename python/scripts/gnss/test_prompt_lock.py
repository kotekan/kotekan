"""Task #47: the served C/N0 must not claim a satellite the prompt tap never saw.

THE DEFECT THESE PIN. cn0_coh is built from deep_snr, and the deep fold RE-SEARCHES rate and
phase -- it re-finds the satellite regardless of where the prompt tap was commanded. On
2026-08-12 15:20-15:45 UTC every chain served ~41 dB-Hz with dll_disc ~= 0.013 while the prompt
taps sat on noise (s4_raw 1.07, cn0_inc 21.3, per-instance deep_snr 1.5-2.5, `present` 0-1 of
11). That window was reported as the array's BEST look of the day. 7-30% of all bright-looking
samples over 08-11/08-12 carry the same signature.

⚠️ These tests must VARY THE AXIS (see chord-a-gate-that-cannot-fail): each asserts that a
blind row and a tracking row come out DIFFERENT, so a flag hardwired to either answer fails.
"""
import math
import unittest

from gnss_broker.publish import PROMPT_RAYLEIGH_S4
from gnss_observables import cn0_dbhz


def prompt_lock(s4_raw, fleet_present):
    """The rule as implemented in publish.py, isolated so the truth table is testable without
    standing up a FleetPublisher. Blind only when BOTH witnesses agree."""
    rayleigh = (s4_raw is not None and s4_raw >= PROMPT_RAYLEIGH_S4)
    return not (rayleigh and not fleet_present)


class TestPromptLockTruthTable(unittest.TestCase):
    """The flag is deliberately conservative: one witness alone never condemns a satellite."""

    def test_blind_row_is_flagged(self):
        # The measured 2026-08-12 blind state: Rayleigh prompt AND below the live floor.
        self.assertFalse(prompt_lock(s4_raw=1.07, fleet_present=False))

    def test_tracking_row_is_not_flagged(self):
        # The measured tracking state from the same day, same fleet.
        self.assertTrue(prompt_lock(s4_raw=0.47, fleet_present=True))

    def test_blind_and_tracking_differ(self):
        """The axis test: a flag stuck at either constant fails here even if both cases above
        were somehow satisfied by accident."""
        self.assertNotEqual(prompt_lock(1.07, False), prompt_lock(0.47, True))

    def test_scintillation_alone_does_not_condemn(self):
        """s4_raw is absolute but strong scintillation drives it up on a REAL signal. If the
        prompt still clears the live noise floor, the satellite is tracking."""
        self.assertTrue(prompt_lock(s4_raw=1.02, fleet_present=True))

    def test_uniformly_strong_fleet_does_not_condemn(self):
        """`present` is RELATIVE (p_pow vs the population median), so a fleet in which every
        satellite is strong reads low presence for all of them. That must not be read as blind
        -- this is exactly why presence cannot be the whole test."""
        self.assertTrue(prompt_lock(s4_raw=0.30, fleet_present=False))

    def test_missing_s4_does_not_condemn(self):
        """A combiner that does not report s4_raw leaves us one witness short. Declining to
        flag is the safe direction: a false 'blind' would hide real data."""
        self.assertTrue(prompt_lock(s4_raw=None, fleet_present=False))

    def test_bar_sits_in_the_measured_gap(self):
        """The bar was derived from sky, not chosen. Blind measured 1.00-1.05, tracking
        0.17-0.82; the bar must fall strictly between, or it is inside a population."""
        self.assertGreater(PROMPT_RAYLEIGH_S4, 0.82)
        self.assertLess(PROMPT_RAYLEIGH_S4, 1.00)


class TestObservablesUsesPublishedCn0(unittest.TestCase):
    """The #35 regression: gnss_observables recomputed cn0 from the row's OVERRIDDEN deep_snr
    instead of taking the broker's cn0_coh_db, re-mixing the estimator populations that #35
    separated. Measured live on gal_e5a: up to +9.7 dB hot, worst on `coh_src: quad:12`."""

    def test_published_value_wins_over_recomputation(self):
        # A quadrature-fallback row: deep_snr is the inflated fleet-population number, while
        # cn0_coh_db is the best single instance's honest one. They must not be interchangeable.
        row = {"cn0_coh_db": 21.98, "deep_snr": 39.26,
               "coherence_s": 1.0486, "coh_src": "quad:12"}
        got = cn0_dbhz(row, row["deep_snr"], row["coherence_s"])
        self.assertAlmostEqual(got, 21.98, places=6)
        naive = 20 * math.log10(39.26) - 10 * math.log10(1.0486)
        self.assertGreater(naive - got, 9.0,
                           "fixture no longer reproduces the inflation this test pins")

    def test_falls_back_only_when_broker_is_old(self):
        """An older broker publishes no cn0_coh_db; we must still emit something rather than
        drop the observable entirely."""
        row = {"deep_snr": 39.26, "coherence_s": 1.0486}
        got = cn0_dbhz(row, row["deep_snr"], row["coherence_s"])
        self.assertAlmostEqual(got, 20 * math.log10(39.26) - 10 * math.log10(1.0486), places=6)

    def test_no_coherence_yields_none(self):
        self.assertIsNone(cn0_dbhz({}, 39.26, 0.0))

    def test_published_and_recomputed_differ_on_this_row(self):
        """Axis test: if the two paths ever agree on this fixture, the test above proves
        nothing about which one is being used."""
        row = {"cn0_coh_db": 21.98, "deep_snr": 39.26, "coherence_s": 1.0486}
        self.assertNotAlmostEqual(cn0_dbhz(row, row["deep_snr"], row["coherence_s"]),
                                  cn0_dbhz({}, row["deep_snr"], row["coherence_s"]), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
