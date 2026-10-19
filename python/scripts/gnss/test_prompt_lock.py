"""The served C/N0: gnss_observables must emit the broker's published cn0_coh_db, not a value
it recomputes from the row's deep_snr.

These tests vary the axis: the fixture row is one on which the published and the recomputed
values differ, so a writer that used either path unconditionally fails one of them.
"""
import math
import unittest

from gnss_observables import cn0_dbhz


class TestObservablesUsesPublishedCn0(unittest.TestCase):
    """publish.py overrides the row's deep_snr with the fleet-coherent or quadrature-fallback
    value, so a C/N0 recomputed from it mixes estimator populations and reads up to ~10 dB hot
    on quad rows; cn0_coh_db is built from the best single instance."""

    def test_published_value_wins_over_recomputation(self):
        # A quadrature-fallback row: deep_snr is the inflated fleet-population number, while
        # cn0_coh_db is the best single instance's honest one. They must not be interchangeable.
        row = {
            "cn0_coh_db": 21.98,
            "deep_snr": 39.26,
            "coherence_s": 1.0486,
            "coh_src": "quad:12",
        }
        got = cn0_dbhz(row, row["deep_snr"], row["coherence_s"])
        self.assertAlmostEqual(got, 21.98, places=6)
        naive = 20 * math.log10(39.26) - 10 * math.log10(1.0486)
        self.assertGreater(
            naive - got,
            9.0,
            "fixture no longer reproduces the inflation this test pins",
        )

    def test_falls_back_only_when_broker_is_old(self):
        """A row without cn0_coh_db falls back to the recomputation rather than dropping the
        observable."""
        row = {"deep_snr": 39.26, "coherence_s": 1.0486}
        got = cn0_dbhz(row, row["deep_snr"], row["coherence_s"])
        self.assertAlmostEqual(
            got, 20 * math.log10(39.26) - 10 * math.log10(1.0486), places=6
        )

    def test_a_published_none_is_not_recomputed(self):
        """publish.py always writes cn0_coh_db and sets it None when the best instance has no
        deep_snr or coherent span; the row's fleet/quad deep_snr must not fill that in."""
        row = {
            "cn0_coh_db": None,
            "deep_snr": 39.26,
            "coherence_s": 1.0486,
            "coh_src": "quad:12",
        }
        self.assertIsNone(cn0_dbhz(row, row["deep_snr"], row["coherence_s"]))

    def test_no_coherence_yields_none(self):
        self.assertIsNone(cn0_dbhz({}, 39.26, 0.0))

    def test_published_and_recomputed_differ_on_this_row(self):
        """Axis test: if the two paths ever agree on this fixture, the test above proves
        nothing about which one is being used."""
        row = {"cn0_coh_db": 21.98, "deep_snr": 39.26, "coherence_s": 1.0486}
        self.assertNotAlmostEqual(
            cn0_dbhz(row, row["deep_snr"], row["coherence_s"]),
            cn0_dbhz({}, row["deep_snr"], row["coherence_s"]),
            places=3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
