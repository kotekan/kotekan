#!/usr/bin/env python3
"""The per-subband spectrum archive (task #25).

The archive exists because a COMBINED number can always be rebuilt from the parts
while the parts can never be recovered from a combined number. These tests pin the
properties that make that true: every point survives, the axes are labelled, and a
failure cannot take the broker down.

Run: python3 test_spectrum_archive.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gps_distributed_broker import make_spectrum_writer   # noqa: E402

# fleet_spectrum's shape: {prn: [(freq_id, amplitude, energy, instance)]}
SPEC = {
    11: [(0, 1.5, 2.25, "cx19/0"), (1, 1.4, 1.96, "cx19/0"),
         (0, 1.6, 2.56, "cx27/1"), (1, 1.3, 1.69, "cx27/1")],
    21: [(0, 0.8, 0.64, "cx19/0"), (3, 0.9, 0.81, "cx42/1")],
}


class TestSpectrumArchive(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _rows(self, path):
        return [json.loads(l) for l in open(path)]

    def test_off_by_default(self):
        """No path, no writer -- and the call site checks exactly this."""
        self.assertIsNone(make_spectrum_writer(""))
        self.assertIsNone(make_spectrum_writer(None))

    def test_every_point_survives(self):
        """THE POINT OF THE ARCHIVE. Six input points must produce six rows -- no
        collapse, no dedup, no per-PRN reduction."""
        p = os.path.join(self.d, "spec.jsonl")
        w = make_spectrum_writer(p)
        n = w(1786400000.0, "1176.45MHz", SPEC)
        self.assertEqual(n, 6)
        rows = self._rows(p)
        self.assertEqual(len(rows), 6)
        got = {(r["prn"], r["freq_id"], r["inst"]) for r in rows}
        want = {(prn, p_[0], p_[3]) for prn, pts in SPEC.items() for p_ in pts}
        self.assertEqual(got, want, "the (prn, freq, instance) axes must be preserved")

    def test_values_are_raw(self):
        """No normalisation, no combine -- amplitudes land exactly as measured."""
        p = os.path.join(self.d, "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC)
        r = [x for x in self._rows(p) if x["prn"] == 11 and x["freq_id"] == 0
             and x["inst"] == "cx19/0"][0]
        self.assertAlmostEqual(r["amp"], 1.5, places=9)
        self.assertAlmostEqual(r["energy"], 2.25, places=9)
        self.assertEqual(r["band"], "b")

    def test_appends_across_polls(self):
        p = os.path.join(self.d, "spec.jsonl")
        w = make_spectrum_writer(p)
        w(1786400000.0, "b", SPEC)
        w(1786400002.0, "b", SPEC)
        rows = self._rows(p)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["t"] for r in rows}), 2, "epochs must stay distinguishable")

    def test_rolls_by_day_without_restart(self):
        """%Y%m%d expands, so a multi-day run splits files on its own."""
        p = os.path.join(self.d, "spec_%Y%m%d.jsonl")
        w = make_spectrum_writer(p)
        w(1786400000.0, "b", SPEC)                 # 2026-08-11
        w(1786400000.0 + 86400 * 2, "b", SPEC)     # two days later
        files = sorted(f for f in os.listdir(self.d) if f.startswith("spec_"))
        self.assertEqual(len(files), 2, "expected one file per UTC day, got %r" % files)

    def test_creates_missing_directory(self):
        p = os.path.join(self.d, "nested", "deeper", "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC)
        self.assertTrue(os.path.exists(p))

    def test_flushes_so_a_reader_sees_rows_immediately(self):
        """An archive nobody can read until the broker exits is not an archive."""
        p = os.path.join(self.d, "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC)
        self.assertEqual(len(self._rows(p)), 6)

    def test_empty_spec_is_harmless(self):
        p = os.path.join(self.d, "spec.jsonl")
        self.assertEqual(make_spectrum_writer(p)(1786400000.0, "b", {}), 0)

    def test_longer_tuples_do_not_truncate(self):
        """A future field appended to the point tuple must not silently drop the row."""
        p = os.path.join(self.d, "spec.jsonl")
        w = make_spectrum_writer(p)
        n = w(1786400000.0, "b", {7: [(2, 1.1, 1.21, "cx19/0", "extra", 42)]})
        self.assertEqual(n, 1)
        self.assertEqual(self._rows(p)[0]["freq_id"], 2)

    def test_combined_value_is_recoverable_from_the_parts(self):
        """The archive's whole justification, asserted rather than assumed."""
        p = os.path.join(self.d, "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC)
        rows = self._rows(p)
        for prn, pts in SPEC.items():
            got = sum(r["energy"] for r in rows if r["prn"] == prn)
            self.assertAlmostEqual(got, sum(x[2] for x in pts), places=9)




class TestArchiveEpoch(unittest.TestCase):
    """The archive must be JOINABLE with the observables record, which means UTC.

    Caught on first deployment: the call site passed `t_now_abs` -- seconds since the
    F-engine's sample-0 anchor, not a Unix epoch -- and the first file was named
    gps_l5_19700102.jsonl with every row stamped in 1970. A per-subband archive that
    cannot be joined to az/el is not a beam-map input.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_filename_and_rows_use_the_utc_passed_in(self):
        p = os.path.join(self.d, "spec_%Y%m%d.jsonl")
        t = 1786400000.0
        want = "spec_%s.jsonl" % time.strftime("%Y%m%d", time.gmtime(t))
        make_spectrum_writer(p)(t, "b", SPEC, t_rx=1234.5)
        files = os.listdir(self.d)
        self.assertEqual(files, [want],
                         "filename did not use the UTC passed in: %r" % files)
        r = json.loads(open(os.path.join(self.d, files[0])).readline())
        self.assertAlmostEqual(r["t"], t, places=3)
        self.assertGreater(r["t"], 1.7e9, "row timestamp is not a plausible Unix epoch")

    def test_receiver_relative_time_is_kept_alongside(self):
        """t_rx is what code phases are referenced to -- keep it, just don't name files by it."""
        p = os.path.join(self.d, "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC, t_rx=1234.5)
        r = json.loads(open(p).readline())
        self.assertAlmostEqual(r["t_rx"], 1234.5, places=3)

    def test_t_rx_optional(self):
        p = os.path.join(self.d, "spec.jsonl")
        make_spectrum_writer(p)(1786400000.0, "b", SPEC)
        self.assertIsNone(json.loads(open(p).readline())["t_rx"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
