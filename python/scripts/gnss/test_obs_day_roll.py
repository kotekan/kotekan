#!/usr/bin/env python3
"""The observables record must be filed under the day it was MEASURED.

THE BUG THIS PINS (2026-09-10): gnss_observables expanded its --out template once, at
startup, and held that handle for the life of the process. A logger begun on 09-05 wrote
<chain>_20260905.jsonl until it was restarted on 09-09 -- four days and 2.0 GB of rows in
a file named for the first of them, while 09-06, 09-07 and 09-08 had no observables file
at all. Nothing failed: the rows were correct, the filename was a lie, and every consumer
that opens a day by name saw an empty day.

Both the --out help text and obs_up.sh already claimed the file rolled at UTC midnight, so
the two places a reader would check to confirm the behaviour both asserted it. That is why
these tests assert the ROLL, not the docstring.

Run: python3 test_obs_day_roll.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_observables import make_obs_writer   # noqa: E402

# 2026-09-09 23:59:59 UTC and the two seconds either side of the boundary.
T_LATE = 1788998399.0
DAY_S = 86400.0


class TestObsDayRoll(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.tmpl = os.path.join(self.d, "gps_l5_%Y%m%d.jsonl")
        self.logged = []
        self.writers = []

    def tearDown(self):
        for w in self.writers:
            w.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def _w(self, tmpl=None):
        w = make_obs_writer(tmpl or self.tmpl, "GPS_L5 [G/gps_l5]", self.logged.append)
        self.writers.append(w)
        return w

    def _rows(self, name):
        with open(os.path.join(self.d, name)) as fh:
            return [json.loads(l) for l in fh]

    def test_rolls_at_utc_midnight(self):
        """THE REGRESSION. One writer, epochs either side of midnight, two files."""
        w = self._w()
        w(T_LATE, {"prn": 1, "t": T_LATE})
        w(T_LATE + 2.0, {"prn": 1, "t": T_LATE + 2.0})
        self.assertEqual(sorted(os.listdir(self.d)),
                         ["gps_l5_20260909.jsonl", "gps_l5_20260910.jsonl"])
        self.assertEqual(len(self._rows("gps_l5_20260909.jsonl")), 1)
        self.assertEqual(len(self._rows("gps_l5_20260910.jsonl")), 1)

    def test_a_row_lands_under_the_day_it_was_measured(self):
        """Filed by the ROW's epoch, not by wall-clock-at-poll: the two differ either
        side of midnight and only the epoch is what consumers join on."""
        w = self._w()
        for k in range(4):
            w(T_LATE - k * DAY_S, {"prn": 7, "t": T_LATE - k * DAY_S})
        for day in ("20260906", "20260907", "20260908", "20260909"):
            rows = self._rows("gps_l5_%s.jsonl" % day)
            self.assertEqual(len(rows), 1, "%s should hold exactly its own row" % day)
            self.assertEqual(day, _utcday(rows[0]["t"]),
                             "row in %s was not measured on that day" % day)

    def test_no_row_is_dropped_or_duplicated_across_a_roll(self):
        """A roll is a reopen, never a filter -- 240 rows in, 240 rows out."""
        w = self._w()
        n = 240
        for k in range(n):                       # 2 h of 30 s epochs across the boundary
            w(T_LATE - 3600.0 + 30.0 * k, {"prn": 3, "k": k})
        got = []
        for name in sorted(os.listdir(self.d)):
            got += [r["k"] for r in self._rows(name)]
        self.assertEqual(sorted(got), list(range(n)))
        self.assertEqual(len(got), n, "rows were lost or duplicated by the roll")

    def test_reopens_are_one_per_day_not_one_per_row(self):
        """The handle is cached: 100 rows in one day must not reopen 100 times."""
        w = self._w()
        for k in range(100):
            w(T_LATE - 3600.0 + k, {"k": k})
        self.assertEqual(len(self.logged), 1, "reopened mid-day: %r" % self.logged)
        self.assertIn("gps_l5_20260909.jsonl", self.logged[0])

    def test_appends_never_truncates(self):
        """A restart mid-day must not eat the morning -- the mode is 'a'."""
        self._w()(T_LATE, {"first": True})
        self._w()(T_LATE, {"second": True})       # a fresh writer, same day
        self.assertEqual(len(self._rows("gps_l5_20260909.jsonl")), 2)

    def test_a_template_with_no_date_codes_still_works(self):
        """--out /tmp/x.jsonl (the default, and every replay) opens exactly one file."""
        p = os.path.join(self.d, "flat.jsonl")
        w = self._w(p)
        w(T_LATE, {"a": 1})
        w(T_LATE + DAY_S, {"a": 2})
        self.assertEqual(len(self.logged), 1)
        self.assertEqual(len(self._rows("flat.jsonl")), 2)

    def test_creates_missing_directories(self):
        w = self._w(os.path.join(self.d, "%Y", "%m", "obs_%Y%m%d.jsonl"))
        w(T_LATE, {"a": 1})
        self.assertTrue(os.path.exists(
            os.path.join(self.d, "2026", "09", "obs_20260909.jsonl")))

    def test_rows_are_compact_single_line_json(self):
        """One row per line, no spaces -- the record is read with a line loop."""
        w = self._w()
        w(T_LATE, {"prn": 1, "az": 12.5, "el": None})
        with open(os.path.join(self.d, "gps_l5_20260909.jsonl")) as fh:
            line = fh.read()
        self.assertEqual(line.count("\n"), 1)
        self.assertNotIn(" ", line)
        self.assertEqual(json.loads(line), {"prn": 1, "az": 12.5, "el": None})


def _utcday(t):
    import time
    return time.strftime("%Y%m%d", time.gmtime(t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
