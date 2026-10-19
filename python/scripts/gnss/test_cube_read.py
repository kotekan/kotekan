#!/usr/bin/env python3
"""Tests for gnss_cube_read.py ls: how the holes in a sender's window index are attributed.

    /home/kvand/gnss/venv-ft/bin/python test_cube_read.py

Offline: iter_frames is replaced by synthetic headers, so each test sets exactly which windows
a sender shipped and what its cumulative dropped_windows counter said.
"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_cube_read as cr  # noqa: E402


def header(idx, dropped):
    return dict(
        chain="cx19//gnss0_n2assemble",
        gpu=0,
        idx=idx,
        dropped=dropped,
        n_prn=2,
        n_bin=3,
        n_elem=4,
        win_samples=3.2e9,
        sample_rate=3.2e9,
    )


class TestLsAttribution(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="test_cube_read-")
        self.path = os.path.join(self.d, "a.raw")
        open(self.path, "wb").close()
        self._orig = cr.iter_frames

    def tearDown(self):
        cr.iter_frames = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def ls(self, frames):
        """cmd_ls output for one sender shipping `frames` = [(idx, cumulative dropped)]."""
        cr.iter_frames = lambda path, want_arrays=True: [
            (header(i, d), None) for i, d in frames
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cr.cmd_ls(Namespace(path=[self.path]))
        return out.getvalue()

    def unclaimed(self, out):
        m = re.search(r"^unclaimed \(downstream\)\s*:\s*(\d+)", out, re.M)
        self.assertIsNotNone(m, out)
        return int(m.group(1))

    def test_a_sender_drop_is_not_downstream(self):
        # Window 2 is missing and the sender's counter rose by one across it: the sender dropped
        # it, so nothing is unclaimed.
        out = self.ls([(0, 0), (1, 0), (3, 1), (4, 1)])
        self.assertIn("nothing was lost in transport", out)
        self.assertNotIn("DOWNSTREAM", out)
        self.assertEqual(self.unclaimed(out), 0, out)

    def test_an_unclaimed_hole_is_downstream(self):
        # The control: the same hole with the counter flat is downstream of the sender.
        out = self.ls([(0, 0), (1, 0), (3, 0), (4, 0)])
        self.assertEqual(self.unclaimed(out), 1, out)
        self.assertIn("every hole is DOWNSTREAM", out)

    def test_a_restarted_counter_claims_nothing(self):
        # The sender restarted across the hole, so its counter fell from 5 to 0: the one hole is
        # unclaimed, never more holes than the index has.
        out = self.ls([(0, 5), (1, 5), (3, 0), (4, 0)])
        self.assertEqual(self.unclaimed(out), 1, out)


if __name__ == "__main__":
    unittest.main()
