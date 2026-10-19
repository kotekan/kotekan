#!/usr/bin/env python3
"""Tests for gnss_cube_compact.py rung: a rung file exists only once it is complete.

    /home/kvand/gnss/venv/bin/python test_cube_compact_rung.py

cubecompact_loop.sh decides whether a day's rungs are done by counting the rung files that
exist, so a cut that dies mid-file must leave no file under the rung's name. Needs h5py, so it
runs in venv, not venv-ft.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_cube_compact as cc  # noqa: E402
from test_cube_compact import T0, frame  # noqa: E402

import h5py  # noqa: E402

N_WIN = 30  # windows in the L0 file: blocks 0, 1 and 2 of a rung12


class TestRungWrite(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="test_cube_compact_rung-")
        day = cc.utc_day(T0)
        rel = os.path.join("unknown", cc.sender_name(frame(0)[0]["chain"]), day + ".h5")
        self.l0 = os.path.join(self.d, "l0", rel)
        self.dst = os.path.join(self.d, "rung12", rel)
        lf = cc.L0File(h5py, self.l0, frame(0)[0], T0, "frame", "unknown", day)
        for i in range(N_WIN):
            h, a = frame(i)
            lf.add(h, a, cc.frame_utc(h, T0))
        lf.flush(["a.raw"])
        lf.close()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def rung(self):
        args = Namespace(files=[self.l0], n=12, out=self.d, l0_root=self.d)
        with contextlib.redirect_stdout(io.StringIO()):
            cc.cmd_rung(args)

    def test_complete_cut(self):
        # The control: an uninterrupted cut leaves the rung, with one row per (block, slot).
        self.rung()
        with h5py.File(self.dst, "r") as g:
            self.assertEqual(g["rows/blk"].shape[0], 3 * 2)
            self.assertEqual(g["blk/n_win_present"][:].tolist(), [12, 12, 6])

    def test_cut_killed_mid_file_leaves_no_rung(self):
        orig = h5py.Group.create_dataset
        calls = []

        def dies_partway(group, *args, **kw):
            calls.append(args[0] if args else kw.get("name"))
            if len(calls) == 4:
                raise OSError("killed mid-file")
            return orig(group, *args, **kw)

        h5py.Group.create_dataset = dies_partway
        try:
            with self.assertRaises(OSError):
                self.rung()
        finally:
            h5py.Group.create_dataset = orig
        self.assertFalse(
            os.path.exists(self.dst), "a partial rung file exists under its real name"
        )


if __name__ == "__main__":
    unittest.main()
