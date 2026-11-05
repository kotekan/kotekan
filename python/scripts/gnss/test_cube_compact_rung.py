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
import numpy as np  # noqa: E402

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

    def rung(self, n=(12,), skip_done=False):
        args = Namespace(
            files=[self.l0],
            n=list(n),
            out=self.d,
            l0_root=self.d,
            skip_done=skip_done,
            max_read_mbps=0.0,
        )
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

    def test_sums_are_the_add_at_sums(self):
        # The cut sums each (block, slot, prn) group in file order; np.add.at in row order is
        # the reference, and the two agree bit for bit.
        self.rung(n=(12, 60))
        with h5py.File(self.l0, "r") as f:
            rows = {k: f["rows/" + k][:] for k in f["rows"]}
        ref = int(np.argmax(rows["incoh"].sum(axis=(0, 1))))
        for n in (12, 60):
            key = np.stack([rows["idx"] // n, rows["slot"], rows["prn"]], axis=1)
            uk, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            want = {}
            for k in ("w", "energy", "incoh"):
                want[k] = np.zeros((len(uk),) + rows[k].shape[1:])
                np.add.at(want[k], inv, rows[k].astype(np.float64))
            c = rows["coh"].astype(np.complex128)
            want["cohpow"] = np.zeros(want["incoh"].shape)
            np.add.at(want["cohpow"], inv, c.real ** 2 + c.imag ** 2)
            want["cohref"] = np.zeros(c.shape[1:], np.complex128)[None].repeat(
                len(uk), 0
            )
            np.add.at(want["cohref"], inv, c * np.conj(c[:, :, ref : ref + 1]))
            dst = self.dst.replace("rung12", "rung%d" % n)
            with h5py.File(dst, "r") as g:
                self.assertEqual(g["rows/blk"][:].tolist(), uk[:, 0].tolist())
                for k, v in want.items():
                    got = g["rows/" + k][:]
                    self.assertTrue(
                        np.array_equal(got, v.astype(got.dtype)), "%s rung%d" % (k, n)
                    )

    def test_one_read_cuts_both_rungs(self):
        self.rung(n=(12, 60))
        with h5py.File(self.dst.replace("rung12", "rung60"), "r") as g:
            self.assertEqual(g.attrs["rung_n"], 60)
            self.assertEqual(g["blk/n_win_present"][:].tolist(), [N_WIN])

    def test_skip_done_skips_only_a_rung_newer_than_its_l0(self):
        self.rung()
        os.utime(self.dst, (1e9, 1e9))
        self.rung(skip_done=True)
        self.assertNotEqual(os.path.getmtime(self.dst), 1e9, "an older rung was skipped")
        t = os.path.getmtime(self.l0)
        os.utime(self.dst, (t + 1, t + 1))
        self.rung(skip_done=True)
        self.assertEqual(os.path.getmtime(self.dst), t + 1, "a newer rung was re-cut")


if __name__ == "__main__":
    unittest.main()
