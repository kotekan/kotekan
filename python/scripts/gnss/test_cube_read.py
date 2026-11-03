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
import struct
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

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


def wire_frame(rng, mp=8, mb=4, ne=3, n_prn=6, n_bin=3, live=(0, 2, 5)):
    """A v3 frame, zero outside the chain's slots and bins and in the slots with no records, as
    the assembler fills them."""
    a = dict(
        phi0=np.zeros(mp, "<f8"),
        lo=np.zeros(mb, "<i4"),
        hi=np.zeros(mb, "<i4"),
        prn=np.zeros(mp, "<i4"),
        nrec=np.zeros(mp, "<i4"),
        nre=np.zeros(mp, "<i4"),
        w=np.zeros((mp, mb), "<f4"),
        en=np.zeros((mp, mb), "<f4"),
        cre=np.zeros((mp, mb, ne), "<f4"),
        cim=np.zeros((mp, mb, ne), "<f4"),
        inc=np.zeros((mp, mb, ne), "<f4"),
    )
    a["lo"][:n_bin] = 6000 + np.arange(n_bin)
    a["hi"][:n_bin] = 6000 + np.arange(n_bin)
    for s in live:
        a["phi0"][s] = rng.normal()
        a["prn"][s], a["nrec"][s], a["nre"][s] = 10 + s, 90 + s, s % 2
        for k in ("w", "en", "cre", "cim", "inc"):
            a[k][s, :n_bin] = rng.normal(size=a[k][s, :n_bin].shape)
    hdr = bytearray(cr.CUBE_HEADER_BYTES[3])
    struct.pack_into("<8q", hdr, 0, 3, 77, 1, 2, 0, n_prn, n_bin, ne)
    struct.pack_into("<2d", hdr, 64, 1.0, 2.0)
    struct.pack_into("<4i", hdr, 80, 0, 1, mp, mb)
    struct.pack_into("<d", hdr, 96, 1.7e9)
    hdr[104:120] = b"cx19/gnss0_cube\0"
    body = b"".join(
        a[k].tobytes()
        for k in (
            "phi0",
            "lo",
            "hi",
            "prn",
            "nrec",
            "nre",
            "w",
            "en",
            "cre",
            "cim",
            "inc",
        )
    )
    return bytes(hdr) + body, a, (n_prn, n_bin, live)


def packed_frame(wire, a, shape):
    """The same frame as GnssCubeWrite packs it (GnssCubeWrite.hpp)."""
    n_prn, n_bin, live = shape
    hdr = bytearray(wire[: cr.CUBE_HEADER_BYTES[3]])
    struct.pack_into("<q", hdr, 0, 4)
    struct.pack_into("<2i", hdr, 88, len(live), n_bin)
    L = list(live)
    parts = [
        a["phi0"][L],
        a["lo"][:n_bin],
        a["hi"][:n_bin],
        a["prn"][L],
        a["nrec"][L],
        a["nre"][L],
        np.array(L, "<i4"),
    ]
    parts += [a[k][L, :n_bin] for k in ("w", "en", "cre", "cim", "inc")]
    return bytes(hdr) + b"".join(np.ascontiguousarray(x).tobytes() for x in parts)


class TestPackedFrames(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="test_cube_read-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write(self, name, frames):
        path = os.path.join(self.d, name)
        with open(path, "wb") as fh:
            for f in frames:
                fh.write(struct.pack("<I", 0) + f)
        return path

    def test_packed_reads_back_as_the_wire_frame(self):
        rng = np.random.default_rng(1)
        frames = [
            wire_frame(rng),
            wire_frame(rng, live=()),
            wire_frame(rng, live=(1, 2, 3, 4, 5)),
        ]
        wire = self.write("w.raw", [f for f, _, _ in frames])
        packed = self.write("p.raw", [packed_frame(*f) for f in frames])
        self.assertLess(os.path.getsize(packed), os.path.getsize(wire))
        got_w, got_p = list(cr.iter_frames(wire)), list(cr.iter_frames(packed))
        self.assertEqual(len(got_w), 3)
        self.assertEqual(len(got_p), 3)
        for (hw, aw), (hp, ap) in zip(got_w, got_p):
            self.assertEqual(hp["version"], 4)
            for k in ("idx", "n_prn", "n_bin", "n_elem", "chain", "utc0"):
                self.assertEqual(hp[k], hw[k], k)
            self.assertEqual(sorted(ap), sorted(aw))
            for k in aw:
                np.testing.assert_array_equal(ap[k], aw[k], err_msg=k)
                self.assertEqual(ap[k].shape, aw[k].shape, k)

    def test_a_packed_slot_outside_the_chain_is_refused(self):
        rng = np.random.default_rng(2)
        f = bytearray(packed_frame(*wire_frame(rng)))
        o = cr.CUBE_HEADER_BYTES[3] + 3 * 8 + 3 * 4 * 2 + 3 * 4 * 3
        struct.pack_into("<i", f, o, 7)  # the first slot index, beyond n_prn 6
        with self.assertRaises(SystemExit):
            list(cr.iter_frames(self.write("bad.raw", [bytes(f)])))


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
