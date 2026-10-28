#!/usr/bin/env python3
"""Tests for gnss_cube_compact.py compact: what reaches L0 and what the manifest records.

    /home/kvand/gnss/venv/bin/python test_cube_compact.py

Offline: the raw reader (gnss_cube_read.iter_frames) is replaced by synthetic frames keyed by
file name, so each test sets exactly which windows, epochs and mtimes a raw file carries. Needs
h5py, so it runs in venv, not venv-ft.
"""
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_cube_compact as cc  # noqa: E402

import h5py  # noqa: E402

T0 = 1790626549.000002861  # an F-engine frame0
SR = 3.2e9  # sample rate; one window is one second
N_PRN, N_BIN, N_ELEM = 2, 3, 4
CHAIN = "cx19//gnss0_n2assemble"


def frame(idx, utc0=T0, chain=CHAIN, t=None):
    """One window with both slots live, dated t (default utc0 + idx s) on the epoch utc0."""
    t = utc0 + idx if t is None else t
    w0 = int(round((t - utc0) * SR))
    h = dict(
        version=3,
        idx=idx,
        wstart0=w0,
        wstart1=w0 + int(SR) - 1,
        dropped=0,
        n_prn=N_PRN,
        n_bin=N_BIN,
        n_elem=N_ELEM,
        win_samples=SR,
        sample_rate=SR,
        gpu=0,
        bin_width=1,
        max_prn=N_PRN,
        max_bins=N_BIN,
        chain=chain,
        utc0=utc0,
    )
    a = dict(
        phi0=np.zeros(N_PRN),
        freq_id_lo=np.arange(N_BIN, dtype=np.int32),
        freq_id_hi=np.arange(N_BIN, dtype=np.int32),
        prn=np.array([3, 7], dtype=np.int32),
        n_rec=np.full(N_PRN, 8, dtype=np.int32),
        n_reanchor=np.zeros(N_PRN, dtype=np.int32),
        w=np.ones((N_PRN, N_BIN), dtype=np.float32),
        energy=np.ones((N_PRN, N_BIN), dtype=np.float32),
        coh_re=np.ones((N_PRN, N_BIN, N_ELEM), dtype=np.float32),
        coh_im=np.zeros((N_PRN, N_BIN, N_ELEM), dtype=np.float32),
        incoh=np.ones((N_PRN, N_BIN, N_ELEM), dtype=np.float32),
    )
    return h, a


class CompactCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="test_cube_compact-")
        self.raw = os.path.join(self.d, "raw")
        self.out = os.path.join(self.d, "out")
        os.makedirs(self.raw)
        self.frames = {}
        self._orig = cc.raw.iter_frames
        cc.raw.iter_frames = lambda path, want_arrays=True: list(
            self.frames[os.path.basename(path)]
        )

    def tearDown(self):
        cc.raw.iter_frames = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def add_raw(self, name, frames, mtime, size=64):
        """A raw file of `size` bytes, last written at `mtime`, whose frames are `frames`."""
        p = os.path.join(self.raw, name)
        with open(p, "wb") as fh:
            fh.write(b"\0" * size)
        os.utime(p, (mtime, mtime))
        self.frames[name] = frames
        return p

    def seed_manifest(self, m):
        os.makedirs(os.path.join(self.out, "l0"), exist_ok=True)
        with open(os.path.join(self.out, "l0", "manifest.json"), "w") as fh:
            json.dump(m, fh)

    def compact(self, **kw):
        args = Namespace(
            raw=self.raw,
            out=self.out,
            pointings=os.path.join(self.d, "no_pointings.yaml"),
            include_open=True,
            redo=False,
            batch=12,
            mtime_tol=60.0,
            newest_first=False,
            max_files=0,
            readers=1,
        )
        vars(args).update(kw)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            cc.cmd_compact(args)
        return cc.load_manifest(self.out)

    def l0_rows(self):
        n = 0
        for p in glob.glob(os.path.join(self.out, "l0", "**", "*.h5"), recursive=True):
            with h5py.File(p, "r") as f:
                n += f["rows/idx"].shape[0]
        return n


class TestLastFileSkipped(CompactCase):
    """Rows buffered since the last batch flush reach L0 when the last todo file is skipped."""

    def test_last_file_empty(self):
        self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.add_raw("b.raw", [], T0 + 3)
        m = self.compact()
        self.assertEqual(m["b.raw"].get("note"), "empty")
        self.assertEqual(self.l0_rows(), 4, "a.raw's rows were dropped")

    def test_last_file_refused(self):
        self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.add_raw("b.raw", [frame(2)], T0 + 3600)
        m = self.compact()
        self.assertIn("mtime gate", m["b.raw"].get("refused", ""))
        self.assertEqual(self.l0_rows(), 4, "a.raw's rows were dropped")


class TestSizeChanged(CompactCase):
    """A file whose size changed since the manifest saw it."""

    def test_folded_file_is_skipped(self):
        # The control: rows of it are in L0 already, so folding it again would duplicate them.
        p = self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.seed_manifest({"a.raw": dict(size=1, mtime=os.stat(p).st_mtime, frames=1)})
        m = self.compact()
        self.assertEqual(m["a.raw"]["size"], 1)
        self.assertEqual(self.l0_rows(), 0)

    def test_redo_folds_it(self):
        p = self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.seed_manifest({"a.raw": dict(size=1, mtime=os.stat(p).st_mtime, frames=1)})
        m = self.compact(redo=True)
        self.assertEqual(m["a.raw"]["size"], 64)
        self.assertEqual(self.l0_rows(), 4)

    def test_refused_file_is_retried(self):
        self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.seed_manifest(
            {
                "a.raw": dict(
                    size=1,
                    mtime=T0 - 3600,
                    frames=1,
                    refused="epoch does not describe file (mtime gate)",
                )
            }
        )
        m = self.compact()
        self.assertNotIn("refused", m["a.raw"])
        self.assertEqual(self.l0_rows(), 4)

    def test_empty_file_is_retried(self):
        # Recorded empty while it was open (--include-open), then grown: nothing of it is in L0.
        p = self.add_raw("a.raw", [frame(0), frame(1)], T0 + 2)
        self.seed_manifest(
            {"a.raw": dict(size=0, mtime=os.stat(p).st_mtime, frames=0, note="empty")}
        )
        m = self.compact()
        self.assertEqual(m["a.raw"]["frames"], 2)
        self.assertEqual(self.l0_rows(), 4)


class TestRebaseInsideFile(CompactCase):
    """A raw file whose frames carry two epochs (an F-engine re-base inside the file)."""

    def setUp(self):
        super().setUp()
        # Senders cross a re-base at different times, so their frames interleave: here the file
        # ends on a sender still on the old epoch, and the last window passes the mtime gate.
        self.add_raw(
            "a.raw",
            [
                frame(0, utc0=T0 + 101, chain="cx20//gnss0_n2assemble"),
                frame(101, utc0=T0),
            ],
            T0 + 102,
        )

    def test_compact_refuses(self):
        m = self.compact()
        self.assertIn("distinct utc0", m["a.raw"].get("refused", ""))
        self.assertEqual(self.l0_rows(), 0)

    def test_verify_refuses(self):
        args = Namespace(
            raw=os.path.join(self.raw, "a.raw"),
            l0=self.out,
            n=10,
            seed=1,
            pointings=os.path.join(self.d, "no_pointings.yaml"),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                cc.cmd_verify(args)
        self.assertIn("distinct utc0", str(e.exception))


def reshape(fa, nb):
    """The frame `fa` (frame()'s output) cut to its first nb bins: a sender regenerated with
    fewer channels."""
    h, a = fa
    h = dict(h, n_bin=nb, max_bins=nb)
    a = dict(a)
    for k in ("freq_id_lo", "freq_id_hi"):
        a[k] = a[k][:nb]
    for k in ("w", "energy", "coh_re", "coh_im", "incoh"):
        a[k] = a[k][:, :nb]
    return h, a


class TestGeometryChangeInsideDay(CompactCase):
    """A sender whose bin count changes inside a day (a config regeneration) continues in a
    tagged file beside the day file, in one batch or across passes, and never stops the pass."""

    def day(self):
        return cc.utc_day(T0)

    def files(self):
        d = os.path.join(self.out, "l0", "unknown", "cx19_gnss0_n2assemble")
        out = {}
        for p in sorted(glob.glob(os.path.join(d, "*.h5"))):
            with h5py.File(p, "r") as f:
                out[os.path.basename(p)] = (
                    int(f.attrs["n_bin"]),
                    f["win/idx"].shape[0],
                )
        return out

    def test_one_batch(self):
        self.add_raw("a.raw", [frame(0), frame(1), reshape(frame(2), 2)], T0 + 3)
        m = self.compact()
        self.assertEqual(m["a.raw"]["frames"], 3)
        self.assertEqual(
            self.files(),
            {self.day() + ".h5": (N_BIN, 2), self.day() + "_2b%de.h5" % N_ELEM: (2, 1)},
        )

    def test_across_passes(self):
        self.add_raw("a.raw", [frame(0)], T0 + 1)
        self.compact()
        self.add_raw("b.raw", [reshape(frame(1), 2), frame(2)], T0 + 3)
        self.compact()
        self.assertEqual(
            self.files(),
            {self.day() + ".h5": (N_BIN, 2), self.day() + "_2b%de.h5" % N_ELEM: (2, 1)},
        )


class TestOrderAndReaders(CompactCase):
    """--newest-first with --max-files folds the newest files and leaves the rest for the next
    pass; read-ahead threads (--readers) change nothing that reaches L0."""

    def setUp(self):
        super().setUp()
        for j in range(5):
            self.add_raw("%d.raw" % j, [frame(2 * j), frame(2 * j + 1)], T0 + 2 * j + 2)

    def test_newest_first_in_passes(self):
        m = self.compact(newest_first=True, max_files=2)
        self.assertEqual(sorted(m), ["3.raw", "4.raw"])
        m = self.compact(newest_first=True, max_files=2)
        self.assertEqual(sorted(m), ["1.raw", "2.raw", "3.raw", "4.raw"])
        m = self.compact(newest_first=True, max_files=2)
        self.assertEqual(len(m), 5)
        self.assertEqual(self.l0_rows(), 20)

    def l0_dump(self):
        out = {}
        for p in glob.glob(os.path.join(self.out, "l0", "**", "*.h5"), recursive=True):
            with h5py.File(p, "r") as f:
                o = np.argsort(f["rows/idx"][:], kind="stable")
                out[os.path.relpath(p, self.out)] = (
                    f["rows/idx"][:][o].tolist(),
                    f["rows/incoh"][:][o].tolist(),
                )
        return out

    def test_readers_change_nothing(self):
        self.compact(readers=1)
        one = self.l0_dump()
        shutil.rmtree(self.out)
        self.compact(readers=3)
        self.assertEqual(self.l0_dump(), one)


if __name__ == "__main__":
    unittest.main()
