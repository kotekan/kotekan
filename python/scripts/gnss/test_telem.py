#!/usr/bin/env python3
"""Tests for the tracker->broker telemetry transport (gnss_broker/telem.py).

Three groups of test here, and the first is the one that matters:

  A. Cross-language. scripts/gnss/telemfmt.cpp is compiled and run, so the parser is checked
     against gnssTelem.hpp itself rather than against a transcription of it. The C++ struct and
     the struct.Struct format string are two independent statements of one layout, and a drift
     between them does not raise: it yields plausible numbers with the wrong meaning.

  B. Behavioural. The window ring, the lag rule, hole-vs-shift on a dropped record, and the
     cross-instance grouping: each is a case where a transport that inferred alignment would
     give a confident wrong answer rather than an error.

  C-F. Transport liveness, each against a real socket: silence is not a broken link (C), the
     socket thread is never stalled by the ring lock (D), a connection the gather closed after
     it lived is reconnected at once (E), and the receive buffer is pinned before connect
     rather than autotuned up from 128 kB (F). A failure of C or D stops the stream for the
     whole fleet.

    python3 python/scripts/gnss/test_telem.py
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import kotekan_src  # noqa: E402
from gnss_broker import telem  # noqa: E402


def _make_frame(
    chain="gps_l5",
    inst="cx19.0",
    win=100,
    seq=0,
    n_rec=4,
    n_prn=4,
    hops_per_record=2048,
    fft_len=16384,
    present=None,
    rows=None,
    utc0=0.0,
):
    """Build one wire frame in Python -- for the behavioural tests only.

    The layout it assumes is the one TestWireFormat proves correct, so these tests are not
    quietly re-deriving the format: if the C++ moves, those tests fail first and these become
    meaningless rather than misleading.
    """
    if present is None:
        present = (1 << n_rec) - 1
    wstart0 = win * n_rec * hops_per_record * fft_len
    n_chan = 5
    chan_ids = [5972 + 16 * k for k in range(n_chan)] + [0] * (telem._MAX_CHAN - n_chan)
    row_total = telem._ROW_FLOATS + n_chan * telem._CHAN_FLOATS
    hdr = telem._HDR.pack(
        telem._MAGIC,
        telem._VERSION,
        n_rec,
        n_prn,
        telem._ROW_FLOATS,
        n_chan,
        32,
        hops_per_record,
        fft_len,
        win,
        seq,
        wstart0,
        utc0,
        present,
        n_chan,
        row_total,
        chain.encode(),
        inst.encode(),
        *chan_ids
    )
    body = [0.0] * (n_rec * n_prn * row_total)
    for r in range(n_rec):
        for p in range(n_prn):
            base = (r * n_prn + p) * row_total
            # the comb: P = 1 exactly on every column, energy 1
            for ch in range(n_chan):
                cb = base + telem._ROW_FLOATS + ch * telem._CHAN_FLOATS
                body[cb + telem.CHAN_RE] = 1.0
                body[cb + telem.CHAN_IM] = 0.0
                body[cb + telem.CHAN_ENERGY] = 1.0
                body[cb + telem.CHAN_E_RE] = 0.5  # E and L distinct from P, so a slot
                body[
                    cb + telem.CHAN_E_ENERGY
                ] = 1.0  # mix-up shows up as a value, not a shape
                body[cb + telem.CHAN_L_RE] = 0.25
                body[cb + telem.CHAN_L_ENERGY] = 1.0
            body[base + telem.REC_PRN] = float(1 + p)
            body[base + telem.REC_P_ENERGY] = 2.0
            body[base + telem.REC_P_RE] = 2.0 * (1 + p)  # A = 1+p exactly
            body[base + telem.REC_P_IM] = 0.0
            body[base + telem.REC_CPHASE] = 0.25  # a per-record increment, cycles
            body[base + telem.REC_TRIM_INC] = 0.01
    if rows:
        for (r, p, slot), v in rows.items():
            body[(r * n_prn + p) * row_total + slot] = v
    return hdr + struct.pack("<%df" % len(body), *body)


class TestWireFormat(unittest.TestCase):
    """A. The parser against the C++ definition."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="telemfmt-")
        src = os.path.join(K, "scripts", "gnss", "telemfmt.cpp")
        cls.bin = os.path.join(cls.tmp, "telemfmt")
        # No compiler is an environment this cannot run in. A compiler that rejects the harness
        # means it and the header disagree, which is what this class exists to catch: fail.
        if shutil.which("c++") is None and not os.environ.get("GNSS_REQUIRE_CXX"):
            raise unittest.SkipTest("no c++ compiler")
        p = subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O1",
                "-I",
                kotekan_src.gnss_cpp(),
                "-o",
                cls.bin,
                src,
            ],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            raise AssertionError(
                "telemfmt.cpp does not compile against %s:\n%s"
                % (kotekan_src.gnss_cpp(), p.stderr)
            )
        cls.frame_path = os.path.join(cls.tmp, "frame.bin")
        r = subprocess.run(
            [cls.bin, cls.frame_path], capture_output=True, text=True, check=True
        )
        cls.meta = json.loads(r.stdout)
        with open(cls.frame_path, "rb") as fh:
            cls.raw = fh.read()

    def test_header_size_and_offsets(self):
        m = self.meta
        self.assertEqual(m["header_bytes"], telem._HDR_BYTES)
        self.assertEqual(m["header_bytes"], m["telem_header_bytes_const"])
        self.assertEqual(telem._HDR.size, telem._HDR_BYTES)
        self.assertEqual(m["record_floats"], telem._ROW_FLOATS)
        self.assertEqual(m["magic"], telem._MAGIC)
        self.assertEqual(m["version"], telem._VERSION)
        # Field offsets, one by one. A same-size layout with two fields transposed passes a
        # size check and fails here, which is the point.
        for name, want in (
            ("win", 24),
            ("seq", 32),
            ("wstart0", 40),
            ("utc0", 48),
            ("present", 56),
            ("chain", 64),
            ("inst", 80),
        ):
            self.assertEqual(
                m["off_" + name], want, "gnssTelem.hpp moved field %r" % name
            )

    def test_parses_cpp_frame(self):
        m = self.meta
        self.assertEqual(len(self.raw), m["frame_bytes"])
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertEqual(f.chain, "gal_e5a")
        self.assertEqual(f.inst, "cx42.1")
        self.assertEqual(f.win, m["win"])
        self.assertEqual(f.seq, m["seq"])
        self.assertEqual(f.wstart0, m["wstart0"])
        self.assertEqual(f.n_rec, m["n_rec"])
        self.assertEqual(f.n_prn, m["n_prn"])
        self.assertEqual(f.hops_per_record, 2048)
        self.assertEqual(f.fft_len, 16384)
        self.assertEqual(f.prns(), [100, 101, 102, 103, 104])

    def test_row_values_and_slot_addressing(self):
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        # telemfmt writes row[f] = 1000*r + 10*p + f, then overwrites PRN and P_ENERGY.
        row = f.row(3, 102)  # r=3, p=2
        self.assertIsNotNone(row)
        self.assertEqual(row[telem.REC_PRN], 102.0)
        self.assertEqual(row[telem.REC_P_ENERGY], 3.0)
        self.assertEqual(row[telem.REC_P_IM], 3000 + 20 + telem.REC_P_IM)
        self.assertEqual(row[telem.REC_CPHASE], 3000 + 20 + telem.REC_CPHASE)

    def test_missing_slot_is_a_hole_not_a_shift(self):
        # telemfmt sets present = 0b1011: slot 2 absent. The records after it must keep their
        # own identity: a transport that appended instead of addressing would hand slot 3's
        # data back as slot 2, and every downstream number would be one record stale with
        # nothing to show for it.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertTrue(f.has_record(0))
        self.assertTrue(f.has_record(1))
        self.assertFalse(f.has_record(2))
        self.assertTrue(f.has_record(3))
        self.assertIsNone(f.row(2, 100))
        self.assertEqual(f.row(3, 100)[telem.REC_P_IM], 3000 + 0 + telem.REC_P_IM)

    def test_comb_labels_and_values(self):
        # telemfmt writes comb[ch] = 100000*r + 1000*p + 10*ch + {1,2} with energy ch+1, and
        # freq_ids on a real stride-16 comb. Every element of the address (record, PRN,
        # channel, slot) is in the value, so a wrong row stride or a wrong comb offset returns
        # a number that provably belongs somewhere else.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        # max_chan is the row stride and it is the sender's, not the format's ceiling; the two
        # are deliberately different in this fixture, so a reader that confuses them fails here.
        self.assertEqual(self.meta["max_chan_const"], telem._MAX_CHAN)
        self.assertLess(self.meta["max_chan"], self.meta["max_chan_const"])
        self.assertEqual(self.meta["chan_floats"], telem._CHAN_FLOATS)
        self.assertEqual(
            self.meta["row_floats"],
            telem._ROW_FLOATS + self.meta["max_chan"] * telem._CHAN_FLOATS,
        )
        self.assertEqual(f.row_total, self.meta["row_floats"])
        self.assertEqual(f.chan_ids, [5972 + 16 * ch for ch in range(5)])
        comb = f.comb_epl(3, 102)  # r=3, p=2
        self.assertEqual(len(comb), 5)
        for ch, (fid, _E, P, _L, (_eE, e, _eL)) in enumerate(comb):
            self.assertEqual(fid, 5972 + 16 * ch)
            self.assertEqual(e, ch + 1)
            # comb_epl() normalises by energy, matching the header slots' G/E convention
            self.assertAlmostEqual(
                P.real, (100000 * 3 + 1000 * 2 + 10 * ch + 1) / (ch + 1), places=3
            )
            self.assertAlmostEqual(
                P.imag, (100000 * 3 + 1000 * 2 + 10 * ch + 2) / (ch + 1), places=3
            )

    def test_comb_epl_carries_all_three_taps(self):
        # E/P/L per channel is the comb DLL's input (combdll.py). telemfmt writes +1/+2 for P,
        # +3/+4 for E, +5/+6 for L, so a tap swap is a wrong value rather than a plausible one.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        epl = f.comb_epl(3, 102)
        self.assertEqual(len(epl), 5)
        for ch, (fid, E, P, L, (eE, eP, eL)) in enumerate(epl):
            self.assertEqual(fid, 5972 + 16 * ch)
            b = 100000 * 3 + 1000 * 2 + 10 * ch
            self.assertAlmostEqual(P.real, (b + 1) / (ch + 1), places=3)
            self.assertAlmostEqual(E.real, (b + 3) / (ch + 1), places=3)
            self.assertAlmostEqual(L.real, (b + 5) / (ch + 1), places=3)
            self.assertEqual((eE, eP, eL), (ch + 1, ch + 1, ch + 1))

    def test_unused_comb_columns_do_not_leak(self):
        # telemfmt ships 6 columns and fills 5 on purpose. The reserved-but-unused one must read
        # as absent, never as the neighbouring row's first channel -- which is what a row stride
        # computed from n_chan instead of the header's max_chan would hand back.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertEqual(f.n_chan, 5)
        self.assertLess(f.n_chan, f.max_chan)
        for r in range(f.n_rec):
            if f.has_record(r):
                for prn in f.prns():
                    self.assertEqual(len(f.comb_epl(r, prn)), 5)

    def test_comb_absent_on_a_missing_record(self):
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertFalse(f.has_record(2))
        self.assertEqual(f.comb_epl(2, 100), [])

    def test_hop_is_the_absolute_fengine_index(self):
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertEqual(f.hop(0), f.wstart0 // 16384)
        self.assertEqual(f.hop(1) - f.hop(0), 2048)
        # The key /get_records uses, so the two feeds are directly comparable and an A/B
        # between them is a measurement.
        self.assertEqual(f.hop(3), f.win * 4 * 2048 + 3 * 2048)


class TestWindowRing(unittest.TestCase):
    """B. The store: grouping, eviction, and the lag rule."""

    def _client(self, depth=8):
        return telem.TelemClient(depth=depth)

    def _feed(self, c, raw):
        c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))

    def test_instances_group_by_window(self):
        c = self._client()
        for inst in ("cx19.0", "cx19.1", "cx42.0"):
            self._feed(c, _make_frame(inst=inst, win=10))
        self.assertEqual(
            sorted(c.frame_set("gps_l5", 10).keys()), ["cx19.0", "cx19.1", "cx42.0"]
        )

    def test_lag_hides_the_newest_window(self):
        # The newest window is always still filling, because the senders are independent
        # processes. Reading it combines a partial fleet with its neighbours' full one, which
        # reads as a fleet that keeps changing size.
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        self._feed(c, _make_frame(inst="cx42.0", win=10))
        self.assertEqual(c.windows("gps_l5", lag=1), [10])
        self.assertEqual(c.windows("gps_l5", lag=0), [10, 11])

    def test_eviction_drops_the_oldest_window_not_the_newest(self):
        # A laggard opening an old window must not push it ahead of newer ones in the ring. If
        # it did, the ring would evict a window the fleet is still filling.
        c = self._client(depth=3)
        for w in (10, 11, 12):
            self._feed(c, _make_frame(inst="cx19.0", win=w))
        self._feed(c, _make_frame(inst="cx42.0", win=9))  # out-of-order open
        self._feed(c, _make_frame(inst="cx19.0", win=13))
        held = c.windows("gps_l5", lag=0)
        self.assertEqual(held, sorted(held))
        self.assertIn(13, held)
        self.assertNotIn(9, held)

    def test_sequence_gaps_are_counted_from_the_senders_own_counter(self):
        # A rate that looks right can still be missing every fourth frame; only the sender's
        # counter can say so.
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10, seq=0))
        self._feed(c, _make_frame(inst="cx19.0", win=11, seq=1))
        self._feed(c, _make_frame(inst="cx19.0", win=14, seq=4))
        self.assertEqual(c.gaps, 2)

    def test_a_header_with_no_records_or_rows_is_refused(self):
        # gnss::telem_shape_ok's bounds. With n_rec or n_prn zero the payload is empty, so the
        # length check alone passes a bare header and the frame would open an empty window.
        # The record limit is read from gnssTelem.hpp, not from the constant under test.
        with open(kotekan_src.gnss_cpp("gnssTelem.hpp")) as fh:
            m = re.search(r"constexpr\s+int\s+TELEM_MAX_REC\s*=\s*(\d+)\s*;", fh.read())
        max_rec = int(m.group(1))
        for kw in ({"n_rec": 0}, {"n_prn": 0}, {"n_rec": max_rec + 1}):
            c = self._client()
            c._parse_and_store(_make_frame(**kw), 0.0)
            self.assertEqual(c.bad, 1, kw)
            self.assertEqual(c.chains(), [], kw)
        for kw in ({}, {"n_rec": max_rec}):
            c = self._client()
            c._parse_and_store(_make_frame(**kw), 0.0)
            self.assertEqual((c.bad, c.chains()), (0, ["gps_l5"]), kw)
        self.assertEqual(telem._MAX_REC, max_rec)

    def test_chains_do_not_mix(self):
        c = self._client()
        self._feed(c, _make_frame(chain="gps_l5", inst="cx19.0", win=10))
        self._feed(c, _make_frame(chain="gal_e5a", inst="cx19.0", win=10))
        self.assertEqual(c.chains(), ["gal_e5a", "gps_l5"])
        self.assertEqual(list(c.frame_set("gal_e5a", 10)), ["cx19.0"])

    # -- epoch resets: an F-engine re-base restarts the window counter -------------------------
    def _fill(self, c, lo, hi, insts=("cx19.0", "cx42.0")):
        for w in range(lo, hi):
            for inst in insts:
                self._feed(c, _make_frame(inst=inst, win=w))

    def test_a_persistent_large_backwards_jump_resets_the_ring(self):
        # Without the reset these frames are sorted to the front of a full ring and evicted
        # first: every one discarded, every instance stale, for as long as the process lives.
        c = self._client(depth=16)
        self._fill(c, 1000, 1010)
        for w in range(
            5, 5 + c.epoch_strikes - 1
        ):  # strikes 1..7: counted, stored as usual
            self._feed(c, _make_frame(inst="cx19.0", win=w))
        self.assertEqual(c.epoch_resets, 0)
        self.assertEqual(c.far_behind, c.epoch_strikes - 1)
        self.assertTrue(
            any(w >= 1000 for w in c._store["gps_l5"])
        )  # old epoch still held
        self._feed(
            c, _make_frame(inst="cx19.0", win=5 + c.epoch_strikes - 1)
        )  # strike 8
        self.assertEqual(c.epoch_resets, 1)
        self.assertEqual(list(c._store["gps_l5"]), [5 + c.epoch_strikes - 1])
        self._feed(c, _make_frame(inst="cx42.0", win=5 + c.epoch_strikes))
        st = c.stats(stale_after_s=0)["chains"]["gps_l5"]
        self.assertEqual(st["live"], 2)
        self.assertEqual(c.stats()["epoch_resets"], 1)

    def test_a_laggard_inside_the_margin_is_still_a_laggard(self):
        c = self._client(depth=16)
        self._fill(c, 1000, 1010)
        self._feed(c, _make_frame(inst="cx43.0", win=1010 - c.epoch_margin))
        self.assertEqual(c.epoch_resets, 0)
        self.assertEqual(c.far_behind, 0)
        self.assertIn(
            1010 - c.epoch_margin, c._store["gps_l5"]
        )  # accepted and re-sorted
        self.assertEqual(list(c._store["gps_l5"]), sorted(c._store["gps_l5"]))

    def test_one_corrupt_header_does_not_reset(self):
        c = self._client(depth=16)
        self._fill(c, 1000, 1010)
        self._feed(c, _make_frame(inst="cx19.0", win=3))  # one wild header
        self._feed(c, _make_frame(inst="cx19.0", win=1010))  # in order again
        for w in range(4, 4 + c.epoch_strikes - 1):  # never reaches a full run
            self._feed(c, _make_frame(inst="cx19.0", win=w))
            self._feed(c, _make_frame(inst="cx42.0", win=1011 + w))
        self.assertEqual(c.epoch_resets, 0)
        self.assertEqual(c.far_behind, c.epoch_strikes)  # counted, never acted on
        self.assertIn(
            1011 + 4 + c.epoch_strikes - 2, c._store["gps_l5"]
        )  # the last in-order window fed


class TestQuietGatherDoesNotReconnect(unittest.TestCase):
    """C. Silence is not a broken link.

    With the fleet down the gather has nothing to forward, and it notices a dead client only
    when a write to it fails, which with no senders never happens. A client that reconnected
    on every read timeout would leave a socket in CLOSE_WAIT at the gather each time, until
    accept() fails with EMFILE and the gather serves nothing at all, REST included.
    """

    def _serve(self, sock, gap_s):
        """One frame, then a gap (longer than the client's read timeout), then a second frame."""
        conn, _ = sock.accept()
        self.conns += 1
        try:
            for raw in (_make_frame(win=1), None, _make_frame(win=2)):
                if raw is None:
                    time.sleep(gap_s)
                    continue
                conn.sendall(struct.pack("<I", len(raw)) + raw)
            time.sleep(1.0)  # hold the connection open so a reconnect would be visible
        except Exception:
            pass
        finally:
            conn.close()

    def test_a_silent_gather_is_waited_out_not_reconnected(self):
        import socket as _s
        import threading

        srv = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        srv.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        self.conns = 0
        t = threading.Thread(target=self._serve, args=(srv, 0.30), daemon=True)
        t.start()

        c = telem.TelemClient(
            host="127.0.0.1",
            port=srv.getsockname()[1],
            depth=8,
            retry_s=0.05,
            read_timeout_s=0.10,
        )
        c.start()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and c.frames < 2:
                time.sleep(0.02)
        finally:
            c.stop()
        srv.close()

        # Both frames arrive, across a gap three times the read timeout...
        self.assertEqual(c.frames, 2, "the frame after the silence never arrived")
        # ...on one connection. The accept count is what the gather pays for: one leaked
        # socket per reconnect.
        self.assertEqual(self.conns, 1, "the client reconnected across a quiet gather")
        self.assertEqual(c.connects, 1)
        self.assertEqual(c.bad, 0)


class TestCoherentSource(unittest.TestCase):
    """B. coherent_source(): the /get_records data from telemetry, in fleet_coherent's shape."""

    def _client(self):
        c = telem.TelemClient(depth=16)
        return c

    def _feed(self, c, raw):
        c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))

    def test_shape_and_normalisation(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx42.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))  # newest: excluded by lag
        got, now = c.coherent_source("gps_l5", prns=[2])
        self.assertEqual(sorted(got), ["cx19.0", "cx42.0"])
        # A = (P_re + i P_im)/P_energy, exactly the combiner's ar = gr/energy.
        hops = got["cx19.0"][2]
        self.assertEqual(len(hops), 4)
        for _hop, (A, e) in hops.items():
            self.assertAlmostEqual(A.real, 2.0)
            self.assertAlmostEqual(e, 2.0)
        self.assertEqual(now, max(hops))

    def test_instances_land_on_identical_hops(self):
        # Two instances that reported the same window must produce the same hop keys: no
        # tolerance, no nearest match. Arrival order and a UTC each instance stamps
        # independently can only approximate that.
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx42.1", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[1])
        self.assertEqual(set(got["cx19.0"][1]), set(got["cx42.1"][1]))

    def test_zero_energy_rows_are_silence_not_zeros(self):
        c = self._client()
        self._feed(
            c,
            _make_frame(inst="cx19.0", win=10, rows={(0, 1, telem.REC_P_ENERGY): 0.0}),
        )
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[2])
        self.assertEqual(
            len(got["cx19.0"][2]), 3
        )  # slot 0 dropped, not carried as 0+0j

    def test_missing_record_slot_does_not_shift_hops(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10, present=0b1101))
        self._feed(c, _make_frame(inst="cx42.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[1])
        a, b = set(got["cx19.0"][1]), set(got["cx42.0"][1])
        self.assertEqual(len(a), 3)
        self.assertTrue(a < b)  # a strict subset: the survivors kept their own hops

    def test_a_frame_without_slot_0_keeps_its_records(self):
        # GnssTelemPack writes rows only into the slots that arrived, so a window whose slot-0
        # record was lost ships slot 0 as zeros, PRN included. The other three records are
        # still there and must still be read.
        zero0 = {(0, p, telem.REC_PRN): 0.0 for p in range(4)}
        zero0.update({(0, p, telem.REC_P_ENERGY): 0.0 for p in range(4)})
        raw = _make_frame(inst="cx19.0", win=10, present=0b1110, rows=zero0)
        f = telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0)
        self.assertEqual(f.prns(), [1, 2, 3, 4])
        self.assertIsNotNone(f.row(1, 2))
        self.assertIsNone(f.row(0, 2))
        self.assertEqual(len(f.comb_epl(3, 2)), 5)
        # only the last slot filled: the map comes from it; no slot filled: no PRNs
        zero012 = {(r, p, telem.REC_PRN): 0.0 for r in range(3) for p in range(4)}
        raw3 = _make_frame(inst="cx19.0", win=10, present=0b1000, rows=zero012)
        f3 = telem.TelemFrame(telem._HDR.unpack_from(raw3, 0), raw3, 0.0)
        self.assertEqual(f3.prns(), [1, 2, 3, 4])
        self.assertIsNotNone(f3.row(3, 4))
        raw_none = _make_frame(inst="cx19.0", win=10, present=0, rows=zero012)
        f_none = telem.TelemFrame(telem._HDR.unpack_from(raw_none, 0), raw_none, 0.0)
        self.assertEqual(f_none.prns(), [])
        c = self._client()
        self._feed(c, raw)
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[2])
        self.assertEqual(sorted(got["cx19.0"][2]), [f.hop(r) for r in (1, 2, 3)])


class TestStaleness(unittest.TestCase):
    """B. Live versus stale.

    A stopped instance keeps its last window forever, so one dead sender would drive the raw
    spread to the number of windows since it died, while over the live instances it stays at 0
    or 1. A health number that alarms on every instance death stops being read, so the verdict
    is over live senders and the stale ones are named.
    """

    def _feed(self, c, raw, rx):
        f = telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, rx)
        c._store_frame(f)

    def test_stale_instance_does_not_inflate_the_spread(self):
        c = telem.TelemClient(depth=64)
        now = time.time()
        # cx43.0 stopped 240 s ago at window 10; the others are current at 994/995.
        self._feed(c, _make_frame(inst="cx43.0", win=10), now - 240.0)
        self._feed(c, _make_frame(inst="cx19.0", win=994), now)
        self._feed(c, _make_frame(inst="cx27.1", win=995), now)
        st = c.stats(stale_after_s=5.0)["chains"]["gps_l5"]
        self.assertEqual(st["instances"], 3)
        self.assertEqual(st["live"], 2)
        self.assertEqual(st["stale"], ["cx43.0"])
        self.assertEqual(
            st["spread"], 1, "the dead instance must not be in the verdict"
        )

    def test_all_stale_reports_no_live_rather_than_a_healthy_spread(self):
        # Silence must never read as health. With every sender stale there is no spread to
        # report, and the row says so instead of omitting the chain.
        c = telem.TelemClient(depth=64)
        old = time.time() - 600.0
        self._feed(c, _make_frame(inst="cx19.0", win=10), old)
        self._feed(c, _make_frame(inst="cx27.0", win=10), old)
        st = c.stats(stale_after_s=5.0)["chains"]["gps_l5"]
        self.assertEqual(st["live"], 0)
        self.assertNotIn("spread", st)
        self.assertEqual(st["stale"], ["cx19.0", "cx27.0"])

    def test_staleness_can_be_disabled(self):
        c = telem.TelemClient(depth=64)
        self._feed(c, _make_frame(inst="cx43.0", win=10), time.time() - 240.0)
        self._feed(c, _make_frame(inst="cx19.0", win=994), time.time())
        st = c.stats(stale_after_s=0)["chains"]["gps_l5"]
        self.assertEqual(st["live"], 2)
        self.assertEqual(st["spread"], 984)  # the raw number, for when it is wanted


class TestStats(unittest.TestCase):
    def test_spread_reports_misalignment_among_LIVE_instances(self):
        # The alignment check, served rather than inferred: three instances all delivering now,
        # one of them on a different window. That is the real fault -- and it is a different
        # thing from an instance that stopped (TestStaleness), which is why the two must not
        # share a number. This one has to keep firing.
        c = telem.TelemClient(depth=16)
        now = time.time()

        def feed(inst, win, rx=None):
            raw = _make_frame(inst=inst, win=win)
            c._store_frame(
                telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, rx or now)
            )

        feed("cx19.0", 10)
        feed("cx42.0", 10)
        self.assertEqual(c.stats()["chains"]["gps_l5"]["spread"], 0)
        feed("cx51.1", 7)  # live, but three windows behind: a genuine misalignment
        st = c.stats()["chains"]["gps_l5"]
        self.assertEqual(st["instances"], 3)
        self.assertEqual(st["live"], 3)
        self.assertEqual(st["stale"], [])
        self.assertEqual(st["spread"], 3)


def _listen():
    import socket as _s

    srv = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
    srv.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
    # A small send buffer, set before accept so every accepted connection inherits it, and set
    # explicitly so the kernel does not autotune it upwards. Production's slack is a fraction
    # of a second at the stream's rate; without this the loopback kernel would quietly hold
    # megabytes for a stalled reader and these tests would pass on the code they are meant to
    # fail on.
    srv.setsockopt(_s.SOL_SOCKET, _s.SO_SNDBUF, 65536)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    return srv


class TestSocketDrainsIndependentOfTheRing(unittest.TestCase):
    """D. The socket thread never waits on the ring.

    The gather closes a client that cannot take a frame within 200 ms, the stream carries
    every chain on one connection, and the kernel buffers hold only a fraction of a second at
    the stream's rate. A reader that parsed and took the ring lock between two recv() calls
    would turn any contention on that lock into a dropped connection and records missing on
    every instance of every chain: a fleet-wide fleet-ADR arc break. Here the ring lock is held
    for 1.5 s while the server pushes ~2.2 MB through a 64 kB send buffer: the send must finish
    while the lock is still held.
    """

    def _serve(self, srv, n):
        conn, _ = srv.accept()
        try:
            for k in range(n):
                raw = _make_frame(win=1000 + k, seq=k)
                conn.sendall(struct.pack("<I", len(raw)) + raw)
            self.t_sent = time.monotonic()
            time.sleep(2.5)
        except Exception as e:
            self.err = e
        finally:
            conn.close()

    def test_a_held_ring_lock_does_not_stall_the_socket(self):
        import threading

        srv = _listen()
        n = 450  # ~2.2 MB of frames vs a 64 kB send buffer
        self.t_sent = None
        self.err = None
        c = telem.TelemClient(
            host="127.0.0.1",
            port=srv.getsockname()[1],
            depth=1024,
            retry_s=0.05,
            read_timeout_s=5.0,
        )
        hold_s = 1.5
        with c._lock:  # every consumer's lock, held for a long pass
            t0 = time.monotonic()
            threading.Thread(target=self._serve, args=(srv, n), daemon=True).start()
            c.start()
            time.sleep(hold_s)
            # While the lock is held the store thread cannot insert a single frame...
            frames_during = c.frames
            pending_during = len(c._pending)
            t_release = time.monotonic()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and c.frames < n:
                time.sleep(0.02)
        finally:
            c.stop()
            srv.close()
        self.assertIsNone(self.err)
        self.assertEqual(frames_during, 0)
        # ...yet the socket was drained: the server finished sending well before the release.
        self.assertIsNotNone(self.t_sent, "the server never finished sending")
        self.assertLess(
            self.t_sent,
            t_release - 0.3,
            "sendall finished only %.2f s before the lock was released: the "
            "socket thread was blocked behind the ring lock"
            % (t_release - self.t_sent),
        )
        self.assertGreaterEqual(pending_during, n // 2)
        # Nothing lost, one connection: the gather would have had no reason to drop us.
        self.assertEqual(c.frames, n)
        self.assertEqual(c.pending_dropped, 0)
        self.assertEqual(c.connects, 1)
        self.assertEqual(c.bad, 0)
        self.assertGreaterEqual(c.pending_peak, n // 2)
        del t0

    def test_the_pending_queue_sheds_the_oldest_and_keeps_the_connection(self):
        import threading

        srv = _listen()
        one = len(_make_frame())
        n = 40
        self.t_sent = None
        self.err = None
        c = telem.TelemClient(
            host="127.0.0.1",
            port=srv.getsockname()[1],
            depth=1024,
            retry_s=0.05,
            read_timeout_s=5.0,
            pending_max_bytes=4 * one,
        )
        with c._lock:
            threading.Thread(target=self._serve, args=(srv, n), daemon=True).start()
            c.start()
            time.sleep(1.0)
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline and c.frames + c.pending_dropped < n:
                time.sleep(0.02)
        finally:
            c.stop()
            srv.close()
        self.assertIsNone(self.err)
        self.assertIsNotNone(self.t_sent)
        # Over the bound the oldest frames go, counted, and the newest window is what lands.
        self.assertGreaterEqual(c.pending_dropped, n - 6)
        self.assertEqual(c.frames + c.pending_dropped, n)
        self.assertIn(1000 + n - 1, c.windows("gps_l5", lag=0))
        self.assertEqual(c.connects, 1)


class TestReconnectPolicy(unittest.TestCase):
    """E. A connection the gather closed after it lived is reconnected at once.

    Every second between the drop and the reconnect is a second of records lost on every
    chain, and the fold cannot bridge it. The backoff is for a gather that is down -- a
    connection refused or one that died young -- where retrying at once would only churn.
    """

    def _serve(self, srv, live_s, n_conns):
        for _ in range(n_conns):
            conn, _ = srv.accept()
            self.accepts.append(time.monotonic())
            try:
                t_end = time.monotonic() + live_s
                k = 0
                while time.monotonic() < t_end:
                    raw = _make_frame(win=2000 + k, seq=k)
                    conn.sendall(struct.pack("<I", len(raw)) + raw)
                    k += 1
                    time.sleep(0.02)
            except Exception:
                pass
            finally:
                conn.close()
                self.closes.append(time.monotonic())

    def _run(self, live_s, fast_after_s, retry_s):
        import threading

        srv = _listen()
        self.accepts, self.closes = [], []
        threading.Thread(target=self._serve, args=(srv, live_s, 2), daemon=True).start()
        c = telem.TelemClient(
            host="127.0.0.1",
            port=srv.getsockname()[1],
            depth=64,
            retry_s=retry_s,
            read_timeout_s=5.0,
            fast_retry_after_s=fast_after_s,
        )
        c.start()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and len(self.accepts) < 2:
                time.sleep(0.01)
        finally:
            c.stop()
            srv.close()
        self.assertEqual(len(self.accepts), 2, "the client never reconnected")
        return c, self.accepts[1] - self.closes[0]

    def test_a_connection_that_lived_is_reconnected_immediately(self):
        c, gap = self._run(live_s=0.5, fast_after_s=0.2, retry_s=2.0)
        self.assertLess(
            gap,
            0.3,
            "waited %.2f s (the backoff) after a live connection was " "closed" % gap,
        )
        self.assertEqual(c.fast_reconnects, 1)

    def test_a_connection_that_died_young_waits_the_backoff(self):
        c, gap = self._run(live_s=0.02, fast_after_s=5.0, retry_s=0.4)
        self.assertGreaterEqual(
            gap,
            0.3,
            "reconnected after %.2f s to a gather that keeps "
            "dropping us at once -- that is the churn the backoff exists "
            "for" % gap,
        )
        self.assertEqual(c.fast_reconnects, 0)


class TestReceiveBufferIsPinned(unittest.TestCase):
    """F. The receive buffer is asked for explicitly, before connect().

    The kernel starts a connection near 128 kB and autotunes up over tens of seconds. This
    stream runs at tens of MB/s, the gather drops a client that cannot take a frame in time,
    and a drop is followed by an immediate reconnect, so leaving the size to autotuning would
    re-enter the smallest-buffer ramp on every drop, at the worst possible moment.
    """

    def _connected_client(self, **kw):
        import threading

        srv = _listen()
        holder = {}

        def serve():
            try:
                conn, _ = srv.accept()
                holder["conn"] = conn
                time.sleep(1.5)
            except Exception:
                pass

        threading.Thread(target=serve, daemon=True).start()
        c = telem.TelemClient(
            host="127.0.0.1",
            port=srv.getsockname()[1],
            depth=8,
            retry_s=0.05,
            read_timeout_s=1.0,
            **kw
        )
        c.start()
        deadline = time.time() + 5.0
        while time.time() < deadline and c.rcvbuf_actual is None:
            time.sleep(0.01)
        c.stop()
        srv.close()
        return c

    def test_the_kernel_grants_what_was_asked_for_within_its_ceiling(self):
        req = 256 * 1024  # small enough to be under any plausible rmem_max
        with open("/proc/sys/net/core/rmem_max") as f:
            rmem_max = int(f.read().strip())
        c = self._connected_client(rcvbuf_bytes=req)
        self.assertIsNotNone(
            c.rcvbuf_actual, "the client never recorded a receive buffer"
        )
        # Linux clamps the request to rmem_max, then stores twice that.
        self.assertGreaterEqual(c.rcvbuf_actual, min(2 * req, rmem_max))
        self.assertEqual(c.stats()["rcvbuf"], c.rcvbuf_actual)

    def test_zero_leaves_the_kernels_autotuning_alone(self):
        c = self._connected_client(rcvbuf_bytes=0)
        # Still reported (it is the number worth reading), just not requested: a fresh
        # connection's default is far below what this stream needs, which is the whole point.
        self.assertIsNotNone(c.rcvbuf_actual)
        self.assertLess(c.rcvbuf_actual, 32 << 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
