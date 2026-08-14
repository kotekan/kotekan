#!/usr/bin/env python3
"""Tests for the task #59 tracker->broker telemetry transport (gnss_broker/telem.py).

TWO KINDS OF TEST HERE, and the first is the one that matters:

  A. CROSS-LANGUAGE. scripts/gnss/telemfmt.cpp is COMPILED AND RUN, so the parser is checked
     against gnssTelem.hpp itself rather than against a transcription of it. The struct and the
     struct.Struct format string are two independent statements of one layout, and a drift
     between them does not raise -- it yields plausible numbers with the wrong meaning. That is
     precisely the class of defect this transport exists to remove, so it would be absurd to
     introduce a new instance of it in the parser.

  B. BEHAVIOURAL. The window ring, the lag rule, hole-vs-shift on a dropped record, and the
     cross-instance grouping -- each written against the specific way its REST predecessor
     failed (#53, #52, #46, #33). Every one of these is a case where the old path produced a
     confident wrong answer rather than an error.

    python3 python/scripts/gnss/test_telem.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

from gnss_broker import telem  # noqa: E402


def _make_frame(chain="gps_l5", inst="cx19.0", win=100, seq=0, n_rec=4, n_prn=4,
                hops_per_record=2048, fft_len=16384, present=None, rows=None, utc0=0.0):
    """Build one wire frame in Python -- for the BEHAVIOURAL tests only.

    The layout it assumes is the one test_wire_format_matches_cpp proves correct, so these
    tests are not quietly re-deriving the format: if the C++ moves, that test fails first and
    these become meaningless rather than misleading.
    """
    if present is None:
        present = (1 << n_rec) - 1
    wstart0 = win * n_rec * hops_per_record * fft_len
    hdr = telem._HDR.pack(telem._MAGIC, telem._VERSION, n_rec, n_prn, telem._ROW_FLOATS,
                          7, 32, hops_per_record, fft_len, win, seq, wstart0, utc0,
                          present, 0, chain.encode(), inst.encode())
    body = [0.0] * (n_rec * n_prn * telem._ROW_FLOATS)
    for r in range(n_rec):
        for p in range(n_prn):
            base = (r * n_prn + p) * telem._ROW_FLOATS
            body[base + telem.REC_PRN] = float(1 + p)
            body[base + telem.REC_P_ENERGY] = 2.0
            body[base + telem.REC_P_RE] = 2.0 * (1 + p)   # A = 1+p exactly
            body[base + telem.REC_P_IM] = 0.0
            body[base + telem.REC_CPHASE] = 0.25          # a per-record INCREMENT, cycles
            body[base + telem.REC_TRIM_INC] = 0.01
    if rows:
        for (r, p, slot), v in rows.items():
            body[(r * n_prn + p) * telem._ROW_FLOATS + slot] = v
    return hdr + struct.pack("<%df" % len(body), *body)


class TestWireFormat(unittest.TestCase):
    """A. The parser against the C++ definition."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="telemfmt-")
        src = os.path.join(K, "scripts", "gnss", "telemfmt.cpp")
        cls.bin = os.path.join(cls.tmp, "telemfmt")
        p = subprocess.run(["c++", "-std=c++17", "-O1",
                            "-I", os.path.join(K, "lib", "stages", "gnss"),
                            "-o", cls.bin, src],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise unittest.SkipTest("cannot compile telemfmt.cpp:\n%s" % p.stderr)
        cls.frame_path = os.path.join(cls.tmp, "frame.bin")
        r = subprocess.run([cls.bin, cls.frame_path], capture_output=True, text=True, check=True)
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
        for name, want in (("win", 24), ("seq", 32), ("wstart0", 40), ("utc0", 48),
                           ("present", 56), ("chain", 64), ("inst", 80)):
            self.assertEqual(m["off_" + name], want, "gnssTelem.hpp moved field %r" % name)

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
        # telemfmt sets present = 0b1011: slot 2 absent. The records after it must keep THEIR
        # OWN identity -- a transport that appended instead of addressing would hand slot 3's
        # data back as slot 2, and every downstream number would be one record stale with
        # nothing to show for it.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertTrue(f.has_record(0))
        self.assertTrue(f.has_record(1))
        self.assertFalse(f.has_record(2))
        self.assertTrue(f.has_record(3))
        self.assertIsNone(f.row(2, 100))
        self.assertEqual(f.row(3, 100)[telem.REC_P_IM], 3000 + 0 + telem.REC_P_IM)

    def test_utc_double_alias(self):
        # UTC is a double aliased over two float slots. A float-by-float parse gets a
        # plausible-looking wrong number here rather than an error.
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertAlmostEqual(f.utc(1, 103), 1786285988.5 + 0.0104857 * 1 + 0.001 * 3, places=6)

    def test_hop_is_the_absolute_fengine_index(self):
        f = telem.TelemFrame(telem._HDR.unpack_from(self.raw, 0), self.raw, 0.0)
        self.assertEqual(f.hop(0), f.wstart0 // 16384)
        self.assertEqual(f.hop(1) - f.hop(0), 2048)
        # The key /get_records used, so the two feeds are directly comparable -- which is what
        # makes an A/B between them a measurement instead of an argument.
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
        self.assertEqual(sorted(c.frame_set("gps_l5", 10).keys()), ["cx19.0", "cx19.1", "cx42.0"])

    def test_lag_hides_the_newest_window(self):
        # #53's defect in one assertion: the newest window is ALWAYS still filling, because the
        # senders are independent processes. Reading it combines a partial fleet with its
        # neighbours' full one, which reads as a fleet that keeps changing size.
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

    def test_chains_do_not_mix(self):
        c = self._client()
        self._feed(c, _make_frame(chain="gps_l5", inst="cx19.0", win=10))
        self._feed(c, _make_frame(chain="gal_e5a", inst="cx19.0", win=10))
        self.assertEqual(c.chains(), ["gal_e5a", "gps_l5"])
        self.assertEqual(list(c.frame_set("gal_e5a", 10)), ["cx19.0"])


class TestCoherentSource(unittest.TestCase):
    """B. The /get_records replacement, in the shape fleet_coherent already consumes."""

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
        # THE WHOLE POINT. Two instances that reported the same window must produce the SAME
        # hop keys -- no tolerance, no nearest-match. Under the REST path this was inferred
        # from arrival order and from a UTC each instance stamped independently (#46 measured
        # 0.105 s of spread in exactly that stamp).
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx42.1", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[1])
        self.assertEqual(set(got["cx19.0"][1]), set(got["cx42.1"][1]))

    def test_zero_energy_rows_are_silence_not_zeros(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10,
                                  rows={(0, 1, telem.REC_P_ENERGY): 0.0}))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[2])
        self.assertEqual(len(got["cx19.0"][2]), 3)  # slot 0 dropped, not carried as 0+0j

    def test_missing_record_slot_does_not_shift_hops(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10, present=0b1101))
        self._feed(c, _make_frame(inst="cx42.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        got, _ = c.coherent_source("gps_l5", prns=[1])
        a, b = set(got["cx19.0"][1]), set(got["cx42.0"][1])
        self.assertEqual(len(a), 3)
        self.assertTrue(a < b)  # a strict subset: the survivors kept their own hops


class TestRecordStream(unittest.TestCase):
    """B. The per-instance, contiguity-checked feed -- the #33 fix.

    The carrier-rate feed used to difference `res_cycles`, a per-instance ACCUMULATOR, across a
    served row that silently changed instance: 4.92 Hz where one instance read 0.82. So the
    replacement must make the instance the key and must SAY where the run breaks.
    """

    def _client(self):
        return telem.TelemClient(depth=16)

    def _feed(self, c, raw):
        c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))

    def test_contiguous_run_reports_no_gaps(self):
        c = self._client()
        for w in (10, 11, 12):
            self._feed(c, _make_frame(inst="cx19.0", win=w))
        s = c.record_stream("gps_l5", "cx19.0", 1)
        self.assertEqual(len(s), 8)  # windows 10 and 11; 12 is held back by the lag
        self.assertEqual([x["gap"] for x in s], [0] * 8)
        self.assertEqual([x["hop"] for x in s], sorted(x["hop"] for x in s))
        self.assertTrue(all(x["dphi_cmd"] == 0.25 for x in s))

    def test_a_break_is_reported_not_smoothed(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=12))  # window 11 never arrived
        self._feed(c, _make_frame(inst="cx19.0", win=13))
        s = c.record_stream("gps_l5", "cx19.0", 1)
        gaps = [x["gap"] for x in s]
        self.assertEqual(gaps[:4], [0, 0, 0, 0])
        self.assertEqual(gaps[4], 4, "a missing window is 4 missing records, and the consumer "
                                     "must be told so rather than left to difference across it")

    def test_one_instance_only(self):
        c = self._client()
        self._feed(c, _make_frame(inst="cx19.0", win=10))
        self._feed(c, _make_frame(inst="cx42.0", win=10))
        self._feed(c, _make_frame(inst="cx19.0", win=11))
        self.assertEqual(len(c.record_stream("gps_l5", "cx19.0", 1)), 4)
        self.assertEqual(len(c.record_stream("gps_l5", "cx42.0", 1)), 4)
        self.assertEqual(c.record_stream("gps_l5", "cx99.9", 1), [])


class TestStats(unittest.TestCase):
    def test_spread_reports_misalignment(self):
        # The alignment check, served rather than inferred: if two instances are on different
        # windows this number says so immediately, instead of the fact surfacing weeks later as
        # a physics anomaly.
        c = telem.TelemClient(depth=16)
        for raw in (_make_frame(inst="cx19.0", win=10), _make_frame(inst="cx42.0", win=10)):
            c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))
        self.assertEqual(c.stats()["chains"]["gps_l5"]["spread"], 0)
        raw = _make_frame(inst="cx51.1", win=7)
        c._store_frame(telem.TelemFrame(telem._HDR.unpack_from(raw, 0), raw, 0.0))
        st = c.stats()["chains"]["gps_l5"]
        self.assertEqual(st["instances"], 3)
        self.assertEqual(st["spread"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
