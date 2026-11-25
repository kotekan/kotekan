#!/usr/bin/env python3
"""Galileo E6-B C/NAV (galileo_cnav, has_predictor): pages, HAS messages and their recovery.

* A page's 486 bits come back through FEC, interleaver and sync, with a valid CRC; a dummy
  header reads as no header.
* The RS(255, 32) generator is systematic and every column is a code word: zero at alpha^1 to
  alpha^223.
* Any MS distinct pages give the message back; a page beyond MS that disagrees refuses it.
* A type 1 message with a mask, orbit, clock and code biases parses to its values; a later
  message on the same mask ID reads with the mask held.
* The predictor gathers pages of one message from two satellites, in windows of either
  polarity, counts each page once, and holds the parsed message.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galileo_cnav as C  # noqa: E402
import has_predictor  # noqa: E402
from galileo_inav import crc24q  # noqa: E402


def _bits(v, n):
    return [(v >> (n - 1 - k)) & 1 for k in range(n)]


def _page(header, body_octets):
    """486 page bits: reserved, header (int), 53 body octets, CRC."""
    b = [0] * 14 + _bits(header, 24) + C.message_bits(body_octets)
    return b + _bits(crc24q(b), 24)


def _header(mt, mid, ms, pid, status=1):
    return status << 22 | mt << 18 | mid << 13 | (ms - 1) << 8 | pid


def _encode(msg, ms):
    """All 255 code rows of an ms x 53 message."""
    m = np.zeros((32, C.OCTETS), dtype=np.int64)
    m[:ms] = msg
    return np.array([C._combine(C.G[k], m) for k in range(255)])


def _mt1(ms):
    """A type 1 message: mask ID 3 holding E04 and E10 on E5a-Q and E6-C, orbit, clock and code
    biases; ms pages of octets, zero filled."""
    b = _bits(1800, 12) + [1, 1, 1, 0, 1, 0] + [0] * 4 + _bits(3, 5) + _bits(2, 5)
    b += _bits(1, 4) + _bits(2, 4)  # one system: Galileo
    b += _bits(1 << 36 | 1 << 30, 40)  # E04, E10
    b += _bits(1 << 11 | 1 << 2, 16)  # signals 4 (E5a-Q) and 13 (E6-C)
    b += [0] + _bits(0, 3) + [0] * 6
    b += _bits(10, 4)  # orbit validity 300 s
    for iod, dr, dit, dct in ((85, -40, 12, -2048), (86, 100, -5, 3)):
        b += (
            _bits(iod, 10)
            + _bits(dr & 0x1FFF, 13)
            + _bits(dit & 0xFFF, 12)
            + _bits(dct & 0xFFF, 12)
        )
    b += _bits(5, 4) + _bits(1, 2)  # clock validity 60 s, multiplier 2
    b += _bits(-30 & 0x1FFF, 13) + _bits(-4096 & 0x1FFF, 13)
    b += _bits(10, 4)
    for v in (22, -7, -1024, 50):
        b += _bits(v & 0x7FF, 11)
    b += [0] * (ms * 424 - len(b))
    return np.array(
        [int("".join(map(str, b[8 * k : 8 * k + 8])), 2) for k in range(ms * C.OCTETS)]
    ).reshape(ms, C.OCTETS)


class TestPage(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        body = rng.integers(0, 256, C.OCTETS)
        bits = _page(_header(1, 7, 4, 99), body)
        sym = 1.0 - 2.0 * C.encode_page(bits)
        out = C.decode_page(sym + 0.4 * rng.standard_normal(len(sym)))
        self.assertEqual(out, bits)
        self.assertTrue(C.check_page(out))
        self.assertEqual(
            C.header(out), {"status": 1, "type": 1, "mid": 7, "ms": 4, "pid": 99}
        )
        self.assertTrue(np.array_equal(C.body_octets(out), body))
        self.assertIsNone(C.header(_page(C.DUMMY, body)))
        self.assertIsNone(C.decode_page(-sym))


class TestReedSolomon(unittest.TestCase):
    def test_generator(self):
        self.assertTrue(np.array_equal(C.G[:32], np.eye(32, dtype=np.int64)))
        for j in (0, 13, 31):
            for i in (1, 77, 223):
                s = 0
                for k, c in enumerate(C.G[:, j]):
                    if c:
                        s ^= int(C._EXP[(C._LOG[c] + i * (254 - k)) % 255])
                self.assertEqual(s, 0)

    def test_recover(self):
        rng = np.random.default_rng(5)
        msg = rng.integers(0, 256, (5, C.OCTETS))
        rows = _encode(msg, 5)
        pages = {p: rows[p - 1] for p in (3, 40, 77, 200, 254, 120)}
        out, n = C.recover(pages, 5)
        self.assertTrue(np.array_equal(out, msg))
        self.assertEqual(n, 1)
        self.assertIsNone(C.recover({p: pages[p] for p in (3, 40, 77, 200)}, 5)[0])
        pages[254] = pages[254] ^ 1
        self.assertIsNone(C.recover(pages, 5)[0])


class TestMt1(unittest.TestCase):
    def test_parse(self):
        masks = {}
        d = C.parse_mt1(C.message_bits(_mt1(2)), masks)
        self.assertEqual((d["toh"], d["mask_id"], d["iod_set"]), (1800, 3, 2))
        self.assertEqual(d["flags"], ["mask", "orbit", "clock", "code_bias"])
        e4, e10 = d["sats"]["E04"], d["sats"]["E10"]
        self.assertEqual(
            e4["orbit"],
            {"iod": 85, "radial_m": -0.1, "in_track_m": 0.096, "cross_track_m": None},
        )
        self.assertEqual(e10["orbit"]["radial_m"], 0.25)
        self.assertEqual((e4["clock_m"], e10["clock_m"]), (-0.15, None))
        self.assertEqual(e4["code_bias_m"], {"E5a-Q": 0.44, "E6-C": -0.14})
        self.assertEqual(e10["code_bias_m"], {"E5a-Q": None, "E6-C": 1.0})
        self.assertIn(3, masks)
        # A clock-only message on mask 3 reads with the mask held.
        b = _bits(1810, 12) + [0, 0, 1, 0, 0, 0] + [0] * 4 + _bits(3, 5) + _bits(2, 5)
        b += _bits(5, 4) + _bits(0, 2) + _bits(4, 13) + _bits(-8 & 0x1FFF, 13)
        d = C.parse_mt1(b, masks)
        self.assertEqual(
            (d["sats"]["E04"]["clock_m"], d["sats"]["E10"]["clock_m"]), (0.01, -0.02)
        )
        self.assertEqual(C.parse_mt1(b, {})["mask"], "not held")


class TestPredictor(unittest.TestCase):
    def test_pages_from_two_satellites(self):
        rng = np.random.default_rng(9)
        rows = _encode(_mt1(2), 2)
        sent = {4: (60, 61, 62), 10: (130, 131, 132)}
        pred = has_predictor.HasPredictor()
        for prn, pids in sent.items():
            sym = np.concatenate(
                [
                    1.0 - 2.0 * C.encode_page(_page(_header(1, 9, 2, p), rows[p - 1]))
                    for p in pids
                ]
            )
            # Two overlapping 2.5 s windows, each of its own polarity.
            for a in (0, 500):
                pol = rng.choice([-1, 1])
                w = sym[a : a + 2500]
                obs = {
                    "utc_ref": 100.0 + a * 0.001,
                    "rec_dt": 0.001,
                    "phase": 0,
                    "br": 1,
                    "bits": [[i, int(pol * v)] for i, v in enumerate(w)],
                }
                pred.ingest(prn, obs)
        self.assertEqual(pred.n_messages, 1)
        self.assertEqual(
            pred.health(4), {"pages": 3, "dummy": 0, "crc_fail": 0, "messages": 1}
        )
        self.assertEqual(pred.health(10)["pages"], 3)
        t, d = pred.newest["orbit"]
        self.assertEqual(d["sats"]["E04"]["orbit"]["iod"], 85)
        self.assertIsNone(pred.ephemeris(4))

    def test_soft_values(self):
        # At this noise (symbol SNR ~3 dB) hard decisions fail both pages; the soft values
        # decode them.
        rng = np.random.default_rng(3)
        sym = np.concatenate(
            [
                1.0
                - 2.0
                * C.encode_page(
                    _page(_header(1, 7, 4, 90 + k), rng.integers(0, 256, C.OCTETS))
                )
                for k in range(2)
            ]
        )
        x = sym + 0.7 * rng.standard_normal(len(sym))
        for k in (0, 1000):
            x[k : k + C.N_SYNC] = sym[k : k + C.N_SYNC]
        obs = {
            "utc_ref": 50.0,
            "rec_dt": 0.001,
            "phase": 0,
            "br": 1,
            "bits": [[i, 1 if v >= 0 else -1] for i, v in enumerate(x)],
        }
        pages = []
        for soft in (None, [int(round(10 * v)) for v in x]):
            pred = has_predictor.HasPredictor()
            pred.ingest(3, dict(obs, soft=soft) if soft else obs)
            pages.append(pred.health(3)["pages"])
        self.assertEqual(pages, [0, 2])


if __name__ == "__main__":
    unittest.main()
