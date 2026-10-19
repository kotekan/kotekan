#!/usr/bin/env python3
"""galileo_inav and the codecs that share it (gps_cnav2).

  * assemble_word takes only nominal pages: an alert page (page type 1) is not a word, even
    with a valid CRC.
  * gps_cnav2.decode_frame given a TOI but no polarity syncs the frame itself.
  * The galileo_inav self-test packs the same field codes in every process, whatever the
    str hash salt.
"""
import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galileo_inav as G  # noqa: E402
import gps_cnav2 as L  # noqa: E402


def _bits(v, n):
    return [(v >> (n - 1 - k)) & 1 for k in range(n)]


class TestAlertPage(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.word = [int(b) for b in rng.integers(0, 2, G.WORD_BITS)]

    def test_nominal_page_assembles(self):
        even, odd = G.build_page(self.word)
        word, crc_ok, _wt = G.assemble_word(even, odd)
        self.assertTrue(crc_ok)
        self.assertEqual(word, self.word)

    def test_alert_page_with_valid_crc_is_rejected(self):
        # page type 1 (alert) in either part, or both, with the CRC recomputed to match
        for parts in ("even", "odd", "both"):
            with self.subTest(alert=parts):
                even, odd = G.build_page(self.word)
                if parts in ("even", "both"):
                    even[1] = 1
                if parts in ("odd", "both"):
                    odd[1] = 1
                crc = G.crc24q(even + odd[: G.ODD_CRC_AT])
                odd[G.ODD_CRC_AT : G.ODD_CRC_AT + 24] = _bits(crc, 24)
                self.assertIsNone(G.assemble_word(even, odd))


def _cnav2_stream(toi, rev):
    """One CNAV-2 frame (TOI `toi`, polarity `rev`) plus the next frame's SF1: (symbols,
    SF2 bits, SF3 bits)."""
    rng = np.random.default_rng(11)

    def crc_append(k):
        m = [int(b) for b in rng.integers(0, 2, k - 24)]
        return m + _bits(G.crc24q(m), 24)

    sf2, sf3 = crc_append(L.K_SF2), crc_append(L.K_SF3)
    deint = np.array(
        sf2
        + [int(b) for b in rng.integers(0, 2, L.N_SF2 - L.K_SF2)]
        + sf3
        + [int(b) for b in rng.integers(0, 2, L.N_SF3 - L.K_SF3)],
        dtype=np.int8,
    )
    il = deint.reshape(L.IL_COLS, L.IL_ROWS).T.ravel()
    syms = np.hstack([L._sf1(toi), il, L._sf1((toi + 1) % L.N_TOI)]).astype(np.int8)
    return syms ^ rev, sf2, sf3


class TestCnav2DecodeFrame(unittest.TestCase):
    def test_toi_without_rev_syncs(self):
        for rev in (0, 1):
            with self.subTest(rev=rev):
                syms, sf2, sf3 = _cnav2_stream(123, rev)
                self.assertEqual(L.decode_frame(syms, toi=123), (123, sf2, sf3))

    def test_toi_without_rev_on_noise(self):
        self.assertIsNone(L.decode_frame(np.zeros(1852, dtype=np.int8), toi=0))


# Runs galileo_inav.py as __main__ and prints the words its self-test hands to
# parse_inav_ephemeris (the packed field codes).
_CAPTURE = r"""
import runpy, sys
seen = []
def prof(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "parse_inav_ephemeris":
        w = frame.f_locals["words_by_type"]
        seen.append(sorted((k, tuple(v)) for k, v in w.items()))
sys.setprofile(prof)
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except SystemExit as e:
    rc = e.code
sys.setprofile(None)
print(repr(seen))
sys.exit(rc)
"""


class TestSelftestReproducible(unittest.TestCase):
    def test_field_codes_independent_of_hash_seed(self):
        out = []
        for seed in ("1", "2"):
            r = subprocess.run(
                [sys.executable, "-c", _CAPTURE, os.path.join(HERE, "galileo_inav.py")],
                env=dict(os.environ, PYTHONHASHSEED=seed),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("ALL SELF-CONSISTENT", r.stdout)
            out.append(r.stdout.strip().splitlines()[-1])
        self.assertNotEqual(out[0], "[]")
        self.assertEqual(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
