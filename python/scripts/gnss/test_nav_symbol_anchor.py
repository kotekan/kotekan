#!/usr/bin/env python3
"""nav_symbol_anchor: a stream's satellite-time index and sign from a decoded message.

Synthetic streams: random symbols with a message placed at a known stream index, the whole
stream times a sign. The anchor must find that index's satellite time and the sign, and must
refuse a mark that does not line up with what the stream holds.
"""

import datetime
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import galileo_fnav
import galileo_inav
import gnss_ephemeris as ge
import nav_symbol_anchor as na

WK = 2446  # GPS week
T0 = 1795540000.0  # a stream's start, UTC


def utc_of_gpst(t):
    return (ge.GPS_EPOCH + datetime.timedelta(seconds=t - ge.GPS_UTC_LEAP)).timestamp()


class TestIndex(unittest.TestCase):
    def test_gps_l2c(self):
        # A CNAV TOW counts to the next message: a 12 s L2C message with TOW n starts at 6n - 12.
        n = 40000
        mark = {
            "signal": "GPS_L2C_CNAV",
            "tow": n,
            "t_host": utc_of_gpst(WK * 604800 + 6 * n - 12 + 0.07),
        }
        self.assertEqual(
            na.sv_index(mark, "G", 0.02), round((WK * 604800 + 6 * n - 12) / 0.02)
        )

    def test_beidou_is_bdt(self):
        # B-CNAV2 SOW 3n is BDT: 14 s behind GPST, so 2800 B2a symbols lower than the GPST count.
        n = 100000
        t_gpst = WK * 604800 + 3 * n + 14.0
        mark = {
            "signal": "BDS_B2A_BCNAV2",
            "tow": n,
            "t_host": utc_of_gpst(t_gpst + 0.07),
        }
        self.assertEqual(na.sv_index(mark, "C", 0.005), round(t_gpst / 0.005) - 2800)


class TestSign(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def stream(self, n, k, held, sign):
        s = self.rng.choice([-1, 1], n).astype(np.int8)
        s[k : k + len(held)] = held
        return (s * sign).astype(np.int8)

    def test_fnav_sync(self):
        sync = 1 - 2 * np.array(galileo_fnav.SYNC)
        buf = self.stream(1500, 700, sync, -1)
        mark = {"t_host": T0 + (1000 + 700) * 0.02}
        self.assertEqual(
            na.mark_sign(mark, "GAL_E5AI_FNAV", buf, 1000, T0, 0.02), (1700, -1, 0)
        )
        mark["t_host"] += 0.02  # one symbol late: no sync there
        self.assertIsNone(na.mark_sign(mark, "GAL_E5AI_FNAV", buf, 1000, T0, 0.02))

    def test_cnav_reencoded(self):
        # Two messages encoded continuously, on air (G2 not inverted); the mark is the second's.
        prev = self.rng.integers(0, 2, 300)
        msg = self.rng.integers(0, 2, 300)
        sym = galileo_inav.conv_encode(np.concatenate([prev, msg]), flush=False).astype(
            int
        )
        sym[1::2] ^= 1
        sym = 1 - 2 * sym
        buf = np.concatenate([self.rng.choice([-1, 1], 200), sym]).astype(np.int8)
        k = 1000 + 200 + 600
        bits = "%075x" % int("".join(str(b) for b in msg), 2)
        mark = {"t_host": T0 + k * 0.02, "bits": bits}
        self.assertEqual(
            na.mark_sign(mark, "GPS_L2C_CNAV", buf, 1000, T0, 0.02), (k, 1, 0)
        )
        self.assertEqual(
            na.mark_sign(mark, "GPS_L2C_CNAV", -buf, 1000, T0, 0.02), (k, -1, 0)
        )
        mark["t_host"] -= 2 * 0.02  # a bit early
        self.assertIsNone(na.mark_sign(mark, "GPS_L2C_CNAV", buf, 1000, T0, 0.02))

    def test_bcnav3_preamble_within_slack(self):
        # B-CNAV3's marks are on the combiner's grid: the frame start is found up to 3 symbols off.
        pat = 1 - 2 * na.beidou_bcnav3.preamble_for(25).astype(int)
        buf = self.stream(3000, 1200, pat, -1)
        mark = {"prn": 25, "t_host": T0 + (1000 + 1202) * 0.001}  # 2 symbols late
        self.assertEqual(
            na.mark_sign(mark, "BDS_B2B_BCNAV3", buf, 1000, T0, 0.001), (2200, -1, 0)
        )
        mark["t_host"] += 0.002  # 4 symbols late: beyond the slack
        self.assertIsNone(na.mark_sign(mark, "BDS_B2B_BCNAV3", buf, 1000, T0, 0.001))

    def test_old_mark_carried_by_whole_frames(self):
        # A B-CNAV3 mark 5 frames before the window: its frame start carried 5000 symbols into
        # it, the satellite-time index with it.
        pat = 1 - 2 * na.beidou_bcnav3.preamble_for(23).astype(int)
        buf = self.stream(1500, 300, pat, 1)  # frame starts at stream 10300 and 11300
        buf[1300 : 1300 + len(pat)] = pat
        mark = {
            "prn": 23,
            "t_host": T0 + 5301 * 0.001,
        }  # one symbol off, 5 frames early
        self.assertEqual(
            na.mark_sign(mark, "BDS_B2B_BCNAV3", buf, 10000, T0, 0.001),
            (10300, 1, 5000),
        )

    def test_anchor_of_a_status_row(self):
        pre = 1 - 2 * np.array(na.beidou_bcnav2.PREAMBLE)
        buf = self.stream(1500, 900, pre, -1)
        n_sym = 5000
        k = n_sym - 1500 + 900
        n = 100000
        t_sym_start = WK * 604800 + 3 * n + 14.0
        mark = {
            "signal": "BDS_B2A_BCNAV2",
            "prn": 25,
            "tow": n,
            "t_host": T0 + k * 0.005,
        }
        # the stream's receive grid puts symbol k at the mark's time; the satellite time is 3n
        row = {
            "prn": 25,
            "n_sym": n_sym,
            "t0": T0,
            "stream_id": int(T0 * 1000),
            "nav_obs": {
                "br": 5,
                "rec_dt": 0.001,
                "bits": [[i, int(s)] for i, s in enumerate(buf)],
            },
        }
        old = na.sv_index
        na.sv_index = lambda m, s, t: round((t_sym_start - 14.0) / 0.005)
        try:
            got = na.anchor_of(row, [mark], "BDS_B2A_BCNAV2", "C", 0.005)
        finally:
            na.sv_index = old
        self.assertEqual(got, (k, round((t_sym_start - 14.0) / 0.005), -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
