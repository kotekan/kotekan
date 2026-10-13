#!/usr/bin/env python3
"""A row's epoch is its F-engine hop, and it must be NOW-ish -- or there is no row.

THE BUG THIS PINS (found 2026-10-02): gal_e5a_20261002.jsonl was created on 09-28 at 20:16:38,
48 s after that day's F-engine re-base. The writer had restarted on the new frame0, but the
broker's fleet hop still came from the previous session's windows, and new frame0 + old hop is
10-02 00:49:00. Files roll on the row's epoch, so those rows headed a file four days early. A row
with no hop at all fell back to the broker's `utc`, which the broker documents as diagnostic-only.

The numbers below are the incident's own.

Run: python3 test_obs_epoch_gate.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_observables import (  # noqa: E402
    row_epoch,
    SAMPLES_PER_HOP as SPH,
    SAMPLE_RATE_HZ as SR,
)

HOP_S = SPH / SR
FRAME0_OLD = 1790346673.000002861  # 09-25 session
FRAME0_NEW = 1790626549.000002861  # 09-28 20:15:49 re-base
WRITE_T = 1790626598.0  # 09-28 20:16:38, when the bad rows were written
OLD_LAST = 1790622264.6  # 09-28 19:04:24.6, the old session's last window


def epoch(r, now):
    return row_epoch(r, FRAME0_NEW, now)


class TestRowEpoch(unittest.TestCase):
    def test_a_current_hop_is_recorded(self):
        hop = int((WRITE_T - 1.5 - FRAME0_NEW) / HOP_S)
        t, h, why = epoch({"fleet_hop": hop}, WRITE_T)
        self.assertIsNone(why)
        self.assertAlmostEqual(t, WRITE_T - 1.5, places=3)
        self.assertEqual(h, hop)

    def test_the_previous_sessions_hop_is_refused(self):
        stale = int((OLD_LAST - FRAME0_OLD) / HOP_S)  # what the broker still served
        t, _, why = epoch({"fleet_hop": stale, "pow_hop": 9435136}, WRITE_T)
        self.assertIsNone(t)
        self.assertEqual(why, "skew")
        # and that refused row would have been stamped 10-02 00:49:00, the incident's stamp
        self.assertAlmostEqual(FRAME0_NEW + stale * HOP_S, 1790902140.6, delta=0.01)

    def test_no_hop_is_no_row(self):
        # The broker's diagnostic `utc` is not an epoch. Even a plausible one is refused.
        for r in (
            {"utc": WRITE_T},
            {"fleet_hop": 0, "utc": WRITE_T},
            {"fleet_hop": None},
        ):
            self.assertEqual(epoch(r, WRITE_T)[2], "no-hop", r)

    def test_the_combs_no_window_marker_is_not_a_hop(self):
        # combdll seeds its aggregate with hop -1; it is truthy, and used to be stamped
        # frame0 - 5 us -- days in the past.
        self.assertEqual(epoch({"fleet_hop": -1}, WRITE_T)[2], "no-hop")

    def test_pow_hop_is_the_fallback_hop(self):
        hop = int((WRITE_T - 2.0 - FRAME0_NEW) / HOP_S)
        self.assertIsNone(epoch({"pow_hop": hop}, WRITE_T)[2])

    def test_old_rows_are_refused_too(self):
        now = FRAME0_NEW + 2 * 3600.0  # two hours into the session
        hop = int((now - 3600.0 - FRAME0_NEW) / HOP_S)  # a row stamped an hour ago
        self.assertEqual(epoch({"fleet_hop": hop}, now)[2], "skew")


if __name__ == "__main__":
    unittest.main()
