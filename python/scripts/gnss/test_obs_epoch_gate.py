#!/usr/bin/env python3
"""A row's epoch is its F-engine hop, and it must be within MAX_EPOCH_SKEW_S of now, or there
is no row.

Right after an F-engine re-base the writer runs on the new frame0 while the broker's fleet hop
can still come from the previous session's windows: new frame0 + old hop lands days in the
future, and since files roll on the row's epoch, those rows would head a future day's file. A
row with no hop has no epoch; the broker's `utc` is diagnostic only.

The constants below are a real re-base: the previous session's frame0 and last window, the
new frame0, and a write 49 s after the re-base.

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
FRAME0_OLD = 1790346673.000002861  # frame0 of the previous session
FRAME0_NEW = 1790626549.000002861  # frame0 after the re-base
WRITE_T = 1790626598.0  # the write, 49 s after the re-base
OLD_LAST = 1790622264.6  # the previous session's last window


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
        # ...which new frame0 + stale hop would have stamped ~3.2 days after the write
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
        # combdll seeds its aggregate with hop -1 ("no window"); it is truthy, and as a hop
        # it would stamp frame0 - 5 us, before the session's first hop.
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
