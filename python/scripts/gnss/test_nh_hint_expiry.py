#!/usr/bin/env python3
"""The nh-offset hint must count DISTINCT detections and must be able to EXPIRE.

THE INCIDENT (2026-08-10). The hint narrows the search to +-nh_hint_span of 20 overlay
phases. Two defects made a wrong hint permanent:

  1. the accumulator re-appended every nh_seen entry on every cycle, so ONE stale detection
     filled the 64-sample window in ~2 minutes and nh_hint_min_samples read it as 64
     independent confirmations -- a count of UPTIME, not of evidence; and
  2. nothing aged the offset out, so a wrong one kept being pushed forever.

Together those close a loop with no restoring force: bad clock -> bad hint -> search
narrowed onto the wrong phase -> no detections -> the clock cannot be re-solved. The only
escape was the code clock random-walking back onto truth (~15 min, observed).

These tests model the accumulator's logic directly rather than the broker's cycle, so they
stay meaningful without a live fleet.
"""
import unittest


class Accum:
    """The nh-offset accumulator: distinct-detection admission + aged window."""

    def __init__(self, min_samples=6, max_age_s=600.0, keep=64):
        self.min_samples, self.max_age_s, self.keep = min_samples, max_age_s, keep
        self.hist = []        # (t, offset)
        self.last_rh = {}     # prn -> ref_hop already folded in
        self.offset = None

    def cycle(self, t, seen):
        """`seen` is prn -> (offset, ref_hop), the broker's nh_seen for this cycle."""
        for prn, (off, rh) in seen.items():
            if self.last_rh.get(prn) == rh:
                continue          # SAME detection as last cycle: not new evidence
            self.last_rh[prn] = rh
            self.hist.append((t, off))
        del self.hist[:-self.keep]
        fresh = [o for (ts, o) in self.hist if t - ts <= self.max_age_s]
        if len(fresh) < self.min_samples:
            self.offset = None    # let the search widen rather than stay confidently wrong
        else:
            self.offset = max(set(fresh), key=fresh.count)
        return self.offset


class TestNhHintExpiry(unittest.TestCase):
    def test_one_stale_detection_cannot_fill_the_window(self):
        """DEFECT 1. A single detection, held while the fleet is starved, must never look
        like a full window of agreement however long the broker runs."""
        a = Accum()
        seen = {3: (7, 1000)}                      # one PRN, one detection, never updated
        for i in range(400):                       # ~13 min at a 2 s interval
            a.cycle(i * 2.0, seen)
        self.assertEqual(len(a.hist), 1, "a stale detection was re-counted as new evidence")
        self.assertIsNone(a.offset, "one detection cleared a 6-sample threshold")

    def test_healthy_fleet_builds_and_holds_the_offset(self):
        """...without breaking the case the hint exists for."""
        a = Accum()
        t = 0.0
        for k in range(10):                        # ten distinct detections, agreeing
            t += 30.0
            a.cycle(t, {p: (7, 1000 + k) for p in (3, 6, 9)})
        self.assertEqual(a.offset, 7)

    def test_offset_expires_when_detections_stop(self):
        """DEFECT 2. Once established, the offset must not outlive its evidence -- this is
        the step that breaks the loop."""
        a = Accum(max_age_s=600.0)
        t = 0.0
        for k in range(10):
            t += 30.0
            a.cycle(t, {p: (7, 1000 + k) for p in (3, 6, 9)})
        self.assertEqual(a.offset, 7)
        frozen = {3: (7, 1009), 6: (7, 1009), 9: (7, 1009)}   # detections stop updating
        t += 300.0
        self.assertEqual(a.cycle(t, frozen), 7, "expired while samples were still fresh")
        t += 400.0                                             # now past max_age_s
        self.assertIsNone(a.cycle(t, frozen),
                          "a wrong hint would keep narrowing the search forever")

    def test_recovery_after_expiry(self):
        """And the loop must be re-enterable: wide search -> detections -> hint rebuilds."""
        a = Accum(max_age_s=600.0)
        t = 0.0
        for k in range(10):
            t += 30.0
            a.cycle(t, {p: (7, 1000 + k) for p in (3, 6, 9)})
        t += 1200.0
        a.cycle(t, {3: (7, 1009)})
        self.assertIsNone(a.offset)
        for k in range(10):                        # the widened search finds the TRUE phase
            t += 30.0
            a.cycle(t, {p: (12, 2000 + k) for p in (3, 6, 9)})
        self.assertEqual(a.offset, 12, "the hint never recovered after expiring")


if __name__ == "__main__":
    unittest.main(verbosity=2)
