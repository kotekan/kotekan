#!/usr/bin/env python3
"""The P2c rotation must actually rotate, must end cleanly, and must not silently vanish.

A hand-picked PRN is one satellite at one geometry; on a transit instrument that produced
two contradictory results and one null (docs 11.31). The rotation exists to accumulate a
DISTRIBUTION -- so the properties worth pinning are that it moves on, that it distinguishes
a completed coast from a satellite lost mid-test, and that it survives a fleet with fewer
satellites than the skip depth.

Models the selection/lifecycle logic directly; the residual arithmetic is the filter's and
is covered by test_state_filter.py.
"""
import unittest


class Rot:
    def __init__(self, hold_s=600.0, skip=4, mask_after=50):
        self.hold_s, self.skip, self.mask_after = hold_s, skip, mask_after
        self.key = None
        self.t0 = 0.0
        self.samples = []
        self.history = []

    def pick(self, state):
        cand = [(n, k) for k, n in state.items() if n >= self.mask_after]
        if not cand:
            return None
        recent = {h["key"] for h in self.history[-self.skip:]}
        fresh = [c for c in cand if c[1] not in recent]
        return max(fresh or cand)[1]

    def tick(self, t, state):
        if self.key is None:
            k = self.pick(state)
            if k is not None:
                self.key, self.t0, self.samples = k, t, []
            return
        if self.key not in state:
            self.history.append({"key": self.key, "why": "sat dropped from state",
                                 "n": len(self.samples)})
            self.key, self.samples = None, []
        elif t - self.t0 >= self.hold_s:
            self.history.append({"key": self.key, "why": "hold complete",
                                 "n": len(self.samples)})
            self.key, self.samples = None, []


class TestP2cRotate(unittest.TestCase):
    def _fleet(self, n=8):
        return {("gps", p): 100 + p for p in range(1, n + 1)}

    def test_rotates_across_satellites(self):
        """The whole point: successive coasts must land on DIFFERENT satellites."""
        r, st = Rot(), self._fleet()
        t = 0.0
        for _ in range(6):
            r.tick(t, st)
            t += 700.0
            r.tick(t, st)
        tested = [h["key"] for h in r.history]
        self.assertGreaterEqual(len(set(tested)), 5,
                                "the rotation re-tested the same satellite: %r" % (tested,))

    def test_hold_completes_and_is_labelled(self):
        r, st = Rot(hold_s=600.0), self._fleet()
        r.tick(0.0, st)
        held = r.key
        self.assertIsNotNone(held)
        r.tick(300.0, st)
        self.assertEqual(r.key, held, "released early")
        r.tick(600.0, st)
        self.assertIsNone(r.key)
        self.assertEqual(r.history[-1]["why"], "hold complete")

    def test_sat_lost_midtest_is_a_distinct_outcome(self):
        """A withheld sat is not fed, so it can age out of the state. That must be reported
        as its own outcome, not look like a completed coast with a short curve."""
        r, st = Rot(), self._fleet()
        r.tick(0.0, st)
        gone = r.key
        del st[gone]
        r.tick(120.0, st)
        self.assertIsNone(r.key)
        self.assertEqual(r.history[-1]["why"], "sat dropped from state")

    def test_small_fleet_still_rotates(self):
        """Fewer satellites than the skip depth must not deadlock the rotation."""
        r, st = Rot(skip=4), self._fleet(n=2)
        t = 0.0
        for _ in range(4):
            r.tick(t, st)
            self.assertIsNotNone(r.key, "rotation stalled on a 2-satellite fleet")
            t += 700.0
            r.tick(t, st)
        self.assertEqual(len(r.history), 4)

    def test_waits_for_established_satellites(self):
        """Masking from startup is the vacuous test -- nothing is withheld until a satellite
        has actually been established in the state."""
        r = Rot(mask_after=50)
        r.tick(0.0, {("gps", 3): 10, ("gps", 6): 20})
        self.assertIsNone(r.key, "withheld a satellite that was never established")
        r.tick(10.0, {("gps", 3): 10, ("gps", 6): 60})
        self.assertEqual(r.key, ("gps", 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
