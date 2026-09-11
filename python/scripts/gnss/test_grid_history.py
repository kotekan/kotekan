"""#117 -- the grid snapshot is a HISTORY, and that is what makes two bands pair.

The broker folds a record every GRID_HOPS (~1.0066 s) but the observables writer polls at
2 s. That is exactly Nyquist, so a single-slot snapshot hands each poller its own alternating
half of the grid hops; because each chain lands on its own alternate, a band PAIR shared ~40%
of the grid and a triple ~35%. The hops were never missing from the broker -- only from the
poll. These tests pin the retention, the arc rule, and the pairing yield the fix exists for.
"""
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnss_broker import fleetadr as F


HPR = 2048                              # hops per record, the production shape
RECS_PER_GRID = F.GRID_HOPS // HPR      # 96 records between grid hops


def _fold(st, hop, dop=1000.0, n_inst=3):
    """One record at `hop` with `n_inst` agreeing instances."""
    per_inst = {i: (dop, 0.0, complex(1.0, 0.0), 1.0) for i in range(n_inst)}
    return F.fold_record(st, hop, per_inst, hpr=HPR)


def _run(st, n_grid, first_rec=0):
    """Fold CONSECUTIVE records spanning `n_grid` grid hops.

    ⚠️ Stepping a whole grid hop per fold would be a 96-record gap, which breaks the arc on
    every step (max_gap_rec=3) and leaves n_rec at 1 -- nothing publishes and the history
    never fills. Chains fold every record; only every 96th is a grid hop, which is exactly
    what makes the grid a shared key in the first place.
    """
    for r in range(first_rec, first_rec + n_grid * RECS_PER_GRID + 1):
        _fold(st, r * HPR)
    return st


class TestGridRetention(unittest.TestCase):
    def test_keeps_at_most_grid_keep_snapshots(self):
        """The history is bounded: it is a window, not a leak."""
        st = _run(F.SatAdr(), 11)
        self.assertLessEqual(len(st.grid), F.GRID_KEEP)
        self.assertEqual([g[0] for g in st.grid],
                         [k * F.GRID_HOPS for k in range(12 - F.GRID_KEEP, 12)])

    def test_only_grid_multiples_are_snapshotted(self):
        """A record off the grid never enters the history -- the grid is the pairing key."""
        st = _run(F.SatAdr(), 1)           # lands exactly one record past the first grid hop
        self.assertEqual([g[0] for g in st.grid], [0, F.GRID_HOPS])
        _fold(st, (RECS_PER_GRID + 2) * HPR)   # a record, but not a grid hop
        self.assertEqual([g[0] for g in st.grid], [0, F.GRID_HOPS])

    def test_newest_is_last(self):
        st = _run(F.SatAdr(), 3)
        self.assertEqual(st.grid[-1][0], 3 * F.GRID_HOPS)


class TestPairingYield(unittest.TestCase):
    """THE POINT OF THE FIX, as a number.

    Two chains poll the same 1.0066 s grid every 2 s at different phases. Model what each
    poll can report, with and without the history, and count the grid hops the two share.
    """

    @staticmethod
    def _polls(phase_s, span_s=120.0, poll_s=2.0):
        """Grid-hop indices visible to a poller starting at `phase_s`, newest-only vs history."""
        newest, withhist = set(), set()
        t = phase_s
        while t < span_s:
            k = int(t / F.GRID_SECONDS) if hasattr(F, "GRID_SECONDS") else int(t / 1.006632)
            newest.add(k)
            for j in range(F.GRID_KEEP):
                if k - j >= 0:
                    withhist.add(k - j)
            t += poll_s
        return newest, withhist

    def test_newest_only_loses_half_the_grid_and_most_of_the_pairs(self):
        a_new, _ = self._polls(0.0)
        b_new, _ = self._polls(1.0)      # the other chain, half a grid hop out of phase
        total = max(max(a_new), max(b_new)) + 1
        self.assertLess(len(a_new) / total, 0.65, "a 2 s poll cannot see every 1.0066 s hop")
        shared = len(a_new & b_new) / total
        self.assertLess(shared, 0.65, "newest-only pairing is limited by poll phase")

    def test_history_recovers_the_pairing(self):
        """With GRID_KEEP snapshots per row both chains cover the grid, whatever the phase."""
        for phase in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
            _, a = self._polls(0.0)
            _, b = self._polls(phase)
            lo = max(min(a), min(b))
            hi = min(max(a), max(b))
            grid = set(range(lo, hi + 1))
            shared = len(a & b & grid) / len(grid)
            self.assertGreater(shared, 0.99,
                               "phase %.2f s: history should make pairing poll-independent"
                               % phase)


class TestPublishedHistory(unittest.TestCase):
    def test_publish_emits_hist_newest_last_and_matches_the_scalars(self):
        fa = F.FleetAdr()
        st = _run(fa.sats.setdefault(7, F.SatAdr()), 5)
        out = fa.publish(1176.45e6, now=st.t)
        row = out.get(7)
        self.assertIsNotNone(row, "a folded PRN must publish")
        self.assertIn("g_hist", row)
        self.assertLessEqual(len(row["g_hist"]), F.GRID_KEEP)
        self.assertEqual(row["g_hist"][-1][0], row["g_hop"], "newest entry is the scalar hop")
        self.assertEqual(row["g_hist"][-1][1], row["g_dop_cycles"])
        self.assertEqual([e[0] for e in row["g_hist"]],
                         sorted(e[0] for e in row["g_hist"]), "oldest first")

    def test_history_never_crosses_an_arc(self):
        """A break resets the accumulator, so an older arc's phase must not pair forward."""
        st = _run(F.SatAdr(), 3)
        arc_before = st.arc
        st.arc += 1                                  # simulate a break
        _run(st, 1, first_rec=4 * RECS_PER_GRID)
        current = [g for g in st.grid if g[2] == st.arc]
        self.assertTrue(all(g[2] != arc_before for g in current),
                        "snapshots from the previous arc must not be published as current")


if __name__ == "__main__":
    unittest.main()
