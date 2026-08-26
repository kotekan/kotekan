"""D0: the survivor-bias case, replayed.

    python3 -m gnss_broker.test_detectors

The test that matters is `test_the_e4_case`: it reconstructs the shape that produced a wrong
verdict on 2026-08-25 and asserts the series reports it honestly where the DLL line did not.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.detectors import QSeries, PRESENT, ABSENT, DROPPED


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def fleet(present_q):
    """A fleet aggregate from {prn: q}; a PRN mapped to None is seeded but not present."""
    return {p: ({"present": True, "q": q} if q is not None else {"present": False})
            for p, q in present_q.items()}


# ---- THE CASE THAT COST A VERDICT ---------------------------------------------------------
def test_the_e4_case():
    print("the E4 case (2026-08-25): a satellite sick enough to leave the sample")

    # Two satellites over a 100-cycle window. PRN 4 is healthy for the first 15 cycles, then
    # goes absent for the remaining 85 -- its sick interval. PRN 9 is healthy throughout.
    # The DLL line would show PRN 4 only while it was healthy, so any q statistic over that
    # line describes a satellite that was fine, for as long as it was fine.
    q = QSeries(window_s=10000.0)
    seeded = {4, 9}
    for i in range(15):
        q.note_cycle(i * 10.0, seeded, fleet({4: 3.0, 9: 3.0}))
    for i in range(15, 100):
        q.note_cycle(i * 10.0, seeded, fleet({4: None, 9: 3.0}))

    n, npres, mean, sd, frac = q.summary(4)
    check(n == 100, "the window keeps ALL 100 cycles for PRN 4, not just the 15 it reported")
    check(npres == 15, "only 15 of them are present samples")
    check(abs(frac - 0.15) < 1e-9, "present_frac is 15%% -- the honest headline")
    check(abs(mean - 3.0) < 1e-9 and sd < 1e-9,
          "and its q over those samples looks PERFECT (3.00 +- 0.00) -- which is the trap: "
          "the survivor statistic is not wrong, it is answering a different question")

    # The comparison that was actually made: PRN 4 against a healthy neighbour, on q alone.
    n9, npres9, mean9, sd9, frac9 = q.summary(9)
    check(abs(sd - sd9) < 1e-9,
          "on q-SD alone the sick satellite is INDISTINGUISHABLE from the healthy one")
    check(frac < frac9,
          "presence fraction is what separates them -- 15%% against 100%%")

    eps = q.episodes(4, min_absence_s=300.0)
    check(len(eps) == 1, "the absence is reported as ONE episode")
    check(abs(eps[0][0] - 150.0) < 1e-9 and abs(eps[0][1] - 990.0) < 1e-9,
          "spanning exactly the sick interval (t=150 s to t=990 s)")
    check(not q.episodes(9, min_absence_s=300.0), "the healthy satellite has no episode")


def test_states():
    print("state labelling")

    q = QSeries()
    q.note_cycle(0.0, {1, 2}, fleet({1: 2.5, 2: None}))
    check(q.hist[1][-1][2] == PRESENT, "in the aggregate and passing the gate: present")
    check(q.hist[2][-1][2] == ABSENT, "seeded but not passing: absent, WITH a sample recorded")
    check(q.hist[2][-1][1] is None, "an absent sample carries no q rather than a fabricated one")

    q.note_cycle(1.0, {1}, fleet({1: 2.5}))
    check(q.hist[2][-1][2] == DROPPED, "no longer seeded: dropped, still in the series")

    # A cycle where the DLL did not run at all must not be read as everyone being healthy.
    q.note_cycle(2.0, {1}, None)
    check(q.hist[1][-1][2] == ABSENT, "no fleet aggregate this cycle reads as ABSENT, not present")


def test_window_and_line():
    print("window and reporting")

    q = QSeries(window_s=50.0)
    for i in range(100):
        q.note_cycle(i * 10.0, {1}, fleet({1: 2.0}))
    n = q.summary(1)[0]
    check(n <= 6, "history older than the window is dropped (%d samples kept)" % n)

    q = QSeries(window_s=1000.0)
    for i in range(10):
        q.note_cycle(i * 10.0, {1, 2}, fleet({1: 2.0, 2: None}))
    line = q.line("gal_e5b")
    check(line is not None and line.startswith("QPOP gal_e5b"), "the line is emitted")
    check(line.index("2:") < line.index("1:"),
          "WORST FIRST: the absent satellite leads, because a PRN-ordered line buries it")

    check(QSeries().line("x") is None, "an empty series emits nothing rather than a bare header")


def test_no_side_effects():
    print("read-only")

    q = QSeries()
    seeded = {1}
    f = fleet({1: 2.0})
    before = (set(seeded), dict(f[1]))
    q.note_cycle(0.0, seeded, f)
    check(set(seeded) == before[0] and dict(f[1]) == before[1],
          "the detector mutates neither the seed set nor the fleet rows it observes")


if __name__ == "__main__":
    print("D0 -- the population-honest q series\n")
    for fn in (test_the_e4_case, test_states, test_window_and_line, test_no_side_effects):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
