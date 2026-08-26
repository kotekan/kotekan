"""D0: the survivor-bias case, replayed.

    python3 -m gnss_broker.test_detectors

The test that matters is `test_the_e4_case`: it reconstructs the shape that produced a wrong
verdict on 2026-08-25 and asserts the series reports it honestly where the DLL line did not.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.detectors import (
    QSeries, BrownoutDetector, LatchDetector, SawtoothDetector,
    PRESENT, ABSENT, DROPPED,
)


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


# ---- D1: THE BROWNOUT ---------------------------------------------------------------------
def test_brownout():
    print("D1: the brownout episode (the 2026-08-25 23:00 shape)")

    d = BrownoutDetector(window_s=600.0, frac=0.6, min_base=4, min_len_s=60.0)
    msgs = []
    for i in range(20):                       # a steady chain at 7 present
        msgs.append(d.note_cycle(i * 10.0, 7))
    check(not any(msgs), "a steady chain says nothing at all")
    check(not d.active(), "and is not browned out")

    # e5b's collapse: 7 -> 3 satellites, held for six minutes.
    opened = [d.note_cycle(200.0 + i * 10.0, 3) for i in range(36)]
    fired = [m for m in opened if m]
    check(d.active(), "the collapse opens an episode")
    check(len(fired) == 1 and "BROWNOUT open" in fired[0],
          "announced ONCE, not once per cycle")
    check("3 present vs 7 baseline" in fired[0], "carrying depth against its own peak")

    closed = d.note_cycle(600.0, 7)
    check(closed and "BROWNOUT closed" in closed, "recovery closes it")
    check(len(d.episodes) == 1, "and it is retained as one labelled episode")
    t0, t1, base, deep = d.episodes[0]
    check(base == 7 and deep == 3, "with the baseline and the worst count")

    # A brief dip is not an episode -- a chain that flickers must not fill the log.
    d2 = BrownoutDetector(min_len_s=60.0)
    for i in range(10):
        d2.note_cycle(i * 10.0, 8)
    check(d2.note_cycle(100.0, 2) is None, "a one-cycle dip announces nothing")
    check(d2.note_cycle(110.0, 8) is None, "and closes silently")
    check(not d2.episodes, "leaving no episode behind")

    # The threshold is relative: a small constellation must not flap it.
    d3 = BrownoutDetector(min_base=4)
    for i in range(6):
        d3.note_cycle(i * 10.0, 3)
    check(d3.note_cycle(70.0, 1) is None and not d3.active(),
          "a baseline under min_base cannot brown out (1 of 3 is not an episode)")


# ---- D2: THE LATCH, MEASURED RATHER THAN ACTED ON -----------------------------------------
def test_latch():
    print("D2: the deep latch, unarmed")

    def build(healthy_cycles, absent_cycles, q=3.0):
        s = QSeries(window_s=100000.0)
        for i in range(healthy_cycles):
            s.note_cycle(i * 10.0, {5}, fleet({5: q}))
        for i in range(healthy_cycles, healthy_cycles + absent_cycles):
            s.note_cycle(i * 10.0, {5}, fleet({5: None}))
        return s, (healthy_cycles + absent_cycles - 1) * 10.0

    s, t = build(30, 60)                      # healthy 300 s, then absent 600 s
    d = LatchDetector()
    hits = d.scan(t, s, browned_out=False)
    check(len(hits) == 1 and hits[0][0] == 5, "a healthy-then-absent satellite is reported")
    check(hits[0][1] >= 300.0, "with how long it has been gone")
    check(abs(hits[0][2] - 3.0) < 1e-9, "and the q it had BEFORE it went")

    check(not d.scan(t + 10.0, s, browned_out=False),
          "reported ONCE per episode, not once per cycle (the cooldown)")

    # The three populations #90's flights actually hit, none of which is a latch:
    s2, t2 = build(30, 10)
    check(not LatchDetector().scan(t2, s2, browned_out=False),
          "absent only 100 s: too short, this is flicker")

    s3, t3 = build(30, 60, q=1.2)
    check(not LatchDetector().scan(t3, s3, browned_out=False),
          "never healthy before it went: a set, not a latch")

    s4, t4 = build(30, 60)
    check(not LatchDetector().scan(t4, s4, browned_out=True),
          "during a chain-wide BROWNOUT nothing is reported -- the chain is the patient")

    # THE STARTUP SOLVE IS NOT A LATCH (flight 3a). Presence flaps while the clock converges.
    s6, t6 = build(30, 60)
    d6 = LatchDetector(startup_hold_s=900.0)
    check(not d6.scan(t6, s6, browned_out=False, uptime_s=200.0),
          "inside the startup hold-off nothing is reported")
    check(d6.suppressed_startup == 1, "and the suppression is COUNTED, not silent")
    check(len(d6.scan(t6, s6, browned_out=False, uptime_s=1800.0)) == 1,
          "the same satellite IS reported once the process is old enough -- suppressing a "
          "startup report must not swallow the later real one")

    # Startup: absent from the first cycle, never seen healthy.
    s5 = QSeries(window_s=100000.0)
    for i in range(60):
        s5.note_cycle(i * 10.0, {5}, fleet({5: None}))
    check(not LatchDetector().scan(590.0, s5, browned_out=False),
          "a satellite that was never present cannot latch (the PRN 34 case)")


# ---- D3: THE HANDOVER SAWTOOTH ------------------------------------------------------------
def test_sawtooth():
    print("D3: the handover sawtooth (#92's P2 population)")

    d = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5)
    msgs = []
    for i in range(30):                        # a trim ramping 0 -> 1.45 chips
        msgs.append(d.note(i * 10.0, 7, i * 0.05))
    check(not any(msgs), "a RAMP alone says nothing -- that is the loop tracking real drift")

    hit = d.note(300.0, 7, 0.02)               # ... then discarded in one cycle
    check(hit and "SAWTOOTH PRN 7" in hit, "the WIPE is what fires")
    check("ramp discarded" in hit, "and it names the mechanism, not just the numbers")
    check(len(d.episodes) == 1, "retained as an episode for #92's P2 population")

    check(d.note(310.0, 7, 0.01) is None, "reported once, not once per cycle")

    # A small trim that wobbles is not a sawtooth: the ramp threshold is what excludes it.
    d2 = SawtoothDetector(ramp_chips=0.5)
    for i in range(20):
        d2.note(i * 10.0, 3, 0.05)
    check(d2.note(200.0, 3, 0.0) is None, "a trim that never ramped past the bar cannot wipe")

    # A brownout takes trims away chain-wide; that is D1's event, not this one.
    d3 = SawtoothDetector(ramp_chips=0.5)
    for i in range(30):
        d3.note(i * 10.0, 9, i * 0.05)
    check(d3.note(300.0, 9, 0.0, browned_out=True) is None,
          "during a brownout nothing is reported -- superposing the two is what made E3 "
          "look heterogeneous")

    # A gradual decay is the LEAK, not a discontinuity.
    d4 = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5)
    for i in range(30):
        d4.note(i * 10.0, 4, i * 0.05)
    slow = [d4.note(300.0 + i * 10.0, 4, 1.45 * (0.9 ** (i + 1))) for i in range(10)]
    check(not any(slow), "a slow decay is the leak, and does not fire")


if __name__ == "__main__":
    print("D0-D3 -- population, brownout, latch, sawtooth\n")
    for fn in (test_the_e4_case, test_states, test_window_and_line, test_no_side_effects,
               test_brownout, test_latch, test_sawtooth):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
