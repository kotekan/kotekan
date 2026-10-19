"""The population detectors: QSeries, BrownoutDetector, SawtoothDetector.

    python3 -m gnss_broker.test_detectors

The central case is `test_the_e4_case`, a satellite sick enough to leave the sample: a q
statistic over its present samples alone describes a satellite that was fine, for as long as it
was fine, so the series must carry the presence fraction beside it.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.detectors import (
    QSeries,
    BrownoutDetector,
    SawtoothDetector,
    PRESENT,
    ABSENT,
    DROPPED,
)


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def fleet(present_q):
    """A fleet aggregate from {prn: q}; a PRN mapped to None is seeded but not present."""
    return {
        p: ({"present": True, "q": q} if q is not None else {"present": False})
        for p, q in present_q.items()
    }


# ---- QSeries: a satellite sick enough to leave the sample ---------------------------------
def test_the_e4_case():
    print("QSeries: a satellite sick enough to leave the sample")

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
    check(
        n == 100,
        "the window keeps all 100 cycles for PRN 4, not just the 15 it reported",
    )
    check(npres == 15, "only 15 of them are present samples")
    check(abs(frac - 0.15) < 1e-9, "present_frac is 15% -- the honest headline")
    check(
        abs(mean - 3.0) < 1e-9 and sd < 1e-9,
        "and its q over those samples looks PERFECT (3.00 +- 0.00) -- which is the trap: "
        "the survivor statistic is not wrong, it is answering a different question",
    )

    # PRN 4 against a healthy neighbour, on q alone.
    n9, npres9, mean9, sd9, frac9 = q.summary(9)
    check(
        abs(sd - sd9) < 1e-9,
        "on q-SD alone the sick satellite is INDISTINGUISHABLE from the healthy one",
    )
    check(frac < frac9, "presence fraction is what separates them -- 15% against 100%")


def test_states():
    print("state labelling")

    q = QSeries()
    q.note_cycle(0.0, {1, 2}, fleet({1: 2.5, 2: None}))
    check(q.hist[1][-1][2] == PRESENT, "in the aggregate and passing the gate: present")
    check(
        q.hist[2][-1][2] == ABSENT,
        "seeded but not passing: absent, WITH a sample recorded",
    )
    check(
        q.hist[2][-1][1] is None,
        "an absent sample carries no q rather than a fabricated one",
    )

    q.note_cycle(1.0, {1}, fleet({1: 2.5}))
    check(q.hist[2][-1][2] == DROPPED, "no longer seeded: dropped, still in the series")

    # A cycle where the DLL did not run at all must not be read as everyone being healthy.
    q.note_cycle(2.0, {1}, None)
    check(
        q.hist[1][-1][2] == ABSENT,
        "no fleet aggregate this cycle reads as ABSENT, not present",
    )


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
    check(
        line.index("2:") < line.index("1:"),
        "worst first: the absent satellite leads, because a PRN-ordered line buries it",
    )

    check(
        QSeries().line("x") is None,
        "an empty series emits nothing rather than a bare header",
    )

    # A satellite that set: kept at 0% while its seeded samples are in the window, then gone,
    # so long-set satellites cannot push live absent ones off the capped line.
    q = QSeries(window_s=50.0)
    for i in range(10):
        q.note_cycle(i * 10.0, {1, 9}, fleet({1: 2.0, 9: 2.0}))
    for i in range(10, 13):
        q.note_cycle(i * 10.0, {1}, fleet({1: 2.0}))
    check(9 in q.hist, "a PRN dropped inside the window is still in the series")
    for i in range(13, 30):
        q.note_cycle(i * 10.0, {1}, fleet({1: 2.0}))
    line = q.line("gal_e5b")
    shown = [p.split(":")[0] for p in line.split(": ", 1)[1].split() if "%" in p]
    check(
        9 not in q.hist and "9" not in shown,
        "a PRN dropped for longer than the window leaves the series and the line: %s"
        % shown,
    )


def test_no_side_effects():
    print("read-only")

    q = QSeries()
    seeded = {1}
    f = fleet({1: 2.0})
    before = (set(seeded), dict(f[1]))
    q.note_cycle(0.0, seeded, f)
    check(
        set(seeded) == before[0] and dict(f[1]) == before[1],
        "the detector mutates neither the seed set nor the fleet rows it observes",
    )


# ---- BrownoutDetector ---------------------------------------------------------------------
def test_brownout():
    print("D1: the brownout episode")

    d = BrownoutDetector(window_s=600.0, frac=0.6, min_base=4, min_len_s=60.0)
    msgs = []
    for i in range(20):  # a steady chain at 7 present
        msgs.append(d.note_cycle(i * 10.0, 7))
    check(not any(msgs), "a steady chain says nothing at all")
    check(not d.active(), "and is not browned out")

    # A collapse from 7 to 3 satellites, held for six minutes.
    opened = [d.note_cycle(200.0 + i * 10.0, 3) for i in range(36)]
    fired = [m for m in opened if m]
    check(d.active(), "the collapse opens an episode")
    check(
        len(fired) == 1 and "BROWNOUT open" in fired[0],
        "announced ONCE, not once per cycle",
    )
    check("3 present vs 7 baseline" in fired[0], "carrying depth against its own peak")

    closed = d.note_cycle(600.0, 7)
    check(closed and "BROWNOUT closed" in closed, "recovery closes it")
    check(
        closed and "3 present at worst vs 7 baseline" in closed,
        "with the baseline and the worst count",
    )

    # A brief dip is not an episode: a chain that flickers must not fill the log.
    d2 = BrownoutDetector(min_len_s=60.0)
    for i in range(10):
        d2.note_cycle(i * 10.0, 8)
    check(d2.note_cycle(100.0, 2) is None, "a one-cycle dip announces nothing")
    check(d2.note_cycle(110.0, 8) is None, "and closes silently")

    # The threshold is relative: a small constellation must not flap it.
    d3 = BrownoutDetector(min_base=4)
    for i in range(6):
        d3.note_cycle(i * 10.0, 3)
    check(
        d3.note_cycle(70.0, 1) is None and not d3.active(),
        "a baseline under min_base cannot brown out (1 of 3 is not an episode)",
    )


# ---- SawtoothDetector: the handover sawtooth ----------------------------------------------
def test_sawtooth():
    print("D3: the handover sawtooth")

    d = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5)
    msgs = []
    for i in range(30):  # a trim ramping 0 -> 1.45 chips
        msgs.append(d.note(i * 10.0, 7, i * 0.05))
    check(
        not any(msgs),
        "a RAMP alone says nothing -- that is the loop tracking real drift",
    )

    hit = d.note(300.0, 7, 0.02)  # ... then discarded in one cycle
    check(hit and "SAWTOOTH PRN 7" in hit, "the WIPE is what fires")
    check("ramp discarded" in hit, "and it names the mechanism, not just the numbers")

    check(d.note(310.0, 7, 0.01) is None, "reported once, not once per cycle")

    # A small trim that wobbles is not a sawtooth: the ramp threshold is what excludes it.
    d2 = SawtoothDetector(ramp_chips=0.5)
    for i in range(20):
        d2.note(i * 10.0, 3, 0.05)
    check(
        d2.note(200.0, 3, 0.0) is None,
        "a trim that never ramped past the bar cannot wipe",
    )

    # A brownout takes trims away chain-wide; that is BrownoutDetector's event, not this one.
    d3 = SawtoothDetector(ramp_chips=0.5)
    for i in range(30):
        d3.note(i * 10.0, 9, i * 0.05)
    check(
        d3.note(300.0, 9, 0.0, browned_out=True) is None,
        "during a brownout nothing is reported -- that wipe is D1's event, not a "
        "handover",
    )

    # A wipe during a railing transit belongs to the transit; counting it would mix geometry
    # into the handover population.
    d5 = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5)
    for i in range(30):
        d5.note(i * 10.0, 5, i * 0.05)
    check(
        d5.note(300.0, 5, 0.02, in_transit=True) is None,
        "a wipe inside a transit is silent",
    )
    # Not "the same wipe fires later": once suppressed, the trim is already gone and there is
    # nothing left to report, which is what makes it the transit's event. What must survive
    # is the detector: a new ramp that wipes on a clear array is still caught.
    for i in range(31, 61):
        d5.note(i * 10.0, 5, (i - 31) * 0.05)
    check(
        d5.note(610.0, 5, 0.02, in_transit=False) is not None,
        "a new ramp wiping after the transit clears is reported -- suppression must not "
        "leave the detector deaf",
    )

    # A gradual decay is the leak, not a discontinuity.
    d4 = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5)
    for i in range(30):
        d4.note(i * 10.0, 4, i * 0.05)
    slow = [d4.note(300.0 + i * 10.0, 4, 1.45 * (0.9 ** (i + 1))) for i in range(10)]
    check(not any(slow), "a slow decay is the leak, and does not fire")


def test_sawtooth_v2():
    print("D3: startup hold-off, the weak-sat qualifier, and the two wipe classes")

    def ramped(**kw):
        d = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5, **kw)
        for i in range(30):
            d.note(i * 10.0, 7, i * 0.05)
        return d

    # Startup hold-off: seeds churn while the clock converges. Inside the hold-off nothing is
    # reported, and the cooldown is not stamped, so the same satellite still reports once the
    # process is old enough.
    d = ramped()
    check(
        d.note(300.0, 7, 0.02, uptime_s=120.0) is None,
        "inside the startup hold-off: silent",
    )
    for i in range(30):
        d.note(400.0 + i * 10.0, 7, i * 0.05)
    check(
        d.note(700.0, 7, 0.02, uptime_s=1800.0) is not None,
        "the same satellite fires later -- suppression must not stamp the cooldown",
    )

    # A trim ramping on a satellite that was absent, or present at q ~1.3, the whole time is
    # chasing churn, not doing sky work.
    d2 = ramped()
    check(
        d2.note(300.0, 7, 0.02, present_frac=0.4) is None,
        "mostly-absent sat: suppressed",
    )
    d3 = ramped()
    check(
        d3.note(300.0, 7, 0.02, present_frac=0.9, q_mean=1.3) is None,
        "present but at the q floor: suppressed",
    )
    d4 = ramped()
    check(
        d4.note(300.0, 7, 0.02, present_frac=0.9, q_mean=3.0) is not None,
        "a healthy locked satellite's wipe is reported",
    )

    # Two classes: a wipe within rebase_window_s of a birth step is a REBASE-WIPE (the seed
    # re-based under the trim); one with no birth step in the window is a BARE-WIPE, typically
    # a slew transfer, a different mechanism.
    d5 = ramped()
    m = d5.note(300.0, 7, 0.02, rebase_age_s=5.0)
    check(m is not None and "REBASE-WIPE" in m, "rebase 5 s before: REBASE-WIPE")
    d6 = ramped()
    m = d6.note(300.0, 7, 0.02)
    check(m is not None and "BARE-WIPE" in m, "no rebase stamp: BARE-WIPE")
    d7 = ramped()
    m = d7.note(300.0, 7, 0.02, rebase_age_s=400.0)
    check(
        m is not None and "BARE-WIPE" in m,
        "a stale birth-step (400 s ago) does not qualify",
    )


def test_sawtooth_cycle_clock():
    print("D3: the startup hold-off counts on the cycle clock")

    # A replay: the cycle clock is the recorded time, far from the replay's own wall clock.
    # uptime() counts from the first cycle it is given, armed trim or not.
    t0 = 1.7e9
    d = SawtoothDetector(ramp_chips=0.5, wipe_frac=0.5, startup_hold_s=900.0)
    msgs = []
    for i in range(31):  # a ramp, then a wipe 300 s after the first cycle
        t = t0 + i * 10.0
        trim = i * 0.05 if i < 30 else 0.02
        msgs.append(d.note(t, 7, trim, uptime_s=d.uptime(t)))
    check(not any(msgs), "a wipe 300 s into the run is inside the hold-off: silent")
    for i in range(31, 70):  # cycles with no armed trim still count
        d.uptime(t0 + i * 10.0)
    for i in range(70, 101):  # a new ramp, wiped 1000 s into the run
        t = t0 + i * 10.0
        trim = (i - 70) * 0.05 if i < 100 else 0.02
        msgs.append(d.note(t, 7, trim, uptime_s=d.uptime(t)))
    check(
        msgs[-1] is not None and "SAWTOOTH PRN 7" in msgs[-1],
        "a wipe 1000 s into the recorded run is reported",
    )
    check(d.uptime(t0 + 1000.0) == 1000.0, "uptime counts from the first cycle")


if __name__ == "__main__":
    print("D0, D1, D3 -- population, brownout, sawtooth\n")
    for fn in (
        test_the_e4_case,
        test_states,
        test_window_and_line,
        test_no_side_effects,
        test_brownout,
        test_sawtooth,
        test_sawtooth_v2,
        test_sawtooth_cycle_clock,
    ):
        fn()
    # ---- the span is reported as censored when it fills the window -------------------
    # A ramp older than window_s reports a span pinned just under it, so the reports would
    # cluster at ~window_s and read as a period in the plant. A censored value is a lower
    # bound and must print as one.
    d = SawtoothDetector(window_s=1800.0, cooldown_s=0.0, startup_hold_s=0.0)
    t0 = 10000.0
    msg = None
    for k in range(0, 200):  # 200 x 20 s = 4000 s, well past the window
        t = t0 + k * 20.0
        trim = 0.02 * k if k < 190 else 0.05  # long ramp, then a wipe
        msg = d.note(t, 1, trim, uptime_s=1e6, present_frac=1.0, q_mean=3.0) or msg
    check(msg is not None, "a ramp longer than the window still reports a wipe")
    check(
        msg is not None and ">=" in msg and "CENSORED" in msg,
        "... and its span is marked as a lower bound, not a measured duration:\n"
        "           %s" % (msg or "")[:150],
    )

    # a short ramp inside the window reports an exact span, with no censoring marker
    d2 = SawtoothDetector(window_s=1800.0, cooldown_s=0.0, startup_hold_s=0.0)
    msg2 = None
    for k in range(0, 20):  # 20 x 20 s = 400 s, well inside
        trim = 0.06 * k if k < 18 else 0.05
        msg2 = (
            d2.note(t0 + k * 20.0, 2, trim, uptime_s=1e6, present_frac=1.0, q_mean=3.0)
            or msg2
        )
    check(
        msg2 is not None and ">=" not in msg2 and "CENSORED" not in msg2,
        "a ramp INSIDE the window reports an exact span, unmarked",
    )

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
