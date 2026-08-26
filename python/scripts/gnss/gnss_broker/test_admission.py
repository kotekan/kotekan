"""#90 admission gate: the 2026-08-25 flight replayed offline.

    python3 -m gnss_broker.test_admission        (or: pytest gnss_broker/test_admission.py)

THE POINT OF THIS FILE. Every guard in `admission.py` was bought by restarting the broker
against the live sky and watching what fired -- four flights in one evening, ending in F3 and
a disarm. Each of those eight fires is a row below, and they now run in milliseconds with no
fleet, no ephemeris, and no sky. If a future change to the gate would have re-admitted PRN 34
inside the startup solve, this file says so before the yaml is touched.

⚠️ THE GATE STRIKES ON THE WALL CLOCK, NOT THE FROZEN CYCLE CLOCK -- so a transcript replay
can never reproduce a #90 fire, and broker_equiv is blind to this logic by construction. That
is precisely why these tests exist rather than a fixture.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.admission import AdmissionGate


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def gate(**kw):
    """An ARMED gate whose PRN 1 is already past rules 1-4, with a healthy population."""
    g = AdmissionGate(armed=True, **kw)
    g.was_present.add(1)
    return g


def healthy(g, t=0.0, n=8):
    """Seed the population history with a steady chain so rule 5 never trips."""
    for i in range(6):
        g.note_population(t + i, n)


# ---- FIRE POPULATION 1: THE STARTUP SOLVE (fires 1-5, flight 3) ------------------------
# At process start every satellite is absent, so the whole constellation accrued strikes.
def test_startup():
    print("startup solve (fires 1-5: rules 3 and 4)")

    # PRN 34, never present, whose SIDELOBE-STABLE tau grew +0.62 -> +0.82 through two fires:
    # a sidelobe repeats its value, so it PASSES the consistency guard. Only was-present stops
    # it. This is the case that proves consistency alone is not enough.
    g = AdmissionGate(armed=True)
    healthy(g)
    d1 = g.decide(34, +0.62, True, 1000.0, 0.0, 8, 3600.0)
    d2 = g.decide(34, +0.82, True, 1120.0, 0.0, 8, 3600.0)
    check(not d1.fire and not d2.fire, "PRN 34 (never present) cannot fire, stable tau or not")
    check(34 not in g.pending, "and it accrues no strike at all")

    # PRN 9: present at q 2.32 at t+28 s, absent by t+2 min, fired legally under was-present
    # alone. The 600 s startup hold-off is what fences it.
    g = gate()
    healthy(g)
    g.note_present(9)
    d1 = g.decide(9, +0.40, True, 100.0, 0.0, 8, 120.0)     # t+2 min: inside the hold-off
    check(not d1.fire and 9 not in g.pending, "PRN 9 at t+120 s: held off, no strike")
    d2 = g.decide(9, +0.40, True, 1000.0, 0.0, 8, 700.0)    # past it: normal strike 1
    check(not d2.fire and d2.reason == "strike1", "the same fit past 600 s takes strike 1")


# ---- FIRE POPULATION 2: THE BROWNOUT (fires 2-5, flight 3b) ----------------------------
# Four fires rode a band-wide e5b presence dip: 7 -> 3 sats over ~6 min, e5a steady.
def test_brownout():
    print("band brownout (flight 3b, rule 5)")

    g = gate()
    for i in range(6):
        g.note_population(float(i) * 60.0, 7)              # the 600 s peak is 7
    d = g.decide(1, +0.50, True, 1000.0, 400.0, 3, 3600.0)  # 3 < 0.6*7
    check(not d.fire and d.reason == "brownout", "7 -> 3 sats suppresses admission")
    check(any(k == "rs-admit-bw" for k, _, _ in d.logs), "and it says so, once per 60 s")
    check(1 not in g.pending, "a suppressed cycle accrues NO strike")

    # The same satellite in a steady chain is judged normally -- the guard is about the
    # CHAIN's population, never about this PRN.
    g = gate()
    healthy(g, n=7)
    d = g.decide(1, +0.50, True, 1000.0, 400.0, 7, 3600.0)
    check(d.reason == "strike1", "a steady chain at the same count strikes normally")

    # A small constellation must not flap the ratio: baseline < 4 disables the guard.
    g = gate()
    for i in range(3):
        g.note_population(float(i), 3)
    d = g.decide(1, +0.50, True, 1000.0, 10.0, 1, 3600.0)
    check(d.reason == "strike1", "baseline under 4 does not brown out (1 of 3 present)")


# ---- FIRE POPULATION 3: THRESHOLD FLICKER (fires 6-8, F3) ------------------------------
# The fires that TRIPPED F3: sats present 100-210 s before the fire, crossing the deep gate
# at 1.1-1.5. presence-clears-strikes plus 60 s decorrelation re-arms faster than F3's 300 s
# window, so the gate cannot tell a flicker from a latch. THIS IS THE OPEN DEFECT: the test
# documents it rather than asserting it is fixed, because v3 (min-absence + recent-lock) is
# on the shelf, not in the code.
def test_threshold_flicker_is_not_yet_fenced():
    print("threshold flicker (fires 6-8, F3 -- documents the OPEN gap)")

    g = gate()
    healthy(g)
    # ⚠️ WALL TIMES ARE REALISTIC (epoch-scale) ON PURPOSE. The post-fire cooldown compares
    # against a 0.0 default, so a test using t_wall of a few hundred seconds would read as
    # "still cooling down" from a fire that never happened -- an artifact of the harness, not
    # of the gate. Cost this file one red run.
    T = 1.787e9
    g.note_present(1)                                       # present at t=0
    d1 = g.decide(1, +0.45, True, T + 100.0, 100.0, 8, 3600.0)  # absent 100 s later: strike 1
    d2 = g.decide(1, +0.45, True, T + 170.0, 170.0, 8, 3600.0)  # 70 s on: decorrelated+consistent
    check(d1.reason == "strike1" and d2.fire,
          "a sat absent only 170 s STILL fires -- F3's gap, v3 is the fix and is not in")


# ---- THE TWO-STRIKE RULE ITSELF (flight 2) ---------------------------------------------
def test_two_strike_rule():
    print("two-strike rule (flight 2)")

    g = gate(); healthy(g)
    check(g.decide(1, +0.50, True, 1000.0, 0.0, 8, 3600.0).reason == "strike1", "first fit strikes")
    d = g.decide(1, +0.52, True, 1030.0, 0.0, 8, 3600.0)
    check(not d.fire and d.reason == "too-fresh", "consistent but 30 s on: HOLD, do not fire")
    check(g.pending[1][1] == 1000.0, "and the pending strike's clock is NOT restarted")
    check(g.decide(1, +0.52, True, 1065.0, 0.0, 8, 3600.0).fire, "65 s on: decorrelated, fires")

    # F2's actual failure: alternating signs on a swinging fit, seconds apart.
    g = gate(); healthy(g)
    g.decide(1, +0.60, True, 1000.0, 0.0, 8, 3600.0)
    d = g.decide(1, -0.55, True, 1090.0, 0.0, 8, 3600.0)
    check(not d.fire and d.reason == "strike1", "a sign flip cannot fire; it REPLACES the strike")
    check(g.pending[1] == (-0.55, 1090.0), "the replacement strike is the new fit")

    # Strike memory must survive non-qualifying cycles -- the flight-2 harness bug cleared it
    # on every ratio dip, which is why the rule looked like it never fired.
    g = gate(); healthy(g)
    g.decide(1, +0.50, True, 1000.0, 0.0, 8, 3600.0)
    g.decide(2, +0.10, True, 1010.0, 0.0, 8, 3600.0)        # a different PRN's cycles
    check(g.pending[1] == (+0.50, 1000.0), "another PRN's traffic does not clear the strike")

    # Expiry: a strike older than max_gap_s is stale, and replaces rather than fires.
    g = gate(); healthy(g)
    g.decide(1, +0.50, True, 1000.0, 0.0, 8, 3600.0)
    d = g.decide(1, +0.50, True, 1700.0, 0.0, 8, 3600.0)    # 700 s > 600 s
    check(not d.fire and g.pending[1][1] == 1700.0, "a 700 s-old strike expires, does not fire")


def test_cooldown_and_presence():
    print("cooldown and presence")

    g = gate(); healthy(g)
    g.decide(1, +0.50, True, 1000.0, 0.0, 8, 3600.0)
    check(g.decide(1, +0.50, True, 1065.0, 0.0, 8, 3600.0).fire, "fires")
    d = g.decide(1, +0.50, True, 1100.0, 0.0, 8, 3600.0)
    check(d.reason == "cooldown", "and is silent for 180 s after")
    check(1 not in g.pending, "the fire consumed its pending strike")
    d = g.decide(1, +0.50, True, 1300.0, 0.0, 8, 3600.0)
    check(d.reason == "strike1", "past the cooldown it may start again")

    # Presence is the natural clear: a present sat is not mid-latch.
    g = gate(); healthy(g)
    g.decide(1, +0.50, True, 1000.0, 0.0, 8, 3600.0)
    g.note_present(1)
    check(1 not in g.pending, "presence clears the pending strike")

    # Rules 1 and 2: disarmed, or unseeded, decides nothing at all.
    g = AdmissionGate(armed=False); healthy(g); g.was_present.add(1)
    check(not g.decide(1, +0.5, True, 1000.0, 0.0, 8, 3600.0).fire, "a DISARMED gate never fires")
    g = gate(); healthy(g)
    check(not g.decide(1, +0.5, False, 1000.0, 0.0, 8, 3600.0).fire, "an unseeded PRN never fires")


def test_population_window():
    print("population history")

    g = gate()
    g.note_population(0.0, 9)
    g.note_population(0.0, 3)                               # same cycle stamp
    check(len(g.population) == 1, "one entry per cycle, not per PRN")
    g.note_population(700.0, 4)                             # 700 s later
    check(len(g.population) == 1 and g.population[0][0] == 700.0,
          "entries older than the 600 s window are dropped")


if __name__ == "__main__":
    print("#90 admission gate -- the 2026-08-25 flight, offline\n")
    for fn in (test_startup, test_brownout, test_threshold_flicker_is_not_yet_fenced,
               test_two_strike_rule, test_cooldown_and_presence, test_population_window):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
