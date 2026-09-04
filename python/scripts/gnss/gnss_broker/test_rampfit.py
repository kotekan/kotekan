"""Ramp estimation: the traps that would produce a wrong verdict rather than a crash.

    python3 -m gnss_broker.test_rampfit

@author Keith Vanderlinde
"""

import sys

from gnss_broker.rampfit import RampTracker


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def feed(r, key, t0, n, dt, y0, slope, step_at=None, step=0.0):
    for i in range(n):
        y = y0 + slope * (i * dt)
        if step_at is not None and i >= step_at:
            y += step
        r.update(key, t0 + i * dt, y)


def test_recovers_a_known_slope():
    print("the fit itself")

    r = RampTracker()
    feed(r, 7, 1000.0, 40, 10.0, -1.20, +0.0009)     # 0.9 mchips/s over 390 s
    f = r.fit(7)
    check(f is not None, "a 390 s window of 40 points qualifies")
    slope, mean, span, n = f
    check(abs(slope - 0.0009) < 1e-9, "the slope is recovered exactly on clean data")
    check(abs(mean - (-1.20 + 0.0009 * 195.0)) < 1e-9, "the mean value comes back with it")
    check(abs(span - 390.0) < 1e-9 and n == 40, "span and count describe what was fitted")

    # A flat series must read as flat, not as a small spurious trend.
    r = RampTracker()
    feed(r, 7, 1000.0, 40, 10.0, 0.5, 0.0)
    check(abs(r.fit(7)[0]) < 1e-12, "a flat trim has ZERO slope")


def test_a_discontinuity_is_not_a_rate():
    print("discontinuity (the wrong-verdict trap)")

    # THE TRAP: a 1.0-chip re-anchor inside a 400 s window is 2.5e-3 chips/s -- about three
    # times a real drift, with the jump's sign. Fitting across it is the failure this guard
    # exists to prevent.
    r = RampTracker()
    feed(r, 7, 1000.0, 40, 10.0, 0.0, 0.0, step_at=20, step=1.0)
    check(r.resets == 1, "the step restarted the window")
    f = r.fit(7)
    check(f is not None, "and the post-step samples still form a fit")
    check(abs(f[0]) < 1e-12, "which reads FLAT -- the jump is not a rate")
    check(f[3] == 20, "the fit uses only the post-step samples")

    # A step at the tolerance boundary must not reset: real drift crossing 0.3 chips
    # gradually would otherwise reset forever and never produce a fit.
    r = RampTracker(step_reset_chips=0.3)
    r.update(7, 0.0, 0.0)
    r.update(7, 10.0, 0.30)
    check(r.resets == 0, "a step exactly at the tolerance does not reset")
    r.update(7, 20.0, 0.61)
    check(r.resets == 1, "one past it does")


def test_a_short_window_is_not_a_measurement():
    print("qualification")

    r = RampTracker(min_points=4, min_span_s=120.0)
    # Four samples over four seconds: enough points, nowhere near enough span. Unguarded this
    # prints a confident slope with essentially infinite variance.
    feed(r, 7, 1000.0, 4, 1.0, 0.0, 0.5)
    check(r.fit(7) is None, "4 points over 4 s does NOT qualify (span guard binds)")

    r = RampTracker()
    feed(r, 7, 1000.0, 3, 100.0, 0.0, 1e-4)
    check(r.fit(7) is None, "3 points over 200 s does not qualify either (count guard)")

    r = RampTracker()
    for _ in range(10):
        r.update(7, 1000.0, 0.5)                     # all at the SAME instant
    check(r.fit(7) is None, "a zero-variance time axis returns None, never divides by zero")

    check(r.fit("never-seen") is None, "an unknown key is None, not an exception")


def test_window_and_lifecycle():
    print("window and lifecycle")

    r = RampTracker(window_s=600.0)
    feed(r, 7, 1000.0, 200, 10.0, 0.0, 0.0)          # 2000 s of history, 600 s window
    check(r.fit(7)[2] <= 600.0, "history older than the window is dropped")

    r = RampTracker()
    feed(r, 7, 1000.0, 40, 10.0, 0.0, 1e-4)
    r.drop(7)
    check(r.fit(7) is None, "drop() forgets a released PRN entirely")

    r = RampTracker()
    feed(r, 7, 1000.0, 40, 10.0, 0.0, 1e-4)
    feed(r, 9, 1000.0, 40, 10.0, 0.0, 1e-4)
    r.retain({7})
    check(r.fit(7) is not None and r.fit(9) is None,
          "retain() keeps the live set and forgets a vanished satellite")


if __name__ == "__main__":
    print("#93 shadow ramp estimation\n")
    for fn in (test_recovers_a_known_slope, test_a_discontinuity_is_not_a_rate,
               test_a_short_window_is_not_a_measurement, test_window_and_lifecycle):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
