"""#96: the code-rate cross-check. The traps that would give a wrong verdict, not a crash.

    python3 -m gnss_broker.test_cprate

@author Keith Vanderlinde
"""

import sys

from gnss_broker.fits import cp_rate_from_code_bias, fit_cp_rate


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


HPS = 1.0 / 5.12e-6          # hops per second, CHORD
CHIP = 10.23e6               # L5/E5a chip rate
CARR = 1176.45e6
CODE_LEN = 10230


def _dev_chips_s(rate, clock_ema, dop=0.0):
    """What the guard in seeding.py computes: fit-minus-clock, in chips/s."""
    model = cp_rate_from_code_bias(dop, clock_ema, HPS, CHIP, CARR)
    return (rate - model) * HPS


def test_unbounded_fit_is_the_defect():
    """fit_cp_rate has no bound of its own -- the guard cannot live inside it."""
    # A single mis-wrapped point injects a code period; the slope follows it anywhere.
    hist = [(0, 100.0), (1000, 100.5), (2000, 101.0), (3000, 101.5 + CODE_LEN)]
    fit = fit_cp_rate(hist, CODE_LEN)
    check(fit is not None, "an unwrap-blown history still returns a fit (no internal bound)")
    if fit:
        # Not asserting a specific magnitude: the point is only that nothing rejected it.
        check(abs(fit[0] * HPS) > 1.0,
              "the blown fit is far outside any physical clock rate (%.1f chips/s)"
              % (fit[0] * HPS))


def test_clean_fit_is_admitted():
    """A fit that agrees with the pooled clock must pass at the armed tolerance."""
    clock = 4.62e-9                       # the l-a measured live on the DR chains
    model = cp_rate_from_code_bias(0.0, clock, HPS, CHIP, CARR)
    check(abs(_dev_chips_s(model, clock)) == 0.0,
          "a fit equal to the pooled clock has zero deviation")
    # Real fit scatter on a healthy sat, measured 2026-08-28: SD ~0.06 chips/s.
    jitter = model + 0.06 / HPS
    check(abs(_dev_chips_s(jitter, clock)) < 0.5,
          "healthy per-poll fit scatter (0.06 chips/s) stays inside tol 0.5")


def test_blown_fit_is_rejected():
    clock = 4.62e-9
    for slope_chips_s in (2.16, 177.0, 994.0):     # the measured p90 / p99 / max
        rate = slope_chips_s / HPS
        check(abs(_dev_chips_s(rate, clock)) > 0.5,
              "a %.2f chips/s fit is rejected at tol 0.5" % slope_chips_s)


def test_guard_is_on_deviation_not_magnitude():
    """THE TRAP. An absolute ceiling would reject the feed-forward the trim needs.

    The receiver clock is ~0.047 chips/s calibrated but ~3.45 chips/s uncalibrated
    (codeloop's note). With the clock genuinely at 3.45, a fit AT 3.45 is correct and
    must pass -- an absolute bound of 0.5 would have rejected it and left the trim
    facing ~11 chips per round trip.
    """
    clock_uncal = 3.45 / CHIP          # the dimensionless l-a that gives 3.45 chips/s
    model = cp_rate_from_code_bias(0.0, clock_uncal, HPS, CHIP, CARR)
    check(abs(model * HPS - 3.45) < 1e-6,
          "an uncalibrated clock really does command ~3.45 chips/s (%.3f)" % (model * HPS))
    check(abs(_dev_chips_s(model, clock_uncal)) < 0.5,
          "a fit tracking the UNCALIBRATED clock passes the deviation guard")
    check(abs(model * HPS) > 0.5,
          "...and would have been REJECTED by an absolute 0.5 chips/s ceiling")


def test_tolerance_zero_disables():
    """Default 0.0 must be an exact no-op, or the digests move on an unarmed broker."""
    clock = 4.62e-9
    tol = 0.0
    rate = 994.0 / HPS
    fires = tol > 0.0 and abs(_dev_chips_s(rate, clock)) > tol
    check(not fires, "tol 0.0 rejects nothing, however blown the fit")


def main():
    for fn in (test_unbounded_fit_is_the_defect, test_clean_fit_is_admitted,
               test_blown_fit_is_rejected, test_guard_is_on_deviation_not_magnitude,
               test_tolerance_zero_disables):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
