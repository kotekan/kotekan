"""The code-rate cross-check (--cp-rate-model-tol): the traps that would give a wrong verdict,
not a crash.

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


HPS = 1.0 / 5.12e-6  # hops per second, CHORD
CHIP = 10.23e6  # L5/E5a chip rate
CODE_LEN = 10230


def _dev_chips_s(rate, clock_ema):
    """What the guard in seeding.py computes: fit-minus-clock, in chips/s."""
    model = cp_rate_from_code_bias(clock_ema, HPS, CHIP)
    return (rate - model) * HPS


def test_unbounded_fit_is_the_defect():
    """fit_cp_rate has no bound of its own, so the guard lives in its caller."""
    # A single mis-wrapped point injects a code period; the slope follows it anywhere.
    hist = [(0, 100.0), (1000, 100.5), (2000, 101.0), (3000, 101.5 + CODE_LEN)]
    fit = fit_cp_rate(hist, CODE_LEN)
    check(
        fit is not None,
        "an unwrap-blown history still returns a fit (no internal bound)",
    )
    if fit:
        # Not asserting a specific magnitude: the point is only that nothing rejected it.
        check(
            abs(fit[0] * HPS) > 1.0,
            "the blown fit is far outside any physical clock rate (%.1f chips/s)"
            % (fit[0] * HPS),
        )


def test_clean_fit_is_admitted():
    """A fit that agrees with the pooled clock must pass at the configured tolerance."""
    clock = 4.62e-9  # a calibrated l-a (dimensionless)
    model = cp_rate_from_code_bias(clock, HPS, CHIP)
    check(
        abs(_dev_chips_s(model, clock)) == 0.0,
        "a fit equal to the pooled clock has zero deviation",
    )
    # Fit scatter on a healthy satellite is ~0.06 chips/s SD.
    jitter = model + 0.06 / HPS
    check(
        abs(_dev_chips_s(jitter, clock)) < 0.5,
        "healthy per-poll fit scatter (0.06 chips/s) stays inside tol 0.5",
    )


def test_blown_fit_is_rejected():
    clock = 4.62e-9
    for slope_chips_s in (2.16, 177.0, 994.0):  # blown-fit slopes, mild to gross
        rate = slope_chips_s / HPS
        check(
            abs(_dev_chips_s(rate, clock)) > 0.5,
            "a %.2f chips/s fit is rejected at tol 0.5" % slope_chips_s,
        )


def test_guard_is_on_deviation_not_magnitude():
    """The trap: an absolute ceiling would reject the feed-forward the trim needs.

    The receiver clock is ~0.047 chips/s calibrated but ~3.45 chips/s uncalibrated
    (codeloop.py). With the clock genuinely at 3.45, a fit at 3.45 is correct and must pass;
    an absolute bound of 0.5 would reject it and leave the trim facing ~11 chips per round
    trip.
    """
    clock_uncal = 3.45 / CHIP  # the dimensionless l-a that gives 3.45 chips/s
    model = cp_rate_from_code_bias(clock_uncal, HPS, CHIP)
    check(
        abs(model * HPS - 3.45) < 1e-6,
        "an uncalibrated clock really does command ~3.45 chips/s (%.3f)"
        % (model * HPS),
    )
    check(
        abs(_dev_chips_s(model, clock_uncal)) < 0.5,
        "a fit tracking the UNCALIBRATED clock passes the deviation guard",
    )
    check(
        abs(model * HPS) > 0.5,
        "...and would have been REJECTED by an absolute 0.5 chips/s ceiling",
    )


def test_flush_cycle_seeds_the_detection():
    """--fit-flush-on-reject 3: on the cycle the third rejected fit flushes the history, the
    fit is the flushed history's, so neither its position and rate nor its maturity may act.

    Driven through seeding.stage_detections_to_seeds on test_escape_dedup's context: a PRN
    whose detections drift 2 chips/s against a pooled clock at 0 (tol 0.5), with a hold-strength
    amplitude, three earlier detections in the history and one per cycle after."""
    from gnss_broker import test_escape_dedup as td

    prn = 9
    ctx = td.seeding_ctx(cp_rate_model_tol=0.5, fit_flush_on_reject=3)
    ctx.cb.code_ema = 0.0
    ctx.status[prn] = {"amp_snr": 20.0}

    def cp_at(h):
        return (td.CP0 + 2.0 * (h - td.H0) / td.HPS) % td.L

    ctx.cpt.hist[prn] = [
        (h, cp_at(h), td.DOP)
        for h in (td.H0 - 3 * td.STEP, td.H0 - 2 * td.STEP, td.H0 - td.STEP)
    ]
    logs = td.run(ctx, prn, [td.H0, td.H0 + td.STEP, td.H0 + 2 * td.STEP], cp_at)
    flushed = [
        i + 1 for i, lines in enumerate(logs) if any("FLUSHED" in m for m in lines)
    ]
    check(
        flushed == [3],
        "the third rejected fit flushes the history (cycle %s)" % flushed,
    )
    seed = ctx.seeds.get(prn) or {}
    owner = getattr(seed, "prov", {}).get("code_phase_chips", (None,))[0]
    check(
        owner != "cp_fit" and seed.get("code_phase_rate") == 0.0,
        "on the flush cycle the detection's seed stands (cp0 by %r, rate %r)"
        % (owner, seed.get("code_phase_rate")),
    )
    check(prn not in ctx.fitted, "...and the PRN is not reported fitted")
    check(
        prn not in ctx.cp_held and not any("HOLD PRN %d" % prn in m for m in logs[2]),
        "...and the flushed fit's six points do not admit a hold",
    )


def main():
    for fn in (
        test_unbounded_fit_is_the_defect,
        test_clean_fit_is_admitted,
        test_blown_fit_is_rejected,
        test_guard_is_on_deviation_not_magnitude,
        test_flush_cycle_seeds_the_detection,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
