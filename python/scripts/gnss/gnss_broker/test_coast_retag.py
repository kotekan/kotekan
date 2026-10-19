"""The coast re-tag keeps the phase the tracker runs at now when the same write changes the
Doppler rate.

    python3 -m gnss_broker.test_coast_retag     (from python/scripts/gnss)

seeding.stage_coast_drop stores the almanac forecast back-propagated to ref_hop, with the
forecast's rate, and re-expresses cp0 so the despread does not move. cp0's partner is the
stored doppler_hz, which a rate change moves by (old rate - new rate) * age on top of the
forecast's news, and the rate itself enters the phase at now through the quadratic term
0.5 * (f_chip/f_carrier) * rate * age^2. A seed without code_phase_at_ref_chips coasts on
(cp0, doppler_hz, doppler_rate_hz_s) alone, so cp0 must absorb both: missed, the phase at now
steps by (f_chip/f_carrier) * d_rate * age * t_abs, thousands of chips two days in.

@author Keith Vanderlinde
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import seeding  # noqa: E402
from gnss_broker.fits import tracker_phase_at  # noqa: E402
from gnss_broker.seed import Seed  # noqa: E402
from gnss_broker.transport import _TR  # noqa: E402

HPS = 195312.5
CHIP = 10.23e6
CARR = 1176.45e6
SGN = 1.0
L = 10230.0
T0 = 172800.0  # two days after sample 0, where t_abs multiplies any Doppler error
H0 = int(T0 * HPS)
AGE = 10.0  # s since the seed's ref_hop
H_NOW = H0 + int(AGE * HPS)
PRN, CP0, DOP0 = 11, 4321.0, 245.0
UTC0 = 1.7e9
# The double-precision floor of a phase rebuilt through t*f_chip ~ 1.8e12 chips (one ulp is
# 2.4e-4 chips); a mistranslation is chips.
TOL = 2e-3

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _wrap(d):
    return ((d + L / 2) % L) - L / 2


def coast(seed, forecast, rate):
    """One stage_coast_drop pass over a seed AGE s past its ref_hop and not re-detected, with
    the almanac predicting (forecast at now, rate); returns the seed."""
    ctx = NS(
        seeds={PRN: seed},
        probe_set=set(),
        up=None,
        dr_state=None,
        best={},
        pred={PRN: (forecast, rate)},
        cb=NS(seed=0.0),
        utc0_sample0=UTC0,
        code_len=L,
        status={},
        have_sig=False,
        hold=NS(prev={}, low_hits={}, miss={}),
        coast_polls=100,
        args=NS(
            almanac=True,
            dr_fengine_axis=False,
            hops_per_sec=HPS,
            chip_rate_hz=CHIP,
            carrier_hz=CARR,
            code_doppler_sign=SGN,
            drop_amplitude=1.0,
            lock_prompt_hold=1.0,
            coast_budget=60.0,
        ),
    )
    _TR._t.v = UTC0 + T0 + AGE  # the cycle's frozen clock: now is AGE past ref_hop
    try:
        seeding.stage_coast_drop(ctx)
    finally:
        _TR._t.v = None
    return ctx.seeds[PRN]


def phase_now(seed):
    return tracker_phase_at(seed, H_NOW, HPS, CHIP, CARR, SGN, L)


def _seed(rate):
    return Seed.born(
        "test",
        epoch=H0,
        code_phase_chips=CP0,
        doppler_hz=DOP0,
        ref_hop=H0,
        doppler_rate_hz_s=rate,
    )


def test_rate_change():
    """0.3 Hz/s -> 0.1 Hz/s at age 10 s, with 0.5 Hz of news in the forecast."""
    seed = _seed(0.3)
    before = phase_now(dict(seed))
    forecast = DOP0 + 0.3 * AGE + 0.5
    after = coast(seed, forecast, 0.1)
    check(
        after["doppler_rate_hz_s"] == 0.1
        and abs(after["doppler_hz"] + 0.1 * AGE - forecast) < 1e-9,
        "the tracker applies the forecast at now (%.6f Hz)"
        % (after["doppler_hz"] + 0.1 * AGE),
    )
    err = _wrap(phase_now(after) - before)
    check(abs(err) < TOL, "the phase at now is kept (moved %+.4f chips)" % err)


def test_rate_change_only():
    """A forecast with no news at now but a new rate still moves cp0's partner."""
    seed = _seed(0.3)
    before = phase_now(dict(seed))
    after = coast(seed, DOP0 + 0.3 * AGE, -0.2)
    applied = after["doppler_hz"] + after["doppler_rate_hz_s"] * AGE
    check(
        abs(applied - (DOP0 + 0.3 * AGE)) < 1e-9,
        "rate alone, 0.3 -> -0.2 Hz/s: the tracker still applies the forecast at now "
        "(%.6f Hz)" % applied,
    )
    err = _wrap(phase_now(after) - before)
    check(abs(err) < TOL, "...and the phase at now is kept (moved %+.4f chips)" % err)


def test_rate_change_negative_sign():
    """The same at code_doppler_sign -1: the Doppler's linear term flips with the sign, the
    rate's quadratic term (propagate_seed's) does not."""
    global SGN
    saved, SGN = SGN, -1.0
    try:
        test_rate_change()
        test_rate_change_only()
    finally:
        SGN = saved


def test_same_rate_unchanged():
    """With the rate unchanged the re-tag is the Doppler's alone, bit for bit as before."""
    from gnss_broker.fits import retag_seed_doppler

    seed = _seed(0.3)
    forecast = DOP0 + 0.3 * AGE + 0.5
    after = coast(seed, forecast, 0.3)
    want = retag_seed_doppler(
        CP0, DOP0 + 0.3 * AGE, forecast, (UTC0 + T0 + AGE) - UTC0, CHIP, CARR, SGN, L
    )
    check(
        after["code_phase_chips"] == want,
        "same rate: cp0 is retag_seed_doppler's (%r vs %r)"
        % (after["code_phase_chips"], want),
    )


def main():
    for fn in (
        test_rate_change,
        test_rate_change_only,
        test_rate_change_negative_sign,
        test_same_rate_unchanged,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
