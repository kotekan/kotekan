"""A whole-period clock error must be visible, and a bootstrap must not take one.

gps_l2c's code is 20 ms; the joint clock is known mod 1 ms. A gps_l2c clock whole milliseconds
off (a cross-band bootstrap from a donor locked on noise) reads to the JOINT-CLK consumer,
comparing modulo 1 ms, as a tiny delta, and adopting it every cycle keeps the wrong clock
fresh, so the bootstrap that would correct it never re-arms. These checks drive the real
consumer with the clock-mod-20-ms record present and absent, and the health rule that gates the
donor, the record and the hint.

    python3 -m gnss_broker.test_clockperiod

@author Keith Vanderlinde
"""

import types

from gnss_broker import deadreckon
from gnss_broker.clockhealth import dr_solve_healthy
from gnss_broker.receiver import Receiver

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


REF_RATE, REF_LEN = 10.23e6, 10230.0  # reference chips (gps_l5's): 1 ms window
L2C_RATE, L2C_LEN = 0.5115e6, 10230.0  # CM: 10230 chips / 20 ms
E6_RATE, E6_LEN = 5.115e6, 5115.0  # E6-C: 5115 chips / 1 ms
JOINT = 150.2  # reference chips = 14.682 us = 7.51 CM chips
T0 = 1000.0


class _FakeJoint(object):
    def __init__(self, clk, sigma=0.047, n=14):
        self.clk, self._sigma, self.L, self.clk_rate = clk, sigma, REF_LEN, 0.0
        self._idx = {("G", i): i for i in range(n)}

    def sigma(self, key=None):
        return self._sigma


def _consume(rate, code_len, legacy, record_s=None, record_age_s=10.0, chain="gps_l2c"):
    """Drive dr_joint_clk once; returns (clk_now after, the JOINT-CLK line)."""
    rx = Receiver(log=lambda m: None)
    js = _FakeJoint(JOINT)
    rx._joint["__receiver__"] = js  # the unit's source (Receiver.joint_unit)
    if record_s is not None:
        rx.contribute_clock_mod_epoch(
            "gps_l5", record_s, 0.02, 6, T0 - record_age_s, healthy=True
        )
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(
            joint_min_sats=4,
            joint_clk_max_chips=5.0,
            joint_clk_max_sigma=0.5,
            joint_clk_max_step_chips=0.3,
            joint_clk_step_hold_s=30.0,
            joint_clk_coast_max_s=0.0,  # the coast is test_jointclk's
            chip_rate_hz=rate,
        ),
        rx=rx,
        band_id="test",
        chain_id=chain,
        t0=T0,
        dr_state={},
        code_len=code_len,
        drp=types.SimpleNamespace(clk_now=legacy),
        joint_state=lambda _rx, _b, _a: js,
    )
    lines = []
    _orig = deadreckon._log_rl
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        deadreckon.dr_joint_clk(ctx)
    finally:
        deadreckon._log_rl = _orig
    return ctx.drp.clk_now, (lines[0] if lines else "")


def test_health_rule():
    now = T0
    good = {"integ": {p: (r, now - 5.0) for p, r in ((1, 0.2), (2, -0.4), (3, 0.7))}}
    check(dr_solve_healthy(good, now), "3 fresh residuals inside a chip: established")
    past = {"integ": dict(good["integ"])}
    past["integ"][4] = (1.8, now - 5.0)
    check(
        dr_solve_healthy(past, now),
        "one tracker past the escape bar, three inside: still established",
    )
    bad = {"integ": dict(good["integ"])}
    bad["integ"][4] = (50.0, now - 5.0)
    check(
        not dr_solve_healthy(bad, now),
        "one residual far off the code (noise): not established",
    )
    scatter = {"integ": {p: (r, now - 5.0) for p, r in ((1, 2.0), (2, 3.5), (3, 0.9))}}
    check(
        not dr_solve_healthy(scatter, now),
        "fewer than two inside a chip: not established",
    )
    thin = {"integ": {1: (0.2, now - 5.0), 2: (0.1, now - 5.0)}}
    check(dr_solve_healthy(thin, now), "two residuals inside a chip: established")
    one = {"integ": {1: (0.2, now - 5.0)}}
    check(
        not dr_solve_healthy(one, now), "one residual: not established (unfalsifiable)"
    )
    stale = {"integ": {p: (r, now - 120.0) for p, (r, _) in good["integ"].items()}}
    check(
        not dr_solve_healthy(stale, now), "fresh enough only 120 s ago: not established"
    )
    check(
        dr_solve_healthy(None, now) is None and dr_solve_healthy({}, now) is None,
        "never judged: no claim (None)",
    )
    empty = {"integ": {}}
    check(
        dr_solve_healthy(empty, now) is False,
        "judged, nothing fresh inside a chip: not established",
    )


def test_l2c_whole_period_is_visible():
    rec = JOINT / REF_RATE  # the NH fit agrees with the joint: 14.682 us mod 20 ms
    # 10 ms off: the 1-ms delta is +0.001 CM, the record says 10 ms.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.509 + 10.0 * 511.5, record_s=rec)
    check(
        "REFUSED" in line
        and "whole-period" in line
        and abs(clk - (7.509 + 5115.0)) < 1e-9,
        "L2C +10 ms with the record: REFUSED, clock untouched (%s)" % line[-90:],
    )
    # 1 ms off: also a whole period of the joint's window.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.509 + 511.5, record_s=rec)
    check(
        "REFUSED" in line and "whole-period" in line,
        "L2C +1 ms with the record: REFUSED",
    )
    # In the right millisecond: adopted and moved within the window.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, record_s=rec)
    check(
        "ADOPTED" in line and abs(clk - 7.51) < 1e-9,
        "L2C in the right ms with the record: ADOPTED -> 7.510 (%s)" % line[:60],
    )
    # The record disagrees with the joint itself by 0.4 ms, past the quarter-window bar on
    # clock_extend_mod's residual: no verdict from it.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.509 + 5115.0, record_s=rec + 0.4e-3)
    check(
        "ADOPTED" in line,
        "record 0.4 ms off the joint: the period check abstains (ADOPTED within the ms)",
    )
    # A stale record (older than 600 s) does not judge.
    clk, line = _consume(
        L2C_RATE, L2C_LEN, 7.509 + 5115.0, record_s=rec, record_age_s=900.0
    )
    check("ADOPTED" in line, "record 900 s old: not consulted (ADOPTED within the ms)")


def test_without_record_and_short_codes_unchanged():
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.509 + 5115.0, record_s=None)
    check(
        "ADOPTED" in line and abs(clk - (7.51 + 5115.0)) < 1e-9,
        "L2C +10 ms with no record: adopted within the ms",
    )
    clk, line = _consume(
        E6_RATE, E6_LEN, 75.3, record_s=JOINT / REF_RATE, chain="gal_e6"
    )
    check(
        "ADOPTED" in line and abs(clk - 75.1) < 1e-9,
        "E6 (1-ms code) with the record present: unchanged, ADOPTED -> 75.100",
    )
    clk, line = _consume(
        REF_RATE, REF_LEN, 150.5, record_s=JOINT / REF_RATE, chain="gal_e5a"
    )
    check(
        "ADOPTED" in line and abs(clk - 150.2) < 1e-9,
        "10.23-Mcps chain with the record present: unchanged, ADOPTED -> 150.200",
    )


def test_receiver_carries_health():
    rx = Receiver(log=lambda m: None)
    rx.contribute_dr_clock(
        "gps_l5",
        "1176.45MHz",
        150.74,
        0.01,
        T0,
        10230,
        chip_rate_hz=REF_RATE,
        healthy=False,
    )
    rec = rx.dr_clock_any_band(exclude="gps_l2c", t_now=T0)
    check(
        rec is not None and rec.extra.get("healthy") is False,
        "an unestablished donor says so in its record",
    )
    rx.contribute_dr_clock(
        "gps_l5", "1176.45MHz", 150.74, 0.01, T0, 10230, chip_rate_hz=REF_RATE
    )
    rec = rx.dr_clock_any_band(exclude="gps_l2c", t_now=T0)
    check(
        rec.extra.get("healthy") is None, "a contributor that makes no claim stays None"
    )


if __name__ == "__main__":
    for t in (
        test_health_rule,
        test_l2c_whole_period_is_visible,
        test_without_record_and_short_codes_unchanged,
        test_receiver_carries_health,
    ):
        print(t.__name__)
        t()
    print("FAIL %d" % len(_fails) if _fails else "OK")
    raise SystemExit(1 if _fails else 0)
