"""The joint clock's gauge is membership-invariant (#94): a satellite joining or leaving the
state does not move the clock level the seeds fly, and the gauge every joint_receiver() call
constructs is the prior gauge that has that property.

The circular-median clock (deadreckon.dr_clock_solve) steps 1-2 chips when its membership
changes; the median-mode gauge of the joint state has the same convention and is run here as
the control, so each property is one the test can see fail.

    python3 -m gnss_broker.test_gauge_churn

@author Keith Vanderlinde
"""

import ast
import os
import random
import sys

from gnss_broker.state_filter import JointReceiverState

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


CLK = 151.0  # chips, the live receiver clock
SIGMA = 0.3  # chips, the per-measurement noise
DT = 2.0  # s, the poll cycle
FLEET = {
    ("E", 3): +2.9,
    ("E", 5): -3.4,
    ("E", 31): +0.8,
    ("C", 39): -1.7,
    ("G", 14): +4.1,
    ("G", 21): -2.7,
}  # chips: the measured model-error scale; sorted, the median pair is (-1.7, +0.8)
JOINER = (("G", 7), +1.4)  # sorted in, the median becomes +0.8: a 1.25-chip step
LEAVER = ("C", 39)  # sorted out, the median becomes +0.8: the same 1.25-chip step


def _cycle(j, fleet, rnd, t):
    j.cycle([(k, CLK + b + rnd.gauss(0.0, SIGMA), SIGMA) for k, b in fleet.items()], t)


def _event(mode, join):
    """Converge FLEET, then add JOINER or let LEAVER age out. Returns (peak |dclk| over the
    45 cycles (90 s) after the event, |dclk| 300 cycles after it, worst |d(clk+b_i)| over the
    satellites that were there throughout, peak |clk_rate| after the event)."""
    rnd = random.Random(812)
    j = JointReceiverState(gauge_mode=mode)
    t = 0.0
    for _ in range(400):
        t += DT
        _cycle(j, FLEET, rnd, t)
    if join:
        after = dict(FLEET)
        after[JOINER[0]] = JOINER[1]
    else:
        after = {k: b for k, b in FLEET.items() if k != LEAVER}
    stayers = {k: j.predicted(k) for k in after if k in FLEET}
    n0 = len(j._idx)
    clk_prev = j.clk
    clk_pre = None
    at_event = 0.0
    i_event = None
    rate_pk = 0.0
    for i in range(760):  # a leaver ages out at 900 s: cycle ~450
        t += DT
        _cycle(j, after, rnd, t)
        if i_event is None and len(j._idx) != n0:
            i_event = i
            clk_pre = clk_prev
        if i_event is None:
            clk_prev = j.clk
        elif i <= i_event + 45:
            at_event = max(at_event, abs(j.clk - clk_pre))
        if i_event is not None:
            rate_pk = max(rate_pk, abs(j.clk_rate))
    if i_event is None:
        return None
    settled = abs(j.clk - clk_pre)
    worst_sum = max(abs(j.predicted(k) - v) for k, v in stayers.items())
    return at_event, settled, worst_sum, rate_pk


def test_join():
    """A satellite joining the membership does not move the prior gauge's clock level."""
    p = _event("prior", join=True)
    m = _event("median", join=True)
    check(p is not None and m is not None, "the joiner is born in both modes")
    check(
        p[0] < 0.10,
        "prior gauge: clk moved %.3f chips in the 90 s after the join" % p[0],
    )
    check(
        m[0] > 0.5,
        "control: the median gauge steps %.3f chips in the 90 s after the join (a test "
        "that cannot fail proves nothing)" % m[0],
    )
    check(
        p[1] < 0.5 and m[1] > 0.8,
        "settled |dclk| after the join: prior %.3f (the soft mean, ~b/N), median %.3f "
        "(the median's jump)" % (p[1], m[1]),
    )


def test_leave():
    """A satellite leaving the membership does not move the prior gauge's clock level."""
    p = _event("prior", join=False)
    m = _event("median", join=False)
    check(p is not None and m is not None, "the leaver ages out in both modes")
    check(
        p[0] < 0.10,
        "prior gauge: clk moved %.3f chips in the 90 s after the drop" % p[0],
    )
    check(
        m[0] > 0.5,
        "control: the median gauge steps %.3f chips in the 90 s after the drop" % m[0],
    )
    check(
        p[1] < 0.5 and m[1] > 0.8,
        "settled |dclk| after the drop: prior %.3f (the soft mean), median %.3f "
        "(the median's jump)" % (p[1], m[1]),
    )


def test_sum_invariant():
    """clk + b_i, what a seed reads, is unchanged by either event for every satellite
    that stays, under both gauges."""
    for join in (True, False):
        for mode in ("prior", "median"):
            r = _event(mode, join)
            check(
                r is not None and r[2] < 0.35,
                "%s gauge, %s: worst survivor |d(clk+b_i)| %.3f chips"
                % (mode, "join" if join else "leave", r[2] if r else -1.0),
            )


def test_rate_shield():
    """Neither event is differentiated into clk_rate: a gauge relaxation is not clock
    motion."""
    for join in (True, False):
        r = _event("prior", join)
        check(
            r is not None and r[3] < 0.002,
            "prior gauge, %s: peak |clk_rate| %.5f chips/s after the event (the receiver "
            "moves under 0.003)" % ("join" if join else "leave", r[3] if r else -1.0),
        )


def test_prior_gauge_is_what_runs():
    """Every joint_receiver() call in the broker constructs the prior gauge, so the
    properties above are the live clock's."""
    here = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(here, "deadreckon.py"),
        os.path.join(here, "ratefeed.py"),
        os.path.join(os.path.dirname(here), "gps_distributed_broker.py"),
    ]
    n = 0
    for path in files:
        with open(path) as f:
            tree = ast.parse(f.read(), path)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "joint_receiver"
            ):
                continue
            n += 1
            modes = [
                ast.literal_eval(kw.value)
                for kw in node.keywords
                if kw.arg == "gauge_mode"
            ]
            check(
                modes == ["prior"],
                "%s:%d joint_receiver(... gauge_mode=%r)"
                % (os.path.basename(path), node.lineno, modes),
            )
    check(n >= 5, "%d joint_receiver() calls found" % n)


def main():
    for fn in (
        test_join,
        test_leave,
        test_sum_invariant,
        test_rate_shield,
        test_prior_gauge_is_what_runs,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
