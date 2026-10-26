"""The shadow state's escape: one satellite's step moves its own bias, not the clock.

    python3 -m gnss_broker.test_trk_escape     (from python/scripts/gnss)

10-26 05:38Z: C35's tracker stepped -1.6 chips; five coherent rejections made it an ESCAPE,
which inflated P[0,0] as well as C35's own b, so the shadow clock took -1.5 chips and every
other satellite then disagreed and escaped in turn. With escape_clock_min_sats (jtrk.py sets
it on TRK_STATE) a lone satellite's step lands in its b; a step that escape_clock_min_sats
satellites share is still a clock event. The live state (None) keeps the old rule, and the
control below shows it is the rule that moved the clock.

@author Keith Vanderlinde
"""

import random

from gnss_broker.state_filter import JointReceiverState

_fails = []
CLK, SIGMA, DT = 151.0, 0.1, 2.0  # the shadow's joint_trk_sigma
FLEET = {
    ("G", 6): +0.12,
    ("G", 13): +0.10,
    ("E", 11): +0.13,
    ("E", 31): -0.11,
    ("C", 22): -0.08,
    ("C", 35): -0.10,
    ("C", 7): -0.25,
    ("G", 28): +0.15,
}


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def run(clock_min_sats, step_keys, step, n_conv=300, n_step=90):
    """Converge FLEET, then step `step_keys` by `step` chips. Returns (peak error of what the
    seeds ride, clk + b_j, over the satellites that did NOT step, while the step lasts; clk
    at the end; b of the first stepped satellite at the end; the state). clk alone is not
    the yardstick: the prior gauge moves it by ~b/N whenever one b moves, with every b
    moving back, and clk + b_j is what a consumer reads."""
    rnd = random.Random(7)
    j = JointReceiverState(gauge_mode="prior", escape_clock_min_sats=clock_min_sats)
    t = 0.0

    peaks = [0.0]

    def cyc(off):
        # one broker cycle: each chain cycles the shared state with its own satellites, as
        # jtrk does, and the clock is read between chains as the other chains seed from it
        nonlocal t
        t += DT
        for sysc in ("G", "E", "C"):
            j.cycle(
                [
                    (k, CLK + b + off.get(k, 0.0) + rnd.gauss(0.0, SIGMA), SIGMA)
                    for k, b in FLEET.items()
                    if k[0] == sysc
                ],
                t,
            )
            peaks[0] = max(
                [peaks[0]]
                + [
                    abs(j.clk + j.bias(k) - (CLK + b + off.get(k, 0.0)))
                    for k, b in FLEET.items()
                    if k not in off and j.bias(k) is not None
                ]
            )

    off = {}
    for _ in range(n_conv):
        cyc(off)
    peaks[0] = 0.0
    off.update({k: step for k in step_keys})
    for _ in range(n_step):
        cyc(off)
    peak = peaks[0]
    return peak, j.clk, j.bias(step_keys[0]), j


def main():
    print("one satellite steps -1.6 chips (C35 at 05:38Z)")
    peak, clk, b, j = run(3, [("C", 35)], -1.6)
    notes = j.drain_notes()
    check(peak < 0.15, "the others' clk + b stay within 0.15 chips (peak %.3f)" % peak)
    check(abs(b - (-0.10 - 1.6)) < 0.2, "and C35's b takes the step (%.2f)" % b)
    check(
        any("bias only" in n for n in notes),
        "logged as a bias-only escape (%d notes)" % len(notes),
    )

    print("two satellites of one chain step -1.6 chips together (the cascade's shape)")
    peak, clk, b, j = run(3, [("C", 35), ("C", 22)], -1.6)
    check(
        peak < 0.15,
        "shadow rule: the others' clk + b stay within 0.15 (peak %.3f)" % peak,
    )
    check(
        abs(clk + b - (CLK - 0.10 - 1.6)) < 0.15,
        "and C35's own clk + b follows its step (%.3f, want %.3f)"
        % (clk + b, CLK - 1.7),
    )
    peak0, _c, b0, _j = run(None, [("C", 35), ("C", 22)], -1.6)
    check(
        peak0 > 1.0,
        "control, the live rule: the clock takes it and every other satellite's clk + b is "
        "off by %.2f chips (C35 b %.2f), so the rule is the lever" % (peak0, b0),
    )

    print("every satellite steps +1.5 chips (a clock event)")
    peak, clk, b, j = run(3, list(FLEET), +1.5)
    check(
        abs(clk - (CLK + 1.5)) < 0.15,
        "the clock follows (%.3f, want %.3f)" % (clk, CLK + 1.5),
    )
    check(abs(b - FLEET[("G", 6)]) < 0.2, "and the biases stay put (G6 b %.2f)" % b)

    if _fails:
        raise SystemExit("FAILED: %d" % len(_fails))
    print("ok")


if __name__ == "__main__":
    main()
