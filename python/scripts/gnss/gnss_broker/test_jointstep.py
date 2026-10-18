"""The JOINT-CLK step guard: a one-off excursion of the joint clock is refused, a level that
holds is followed after the hold, and neither a walk nor the legacy median's churn meets it.

The sequences are gal_e5a JOINT-CLK lines (legacy, joint, sigma, n at the line's own time): a
C33 pass with a one-line 0.806-chip excursion, which adopted would move every 10.23-Mcps chain's
seeds off their prompts; a slow C13 walk that must be followed; and a 1-chip level that holds.

    python3 -m gnss_broker.test_jointstep

@author Keith Vanderlinde
"""

import sys
import types

from gnss_broker import deadreckon
from gnss_broker.receiver import Receiver

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


REF_RATE, REF_LEN = 10.23e6, 10230.0
STEP, HOLD = 0.3, 30.0

# (time, legacy, joint, sigma, n): the C33 pass. Sigma crosses the 0.50 bound at 12:56:17; the
# 4-satellite solve at 12:58:17 passes the bounds 0.664 chips below the last adopted joint clock
# and is gone by the next line.
C33 = [
    ("12:55:17.859", 149.593, 149.466, 0.416, 6),
    ("12:55:47.867", 149.593, 149.450, 0.465, 6),
    ("12:56:17.875", 149.593, 149.435, 0.516, 6),
    ("12:56:47.876", 149.593, 149.420, 0.567, 6),
    ("12:57:17.879", 149.593, 149.404, 0.625, 6),
    ("12:57:47.894", 149.593, 149.389, 0.680, 6),
    ("12:58:17.919", 149.593, 148.786, 0.228, 4),
    ("12:58:53.899", 149.593, 149.527, 0.192, 4),
    ("12:59:23.903", 149.593, 149.515, 0.161, 4),
    ("12:59:53.911", 149.593, 149.731, 0.143, 5),
    ("13:00:23.924", 149.593, 149.679, 0.154, 5),
    ("13:00:53.947", 149.593, 149.182, 0.158, 5),
]

# The C13 run: the joint walks +0.56 -> +0.98 -> +0.58 chips from the legacy over 8 min.
C13 = [
    ("11:50:28.393", 150.019, 150.085, 0.104, 6),
    ("11:50:58.404", 150.009, 150.045, 0.100, 7),
    ("11:51:28.409", 149.763, 149.899, 0.095, 7),
    ("11:51:58.417", 149.838, 149.811, 0.097, 7),
    ("11:52:28.419", 149.838, 149.727, 0.097, 7),
    ("11:52:58.421", 149.838, 149.902, 0.096, 7),
    ("11:53:28.425", 149.838, 149.987, 0.097, 7),
    ("11:53:58.443", 149.838, 150.060, 0.099, 7),
    ("11:54:28.448", 149.838, 150.117, 0.102, 7),
    ("11:54:58.460", 149.838, 150.253, 0.108, 7),
    ("11:55:28.452", 149.838, 150.296, 0.112, 7),
    ("11:55:58.462", 149.838, 150.396, 0.111, 7),
    ("11:56:28.458", 149.838, 150.410, 0.113, 7),
    ("11:56:58.466", 149.838, 150.381, 0.097, 8),
    ("11:57:28.465", 149.838, 150.235, 0.085, 8),
    ("11:57:58.466", 149.838, 150.234, 0.084, 8),
    ("11:58:28.474", 149.838, 150.254, 0.084, 8),
    ("11:58:58.479", 149.838, 150.215, 0.086, 8),
    ("11:59:28.492", 149.838, 150.235, 0.087, 8),
    ("11:59:58.493", 149.838, 150.317, 0.089, 8),
    ("12:00:28.509", 149.838, 150.523, 0.097, 8),
    ("12:00:58.504", 149.838, 150.428, 0.105, 8),
    ("12:01:28.513", 149.838, 150.661, 0.119, 8),
    ("12:01:58.516", 149.838, 150.681, 0.139, 8),
    ("12:02:28.519", 149.838, 150.750, 0.166, 8),
    ("12:02:58.525", 149.838, 150.820, 0.198, 8),
    ("12:03:28.532", 149.838, 150.474, 0.190, 8),
    ("12:03:58.537", 149.838, 150.413, 0.201, 8),
    ("12:04:28.553", 149.838, 150.250, 0.188, 8),
    ("12:04:58.551", 149.838, 150.267, 0.186, 8),
    ("12:05:28.554", 149.838, 150.148, 0.192, 8),
    ("12:05:58.561", 149.838, 150.049, 0.199, 8),
    ("12:06:28.564", 149.838, 150.036, 0.201, 8),
    ("12:06:58.579", 149.838, 149.814, 0.196, 8),
    ("12:07:28.572", 149.838, 150.213, 0.134, 8),
    ("12:07:58.592", 149.838, 150.280, 0.121, 8),
    ("12:08:28.584", 149.838, 150.319, 0.118, 8),
    ("12:08:58.585", 149.838, 150.264, 0.118, 8),
    ("12:09:28.598", 150.194, 150.346, 0.116, 8),
    ("12:09:58.600", 150.300, 150.348, 0.116, 8),
    ("12:10:28.603", 150.220, 150.233, 0.099, 8),
    ("12:10:58.608", 150.449, 150.261, 0.088, 8),
    ("12:11:28.617", 150.323, 150.297, 0.083, 8),
    ("12:11:58.625", 150.373, 150.261, 0.080, 8),
]

# The joint drops ~1 chip, holds it for ~90 s and comes back.
EXC = [
    ("07:50:25.659", 150.351, 150.224, 0.389, 5),
    ("07:50:55.636", 150.351, 149.335, 0.243, 5),
    ("07:51:25.642", 150.351, 149.209, 0.267, 5),
    ("07:51:55.639", 150.351, 149.497, 0.197, 5),
    ("07:52:25.647", 150.351, 150.257, 0.143, 4),
    ("07:52:55.656", 150.351, 150.289, 0.138, 4),
    ("07:53:25.650", 150.351, 150.371, 0.126, 5),
]


def _sod(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class _FakeJoint(object):
    def __init__(self, clk, sigma, n):
        self.clk, self._sigma, self.L, self.clk_rate = clk, sigma, REF_LEN, 0.0
        self._idx = {("G", i): i for i in range(n)}

    def sigma(self, key=None):
        return self._sigma


class _Chain(object):
    """One consumer chain at 10.23 Mcps, its dr_state carried from cycle to cycle."""

    def __init__(self):
        rx = Receiver(log=lambda m: None)
        self.ctx = types.SimpleNamespace(
            args=types.SimpleNamespace(
                joint_min_sats=4,
                joint_clk_max_chips=5.0,
                joint_clk_max_sigma=0.5,
                joint_clk_max_step_chips=STEP,
                joint_clk_step_hold_s=HOLD,
                # The step guard is judged on the legacy clock here; the coast a refusal
                # rides is test_jointclk's.
                joint_clk_coast_max_s=0.0,
                chip_rate_hz=REF_RATE,
            ),
            rx=rx,
            band_id="test",
            code_len=REF_LEN,
            t0=0.0,
            dr_state={},
            drp=types.SimpleNamespace(clk_now=0.0),
            joint_state=None,
        )

    def cycle(self, t, legacy, joint, sigma, n):
        """One dr_joint_clk call; returns (clk_now after, verdict)."""
        c = self.ctx
        c.t0, c.drp.clk_now = t, legacy
        js = _FakeJoint(joint, sigma, n)
        c.joint_state = lambda _rx, _b, _a: js
        c.rx._joint["__receiver__"] = js  # the unit's source (Receiver.joint_unit)
        lines = []
        orig = deadreckon._log_rl
        deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
        try:
            deadreckon.dr_joint_clk(c)
        finally:
            deadreckon._log_rl = orig
        line = lines[0] if lines else ""
        if "-> ADOPTED" in line:
            v = "ADOPTED"
        elif "REFUSED (step" in line:
            v = "STEP"
        elif "REFUSED (bounds" in line:
            v = "BOUNDS"
        else:
            v = line
        return c.drp.clk_now, v

    def run(self, rows):
        return [(ts,) + self.cycle(_sod(ts), *r) for ts, *r in rows]


def test_c33_outlier_refused():
    print(
        "C33 12:58:17Z: the 0.806-chip one-off is refused, the -0.07 that follows adopted"
    )
    out = _Chain().run(C33)
    verdicts = [v for _, _, v in out]
    want = ["ADOPTED"] * 2 + ["BOUNDS"] * 4 + ["STEP"] + ["ADOPTED"] * 4 + ["STEP"]
    check(verdicts == want, "verdicts %s" % verdicts)
    clk = dict((ts, c) for ts, c, _ in out)
    check(
        abs(clk["12:58:17.919"] - 149.593) < 1e-9,
        "12:58:17 refused: the chain stays on the legacy clock (%.3f)"
        % clk["12:58:17.919"],
    )
    check(
        abs(clk["12:58:53.899"] - 149.527) < 1e-9,
        "12:58:53 adopted at once: 0.077 chips from the last adopted joint clock (149.450)",
    )
    check(
        abs(clk["13:00:53.947"] - 149.593) < 1e-9,
        "13:00:53 -0.411 (0.497 below the last adopted) refused the same way",
    )
    # At the 2-s chain cycle the excursion is a decay, 148.79 at 12:58:14 back to 149.53 by
    # 12:58:52. No 0.3-chip level of it holds 30 s, so the chain stays on the legacy clock until
    # the joint is back within 0.3 of the last adopted level and then follows it home: the
    # largest step the seeds see is the 0.3 tolerance plus the legacy's own 0.14 offset, not
    # 0.806.
    ch = _Chain()
    ch.cycle(_sod("12:55:47.867"), 149.593, 149.450, 0.465, 6)
    t, joint, steps, adopted = _sod("12:58:14"), 148.79, [], []
    clk = 149.593
    while t < _sod("12:58:54"):
        new, v = ch.cycle(t, 149.593, joint, 0.23, 4)
        steps.append(new - clk)
        if v == "ADOPTED":
            adopted.append(joint)
        clk, t, joint = new, t + 2.0, min(joint + 0.04, 149.53)
    check(
        min(adopted) > 149.450 - STEP - 1e-9 and max(abs(x) for x in steps) < 0.45,
        "2-s cadence: first adoption at joint %.3f, largest step %.3f chips"
        % (min(adopted), max(abs(x) for x in steps)),
    )


def test_c13_run_followed():
    print("C13 12:00Z: a +0.56 -> +0.98 -> +0.58 run over 8 min is followed")
    out = _Chain().run(C13)
    refused = [ts for ts, _, v in out if v != "ADOPTED"]
    check(
        refused == ["12:03:28.532", "12:07:28.572"],
        "at the log's 30-s cadence only the two 0.35/0.40 steps wait one line (%s)"
        % refused,
    )
    after = dict(zip([ts for ts, _, _ in out], [v for _, _, v in out][1:] + [None]))
    check(
        all(after[ts] == "ADOPTED" for ts in refused),
        "...and each is adopted on the next line: the level held",
    )
    # The chain cycle is 2 s: between the lines the joint walks, and no 2-s step of this run
    # reaches 0.3 chips, so at the live cadence the guard is never met.
    ch = _Chain()
    verdicts = []
    for (ts0, lg0, j0, s0, n0), (ts1, lg1, j1, s1, n1) in zip(C13, C13[1:]):
        t0, t1 = _sod(ts0), _sod(ts1)
        t = t0
        while t < t1:
            f = (t - t0) / (t1 - t0)
            verdicts.append(ch.cycle(t, lg0, j0 + f * (j1 - j0), s0, n0)[1])
            t += 2.0
    check(
        all(v == "ADOPTED" for v in verdicts),
        "%d 2-s cycles through the run: every one adopted" % len(verdicts),
    )


def test_level_followed_after_hold():
    print("07:50Z: a 1-chip level that holds 90 s is adopted 30 s late, both ways")
    out = _Chain().run(EXC)
    verdicts = [v for _, _, v in out]
    want = ["ADOPTED", "STEP", "ADOPTED", "ADOPTED", "STEP", "ADOPTED", "ADOPTED"]
    check(verdicts == want, "verdicts %s" % verdicts)


def test_legacy_churn_is_not_a_step():
    print("the legacy median steps 0.5 chips while the joint stands: no refusal")
    ch = _Chain()
    verdicts = [
        ch.cycle(t, lg, 150.20, 0.09, 6)[1]
        for t, lg in ((0.0, 150.10), (2.0, 150.55), (4.0, 150.05), (6.0, 150.60))
    ]
    check(
        all(v == "ADOPTED" for v in verdicts),
        "delta -0.10 / +0.35 / -0.15 / +0.40 on a still joint: %s" % verdicts,
    )


def test_hold_restarts():
    print("the hold: a bounds refusal restarts it, a drifting candidate re-anchors")
    ch = _Chain()
    ch.cycle(0.0, 150.0, 150.00, 0.09, 6)
    v = [ch.cycle(t, 150.0, 150.50, 0.09, 6)[1] for t in (2.0, 12.0, 22.0)]
    v.append(ch.cycle(24.0, 150.0, 150.50, 0.90, 6)[1])  # sigma over the bound
    v.append(ch.cycle(26.0, 150.0, 150.50, 0.09, 6)[1])
    v.append(ch.cycle(40.0, 150.0, 150.50, 0.09, 6)[1])  # 14 s since the restart
    v.append(ch.cycle(56.0, 150.0, 150.50, 0.09, 6)[1])  # 30 s since the restart
    check(
        v == ["STEP"] * 3 + ["BOUNDS", "STEP", "STEP", "ADOPTED"],
        "+0.5 held 22 s, refused by sigma, held 30 s more: %s" % v,
    )
    ch = _Chain()
    ch.cycle(0.0, 150.0, 150.00, 0.09, 6)
    v = [ch.cycle(t, 150.0, j, 0.09, 6)[1] for t, j in ((2.0, 150.5), (20.0, 150.85))]
    v.append(ch.cycle(32.0, 150.0, 150.85, 0.09, 6)[1])  # 30 s after the first +0.5
    v.append(ch.cycle(50.0, 150.0, 150.85, 0.09, 6)[1])  # 30 s after the +0.85
    check(
        v == ["STEP", "STEP", "STEP", "ADOPTED"],
        "+0.5 then +0.85 (0.35 apart): the hold starts over at the second level: %s"
        % v,
    )
    # A wrap of the joint's window is not a step.
    ch = _Chain()
    ch.cycle(0.0, 10229.9, 10229.95, 0.09, 6)
    clk, v = ch.cycle(2.0, 0.1, 0.05, 0.09, 6)
    check(
        v == "ADOPTED" and abs(clk - 0.05) < 1e-9, "10229.95 -> 0.05: adopted (%s)" % v
    )


def main():
    for fn in (
        test_c33_outlier_refused,
        test_c13_run_followed,
        test_level_followed_after_hold,
        test_legacy_churn_is_not_a_step,
        test_hold_restarts,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
