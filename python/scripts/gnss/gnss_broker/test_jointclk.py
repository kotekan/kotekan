"""The joint-clock consumer ("joint-consume: clk"): one unit before any comparison.

The joint state holds the receiver clock in its FEEDERS' chips (10.23 Mcps: gps_l5 detections,
gal_e5a model-primary); every consumer compares it with its own clock in its own chips. These
checks pin the conversion with the live numbers (joint 150.2 reference chips = 14.7 us =
7.51 L2C-CM chips = 75.1 E6 chips), the bounds' unit (reference chips on every chain), the
window a longer code keeps, and that the 10.23-Mcps chains are bit-for-bit unchanged.

    python3 -m gnss_broker.test_jointclk

@author Keith Vanderlinde
"""

import ast
import os
import random
import struct
import sys
import types

from gnss_broker import deadreckon
from gnss_broker.fits import cp_rate_from_code_bias
from gnss_broker.receiver import Receiver

BROKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gps_distributed_broker.py",
)
DEADRECKON = deadreckon.__file__

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


REF_RATE, REF_LEN = 10.23e6, 10230.0  # the feeders' chips (gps_l5, gal_e5a)
L2C_RATE, L2C_LEN = 0.5115e6, 10230.0  # CM: 10230 chips / 20 ms
E6_RATE, E6_LEN = 5.115e6, 5115.0  # E6-C: 5115 chips / 1 ms
MAX_CHIPS, MAX_SIGMA = 5.0, 0.5  # --joint-clk-max-chips / --joint-clk-max-sigma
delta = Receiver.joint_clk_delta


def _bits(x):
    return struct.pack("<d", x)


def _old_delta(joint, legacy, code_len):
    """The consumer's arithmetic before the conversion, verbatim."""
    return ((joint - legacy + code_len / 2.0) % code_len) - code_len / 2.0


class _FakeJoint(object):
    """The three things the consumer reads from the joint state."""

    def __init__(self, clk, sigma, n=14):
        self.clk = clk
        self._sigma = sigma
        self._idx = {("G", i): i for i in range(n)}

    def sigma(self, key=None):
        return self._sigma


def _consume(
    rate,
    code_len,
    legacy,
    joint_clk,
    sigma=0.047,
    units=(("gps_l5", REF_RATE, REF_LEN),),
):
    """Drive the real consumer once; returns (clk_now after, logged line)."""
    rx = Receiver(log=lambda m: None)
    for chain, r, ln in units:
        rx.joint_declare_unit(chain, r, ln)
    js = _FakeJoint(joint_clk, sigma)
    args = types.SimpleNamespace(
        joint_min_sats=4,
        joint_clk_max_chips=MAX_CHIPS,
        joint_clk_max_sigma=MAX_SIGMA,
        chip_rate_hz=rate,
    )
    ctx = types.SimpleNamespace(
        args=args,
        rx=rx,
        band_id="test",
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


def test_live_numbers_convert():
    # Live broker, same instant: joint 150.442, gps_l2c legacy 7.513, gal_e6 legacy 75.116.
    # Unconverted these read delta +142.9 and +75.3 and were refused on every cycle.
    d, dj = delta(150.442, REF_RATE, REF_LEN, 7.513, L2C_RATE, L2C_LEN)
    check(
        abs(d - 0.0091) < 1e-9 and abs(dj - 0.182) < 1e-9,
        "L2C: joint 150.442 ref = 7.5221 CM chips -> delta %+.4f CM = %+.3f ref"
        % (d, dj),
    )
    d, dj = delta(150.435, REF_RATE, REF_LEN, 75.116, E6_RATE, E6_LEN)
    check(
        abs(d - 0.1015) < 1e-9 and abs(dj - 0.203) < 1e-9,
        "E6: joint 150.435 ref = 75.2175 E6 chips -> delta %+.4f E6 = %+.3f ref"
        % (d, dj),
    )
    check(
        abs(_old_delta(150.442, 7.513, L2C_LEN) - 142.929) < 1e-9,
        "the unconverted comparison reproduces the live L2C refusal (+142.929)",
    )


def test_consumer_adopts_l2c_and_e6():
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.51, 150.2)
    check(
        abs(clk - 7.51) < 1e-9 and "-> ADOPTED" in line,
        "L2C legacy 7.51 / joint 150.2 ref: ADOPTED at 7.510 (%s)" % line,
    )
    check(
        "[x0.0500: joint 150.200 delta +0.000 sigma 0.047 in 10.230-Mcps chips" in line,
        "...and the line carries the reference-chip view the bounds test",
    )
    clk, line = _consume(E6_RATE, E6_LEN, 75.1, 150.2)
    check(
        abs(clk - 75.1) < 1e-9 and "-> ADOPTED" in line,
        "E6 legacy 75.1 / joint 150.2 ref: ADOPTED at 75.100 (%s)" % line,
    )
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.545, 151.111)
    check(
        abs(clk - 7.55555) < 1e-9 and "ADOPTED" in line,
        "first post-restart L2C adoption from the live log: 7.545 -> 7.5556 (+0.011 CM)",
    )


def test_step_refused_in_both_units():
    # gps_l5's legacy clock stepped +500 reference chips; the cross-band bootstrap hands
    # L2C the converted step (650.2 ref = 32.51 CM). The joint did not move.
    d, dj = delta(150.2, REF_RATE, REF_LEN, 650.2 * 0.05, L2C_RATE, L2C_LEN)
    check(
        abs(d + 25.0) < 1e-9 and abs(dj + 500.0) < 1e-9,
        "L2C +500-ref step: delta %+.3f CM = %+.3f ref" % (d, dj),
    )
    clk, line = _consume(L2C_RATE, L2C_LEN, 32.51, 150.2)
    check(
        abs(clk - 32.51) < 1e-12
        and "REFUSED" in line
        and "delta -25.000" in line
        and "delta -500.000" in line,
        "...REFUSED, clock left on the legacy step, delta logged in both units (%s)"
        % line,
    )
    clk, line = _consume(E6_RATE, E6_LEN, 325.1, 150.2)
    check(
        abs(clk - 325.1) < 1e-12
        and "delta -250.000" in line
        and "delta -500.000" in line,
        "E6 +500-ref step: REFUSED at delta -250 E6 = -500 ref",
    )
    # 5 reference chips is the bound on every chain: 4.9 ref passes, 5.1 does not.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.51 - 4.9 * 0.05, 150.2)
    check("ADOPTED" in line, "L2C 4.9 ref chips (0.245 CM) off: ADOPTED")
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.51 - 5.1 * 0.05, 150.2)
    check(
        "REFUSED" in line,
        "L2C 5.1 ref chips (0.255 CM) off: REFUSED -- the bound is ref chips",
    )


def test_sigma_bound_is_reference_chips():
    # Straight after a broker restart the joint read sigma 0.627 (n 10). In CM chips that is
    # 0.031 and would pass a 0.5 bound; the bound is reference chips, so it must refuse.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.522, 150.379, sigma=0.627)
    check(
        abs(clk - 7.522) < 1e-12 and "REFUSED" in line,
        "L2C sigma 0.627 ref (0.031 CM): REFUSED",
    )
    clk, line = _consume(E6_RATE, E6_LEN, 75.221, 150.379, sigma=0.627)
    check("REFUSED" in line, "E6 sigma 0.627 ref: REFUSED")


def test_windows():
    # L2C's code is 20 ms, the joint is known mod 1 ms (511.5 CM chips): the legacy clock
    # resolves which millisecond, the joint only moves us within it.
    for k in (1, 7, 19):
        leg = 7.60 + 511.5 * k
        d, dj = delta(150.2, REF_RATE, REF_LEN, leg, L2C_RATE, L2C_LEN)
        clk, line = _consume(L2C_RATE, L2C_LEN, leg, 150.2)
        check(
            abs(d + 0.09) < 1e-9
            and abs(clk - (7.51 + 511.5 * k)) < 1e-9
            and "ADOPTED" in line,
            "L2C legacy in ms %d of 20 (%.2f): delta %+.3f CM, stays in ms %d"
            % (k, leg, d, k),
        )
    # Our code wraps under the legacy clock: 10229.90 CM is -0.10; joint 1.0 ref = +0.05 CM.
    clk, line = _consume(L2C_RATE, L2C_LEN, 10229.90, 1.0)
    check(
        abs(clk - 0.05) < 1e-9 and "ADOPTED" in line,
        "L2C legacy 10229.90 (-0.10) vs joint +0.05 CM: adopted across our wrap -> 0.050",
    )
    # The joint wraps under its own window: 10229.5 ref is -0.5 ref = -0.025 CM.
    clk, line = _consume(L2C_RATE, L2C_LEN, 0.01, 10229.5)
    check(
        abs(clk - (L2C_LEN - 0.025)) < 1e-9 and "ADOPTED" in line,
        "joint 10229.5 ref (-0.5) vs L2C +0.01 CM: -> -0.025 CM (%.4f)" % clk,
    )
    clk, line = _consume(E6_RATE, E6_LEN, 5114.0, -2.0)
    check(
        abs(clk - 5114.0) < 1e-9 and "ADOPTED" in line,
        "joint -2.0 ref vs E6 legacy 5114.0 (-1.0 E6): the same time, delta 0",
    )
    # A delta near half the joint window is a wrap alias, never a small correction.
    d, dj = delta(150.2 + 5114.0, REF_RATE, REF_LEN, 7.51, L2C_RATE, L2C_LEN)
    check(
        abs(dj - 5114.0) < 1e-6,
        "delta at the half-window edge stays %+.1f ref (refused)" % dj,
    )
    d, dj = delta(150.2 + 5116.0, REF_RATE, REF_LEN, 7.51, L2C_RATE, L2C_LEN)
    check(
        abs(dj + 5114.0) < 1e-6, "...and just past it wraps to %+.1f ref (refused)" % dj
    )


def test_ten_mcps_unchanged():
    """Factor exactly 1.0 and window == our code: the old arithmetic, bit for bit."""
    rnd = random.Random(143)
    vals = [
        0.0,
        0.5,
        150.2,
        5114.9999,
        5115.0,
        5115.0001,
        10229.999,
        -0.0,
        -150.2,
        10230.0,
        20460.0,
        -10230.0,
        1e-12,
        5115.0 - 1e-12,
    ]
    pairs = [(a, b) for a in vals for b in vals]
    pairs += [
        (rnd.uniform(-3e4, 3e4), rnd.uniform(0.0, REF_LEN)) for _ in range(100000)
    ]
    pairs += [(150.0 + rnd.gauss(0, 3), 150.0 + rnd.gauss(0, 3)) for _ in range(100000)]
    bad = 0
    for j, leg in pairs:
        old = _old_delta(j, leg, REF_LEN)
        d, dj = delta(j, REF_RATE, REF_LEN, leg, REF_RATE, REF_LEN)
        if (
            _bits(d) != _bits(old)
            or _bits(dj) != _bits(old)
            or _bits((leg + d) % REF_LEN) != _bits((leg + old) % REF_LEN)
        ):
            bad += 1
    check(
        bad == 0,
        "10.23 Mcps: %d pairs, delta and adopted clock bit-identical (%d differ)"
        % (len(pairs), bad),
    )
    # The whole consumer, against the pre-conversion line for the same inputs.
    for leg, jc, sg in (
        (150.238, 150.435, 0.047),
        (10079.332, 150.660, 0.062),
        (150.192, 148.406, 1.761),
        (497.211, 146.899, 0.140),
    ):
        old = _old_delta(jc, leg, REF_LEN)
        ok = sg <= MAX_SIGMA and abs(old) <= MAX_CHIPS
        want = (
            "JOINT-CLK: legacy %.3f joint %.3f chips (delta %+.3f, sigma %.3f, n %d) -> %s"
            % (
                leg,
                jc % REF_LEN,
                old,
                sg,
                14,
                "ADOPTED"
                if ok
                else "REFUSED (bounds %.1f chips / %.2f sigma)"
                % (MAX_CHIPS, MAX_SIGMA),
            )
        )
        clk, line = _consume(REF_RATE, REF_LEN, leg, jc, sigma=sg)
        want_clk = (leg + old) % REF_LEN if ok else leg
        check(
            line == want and _bits(clk) == _bits(want_clk),
            "10.23 Mcps live case legacy %.3f joint %.3f: same line, same clock"
            % (leg, jc),
        )


def test_unit_declaration():
    rx = Receiver(log=lambda m: None)
    check(rx.joint_unit() is None, "no feeder declared: no unit")
    rx.joint_declare_unit("gps_l5", REF_RATE, 10230)
    rx.joint_declare_unit("gal_e5a", REF_RATE, 10230.0)
    check(
        rx.joint_unit() == (REF_RATE, REF_LEN), "two 10.23-Mcps feeders agree: one unit"
    )
    rx.joint_declare_unit("gps_l2c", L2C_RATE, L2C_LEN)
    check(rx.joint_unit() is None, "a feeder at another chip rate: MIXED, no unit")
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.51, 150.2, units=())
    check(
        clk == 7.51 and "REFUSED" in line and "undeclared or mixed" in line,
        "consumer with no declared unit: REFUSED, clock untouched",
    )
    clk, line = _consume(
        REF_RATE,
        REF_LEN,
        150.1,
        150.2,
        units=(("gps_l5", REF_RATE, REF_LEN), ("x", L2C_RATE, L2C_LEN)),
    )
    check(clk == 150.1 and "REFUSED" in line, "consumer with mixed feeders: REFUSED")
    js_small = _FakeJoint(150.2, 0.047, n=3)
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(
            joint_min_sats=4,
            joint_clk_max_chips=MAX_CHIPS,
            joint_clk_max_sigma=MAX_SIGMA,
            chip_rate_hz=L2C_RATE,
        ),
        rx=Receiver(log=lambda m: None),
        band_id="t",
        code_len=L2C_LEN,
        drp=types.SimpleNamespace(clk_now=7.0),
        joint_state=lambda *a: js_small,
    )
    deadreckon.dr_joint_clk(ctx)
    check(ctx.drp.clk_now == 7.0, "a state below --joint-min-sats is not consumed")


def rate_consumer_divisors(path=BROKER):
    """(number of joint-rate consumer blocks, divisors of every `clk_rate / x` inside them,
    whether the block reads Receiver.joint_unit)."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    blocks = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "'rate' in joint_consume"
    ]
    divs = [
        ast.unparse(d.right)
        for b in blocks
        for d in ast.walk(b)
        if isinstance(d, ast.BinOp)
        and isinstance(d.op, ast.Div)
        and ast.unparse(d.left).endswith("clk_rate")
    ]
    return len(blocks), divs, any("joint_unit()" in ast.unparse(b) for b in blocks)


def test_rate_consumer_unit():
    """joint-consume 'rate': clk_rate is JOINT chips/s, so the fractional rate is clk_rate over
    the JOINT chip rate; cp_rate_from_code_bias then multiplies by OUR chip rate. Dividing by our
    own rate instead seeded L2C 20x and E6 2x the code rate every other chain gets."""
    n, divs, reads_unit = rate_consumer_divisors()
    check(
        n == 1 and len(divs) >= 2 and all(d == "_ju[0]" for d in divs) and reads_unit,
        "the rate consumer divides clk_rate by the joint unit (%d block(s), divisors %s)"
        % (n, divs),
    )
    # The same receiver clock rate is the same time-rate on every chain: -0.0002 joint chips/s.
    hps = 195312.5
    rates = []
    for rate in (REF_RATE, E6_RATE, L2C_RATE):
        cph = cp_rate_from_code_bias(0.0, -0.0002 / REF_RATE, hps, rate, 1.0)
        rates.append(cph * hps / rate)  # own chips/hop -> seconds per second
    check(
        max(rates) - min(rates) < 1e-24,
        "seeded code rate: one fractional rate on 10.23 / 5.115 / 0.5115 Mcps (%s)"
        % rates,
    )


def seed_offset_site(path=DEADRECKON):
    """The per-PRN joint-vs-legacy comparison in dr_seed's seeding loop: (the call that
    computes _d3, the expression _ok3 bounds, every modulus applied to a joff-leg_off
    difference)."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "dr_seed"
    ]
    calls, oks, mods = [], [], []
    for n in ast.walk(fn[0]) if fn else ():
        if isinstance(n, ast.Assign):
            tgt = ast.unparse(n.targets[0])
            if tgt.strip("()").startswith("_d3"):
                calls.append(
                    (
                        tgt,
                        ast.unparse(n.value.func)
                        if isinstance(n.value, ast.Call)
                        else ast.unparse(n.value),
                    )
                )
            if tgt == "_ok3":
                oks.append(ast.unparse(n.value))
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Mod)
            and "_joff - _leg_off" in ast.unparse(n.left)
        ):
            mods.append(ast.unparse(n))
    return calls, oks, mods


def test_seed_offset_per_prn():
    """The per-PRN SEED-OFFSET comparison converts like the consumer: the old one compared
    raw chips and logged every gps_l2c line at +142.8 and every gal_e6 line at +75.7 chips
    REFUSED on the live broker while both chains adopted the joint clock 99.9% of the time."""
    calls, oks, mods = seed_offset_site()
    check(
        calls == [("(_d3, _d3j)", "ctx.rx.joint_clk_delta")] and not mods,
        "dr_seed takes _d3 from Receiver.joint_clk_delta, no raw joff-leg_off wrap (%s %s)"
        % (calls, mods),
    )
    check(
        oks == ["abs(_d3j) <= ctx.args.joint_slew_max_chips"],
        "...and bounds it in the joint's chips (%s)" % oks,
    )
    # The live lines: joint +150.310 vs legacy +7.529 (L2C PRN 18), +150.850 vs +75.148
    # (E6 PRN 10). Both were a few tenths of a reference chip apart, not 142.8 / 75.7.
    d, dj = delta(150.310, REF_RATE, REF_LEN, 7.529, L2C_RATE, L2C_LEN)
    check(
        abs(d + 0.0135) < 1e-9 and abs(dj + 0.270) < 1e-9 and abs(dj) <= MAX_CHIPS,
        "L2C PRN 18: delta %+.4f CM = %+.3f ref, inside the 5-chip bound" % (d, dj),
    )
    d, dj = delta(150.850, REF_RATE, REF_LEN, 75.148, E6_RATE, E6_LEN)
    check(
        abs(d - 0.277) < 1e-9 and abs(dj - 0.554) < 1e-9 and abs(dj) <= MAX_CHIPS,
        "E6 PRN 10: delta %+.4f E6 = %+.3f ref, inside the 5-chip bound" % (d, dj),
    )
    check(
        abs(_old_delta(150.310, 7.529, L2C_LEN) - 142.781) < 1e-9
        and abs(_old_delta(150.850, 75.148, E6_LEN) - 75.702) < 1e-9,
        "the unconverted comparison reproduces both live REFUSED lines (+142.781, +75.702)",
    )


def main():
    for fn in (
        test_live_numbers_convert,
        test_consumer_adopts_l2c_and_e6,
        test_step_refused_in_both_units,
        test_sigma_bound_is_reference_chips,
        test_windows,
        test_ten_mcps_unchanged,
        test_unit_declaration,
        test_rate_consumer_unit,
        test_seed_offset_per_prn,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
