"""The joint-clock consumer ("joint-consume: clk"): one unit before any comparison.

The joint state holds the receiver clock in reference chips (10.23 Mcps, gps_l5's,
receiver.JOINT_REF_RATE_HZ) whatever chain fed it: every feeder converts in and every
consumer compares it with its own clock in its own chips. These checks pin the conversion
with live-sized numbers (joint 150.2 reference chips = 14.7 us = 7.51 L2C-CM chips = 75.1
E6 chips), the bounds' unit (reference chips on every chain), the window a longer code
keeps, that on 10.23-Mcps chains the conversion is bit for bit the plain wrapped difference,
that a feed at any chip rate lands in the same state, and the coast a refused joint clock
rides (clk + clk_rate dt, never the legacy median). Also the feed itself: dr_joint_shadow
hands the state each detection once, however long the search re-serves it, and the state is
born at clk 0 whoever creates it.

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
from gnss_broker.receiver import JOINT_REF_RATE_HZ, Receiver

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


REF_RATE, REF_LEN = JOINT_REF_RATE_HZ, 10230.0  # reference chips (gps_l5's)
L2C_RATE, L2C_LEN = 0.5115e6, 10230.0  # CM: 10230 chips / 20 ms
E6_RATE, E6_LEN = 5.115e6, 5115.0  # E6-C: 5115 chips / 1 ms
MAX_CHIPS, MAX_SIGMA = 5.0, 0.5  # joint_clk_max_chips / joint_clk_max_sigma
delta = Receiver.joint_clk_delta


def _bits(x):
    return struct.pack("<d", x)


def _old_delta(joint, legacy, code_len):
    """The unconverted wrapped difference, valid only when both clocks share one unit."""
    return ((joint - legacy + code_len / 2.0) % code_len) - code_len / 2.0


class _FakeJoint(object):
    """What the consumer reads from the joint state: clk, clk_rate and sigma in reference
    chips, the modulus L, and the satellite index."""

    def __init__(self, clk, sigma, n=14, clk_rate=0.0, L=REF_LEN):
        self.clk = clk
        self.clk_rate = clk_rate
        self.L = L
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
    clk_rate=0.0,
    dr_state=None,
    coast_max_s=300.0,
    t0=100.0,
):
    """Drive the real consumer once; returns (clk_now after, logged line). dr_state, when
    given, is the chain's and is mutated as the consumer would (jclk_* stamps)."""
    rx = Receiver(log=lambda m: None)
    js = _FakeJoint(joint_clk, sigma, clk_rate=clk_rate)
    # The receiver's state is the unit's source (Receiver.joint_unit reads its L), so the
    # double stands where joint_receiver() would have put the real one.
    rx._joint["__receiver__"] = js
    args = types.SimpleNamespace(
        joint_min_sats=4,
        joint_clk_max_chips=MAX_CHIPS,
        joint_clk_max_sigma=MAX_SIGMA,
        joint_clk_max_step_chips=0.3,
        joint_clk_step_hold_s=30.0,
        joint_clk_coast_max_s=coast_max_s,
        chip_rate_hz=rate,
    )
    ctx = types.SimpleNamespace(
        args=args,
        rx=rx,
        band_id="test",
        code_len=code_len,
        t0=t0,
        dr_state={} if dr_state is None else dr_state,
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
    # One instant of the live broker: joint 150.442, gps_l2c legacy 7.513, gal_e6 legacy
    # 75.116. Compared unconverted they read delta +142.9 and +75.3 and would be refused.
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
        "the unconverted comparison reads L2C +142.929, past the bound",
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
        "L2C legacy 7.545 / joint 151.111 ref: adopted at 7.5556 (+0.011 CM)",
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
    # A joint sigma of 0.627 reference chips (as after a broker restart) is 0.031 in CM chips
    # and would pass a 0.5 bound there; the bound is reference chips, so it must refuse.
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
    """Factor exactly 1.0 and window == our code: the plain wrapped difference, bit for bit."""
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
    # The whole consumer, against the line the unconverted arithmetic gives for the same inputs.
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
            "10.23 Mcps consumer at legacy %.3f joint %.3f: same line, same clock"
            % (leg, jc),
        )


_TB = {("G", 3): +2.9, ("G", 5): -3.4, ("G", 14): +0.8, ("G", 21): -1.7, ("G", 30): 1.1}


def _feed(js, rate, clk_ref, n=40, t0=0.0):
    """Feed a fleet at the true clock clk_ref (reference chips) through a chain at this chip
    rate, converting in as dr_joint_shadow does (offsets and sigma times k)."""
    k = JOINT_REF_RATE_HZ / rate
    t = t0
    for _ in range(n):
        t += 2.0
        js.cycle([(key, (clk_ref + b) / k * k, 0.3 * k) for key, b in _TB.items()], t)
    return t


def joint_feed_site(path=DEADRECKON):
    """The measurement tuples dr_joint_shadow hands _js.cycle and the MINNOV / P2C residual
    expressions: every offset must go in times _k and every residual come out over _k."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "dr_joint_shadow"
    ]
    fed, resid = [], []
    for n in ast.walk(fn[0]) if fn else ():
        if (
            isinstance(n, ast.Call)
            and ast.unparse(n.func).endswith(".cycle")
            and isinstance(n.args[0], ast.ListComp)
        ):
            fed.append(ast.unparse(n.args[0].elt))
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("_mi", "_r"):
            resid.append(ast.unparse(n.value))
    return fed, resid


def test_one_unit():
    # (a) the same receiver clock through an L2C-CM feeder and a 10.23-Mcps feeder of the
    # same 20-ms period: one state, read the same in reference chips.
    rx_ref, rx_l2c = Receiver(log=lambda m: None), Receiver(log=lambda m: None)
    js_ref = rx_ref.joint_receiver(
        "l5", 204600.0, REF_RATE, gauge_mode="prior", clk0=150.0
    )
    js_l2c = rx_l2c.joint_receiver(
        "l2c", L2C_LEN, L2C_RATE, gauge_mode="prior", clk0=7.5 * 20.0
    )
    check(
        js_l2c.L == 204600.0 and js_ref.L == js_l2c.L,
        "L2C-CM 10230 chips at 0.5115 Mcps is a 204600-reference-chip modulus (%g)"
        % js_l2c.L,
    )
    _feed(js_ref, REF_RATE, 150.2)
    _feed(js_l2c, L2C_RATE, 150.2)
    check(
        abs(js_ref.clk - js_l2c.clk) < 1e-6 and abs(js_l2c.clk - 150.2) < 0.3,
        "an L2C-rate feed reads the same clk in reference chips (%.4f vs %.4f)"
        % (js_l2c.clk, js_ref.clk),
    )
    check(
        rx_l2c.joint_unit() == (REF_RATE, 204600.0),
        "joint_unit() is the reference rate and the modulus in reference chips",
    )
    check(Receiver(log=lambda m: None).joint_unit() is None, "...None with no state")
    # (b) a second constellation joining at another chip rate neither lengthens the modulus
    # nor changes clk; a 1-ms code at E6's rate is the same 10230-chip modulus.
    rx = Receiver(log=lambda m: None)
    js = rx.joint_receiver("l5", REF_LEN, REF_RATE, gauge_mode="prior", clk0=150.0)
    t = _feed(js, REF_RATE, 150.2)
    clk0, L0 = _bits(js.clk), js.L
    js2 = rx.joint_receiver("l2c", L2C_LEN, L2C_RATE, gauge_mode="prior", clk0=0.0)
    js3 = rx.joint_receiver("e6", E6_LEN, E6_RATE, gauge_mode="prior", clk0=0.0)
    check(
        js2 is js and js3 is js and js.L == L0 == REF_LEN and _bits(js.clk) == clk0,
        "L2C (20 ms) and E6 (1 ms) joining leave the 10230-chip modulus and clk untouched",
    )
    _feed(js, E6_RATE, 150.2, t0=t)
    check(
        abs(js.clk - 150.2) < 0.3,
        "...and E6 measurements converted in keep clk at the receiver's %.3f" % js.clk,
    )
    js4 = rx.joint_receiver("x", 2557.5, E6_RATE, gauge_mode="prior")
    check(js4.L == 5115.0, "a shorter period (0.5 ms at E6's rate) shortens it to 5115")
    # (c) the feed site converts in and reads back out.
    fed, resid = joint_feed_site()
    check(
        fed == ["((ctx.drp.tag, p), d * _k, ctx.args.joint_sigma * _k)"]
        and len(resid) == 2
        and all(r.endswith(") / _k") and "* _k - _js.predicted(" in r for r in resid),
        "dr_joint_shadow feeds d*k, sigma*k and reads MINNOV / P2C back over k (%s %s)"
        % (fed, resid),
    )
    js_small = _FakeJoint(150.2, 0.047, n=3)
    rx_small = Receiver(log=lambda m: None)
    rx_small._joint["__receiver__"] = js_small
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(
            joint_min_sats=4,
            joint_clk_max_chips=MAX_CHIPS,
            joint_clk_max_sigma=MAX_SIGMA,
            joint_clk_max_step_chips=0.3,
            joint_clk_step_hold_s=30.0,
            joint_clk_coast_max_s=300.0,
            chip_rate_hz=L2C_RATE,
        ),
        rx=rx_small,
        band_id="t",
        code_len=L2C_LEN,
        t0=100.0,
        dr_state={},
        drp=types.SimpleNamespace(clk_now=7.0),
        joint_state=lambda *a: js_small,
    )
    deadreckon.dr_joint_clk(ctx)
    check(ctx.drp.clk_now == 7.0, "a state below joint_min_sats is not consumed")


def test_coast_on_refusal():
    """A refused joint clock coasts on the last adopted one at the rate stamped at that
    adoption inside --joint-clk-coast-max-s, never on the legacy clock and never on the
    state's current rate; the stamps do not move; the extrapolation is bounded."""
    # L2C: adopted 150.2 ref at t=60 with the rate +0.001 ref chips/s stamped, joint now
    # 160.2 (bounds refusal) and its live rate a wild +0.5: 40 s later the coast is 150.24
    # ref = 7.512 CM on the STAMPED rate, not the legacy 7.60 and not 150.2 + 20.
    ds = {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0, "jclk_last_rate": 0.001}
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.5, dr_state=ds)
    check(
        abs(clk - 7.512) < 1e-9 and "-> COAST (refused: bounds" in line,
        "L2C bounds refusal 40 s after adoption: COAST to 7.512 CM (%s)" % line,
    )
    check(
        "+0.040 joint chips over 40 s at clk_rate +0.0010 joint chips/s" in line,
        "...the line carries the extrapolation and the rate in joint chips",
    )
    check(
        ds == {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0, "jclk_last_rate": 0.001},
        "...and no stamp moved: jclk_t, jclk_last, jclk_last_t and jclk_last_rate as adopted",
    )
    # 10.23 Mcps, the step guard's refusal: the joint moved 0.9 from the last adopted level.
    ds = {"jclk_last": 150.2, "jclk_last_t": 90.0, "jclk_t": 90.0, "jclk_last_rate": -0.0004}
    clk, line = _consume(
        REF_RATE, REF_LEN, 150.238, 151.1, clk_rate=-0.0004, dr_state=ds
    )
    check(
        abs(clk - (150.2 - 0.004)) < 1e-9 and "COAST (refused: step +0.900" in line,
        "10.23 Mcps step refusal 10 s after adoption: COAST to 150.196 (%s)" % line,
    )
    check(
        ds["jclk_cand"] == (151.1, 100.0), "...the step guard's candidate still armed"
    )
    # E6: the coast is in joint chips, converted like an adoption (150.24 ref = 75.12 E6).
    ds = {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0, "jclk_last_rate": 0.001}
    clk, line = _consume(E6_RATE, E6_LEN, 75.3, 160.2, clk_rate=0.001, dr_state=ds)
    check(abs(clk - 75.12) < 1e-9 and "COAST" in line, "E6: COAST to 75.120 E6 chips")
    # The bound: a stamped rate of 0.05 ref chips/s over 200 s is 10 chips of extrapolation,
    # twice what an adoption may move the seeds; the coast is declined and says so.
    ds = {"jclk_last": 150.2, "jclk_last_t": -100.0, "jclk_t": -100.0, "jclk_last_rate": 0.05}
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.0, dr_state=ds)
    check(
        clk == 7.60
        and "-> REFUSED (bounds 5.0 chips / 0.50 sigma; coast declined: +10.0 joint chips "
        "of extrapolation over 200 s is past the 5.0-chip bound)" in line,
        "10 chips of extrapolation: the coast is declined, legacy clock (%s)" % line[-90:],
    )
    # An adoption without the rate stamp (a dr_state from before 10-18): no coast.
    ds = {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0}
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.001, dr_state=ds)
    check(clk == 7.60 and "-> REFUSED (bounds" in line, "no stamped rate: REFUSED")
    # Past the window: REFUSED on the legacy clock, exactly as before.
    ds = {"jclk_last": 150.2, "jclk_last_t": 100.0 - 300.1, "jclk_t": 100.0 - 300.1,
          "jclk_last_rate": 0.001}
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.001, dr_state=ds)
    check(
        clk == 7.60 and line.endswith("-> REFUSED (bounds 5.0 chips / 0.50 sigma)"),
        "300.1 s after the adoption: REFUSED, legacy clock (%s)" % line,
    )
    # No adoption behind it: REFUSED.
    clk, line = _consume(L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.001)
    check(clk == 7.60 and "-> REFUSED (bounds" in line, "no prior adoption: REFUSED")
    # --joint-clk-coast-max-s 0: nothing coasts.
    ds = {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0, "jclk_last_rate": 0.001}
    clk, line = _consume(
        L2C_RATE, L2C_LEN, 7.60, 160.2, clk_rate=0.001, dr_state=ds, coast_max_s=0.0
    )
    check(clk == 7.60 and "-> REFUSED (bounds" in line, "coast window 0: REFUSED")
    # An adoption stamps all four; a coast then ages against jclk_last_t, not jclk_t, and
    # runs on the stamped rate (+0.002) while the state's has gone to +0.5.
    ds = {}
    clk, line = _consume(
        REF_RATE, REF_LEN, 150.238, 150.435, clk_rate=0.002, dr_state=ds, t0=100.0
    )
    check(
        "ADOPTED" in line
        and ds["jclk_t"] == 100.0
        and ds["jclk_last_t"] == 100.0
        and ds["jclk_last"] == 150.435
        and ds["jclk_last_rate"] == 0.002,
        "an adoption stamps jclk_t, jclk_last, jclk_last_t and jclk_last_rate",
    )
    clk, line = _consume(
        REF_RATE, REF_LEN, 150.238, 170.0, clk_rate=0.5, dr_state=ds, t0=399.0
    )
    check(
        abs(clk - (150.435 + 0.002 * 299.0)) < 1e-9
        and "COAST" in line
        and "at clk_rate +0.0020" in line
        and ds["jclk_t"] == 100.0,
        "299 s on: COAST = adopted level + stamped rate x 299 s, jclk_t still 100 "
        "(the freshness ages); the state's +0.5 is not used",
    )
    # A state not fit to consume (thin: 3 sats under joint_min_sats 4) coasts on the same
    # stamps inside the same window and bound, and says why in its own line.
    for n, why in ((3, "3 sats < 4"), (None, "absent, deaf or under joint_min_sats")):
        js = _FakeJoint(160.2, 0.047, n=n or 1, clk_rate=0.5)
        rx = Receiver(log=lambda m: None)
        rx._joint["__receiver__"] = js
        ds = {"jclk_last": 150.2, "jclk_last_t": 60.0, "jclk_t": 60.0,
              "jclk_last_rate": 0.001}
        ctx = types.SimpleNamespace(
            args=types.SimpleNamespace(
                joint_min_sats=4,
                joint_clk_max_chips=MAX_CHIPS,
                joint_clk_max_sigma=MAX_SIGMA,
                joint_clk_max_step_chips=0.3,
                joint_clk_step_hold_s=30.0,
                joint_clk_coast_max_s=300.0,
                chip_rate_hz=L2C_RATE,
            ),
            rx=rx,
            band_id="test",
            code_len=L2C_LEN,
            t0=100.0,
            dr_state=ds,
            drp=types.SimpleNamespace(clk_now=7.60),
            joint_state=(lambda _r, _b, _a: js) if n else (lambda _r, _b, _a: None),
        )
        lines = []
        _orig = deadreckon._log_rl
        deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
        try:
            deadreckon.dr_joint_clk(ctx)
        finally:
            deadreckon._log_rl = _orig
        check(
            abs(ctx.drp.clk_now - 7.512) < 1e-9
            and len(lines) == 1
            and ("joint state not fit to consume (%s) -> COAST" % why) in lines[0]
            and ds["jclk_t"] == 60.0,
            "a state %s coasts to 7.512 CM on the stamps, nothing restamped (%s)"
            % (why, lines[0][:80] if lines else "no line"),
        )


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
    # The unit of the state the chain reads (deadreckon.joint_read): joint_unit(_jw).
    return (
        len(blocks),
        divs,
        any(
            "deadreckon.joint_read(_ctx)" in ast.unparse(b)
            and "rx.joint_unit(_jw)" in ast.unparse(b)
            for b in blocks
        ),
    )


def test_rate_consumer_unit():
    """joint-consume 'rate': clk_rate is in joint chips/s, so the fractional rate is clk_rate
    over the joint chip rate; cp_rate_from_code_bias then multiplies by our chip rate. Dividing
    by our own rate instead would seed L2C 20x and E6 2x the code rate every other chain gets."""
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
        cph = cp_rate_from_code_bias(-0.0002 / REF_RATE, hps, rate)
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
    """The per-PRN SEED-OFFSET comparison converts like the consumer. Compared in raw chips,
    every gps_l2c line would read ~+142.8 and every gal_e6 line ~+75.7 chips and be refused,
    while both chains adopt the joint clock."""
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
    # Two live cases: joint +150.310 vs legacy +7.529 (L2C PRN 18), +150.850 vs +75.148
    # (E6 PRN 10). Both are a few tenths of a reference chip apart, not 142.8 / 75.7.
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
        "the unconverted comparison would leave both REFUSED (+142.781, +75.702)",
    )


class _PredJoint(_FakeJoint):
    """A joint state that also predicts each satellite's offset (clk + b_i), as the SEED-OFFSET
    summary reads it."""

    def __init__(self, pred, sigma=0.05):
        _FakeJoint.__init__(self, 150.2, sigma, n=0)
        self._pred = dict(pred)
        self._idx = {k: i for i, k in enumerate(self._pred)}

    def predicted(self, key):
        return self._pred[key]


def _seed_offset_summary(joint_pred, legacy_clk, rate=REF_RATE, code_len=REF_LEN):
    """Drive dr_seed once with an empty model sky, so only the clock and the 30-s SEED-OFFSET
    summary run; returns the summary line. The seed modulus is L5's 20 code periods."""
    rx = Receiver(log=lambda m: None)
    js = _PredJoint({("G", p): v for p, v in joint_pred.items()})
    rx._joint["__receiver__"] = js
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(chip_rate_hz=rate, mask_deg=5.0),
        rx=rx,
        band_id="test",
        code_len=code_len,
        dr_state={"clk": legacy_clk, "clk_t": 100.0, "seeded": set()},
        drp=types.SimpleNamespace(
            drift=0.0, now_w=100.0, la=0.0, tag="G", mod=20 * code_len, pd={}
        ),
        joint_consume=(),
        joint_state=lambda _rx, _b, _a: js,
        bsat=types.SimpleNamespace(get=lambda p, t: 0.0),
        dr_min_prn=0,
        capable=None,
    )
    lines = []
    _orig = deadreckon._log, deadreckon._log_rl
    deadreckon._log = lambda msg: lines.append(msg)
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        deadreckon.dr_seed(ctx)
    finally:
        deadreckon._log, deadreckon._log_rl = _orig
    return ([m for m in lines if m.startswith("SEED-OFFSET")] or [""])[0]


def test_seed_offset_summary_wrap():
    """The SEED-OFFSET summary wraps joint-minus-legacy in the window both are defined in,
    not at the seed modulus: a legacy clock just past a code wrap and a joint offset just
    before it are 0.2 chips apart, not a whole code period."""
    line = _seed_offset_summary({3: 10229.9, 5: 10230.0 - 0.05}, 0.1)
    check(
        "median +0.200 chips" in line and "PRN3 -0.20" in line,
        "10.23 Mcps, legacy 0.1 vs joint 10229.9 across the wrap: -0.20 (%s)" % line,
    )
    line = _seed_offset_summary({3: 151.0}, 150.2)
    check("PRN3 +0.80" in line, "...and +0.80 away from the wrap (%s)" % line)


def _consumer_names_tested(path):
    """The string constants `'<name>' in [ctx.]joint_consume` tests for in `path`."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    return {
        n.left.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare)
        and isinstance(n.left, ast.Constant)
        and len(n.ops) == 1
        and isinstance(n.ops[0], ast.In)
        and ast.unparse(n.comparators[0]) in ("joint_consume", "ctx.joint_consume")
    }


def test_consume_names():
    """--joint-consume refuses a name no consumer tests for: any name runs the joint solve,
    so a typo ('clock', 'clk,rat') would run it and consume nothing."""
    from gnss_broker import cli

    parse = getattr(cli, "parse_joint_consume", None)
    check(parse is not None, "cli.parse_joint_consume exists")
    if parse is None:
        return
    check(parse("clk,rate") == {"clk", "rate"}, "'clk,rate' -> {clk, rate}")
    check(parse(" rate , slew ") == {"rate", "slew"}, "' rate , slew ' -> {rate, slew}")
    check(parse("") == set(), "'' -> no consumer")
    for bad, name in (("clock", "'clock'"), ("clk,rat", "'rat'")):
        try:
            parse(bad)
            msg = None
        except SystemExit as e:
            msg = str(e)
        check(
            msg is not None and name in msg,
            "%r is refused at startup, naming %s (%s)" % (bad, name, msg),
        )
    tested = _consumer_names_tested(BROKER) | _consumer_names_tested(DEADRECKON)
    check(
        tested == set(cli.JOINT_CONSUMERS),
        "the names the consumers test for are JOINT_CONSUMERS (%s vs %s)"
        % (sorted(tested), sorted(cli.JOINT_CONSUMERS)),
    )
    with open(BROKER) as f:
        tree = ast.parse(f.read(), BROKER)
    sites = [
        ast.unparse(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "joint_consume" for t in n.targets)
    ]
    check(
        sites == ["parse_joint_consume(args.joint_consume)"],
        "the broker builds joint_consume only through parse_joint_consume (%s)" % sites,
    )


_FEED = {3: +0.6, 5: -0.4, 14: +0.2, 21: -0.9, 30: +0.5}  # b_sat per fed PRN, chips
TRUE_CLK = 150.2  # reference chips


def _shadow_ctx(rx, warmup_s=0.0, clk=0.0):
    """A 10.23-Mcps chain dr_joint_shadow runs on: every detection above joint_min_snr and
    tracked, no P2C hold, the JOINT[shadow] summary line off. warmup_s 1e18 keeps the feed
    withheld (the broker started just now)."""
    return types.SimpleNamespace(
        args=types.SimpleNamespace(
            joint_shadow=True,
            chip_rate_hz=REF_RATE,
            joint_feed_warmup_s=warmup_s,
            joint_min_snr=10.0,
            joint_mask_after=10,
            joint_sigma=0.3,
        ),
        band_id="1176.45MHz",
        code_len=REF_LEN,
        rx=rx,
        dr_state={"clk": clk, "joint_log_next": 1e18},
        uptime_s=0.0,
        minnov_hist={},
        track_ok=lambda p: True,
        p2c_hold=lambda js, key: False,
        p2c_tick=lambda js, t: None,
    )


def _shadow_pass(ctx, t, offs):
    """One broker pass. offs is the search's table as this pass sees it, {prn: (ref_hop,
    offset in chips)}: a PRN it re-detected carries a new ref_hop, one it re-serves the old."""
    ctx.t0 = t
    ctx.best = {p: (100.0, 0.0, 0.0, h) for p, (h, _) in offs.items()}
    ctx.drp = types.SimpleNamespace(
        offs=[(p, d) for p, (_, d) in sorted(offs.items())],
        tag="G",
        t_now_abs=t,
        now_w=t,
    )
    deadreckon.dr_joint_shadow(ctx)


def test_feed_once_per_detection():
    """dr_joint_shadow feeds the joint state each detection once. The search re-serves a
    detection, with its ref_hop, until its next pass; fed at every pass, one detection would
    count as many in _n (joint_mask_after's count), and a re-served outlier would fill the
    escape's rejection run with copies of itself, which agree perfectly."""
    logs = []
    _orig = deadreckon._log_rl
    deadreckon._log_rl = lambda key, msg, every_s=10.0: logs.append(msg)
    try:
        rx = Receiver(log=lambda m: None)
        ctx = _shadow_ctx(rx)
        t, h = 1000.0, 0
        for _ in range(40):  # every PRN re-detected at every pass
            t, h = t + 2.0, h + 1000
            _shadow_pass(ctx, t, {p: (h, TRUE_CLK + b) for p, b in _FEED.items()})
        js = rx.joint_receiver("1176.45MHz", REF_LEN, REF_RATE, gauge_mode="prior")
        n0 = sorted(js._n.values())
        err = abs(js.predicted(("G", 3)) - (TRUE_CLK + _FEED[3]))
        check(
            n0 == [40] * len(_FEED) and err < 0.3,
            "a new detection at each of 40 passes: 40 updates per PRN (%s), PRN 3's "
            "clk + b within %.3f chips of the truth" % (n0, err),
        )
        h3 = h + 1  # PRN 3's last detection, re-served for 10 passes
        for _ in range(10):
            t, h = t + 2.0, h + 1000
            offs = {p: (h, TRUE_CLK + b) for p, b in _FEED.items()}
            offs[3] = (h3, TRUE_CLK + _FEED[3])
            _shadow_pass(ctx, t, offs)
        check(
            js._n[("G", 3)] == 41 and js._n[("G", 5)] == 50,
            "one detection re-served for 10 passes counts once (n %d, 41), a PRN "
            "re-detected at each pass ten times (n %d, 50)"
            % (js._n[("G", 3)], js._n[("G", 5)]),
        )
        del logs[:]
        rej0, h5 = js.rejected, h + 1
        for _ in range(10):  # PRN 5 detected 30 chips off, re-served for 10 passes
            t, h = t + 2.0, h + 1000
            offs = {p: (h, TRUE_CLK + b) for p, b in _FEED.items()}
            offs[5] = (h5, TRUE_CLK + _FEED[5] + 30.0)
            _shadow_pass(ctx, t, offs)
        esc = [m for m in logs if "ESCAPE" in m]
        check(
            js.rejected - rej0 == 1 and not esc,
            "a 30-chip outlier re-served for 10 passes is one rejection and no escape "
            "(%d rejection(s); %s)" % (js.rejected - rej0, esc),
        )
    finally:
        deadreckon._log_rl = _orig


def _joint_receiver_calls(path):
    """(line, keyword names) of every joint_receiver(...) call in `path`."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    return [
        (n.lineno, [k.arg for k in n.keywords])
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("joint_receiver")
    ]


def _drain_before_shadow(path=DEADRECKON):
    """The test of the If statement right before `dr_joint_shadow(ctx)` in
    stage_dead_reckon, when that If calls joint_receiver(), else None."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "stage_dead_reckon"
    ]
    for n in ast.walk(fn[0]) if fn else ():
        for body in (getattr(n, "body", None), getattr(n, "orelse", None)):
            if not isinstance(body, list):
                continue
            for i, st in enumerate(body):
                if (
                    i > 0
                    and isinstance(st, ast.Expr)
                    and ast.unparse(st.value) == "dr_joint_shadow(ctx)"
                    and isinstance(body[i - 1], ast.If)
                    and _joint_receiver_in(body[i - 1])
                ):
                    return ast.unparse(body[i - 1].test)
    return None


def _joint_receiver_in(node):
    return any(
        isinstance(c, ast.Call) and ast.unparse(c.func).endswith("joint_receiver")
        for c in ast.walk(node)
    )


def test_state_born_at_zero():
    """The receiver-wide joint state is created by whichever caller runs first, and no broker
    caller passes clk0, so it is born at clk 0. On the live path stage_dead_reckon's notes
    drain creates it, right before dr_joint_shadow and under a superset of its condition, so
    the shadow never warm-started it: with no clk0 of its own the live path is unchanged."""
    pkg = os.path.dirname(DEADRECKON)
    srcs = [BROKER] + sorted(
        os.path.join(pkg, f)
        for f in os.listdir(pkg)
        if f.endswith(".py") and not f.startswith("test_") and f != "selftest.py"
    )
    warm = [
        "%s:%d" % (os.path.basename(p), ln)
        for p in srcs
        for ln, kws in _joint_receiver_calls(p)
        if "clk0" in kws
    ]
    check(not warm, "no broker joint_receiver() call passes clk0 (%s)" % warm)
    drain = _drain_before_shadow()
    with open(DEADRECKON) as f:
        tree = ast.parse(f.read(), DEADRECKON)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "dr_joint_shadow"
    ][0]
    cond = ast.unparse([st for st in fn.body if isinstance(st, ast.If)][0].test)
    check(
        drain == "ctx.args.rrate_state or ctx.args.joint_shadow"
        and cond == "ctx.args.joint_shadow and ctx.drp.offs",
        "the notes drain creates the state right before dr_joint_shadow, under a "
        "superset of its condition (%r / %r)" % (drain, cond),
    )
    _orig = deadreckon._log_rl
    deadreckon._log_rl = lambda key, msg, every_s=10.0: None
    try:
        # As on the live path: drained first, then the shadow, inside its warmup so it
        # feeds nothing, on a chain whose dead-reckon clock reads 151.7 chips.
        rx = Receiver(log=lambda m: None)
        rx.joint_receiver("1176.45MHz", REF_LEN, REF_RATE, gauge_mode="prior")
        _shadow_pass(_shadow_ctx(rx, warmup_s=1e18, clk=151.7), 1000.0, {3: (1, 151.7)})
        js = rx._joint.get("__receiver__")
        check(
            js.clk == 0.0, "drained first, then the shadow: born at clk %r (0)" % js.clk
        )
        rx = Receiver(log=lambda m: None)
        _shadow_pass(_shadow_ctx(rx, warmup_s=1e18, clk=151.7), 1000.0, {3: (1, 151.7)})
        js = rx._joint.get("__receiver__")
        check(
            js is not None and js.clk == 0.0,
            "the shadow creating it itself: born at clk %r (0), no warm start"
            % (js.clk if js is not None else None),
        )
    finally:
        deadreckon._log_rl = _orig


def main():
    for fn in (
        test_consume_names,
        test_live_numbers_convert,
        test_consumer_adopts_l2c_and_e6,
        test_step_refused_in_both_units,
        test_sigma_bound_is_reference_chips,
        test_windows,
        test_ten_mcps_unchanged,
        test_one_unit,
        test_coast_on_refusal,
        test_rate_consumer_unit,
        test_seed_offset_per_prn,
        test_seed_offset_summary_wrap,
        test_feed_once_per_detection,
        test_state_born_at_zero,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
