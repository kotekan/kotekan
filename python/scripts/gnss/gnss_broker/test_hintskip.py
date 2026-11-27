"""--hint-skip-dr, its handover, and the dead-reckon `joint-consume: rate` consumer.

    python3 -m gnss_broker.test_hintskip

dr_can_seed says whether a chain can dead-reckon a PRN (fresh clock, a model clock known to
half a chip, not untrusted); with --hint-skip-dr, stage_narrow_search stops hinting a PRN that
is also locked (dr_owns_unhinted), and dr_seed takes over its search seed and slews it.
dr_joint_rate turns the joint state's clk_rate into (l-a) through the state's unit, under
joint_min_sats and joint_max_rate_ppm.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import deadreckon, searchhint
from gnss_broker.test_jointsource import make_rx, run_dr_seed

_fails = []

T0 = 1.79e9
CHIP_HZ = 10.23e6


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_ctx():
    """GPS PRN 5 on a broadcast orbit, PRN 7 untrusted, and GuoWang-like PRN 9 on a catalogue
    record whose clock is known to 1 us; the clock solved 10 s ago."""
    pd = {
        ("G", 5): {"el": 40.0},
        ("G", 7): {"el": 50.0},
        ("G", 9): {"el": 30.0, "clk_sigma_s": 1e-6},
    }
    return _Obj(
        args=_Obj(
            narrow_search=True,
            almanac=True,
            hint_skip_dr=True,
            search_margin_hz=50.0,
            search_margin_wide_hz=500.0,
            nh_hint=False,
            chip_rate_hz=CHIP_HZ,
            joint_min_sats=4,
            joint_max_rate_ppm=0.01,
            lock_q=2.2,
            lock_prompt_hold=3.0,
        ),
        pred={p: (-1000.0 + p, 0.0, 30.0, 2.2e7, 0.0) for p in (5, 7, 9, 11)},
        up=None,
        unpredicted=set(),
        warm={"i": 0, "t": float("-inf"), "cur": []},
        capable=None,
        cb=_Obj(ema=0.0, stale=False, value=0.0, code_ema=None),
        dr_state={"clk": 100.0, "clk_meas_t": T0 - 10.0},
        drp=_Obj(pd=pd, tag="G"),
        dr_min_prn=1,
        dr_untrusted={7: "model residual"},
        hold=_Obj(q={5: 3.0, 7: 3.0, 9: 3.0, 11: 3.0}, prev={}),
        joint_consume=set(),
        detectors=["d0"],
        t0=T0,
    )


def hinted(ctx):
    """The PRNs stage_narrow_search posts, and its log lines."""
    posts, lines = [], []
    saved = (searchhint._post, searchhint._log, searchhint._log_rl)
    searchhint._post = lambda url, body: posts.append(body)
    searchhint._log = lines.append
    searchhint._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        searchhint.stage_narrow_search(ctx)
    finally:
        searchhint._post, searchhint._log, searchhint._log_rl = saved
    return sorted(h["prn"] for h in posts[0]), lines


def joint_rate(n_sats, clk_rate, ref_hz=CHIP_HZ):
    """dr_joint_rate on a stub joint state of n_sats satellites at clk_rate (joint chips/s)."""
    ctx = make_ctx()
    state = _Obj(_idx=[("G", p) for p in range(n_sats)], clk_rate=clk_rate)
    ctx.rx = _Obj(joint_unit=lambda which: (ref_hz, 1.0))
    saved = (deadreckon.joint_read, deadreckon._log_rl)
    deadreckon.joint_read = lambda c: (state, "trk", "[trk]")
    deadreckon._log_rl = lambda key, msg, every_s=10.0: None
    try:
        return deadreckon.dr_joint_rate(ctx)
    finally:
        deadreckon.joint_read, deadreckon._log_rl = saved


def handover(q, clk_sigma_s=0.0, best=None):
    """A model-born PRN 3 turned into a locked, held search seed on a chain with a search, then
    one dr_seed pass with --hint-skip-dr: (ctx, log lines, the seed's phase before)."""
    argkw = {"lock_q": 2.2, "hint_skip_dr": True}
    _o, _c, _l, ctx = run_dr_seed(make_rx("fed"), "trk", {3: 0.0, 9: 0.0}, argkw=argkw)
    ctx.dr_state["seeded"].discard(3)
    ctx.cp_held = {3}
    ctx.detectors = ["d0"]
    ctx.best = best or {}
    ctx.hold.q[3] = q
    ctx.dr_state["clk_meas_t"] = ctx.t0
    ctx.drp.pd[("G", 3)]["clk_sigma_s"] = clk_sigma_s
    before = ctx.seeds[3]["code_phase_at_ref_chips"]
    lines = []
    saved = deadreckon._log, deadreckon._log_rl, deadreckon.seed_phase_at_ref
    deadreckon._log = lines.append
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    deadreckon.seed_phase_at_ref = lambda ph, *a: ph
    try:
        deadreckon.dr_seed(ctx)
    finally:
        deadreckon._log, deadreckon._log_rl, deadreckon.seed_phase_at_ref = saved
    return ctx, lines, before


def handover_checks():
    ctx, lines, _b = handover(3.0)
    check(
        3 in ctx.dr_state["seeded"] and 3 not in ctx.cp_held,
        "a locked, unhinted search seed is taken over and its hold released",
    )
    check(any("HANDOVER PRN 3" in m for m in lines), "...and the log says so")
    check(deadreckon.seeds_are_model_owned(ctx, 3), "...and the model owns it")
    slew = [m for m in lines if m.startswith("dead-reckon SLEW PRN 3:")]
    check(
        slew
        and "(cap 0.05)" in slew[0]
        and ctx.seeds[3].owners().startswith("dop=dr_slew cp0=dr_slew"),
        "...and it is slewed at the cap, not re-born (%s)" % (slew[:1],),
    )
    ctx, lines, _b = handover(1.4)
    check(3 not in ctx.dr_state["seeded"], "an unlocked one stays the search's")
    ctx, lines, _b = handover(3.0, clk_sigma_s=1e-6)
    check(3 not in ctx.dr_state["seeded"], "so does one the model cannot seed")
    ctx, lines, _b = handover(3.0, best={3: (20.0,)})
    check(3 not in ctx.dr_state["seeded"], "and one detected this cycle")


def drift_checks():
    ctx = make_ctx()
    ctx.args.dr_drift_source = "bias"
    ctx.args.carrier_hz = 1176.45e6
    ctx.cb.ema = 1.5
    ctx.dr_state["drift"] = 0.02
    ctx.drp.la = 0.0
    ctx.rx = _Obj(joint_unit=lambda which: (CHIP_HZ, 1.0))
    state = _Obj(_idx=[("G", p) for p in range(6)], clk_rate=0.001)
    saved = (deadreckon.joint_read, deadreckon._log_rl)
    deadreckon.joint_read = lambda c: (state, "trk", "[trk]")
    deadreckon._log_rl = lambda key, msg, every_s=10.0: None
    try:
        for consume, want, what in (
            (
                set(),
                1.5 * CHIP_HZ / 1176.45e6,
                "the carrier bias (--dr-drift-source bias)",
            ),
            ({"rate"}, 1.5 * CHIP_HZ / 1176.45e6, "rate alone leaves it"),
            ({"clk", "rate"}, 1.5 * CHIP_HZ / 1176.45e6, "clk,rate, no --dr-drift-joint"),
        ):
            ctx.joint_consume = consume
            got = deadreckon.dr_drift(ctx)
            check(abs(got - want) < 1e-12, "%s (%.6f chips/s)" % (what, got))
        ctx.args.dr_drift_joint = True
        ctx.joint_consume = {"clk", "rate"}
        got = deadreckon.dr_drift(ctx)
        check(abs(got - 0.001) < 1e-12, "--dr-drift-joint: the joint rate (%.6f)" % got)
        ctx.args.dr_drift_joint = False
        ctx.args.dr_drift_source = "solve"
        ctx.joint_consume = set()
        check(deadreckon.dr_drift(ctx) == 0.02, "the measured drift otherwise")
    finally:
        deadreckon.joint_read, deadreckon._log_rl = saved


def main():
    print("\ndr_can_seed")
    ctx = make_ctx()
    check(deadreckon.dr_can_seed(ctx, 5), "a broadcast orbit on a fresh clock")
    check(not deadreckon.dr_can_seed(ctx, 7), "not an untrusted PRN")
    check(not deadreckon.dr_can_seed(ctx, 9), "not a clock known to 10 chips")
    check(not deadreckon.dr_can_seed(ctx, 11), "not a PRN the model does not predict")
    ctx.dr_state["clk_meas_t"] = T0 - 400.0
    check(not deadreckon.dr_can_seed(ctx, 5), "not on a clock 400 s stale")
    ctx.dr_state["jclk_t"] = T0 - 5.0
    check(deadreckon.dr_can_seed(ctx, 5), "a joint-clock adoption keeps it fresh")
    ctx.dr_state["clk"] = None
    check(not deadreckon.dr_can_seed(ctx, 5), "not before the clock is solved")

    print("\n--hint-skip-dr")
    prns, lines = hinted(make_ctx())
    check(prns == [7, 9, 11], "hints only what the model cannot seed (%s)" % prns)
    check(
        any("1 locked and dead-reckoned" in m for m in lines),
        "the log counts the skipped PRN",
    )
    ctx = make_ctx()
    ctx.hold.q[5] = 1.4
    check(hinted(ctx)[0] == [5, 7, 9, 11], "an unlocked PRN stays hinted")
    ctx = make_ctx()
    ctx.hold.q[5], ctx.hold.prev[5] = 1.4, 8.0
    check(
        hinted(ctx)[0] == [7, 9, 11], "q under lock_q but the prompt held: not hinted"
    )
    ctx = make_ctx()
    del ctx.hold.q[5]
    ctx.dr_state["seeded"] = set()
    check(hinted(ctx)[0] == [5, 7, 9, 11], "no fleet reading, not yet owned: hinted")
    ctx.dr_state["seeded"] = {5}
    check(hinted(ctx)[0] == [7, 9, 11], "no fleet reading, already owned: kept")
    ctx = make_ctx()
    ctx.joint_consume = {"clk", "rate", "slew"}
    ctx.args.joint_source = "trk"
    ctx.dr_state["jsrc"] = {"mode": "search", "fresh_t": None}
    check(
        hinted(ctx)[0] == [5, 7, 9, 11],
        "on the tracker-fed clock but reading the search-fed state: everything hinted",
    )
    ctx.dr_state["jsrc"]["mode"] = "trk"
    check(hinted(ctx)[0] == [7, 9, 11], "...and once it reads TRK_STATE, the skip")
    ctx = make_ctx()
    ctx.args.hint_skip_dr = False
    check(hinted(ctx)[0] == [5, 7, 9, 11], "off: every PRN is hinted")
    ctx = make_ctx()
    ctx.dr_state["clk"] = None
    check(hinted(ctx)[0] == [5, 7, 9, 11], "no clock: every PRN is hinted")

    print("\nhandover")
    handover_checks()

    print("\ndr_joint_rate")
    la = joint_rate(6, 0.001)
    check(
        la is not None and abs(la - 0.001 / CHIP_HZ) < 1e-18,
        "clk_rate / unit (%r)" % la,
    )
    la = joint_rate(6, 0.001, ref_hz=2 * CHIP_HZ)
    check(
        la is not None and abs(la - 0.0005 / CHIP_HZ) < 1e-18,
        "through the state's unit, not this chain's chip rate",
    )
    check(joint_rate(3, 0.001) is None, "refused under joint_min_sats")
    check(
        joint_rate(6, 0.2 * CHIP_HZ * 1e-6) is None, "refused beyond joint_max_rate_ppm"
    )

    print("\ndr_drift")
    drift_checks()

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: --hint-skip-dr and the dead-reckon rate consumer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
