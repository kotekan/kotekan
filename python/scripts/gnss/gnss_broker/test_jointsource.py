"""--joint-source: which joint state a chain's consumers read (#191 step 3).

    python3 -m gnss_broker.test_jointsource     (from python/scripts/gnss)

docs/CHORD_JOINT_TRACKING.md 6.3 step 3. Two JointReceiverStates live in the Receiver:
JOINT_STATE (gps_l5's search detections feed it) and TRK_STATE (the tracker residuals,
jtrk.py). A chain with --joint-source trk has every consumer (clk, rate, slew, and the
JOINT-CLK / SEED-OFFSET lines) read TRK_STATE in TRK_STATE's unit; the SatBiasFilter no longer
moves the seed of a satellite the slew drives from it; a satellite TRK_STATE does not hold keeps
clk_now + b_sat. While TRK_STATE is unfit or stale (its feed held through a transit) the chain
reads JOINT_STATE exactly as a search chain does, and returns after --joint-trk-return-s fresh;
with neither fit it is refused and coasted on, never raised into the loop.

The broker's own code is exercised, not a copy of it: the `_joint_state` closure, the rate
consumer block and the startup line are lifted out of gps_distributed_broker.main by the AST
and run as written; deadreckon.dr_seed runs whole (clock consumer, SEED-OFFSET summary, the
per-satellite slew offset, the birth), with every log line formatted.
"""

import ast
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import cli, deadreckon, receiver  # noqa: E402
from gnss_broker.receiver import JOINT_STATE, TRK_STATE, Receiver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")

REF = 10.23e6
L = 10230.0
MOD = 20 * L
HPS = 195312.5
CARR = 1176.45e6
BAND = "1176.45MHz"
LEGACY_CLK = 150.0
SEARCH_CLK, SEARCH_B = 150.2, {3: +0.6, 5: -0.4, 9: +0.3, 14: +0.2, 21: -0.9, 30: +0.5}
TRK_CLK, TRK_B = 150.9, {3: +0.1, 5: -0.2, 14: 0.0, 21: +0.3, 30: -0.1}  # no PRN 9
MODEL = 1000.0  # cp_predicted, constant: a born seed's phase is MODEL + its offset
# The chain's cycle time: 12 s after make_rx's last feed (78 s), so TRK_STATE is fresh.
T_NOW = 90.0

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


# -- the broker's own code, lifted from main() --------------------------------------------
def _main_tree():
    with open(BROKER) as f:
        tree = ast.parse(f.read(), BROKER)
    return [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    ][0]


def _run_nodes(nodes, ns):
    exec(compile(ast.Module(body=list(nodes), type_ignores=[]), BROKER, "exec"), ns)
    return ns


def broker_joint_state():
    """gps_distributed_broker.main's _joint_state, as written, with CODE_LEN = L."""
    fn = [
        n
        for n in ast.walk(_main_tree())
        if isinstance(n, ast.FunctionDef) and n.name == "_joint_state"
    ]
    assert len(fn) == 1, "one _joint_state in main"
    return _run_nodes(fn, {"receiver": receiver, "CODE_LEN": L})["_joint_state"]


def _if_block(test_src):
    blocks = [
        n
        for n in ast.walk(_main_tree())
        if isinstance(n, ast.If) and ast.unparse(n.test) == test_src
    ]
    assert len(blocks) == 1, "one block `if %s` in main (%d)" % (test_src, len(blocks))
    return blocks[0]


def run_rate_consumer(rx, args, joint_consume, code_ema, t_now=None, dr_state=None):
    """main's `if 'rate' in joint_consume:` block as written: (cb_to_seed, log lines). The
    chain's context carries what deadreckon.joint_read reads: the broker's _joint_state, the
    axis time (T_NOW by default) and dr_state."""
    lines = []
    ctx = types.SimpleNamespace(
        args=args,
        rx=rx,
        band_id=BAND,
        joint_state=broker_joint_state(),
        dr_state={} if dr_state is None else dr_state,
        drp=types.SimpleNamespace(t_now_abs=T_NOW if t_now is None else t_now),
    )
    ns = {
        "joint_consume": joint_consume,
        "_joint_state": broker_joint_state(),
        "_ctx": ctx,
        "deadreckon": deadreckon,
        "rx": rx,
        "band_id": BAND,
        "args": args,
        "receiver": receiver,
        "_cb": types.SimpleNamespace(code_ema=code_ema),
        "cb_to_seed": code_ema,
        "_log_rl": lambda key, msg, every_s=10.0: lines.append(msg),
    }
    saved = deadreckon._log, deadreckon._log_rl
    deadreckon._log = lines.append
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        _run_nodes([_if_block("'rate' in joint_consume")], ns)
    finally:
        deadreckon._log, deadreckon._log_rl = saved
    return ns["cb_to_seed"], lines


def run_startup_line(args, joint_consume):
    lines = []
    ns = {"joint_consume": joint_consume, "args": args, "_log": lines.append}
    _run_nodes([_if_block("joint_consume or args.joint_source != 'search'")], ns)
    return lines


# -- the receiver and the chain ------------------------------------------------------------
# The parser's defaults for the settled gate, so the tests run what a chain runs.
_DEF = cli.build_parser("x").parse_args(["--trackers", "t"])
# TRK_STATE as its feeder builds it (jtrk.stage_joint_trk).
TRK_KW = dict(gauge_mode="prior", escape_clock_min_sats=3)


def _feed(js, clk, b, n=40, t_start=0.0, step=None):
    """n cycles 2 s apart from t_start; `step` {prn: chips} added to those satellites."""
    for i in range(n):
        js.cycle(
            [
                (("G", p), clk + v + (step or {}).get(p, 0.0), 0.1)
                for p, v in sorted(b.items())
            ],
            t_start + 2.0 * i,
        )


def make_rx(trk="fed", search="fed"):
    """A Receiver with the search-fed state converged on SEARCH_* ("fed") or on two satellites,
    under joint_min_sats ("thin"), and the tracker-fed one converged on TRK_* ("fed"), born
    and never fed ("empty"), deaf ("deaf"), or absent. Both are fed 0..78 s, so at T_NOW the
    tracker-fed one is fresh (--joint-trk-stale-s)."""
    rx = Receiver(log=lambda m: None)
    _feed(
        rx.joint_receiver(BAND, L, REF, gauge_mode="prior"),
        SEARCH_CLK,
        SEARCH_B if search == "fed" else {3: SEARCH_B[3], 9: SEARCH_B[9]},
    )
    if trk in ("fed", "deaf"):
        js = rx.joint_receiver(BAND, L, REF, which=TRK_STATE, **TRK_KW)
        _feed(js, TRK_CLK, TRK_B)
        js.deaf = trk == "deaf"
    elif trk == "empty":
        rx.joint_receiver(BAND, L, REF, which=TRK_STATE, **TRK_KW)
    return rx


def chain_args(source, **kw):
    a = dict(
        joint_source=source,
        hops_per_sec=HPS,
        chip_rate_hz=REF,
        carrier_hz=CARR,
        code_doppler_sign=1.0,
        doppler_sign=1.0,
        mask_deg=5.0,
        lock_q=0.0,
        lock_prompt_hold=1e9,
        lock_snr=3.0,
        detectors_monitor=False,
        dr_repin_s=600.0,
        seed_phase_transport=True,
        joint_min_sats=4,
        joint_clk_max_sigma=0.5,
        joint_clk_max_chips=5.0,
        joint_clk_max_step_chips=0.3,
        joint_clk_step_hold_s=30.0,
        joint_clk_coast_max_s=300.0,
        joint_slew_max_chips=5.0,
        joint_max_rate_ppm=0.01,
        joint_trk_bsat_sigma_max=_DEF.joint_trk_bsat_sigma_max,
        joint_trk_escape_holdoff_s=_DEF.joint_trk_escape_holdoff_s,
        joint_trk_bsat_dev_max=_DEF.joint_trk_bsat_dev_max,
        joint_trk_stale_s=_DEF.joint_trk_stale_s,
        joint_trk_return_s=_DEF.joint_trk_return_s,
        signal="gal_e5a",
    )
    a.update(kw)
    return types.SimpleNamespace(**a)


def run_dr_seed(
    rx,
    source,
    bsat,
    consume=("clk", "slew"),
    prns=(3, 9),
    ds=None,
    t0=None,
    argkw=None,
):
    """dr_seed once on a 10.23-Mcps chain with no search, every PRN of `prns` born from the
    model: returns ({prn: offset added to the model}, clk_now, log lines, ctx). t0 is also the
    axis time (drp.t_now_abs), T_NOW unless given. ds seeds dr_state; its nested dicts (the
    joint_read mode) are carried by reference, so a run fed the last run's ctx.dr_state
    continues it."""
    t0 = T_NOW if t0 is None else t0
    args = chain_args(source, **(argkw or {}))
    if source is None:
        del args.joint_source  # a namespace from before the flag (the older tests')
    ctx = types.SimpleNamespace(
        args=args,
        t0=t0,
        band_id=BAND,
        code_len=L,
        dr_min_prn=0,
        capable=None,
        best={},
        mp_flipped=set(),
        probe_set=set(),
        cp_held={},
        dr_untrusted={},
        detectors=[],
        seeds={},
        hold=types.SimpleNamespace(q={}, prev={}, low_hits={}),
        status={},
        sig_of=lambda st: 0.0,
        dr_eph_mod=types.SimpleNamespace(C_LIGHT=299792458.0),
        cb=types.SimpleNamespace(seed=0.0),
        cp_predicted=lambda v, t: MODEL,
        bsat=types.SimpleNamespace(get=lambda p, t: bsat.get(p, 0.0)),
        joint_consume=set(consume),
        joint_state=broker_joint_state(),
        rx=rx,
        dls=types.SimpleNamespace(trim={}, last={}),
        birth_steps={},
        dr_state=dict(ds or {}, clk=LEGACY_CLK, clk_t=t0, seeded=set(), pin={}),
        drp=types.SimpleNamespace(
            pd={("G", p): {"el": 40.0, "range_rate_mps": 0.0} for p in prns},
            pd2={},
            tag="G",
            mod=MOD,
            t_fc_abs=t0,
            t_now_abs=t0,
            now_w=t0,
            hold=False,
            drift=0.0,
            la=0.0,
            slew_cap=0.05,
            slew_k=0.5,
            clk_now=None,
        ),
    )
    lines = []
    saved = deadreckon._log, deadreckon._log_rl, deadreckon.seed_phase_at_ref
    deadreckon._log = lines.append
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    # The phase a birth ships is seed_phase_at_ref((model + offset) % mod, ...); passed
    # through unchanged, the seed carries model + offset itself.
    deadreckon.seed_phase_at_ref = lambda ph, *a: ph
    try:
        deadreckon.dr_seed(ctx)
    finally:
        deadreckon._log, deadreckon._log_rl, deadreckon.seed_phase_at_ref = saved
    offs = {
        p: ctx.seeds[p]["code_phase_at_ref_chips"] - MODEL
        for p in prns
        if p in ctx.seeds
    }
    return offs, ctx.drp.clk_now, lines, ctx


def _lines(lines, head):
    return [m for m in lines if m.startswith(head)]


# -- the tests -----------------------------------------------------------------------------
def test_flag_and_mapping():
    ap = cli.build_parser("x")
    a0 = ap.parse_args(["--trackers", "t"])
    a1 = ap.parse_args(["--trackers", "t", "--joint-source", "trk"])
    check(a0.joint_source == "search", "--joint-source defaults to 'search'")
    check(a1.joint_source == "trk", "--joint-source trk parses")
    try:
        ap.parse_args(["--trackers", "t", "--joint-source", "trak"])
        refused = False
    except SystemExit:
        refused = True
    check(refused, "a mistyped --joint-source is refused at startup")
    check(
        receiver.joint_which(a0) == JOINT_STATE
        and receiver.joint_which(types.SimpleNamespace()) == JOINT_STATE
        and receiver.joint_which(a1) == TRK_STATE,
        "joint_which: search / no flag -> JOINT_STATE, trk -> TRK_STATE",
    )
    check(
        a1.joint_trk_stale_s == 30.0 and a1.joint_trk_return_s == 30.0,
        "--joint-trk-stale-s and --joint-trk-return-s default to 30 s",
    )


def test_unit_per_state():
    """joint_unit names a state: the two moduli can differ (each shortens to its own callers'
    shortest code), and the trk consumer must convert in TRK_STATE's."""
    rx = Receiver(log=lambda m: None)
    rx.joint_receiver(BAND, 5115.0, 5.115e6, gauge_mode="prior")  # E6: 10230 ref chips
    rx.joint_receiver(
        BAND, 10230.0, 0.5115e6, which=TRK_STATE, gauge_mode="prior"
    )  # CM
    check(
        rx.joint_unit() == (REF, 10230.0)
        and rx.joint_unit(JOINT_STATE) == (REF, 10230.0)
        and rx.joint_unit(TRK_STATE) == (REF, 204600.0),
        "joint_unit() is the live state's; joint_unit(TRK_STATE) the shadow's (%s %s)"
        % (rx.joint_unit(), rx.joint_unit(TRK_STATE)),
    )
    check(
        Receiver(log=lambda m: None).joint_unit(TRK_STATE) is None,
        "...None while that state does not exist",
    )


def test_joint_state_closure():
    js_fn = broker_joint_state()
    rx = make_rx("fed")
    check(
        js_fn(rx, BAND, chain_args("search")) is rx._joint[JOINT_STATE],
        "search: the broker's _joint_state hands out JOINT_STATE",
    )
    check(
        js_fn(rx, BAND, chain_args("trk")) is rx._joint[TRK_STATE],
        "trk: the broker's _joint_state hands out TRK_STATE",
    )
    for kind in ("empty", "deaf", "absent"):
        rx = make_rx(kind)
        got = js_fn(rx, BAND, chain_args("trk"))
        check(
            (kind != "absent" or TRK_STATE not in rx._joint)
            and got is None
            and js_fn(rx, BAND, chain_args("search")) is rx._joint[JOINT_STATE],
            "trk state %s: refused (None), the search one still served" % kind,
        )
    rx = make_rx("fed")
    check(
        js_fn(rx, BAND, chain_args("trk"), which=JOINT_STATE) is rx._joint[JOINT_STATE]
        and js_fn(rx, BAND, chain_args("search"), which=None) is rx._joint[JOINT_STATE],
        "which= overrides --joint-source (joint_read's fallback); None keeps it",
    )
    boom = types.SimpleNamespace(
        joint_consumable=Receiver.joint_consumable.__get__(
            types.SimpleNamespace(joint_receiver=lambda *a, **k: 1 / 0)
        )
    )
    check(
        js_fn(boom, BAND, chain_args("trk")) is None, "a raising state lookup is None"
    )


def test_search_unchanged():
    """--joint-source search (and an args with no flag) reads JOINT_STATE exactly as before:
    the offsets are the legacy-based arithmetic bit for bit, b_sat inside the legacy base."""
    rx = make_rx("fed")
    js = rx._joint[JOINT_STATE]
    bsat = {3: 0.37, 9: -0.21}
    offs, clk_now, lines, _ = run_dr_seed(rx, "search", bsat)
    want_clk = (
        LEGACY_CLK + rx.joint_clk_delta(js.clk, REF, L, LEGACY_CLK, REF, L)[0]
    ) % L
    check(
        clk_now == want_clk, "clk consumer adopts JOINT_STATE's clock (%.4f)" % clk_now
    )
    for p in (3, 9):
        leg = clk_now + bsat[p]
        want = leg + rx.joint_clk_delta(js.predicted(("G", p)), REF, L, leg, REF, L)[0]
        want = (
            MODEL + want
        ) % MOD - MODEL  # through the seed's modulus, as dr_seed does
        check(
            offs.get(p) == want,
            "PRN %d: offset legacy + wrap(joint - legacy) bit for bit (%r vs %r)"
            % (p, offs.get(p), want),
        )
    jc = _lines(lines, "JOINT-CLK")
    so = _lines(lines, "SEED-OFFSET")
    check(
        len(jc) == 1 and jc[0].startswith("JOINT-CLK: legacy") and "ADOPTED" in jc[0],
        "JOINT-CLK line untagged and ADOPTED (%s)" % (jc[:1],),
    )
    check(
        so and all("[trk]" not in m and "vs clk" not in m for m in so),
        "SEED-OFFSET lines untagged, diff against the legacy offset (%d lines)"
        % len(so),
    )
    offs2, _c, _l, _ = run_dr_seed(rx, "search", {3: 0.0, 9: 0.0})
    check(
        abs(offs2[3] - offs[3]) < 1e-9,
        "search: the offset is the joint's whatever b_sat is (wrap of the delta)",
    )
    offs3, c3, l3, _ = run_dr_seed(make_rx("fed"), None, bsat)
    check(
        offs3 == offs and c3 == clk_now and l3 == lines,
        "an args with no joint_source: the same offsets, clock and lines",
    )


def test_trk_consumers_read_trk():
    rx = make_rx("fed")
    jt, js = rx._joint[TRK_STATE], rx._joint[JOINT_STATE]
    bsat = {3: 0.37, 9: -0.21}
    offs, clk_now, lines, ctx = run_dr_seed(rx, "trk", bsat)
    want_clk = (
        LEGACY_CLK + rx.joint_clk_delta(jt.clk, REF, L, LEGACY_CLK, REF, L)[0]
    ) % L
    check(
        clk_now == want_clk and abs(clk_now - js.clk) > 0.3,
        "clk consumer adopts TRK_STATE's clock %.3f (not the search one's %.3f)"
        % (clk_now, js.clk),
    )
    w3 = (
        clk_now + rx.joint_clk_delta(jt.predicted(("G", 3)), REF, L, clk_now, REF, L)[0]
    )
    check(
        offs.get(3) == (MODEL + w3) % MOD - MODEL
        and abs(offs[3] - (TRK_CLK + TRK_B[3])) < 0.05,
        "PRN 3 (in TRK_STATE): offset = TRK clk + b_3 = %.3f, no b_sat (got %r)"
        % (w3, offs.get(3)),
    )
    check(
        offs.get(9) == (MODEL + clk_now + bsat[9]) % MOD - MODEL,
        "PRN 9 (not in TRK_STATE): keeps clk_now + b_sat = %.3f (got %r)"
        % (clk_now + bsat[9], offs.get(9)),
    )
    # The SatBiasFilter moves the fallback satellite's seed, never the TRK-driven one's.
    offs_b, _c, _l, _ = run_dr_seed(make_rx("fed"), "trk", {3: -1.4, 9: +0.9})
    check(
        offs_b.get(3) == offs.get(3),
        "b_sat 0.37 -> -1.40 does not move PRN 3's seed (%r vs %r)"
        % (offs_b.get(3), offs[3]),
    )
    check(
        abs(offs_b.get(9) - offs.get(9) - 1.11) < 1e-9,
        "...and does move PRN 9's, the fallback (by %+.3f)"
        % (offs_b.get(9) - offs.get(9)),
    )
    jc = _lines(lines, "JOINT-CLK")
    so = _lines(lines, "SEED-OFFSET")
    check(
        len(jc) == 1
        and jc[0].startswith("JOINT-CLK[trk]: legacy")
        and "ADOPTED" in jc[0],
        "JOINT-CLK[trk] ... ADOPTED (%s)" % (jc[:1],),
    )
    check(
        any(m.startswith("SEED-OFFSET[trk] %s" % BAND) for m in so)
        and any(
            m.startswith("SEED-OFFSET[trk] PRN 3") and "diff vs clk" in m for m in so
        )
        and not any(m.startswith("SEED-OFFSET[trk] PRN 9") for m in so),
        "SEED-OFFSET[trk] summary and per-PRN line (diff vs clk) (%s)" % so,
    )
    # trk without the slew consumer: the seed keeps the legacy offset, b_sat included.
    offs_c, _c, _l, _ = run_dr_seed(make_rx("fed"), "trk", bsat, consume=("clk",))
    check(
        offs_c.get(3) == (MODEL + clk_now + bsat[3]) % MOD - MODEL,
        "trk with clk only (no slew): PRN 3 keeps clk_now + b_sat",
    )


def test_trk_slew_bound_falls_back():
    """A TRK offset beyond joint_slew_max_chips of the clock is refused per satellite and the
    seed takes the legacy offset, b_sat included, exactly as on the search path."""
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    k = ("G", 3)
    jt.x[jt._idx[k]] += 9.0  # b_3 = 9.1 chips: past the 5-chip bound
    # The median gate (trk_unsettled) would refuse it first; widened so the bound is reached.
    offs, clk_now, lines, _ = run_dr_seed(
        rx,
        "trk",
        {3: 0.37, 9: 0.0},
        consume=("slew",),
        argkw={"joint_trk_bsat_dev_max": 20.0},
    )
    check(
        clk_now == LEGACY_CLK
        and offs.get(3) == (MODEL + LEGACY_CLK + 0.37) % MOD - MODEL,
        "refused PRN 3 rides clk_now + b_sat (%r)" % offs.get(3),
    )
    so = [m for m in lines if m.startswith("SEED-OFFSET[trk] PRN 3")]
    check(
        so and "REFUSED" in so[0],
        "...and its SEED-OFFSET[trk] line says REFUSED (%s)" % so,
    )


def test_trk_unfit_falls_back_or_coasts():
    """Right after a restart TRK_STATE is born empty (n 0, sigma 200), or deaf, or absent. With
    the search-fed state fit, the trk chain reads it exactly as a search chain does, its lines
    tagged [trk->search: why]. With that one unfit too the chain is refused, coasts on its last
    adoption when it has one, and every satellite keeps clk_now + b_sat. No exception reaches
    the loop."""
    bsat = {3: 0.37, 9: -0.21}
    want = {"empty": "unfit: n 0 < 4", "deaf": "unfit: deaf", "absent": "unfit: absent"}
    for kind in ("empty", "deaf", "absent"):
        tag = "[trk->search: %s]" % want[kind]
        try:
            offs, clk_now, lines, _ = run_dr_seed(make_rx(kind), "trk", bsat)
            raised = None
        except Exception as e:  # the regression this test is for
            raised = e
        check(
            raised is None, "trk state %s: dr_seed does not raise (%s)" % (kind, raised)
        )
        if raised is not None:
            continue
        offs_s, clk_s, lines_s, _ = run_dr_seed(make_rx(kind), "search", bsat)
        sw = _lines(lines, "JOINT-SOURCE")
        check(
            clk_now == clk_s and offs == offs_s and clk_now != LEGACY_CLK,
            "trk state %s, search state fit: a search chain's clock and offsets (%r)"
            % (kind, clk_now),
        )
        check(
            [m.replace(tag, "") for m in lines if not m.startswith("JOINT-SOURCE")]
            == lines_s
            and _lines(lines, "JOINT-CLK%s: legacy" % tag)
            and len(sw) == 1
            and "trk -> search (%s)" % want[kind] in sw[0],
            "...its lines, tagged %s, and one JOINT-SOURCE line (%s)" % (tag, sw),
        )
        # Both unfit (the search-fed state under joint_min_sats): refused, legacy clock.
        rx = make_rx(kind, search="thin")
        offs, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat)
        check(
            clk_now == LEGACY_CLK
            and all(
                offs[p] == (MODEL + LEGACY_CLK + bsat[p]) % MOD - MODEL for p in (3, 9)
            )
            and not _lines(lines, "SEED-OFFSET"),
            "trk state %s, search state thin, no adoption behind: legacy clock, every seed "
            "clk_now + b_sat, no SEED-OFFSET line" % kind,
        )
        # With an adoption 60 s behind, the refusal coasts, in the unit of the state it reads.
        ds = {
            "jclk_t": T_NOW - 60.0,
            "jclk_last": 150.8,
            "jclk_last_t": T_NOW - 60.0,
            "jclk_last_rate": 0.001,
        }
        rx = make_rx(kind, search="thin")
        offs, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat, ds=ds)
        jc = _lines(lines, "JOINT-CLK")
        check(
            abs(clk_now - (150.8 + 0.06)) < 1e-9
            and len(jc) == 1
            and jc[0].startswith("JOINT-CLK%s: legacy" % tag)
            and "not fit to consume" in jc[0]
            and "COAST" in jc[0],
            "trk state %s, search thin, adopted 60 s ago: COAST to %.3f (%s)"
            % (kind, clk_now, jc[:1]),
        )
        check(
            all(abs(offs[p] - (clk_now + bsat[p])) < 1e-9 for p in (3, 9))
            and (kind != "absent" or TRK_STATE not in rx._joint),
            "...the seeds ride the coasted clock + b_sat; an absent TRK_STATE is not "
            "created by the consumer (its feeder builds it with escape_clock_min_sats)",
        )
    # A state born and fed once (4 satellites, still cold) is fit and fresh, so it is read,
    # and the clk consumer's sigma bound refuses it.
    rx = make_rx("empty")
    jt = rx._joint[TRK_STATE]
    jt.cycle([(("G", p), TRK_CLK + TRK_B[p], 0.1) for p in (3, 5, 14, 21)], 0.0)
    _o, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(9,), t0=10.0)
    jc = _lines(lines, "JOINT-CLK")
    check(
        jt.sigma() > 0.5
        and clk_now == LEGACY_CLK
        and len(jc) == 1
        and jc[0].startswith("JOINT-CLK[trk]")
        and "REFUSED (bounds" in jc[0],
        "a once-fed TRK state (n 4, sigma %.2f): the clk consumer's sigma bound refuses it (%s)"
        % (jt.sigma(), jc[:1]),
    )


def _w(off):
    """An offset as a born seed carries it (through the seed's modulus)."""
    return (MODEL + off) % MOD - MODEL


def _trk_off(rx, clk_now, prn):
    jt = rx._joint[TRK_STATE]
    return _w(
        clk_now
        + rx.joint_clk_delta(jt.predicted(("G", prn)), REF, L, clk_now, REF, L)[0]
    )


def test_trk_settled_gate():
    """A seed rides TRK_STATE's clk + b only while b is settled: sigma(key) under
    --joint-trk-bsat-sigma-max, not rejecting, no escape within --joint-trk-escape-holdoff-s.
    Each leg is driven through dr_seed; a satellite that fails one keeps clk_now + b_sat."""
    bsat = {3: 0.37, 5: -0.25, 7: 0.15}
    k3 = ("G", 3)
    check(
        _DEF.joint_trk_bsat_sigma_max == 0.08
        and _DEF.joint_trk_escape_holdoff_s == 60.0,
        "defaults: sigma 0.08 reference chips, holdoff 60 s",
    )

    # Settled: passes at the default (sigma ~0.04).
    rx = make_rx("fed")
    s3 = rx._joint[TRK_STATE].sigma(k3)
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 5))
    check(
        s3 < 0.08
        and offs[3] == _trk_off(rx, clk, 3)
        and offs[5] == _trk_off(rx, clk, 5),
        "settled (sigma %.3f): PRN 3 and 5 ride clk + b" % s3,
    )

    # The sigma leg, both sides: a bound under the row's sigma refuses it.
    rx = make_rx("fed")
    offs, clk, lines, _ = run_dr_seed(
        rx, "trk", bsat, prns=(3,), argkw={"joint_trk_bsat_sigma_max": 0.03}
    )
    so = _lines(lines, "SEED-OFFSET[trk] PRN 3")
    check(
        offs[3] == _w(clk + bsat[3]) and so and "b not settled (sigma 0.04" in so[0],
        "sigma bound 0.03 under the row's %.3f: PRN 3 keeps clk_now + b_sat (%s)"
        % (s3, so),
    )
    # A row born one update ago reads ~0.10: refused at the default, its neighbours not.
    rx = make_rx("fed")
    b7 = dict(TRK_B)
    b7[7] = 0.2
    _feed(rx._joint[TRK_STATE], TRK_CLK, b7, n=1, t_start=80.0)
    s7 = rx._joint[TRK_STATE].sigma(("G", 7))
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 7))
    check(
        s7 >= 0.08 and offs[7] == _w(clk + bsat[7]) and offs[3] == _trk_off(rx, clk, 3),
        "newborn PRN 7 (sigma %.3f) keeps clk_now + b_sat; PRN 3 rides clk + b" % s7,
    )

    # The rejecting leg: two cycles of a +3-chip step on PRN 3, rejected, sigma still small.
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    _feed(jt, TRK_CLK, TRK_B, n=2, t_start=80.0, step={3: 3.0})
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 5), t0=84.0)
    so = _lines(lines, "SEED-OFFSET[trk] PRN 3")
    check(
        jt.rejecting(k3)
        and jt.sigma(k3) < 0.08
        and offs[3] == _w(clk + bsat[3])
        and offs[5] == _trk_off(rx, clk, 5)
        and so
        and "(rejecting)" in so[0],
        "PRN 3 rejecting (sigma %.3f): keeps clk_now + b_sat; PRN 5 rides clk + b (%s)"
        % (jt.sigma(k3), so),
    )

    # The escape leg: PRN 3 steps +2 chips and stays; after the (bias-only) escape its b holds
    # the step and its sigma is small again within updates. Inside the holdoff it keeps the
    # legacy offset; past it, it rides clk + b.
    def escaped(after=None):
        rx = make_rx("fed")
        jt = rx._joint[TRK_STATE]
        _feed(jt, TRK_CLK, TRK_B, n=12, t_start=80.0, step={3: 2.0})
        t_last = 80.0 + 2.0 * 11
        e = jt.since_escape(k3, t_last)
        t_esc = None if e is None else t_last - e
        if after is not None and t_esc is not None:
            # Fed on to 2 s before the consumer's cycle at t_esc + after, so it is fresh there.
            n = int((t_esc + after - 2.0 - t_last) // 2.0)
            _feed(jt, TRK_CLK, TRK_B, n=n, t_start=t_last + 2.0, step={3: 2.0})
        return rx, jt, t_esc

    rx, jt, t_esc = escaped()
    check(
        t_esc is not None
        and abs(jt.clk - TRK_CLK) < 0.2
        and abs(jt.predicted(k3) - (TRK_CLK + TRK_B[3] + 2.0)) < 0.2
        and jt.sigma(k3) < 0.08
        and not jt.rejecting(k3),
        "fixture: PRN 3 escaped at t=%s, bias only (clk %.3f), b holds the step, sigma "
        "%.3f" % (t_esc, jt.clk, jt.sigma(k3)),
    )
    if t_esc is None:
        return
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 5), t0=t_esc + 30.0)
    so = _lines(lines, "SEED-OFFSET[trk] PRN 3")
    check(
        offs[3] == _w(clk + bsat[3])
        and offs[5] == _trk_off(rx, clk, 5)
        and so
        and "escaped 30 s ago" in so[0],
        "30 s after the escape: PRN 3 keeps clk_now + b_sat, PRN 5 rides clk + b (%s)"
        % so,
    )
    # Past the holdoff the step is still in b, 2 chips off its peers: the median leg keeps
    # PRN 3 on the legacy offset at the default bound, and a bound wider than the step lets it
    # ride (the holdoff alone would have).
    rx, jt, t_esc = escaped(after=61.0)
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 5), t0=t_esc + 61.0)
    so = _lines(lines, "SEED-OFFSET[trk] PRN 3")
    check(
        _DEF.joint_trk_bsat_dev_max == 0.5
        and offs[3] == _w(clk + bsat[3])
        and offs[5] == _trk_off(rx, clk, 5)
        and so
        and "off the G median" in so[0],
        "61 s after, default bound 0.5: PRN 3 keeps clk_now + b_sat, PRN 5 rides (%s)"
        % so,
    )
    rx, jt, t_esc = escaped(after=61.0)
    offs, clk, lines, _ = run_dr_seed(
        rx,
        "trk",
        bsat,
        prns=(3, 5),
        t0=t_esc + 61.0,
        argkw={"joint_trk_bsat_dev_max": 3.0},
    )
    check(
        offs[3] == _trk_off(rx, clk, 3) and abs(offs[3] - clk - TRK_B[3] - 2.0) < 0.2,
        "61 s after, bound 3.0: PRN 3 rides clk + b, the step included (%+.3f over clk)"
        % (offs[3] - clk),
    )
    # Too few peers to judge: a constellation of 3 in the state leaves each 2 peers.
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    for k in [k for k in list(jt._idx) if k[1] not in (3, 5, 14)]:
        jt._idx.pop(k)
    check(
        deadreckon.trk_unsettled(jt, k3, 200.0, chain_args("trk"))
        == "2 peer(s) in G, need 3",
        "2 peers: unsettled (%s)"
        % deadreckon.trk_unsettled(jt, k3, 200.0, chain_args("trk")),
    )
    # The helper itself, on inputs the caller should never hand it.
    check(
        deadreckon.trk_unsettled(jt, k3, None, chain_args("trk")) == "no epoch"
        and deadreckon.trk_unsettled(None, k3, 1.0, chain_args("trk")).startswith(
            "error:"
        ),
        "trk_unsettled never raises: no epoch, a broken state -> unsettled",
    )


def test_last_accepted():
    """JointReceiverState.last_accepted, what staleness is keyed on: the newest accepted
    measurement over the rows the state holds. predict() advances the epoch and not it, and
    a timestamp left by a row the state no longer holds neither makes a held state look
    fresh nor a fed one stale."""
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    a = chain_args("trk")
    check(jt.last_accepted() == 78.0, "fed 0..78 s: %r" % jt.last_accepted())
    jt.predict(500.0)
    check(
        jt._t == 500.0 and jt.last_accepted() == 78.0,
        "predict(500) moves the epoch (%r), not the newest accepted measurement (%r)"
        % (jt._t, jt.last_accepted()),
    )
    jt._t_seen[("G", 99)] = 499.0  # a row gone from the state, its stamp newer than all
    check(
        jt.last_accepted() == 78.0
        and deadreckon.trk_unfit(jt, rx, 500.0, a) == "stale 422 s",
        "a dropped row's newer stamp does not make the held state fresh (%r, %r)"
        % (jt.last_accepted(), deadreckon.trk_unfit(jt, rx, 500.0, a)),
    )
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    jt.cycle([(("G", 5), TRK_CLK + TRK_B[5], 0.1)], 100.0)  # one live row fed at 100 s
    jt._t_seen[("G", 98)] = 1.0  # a row gone from the state, its stamp older than all
    check(
        jt.last_accepted() == 100.0 and deadreckon.trk_unfit(jt, rx, 110.0, a) is None,
        "a live row's newest stamp counts, an older dropped one does not (%r)"
        % jt.last_accepted(),
    )
    del jt._t_seen[("G", 98)]
    # The state's own age-out (max_age_s): rows unfed for 900 s leave, and their stamps with
    # them.
    jt.cycle([(("G", 5), TRK_CLK + TRK_B[5], 0.1)], 1000.0)
    check(
        sorted(jt._idx) == [("G", 5)] and jt.last_accepted() == 1000.0,
        "after the age-out only G05 is held, last accepted 1000 s (%s, %r)"
        % (sorted(jt._idx), jt.last_accepted()),
    )
    check(
        receiver.Receiver(log=lambda m: None)
        .joint_receiver(BAND, L, REF, which=TRK_STATE, **TRK_KW)
        .last_accepted()
        is None,
        "a state never fed: None",
    )


# The held tracker feed of 10-28 12:43-12:55Z (the G32 transit): jtrk holds TRK_STATE's feed
# for HOLD_S while gps_l5's search keeps feeding JOINT_STATE and the receiver clock walks 0.6
# chips. TRK_STATE reads the clock `step` chips above JOINT_STATE (their gauges differ).
HOLD_T0, HOLD_S = 100.0, 780.0


def _held_feed_run(step, predict_hold=False, rate_at=None):
    """Both states fed every 2 s, TRK_STATE held over (HOLD_T0, HOLD_T0 + HOLD_S) (with
    predict() each cycle under predict_hold, as jtrk with a70103f99), and the trk chain's
    dr_seed run whole every cycle from HOLD_T0 - 10 s to 70 s after the hold, its dr_state
    carried. Returns {t: record} and the rate consumer's (cb, lines, clk_rate) at rate_at."""
    rx = Receiver(log=lambda m: None)
    js = rx.joint_receiver(BAND, L, REF, gauge_mode="prior")
    jt = rx.joint_receiver(BAND, L, REF, which=TRK_STATE, **TRK_KW)
    units = []
    unit = rx.joint_unit

    def joint_unit(which=JOINT_STATE):
        units.append(which)
        return unit(which)

    rx.joint_unit = joint_unit
    t_end = HOLD_T0 + HOLD_S + 70.0
    walk = 0.6 / HOLD_S
    bsat = {3: 0.37, 9: -0.21}
    ds, out, rate = {}, {}, None
    t = 0.0
    while t <= t_end:
        clock = SEARCH_CLK + walk * min(max(0.0, t - HOLD_T0), HOLD_S)
        js.cycle([(("G", p), clock + b, 0.1) for p, b in sorted(SEARCH_B.items())], t)
        if HOLD_T0 < t < HOLD_T0 + HOLD_S:
            if predict_hold:
                jt.predict(t)
        else:
            jt.cycle(
                [(("G", p), clock + step + b, 0.1) for p, b in sorted(TRK_B.items())], t
            )
        if t >= HOLD_T0 - 10.0 and (
            t <= HOLD_T0 + 70.0 or t >= HOLD_T0 + HOLD_S - 4.0 or t % 100.0 == 0.0
        ):
            del units[:]
            rec = dict(
                clock=clock,
                js_clk=js.clk,
                jt_clk=jt.clk,
                js_pred={p: js.predicted(("G", p)) for p in (3, 9)},
                epoch=jt._t,
            )
            offs, clk_now, lines, ctx = run_dr_seed(rx, "trk", bsat, t0=t, ds=ds)
            ds = ctx.dr_state
            rec.update(
                offs=offs,
                clk_now=clk_now,
                lines=lines,
                units=set(units),
                mode=ds["jsrc"]["mode"],
            )
            out[t] = rec
            if t == rate_at:
                rate = run_rate_consumer(
                    rx, chain_args("trk"), {"rate"}, 1e-9, t_now=t, dr_state=ds
                ) + (js.clk_rate,)
        t += 2.0
    return out, rate


def _adopted(clk):
    """clk_now after an adoption of the joint clock clk from the legacy clock."""
    return (
        LEGACY_CLK + Receiver.joint_clk_delta(clk, REF, L, LEGACY_CLK, REF, L)[0]
    ) % L


def _search_off(clk_now, jpred, b):
    """The search path's slew offset: legacy (clk_now + b_sat) plus the wrapped joint delta."""
    leg = clk_now + b
    return _w(leg + Receiver.joint_clk_delta(jpred, REF, L, leg, REF, L)[0])


def test_trk_held_feed_falls_back():
    """A trk chain through a held tracker feed (10-28 G32): TRK_STATE is read until it has
    accepted nothing for --joint-trk-stale-s, then the chain reads JOINT_STATE exactly as a
    search chain does (unit, slew on clk_now + b_sat, the clock adopted from it, the rate)
    through the hold, and reads TRK_STATE again only once it has been fresh for
    --joint-trk-return-s. A ~0.1-chip step between the two clocks is under
    joint_clk_max_step_chips (0.3), so each switch is adopted at once."""
    t0, t1 = HOLD_T0, HOLD_T0 + HOLD_S
    stale_s, ret_s = _DEF.joint_trk_stale_s, _DEF.joint_trk_return_s
    t_sw, t_back = t0 + stale_s + 2.0, t1 + ret_s
    check(stale_s == 30.0 and ret_s == 30.0, "defaults: stale 30 s, return 30 s")
    out, rate = _held_feed_run(0.15, rate_at=t0 + 300.0)
    bsat = {3: 0.37, 9: -0.21}

    def head(r):
        jc = _lines(r["lines"], "JOINT-CLK")
        return jc[0] if len(jc) == 1 else repr(jc)

    sw = [
        (t, m)
        for t, r in sorted(out.items())
        for m in _lines(r["lines"], "JOINT-SOURCE")
    ]
    check(
        len(sw) == 2
        and sw[0][0] == t_sw
        and "trk -> search (stale %.0f s)" % (t_sw - t0) in sw[0][1]
        and sw[1][0] == t_back
        and "search -> trk (fit and fresh for %.0f s; clock step " % ret_s in sw[1][1]
        and re.search(r"\(clock step [-+]\d+\.\d{3} reference chips\)$", sw[0][1]) is not None
        and re.search(r"; clock step [-+]\d+\.\d{3} reference chips\)$", sw[1][1]) is not None,
        "one JOINT-SOURCE line each way, at %.0f s and %.0f s (%s)"
        % (t_sw, t_back, sw),
    )
    pre = [r for t, r in out.items() if t <= t0 + stale_s]
    check(
        pre
        and all(
            r["mode"] == "trk"
            and r["units"] == {TRK_STATE}
            and head(r).startswith("JOINT-CLK[trk]: legacy")
            and "ADOPTED" in head(r)
            and r["clk_now"] == _adopted(r["jt_clk"])
            for r in pre
        ),
        "up to %.0f s stale: TRK_STATE read, in its unit, its clock adopted (%d cycles)"
        % (stale_s, len(pre)),
    )
    held = [r for t, r in out.items() if t_sw <= t < t1]
    bad = [
        (r["clk_now"], head(r))
        for r in held
        if not (
            r["mode"] == "search"
            and r["units"] == {JOINT_STATE}
            and head(r).startswith("JOINT-CLK[trk->search: stale ")
            and "ADOPTED" in head(r)
            and r["clk_now"] == _adopted(r["js_clk"])
            and all(
                r["offs"][p] == _search_off(r["clk_now"], r["js_pred"][p], bsat[p])
                for p in (3, 9)
            )
            and not any("vs clk" in m or "not settled" in m for m in r["lines"])
        )
    ]
    check(
        len(held) > 20 and not bad,
        "%.0f s to the end of the hold: JOINT_STATE read in its unit, its clock adopted, every "
        "seed on clk_now + b_sat + the wrapped joint delta (%d cycles; bad %s)"
        % (t_sw, len(held), bad[:2]),
    )
    last, first = out[t1 - 2.0], out[t_sw]
    moved = last["clk_now"] - first["clk_now"]
    check(
        abs(moved - (last["clock"] - first["clock"])) < 0.05
        and last["clk_now"] - last["jt_clk"] > 0.4,
        "through the hold clk_now moves %+.3f chips with the receiver clock (%+.3f) and ends "
        "%+.3f from the held TRK clock"
        % (moved, last["clock"] - first["clock"], last["clk_now"] - last["jt_clk"]),
    )
    sw_rec = out[t_sw]
    check(
        0.0 < out[t_sw - 2.0]["clk_now"] - sw_rec["clk_now"] < 0.3,
        "the switch steps clk_now %+.3f chips, under joint_clk_max_step_chips: adopted at once"
        % (sw_rec["clk_now"] - out[t_sw - 2.0]["clk_now"]),
    )
    cb, rl, js_rate = rate
    check(
        cb == js_rate / REF
        and any(
            m.startswith("code-rate clock from the JOINT[trk->search: stale")
            for m in rl
        ),
        "the rate consumer reads JOINT_STATE's clk_rate during the hold (%r; %s)"
        % (cb, rl),
    )
    back = [r for t, r in out.items() if t1 <= t < t_back]
    check(
        back
        and all(
            r["mode"] == "search"
            and r["units"] == {JOINT_STATE}
            and "[trk->search: trk fresh " in head(r)
            and r["clk_now"] == _adopted(r["js_clk"])
            for r in back
        ),
        "fed again at %.0f s: still JOINT_STATE until %.0f s fresh (%s)"
        % (t1, ret_s, head(back[-1]) if back else None),
    )
    after = [r for t, r in out.items() if t >= t_back]
    check(
        after
        and all(
            r["mode"] == "trk"
            and r["units"] == {TRK_STATE}
            and head(r).startswith("JOINT-CLK[trk]: legacy")
            and r["clk_now"] == _adopted(r["jt_clk"])
            for r in after
        ),
        "from %.0f s: TRK_STATE read again, its clock adopted at once (%s)"
        % (t_back, head(after[0]) if after else None),
    )

    # jtrk with a70103f99: a held cycle calls predict(), so the epoch advances; the staleness
    # is the measurements', and the chain switches at the same cycle and reads the same clock.
    out_p, _ = _held_feed_run(0.15, predict_hold=True)
    sw_p = [t for t, r in sorted(out_p.items()) if _lines(r["lines"], "JOINT-SOURCE")]
    check(
        out_p[t0 + 50.0]["epoch"] == t0 + 50.0
        and out[t0 + 50.0]["epoch"] == t0
        and sw_p == [t_sw, t_back]
        and all(
            out_p[t]["clk_now"] == out[t]["clk_now"] for t in out if t_sw <= t < t1
        ),
        "held with predict(): epoch %.0f s, yet stale at %s as without it"
        % (out_p[t0 + 50.0]["epoch"], sw_p),
    )


def test_trk_switch_step_held():
    """A switch step beyond joint_clk_max_step_chips goes through the clk consumer's step
    guard like any other: refused for joint_clk_step_hold_s while the chain coasts on the
    clock it last adopted, then adopted. Both ways."""
    t0, t1 = HOLD_T0, HOLD_T0 + HOLD_S
    t_sw, t_back = t0 + 32.0, t1 + 30.0
    hold = chain_args("trk").joint_clk_step_hold_s
    out, _ = _held_feed_run(0.5)

    def head(t):
        jc = _lines(out[t]["lines"], "JOINT-CLK")
        return jc[0] if len(jc) == 1 else repr(jc)

    before = out[t_sw - 2.0]["clk_now"]
    check(
        all(
            "[trk->search: stale" in head(t)
            and "COAST (refused: step " in head(t)
            and abs(out[t]["clk_now"] - before) < 0.01
            for t in out
            if t_sw <= t < t_sw + hold
        )
        and "ADOPTED" in head(t_sw + hold)
        and out[t_sw + hold]["clk_now"] == _adopted(out[t_sw + hold]["js_clk"]),
        "to JOINT_STATE (step %+.3f): coast on the last TRK adoption %.0f s, then adopted (%s)"
        % (out[t_sw + hold]["clk_now"] - before, hold, head(t_sw)),
    )
    before = out[t_back - 2.0]["clk_now"]
    check(
        all(
            head(t).startswith("JOINT-CLK[trk]")
            and "COAST (refused: step " in head(t)
            and abs(out[t]["clk_now"] - before) < 0.05
            for t in out
            if t_back <= t < t_back + hold
        )
        and "ADOPTED" in head(t_back + hold)
        and out[t_back + hold]["clk_now"] == _adopted(out[t_back + hold]["jt_clk"]),
        "back to TRK_STATE: coast %.0f s, then adopted (%s)" % (hold, head(t_back)),
    )


def test_none_fields_format():
    """No epoch, no dr_state, no --signal, a raising state lookup, a missing option: every
    path returns and every line formats, None fields included."""
    a = chain_args("trk", signal=None)
    rx = make_rx("fed")
    jt = rx._joint[TRK_STATE]
    check(
        deadreckon.trk_unfit(jt, rx, None, a) == "no epoch"
        and deadreckon.trk_unfit(None, types.SimpleNamespace(), 1.0, a).startswith(
            "error:"
        )
        and deadreckon.trk_unfit(
            Receiver(log=lambda m: None).joint_receiver(
                BAND, L, REF, which=TRK_STATE, **TRK_KW
            ),
            rx,
            1.0,
            a,
        )
        == "stale: nothing accepted",
        "trk_unfit: no epoch, a broken receiver, nothing accepted",
    )
    lines = []
    saved = deadreckon._log, deadreckon._log_rl
    deadreckon._log = lines.append
    deadreckon._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    try:
        base = dict(rx=rx, band_id=BAND, joint_state=broker_joint_state())
        ctx = types.SimpleNamespace(
            args=a, dr_state=None, drp=types.SimpleNamespace(t_now_abs=None), **base
        )
        got = deadreckon.joint_read(ctx)
        check(
            got == (rx._joint[JOINT_STATE], JOINT_STATE, "[trk->search: no epoch]")
            and not lines,
            "no epoch, no dr_state: JOINT_STATE, no switch line (%s; %s)"
            % (got[1:], lines),
        )
        ctx = types.SimpleNamespace(
            args=a, dr_state={}, drp=types.SimpleNamespace(t_now_abs=None), **base
        )
        got = deadreckon.joint_read(ctx)
        check(
            got[1] == JOINT_STATE
            and len(lines) == 1
            and lines[0].startswith(
                "JOINT-SOURCE %s: trk -> search (no epoch): the consumers read the search-fed "
                "joint state until the tracker-fed one has passed for 30 s (clock step " % BAND
            ),
            "no --signal: the switch line names the band (%s)" % lines,
        )
        del lines[:]
        boom = dict(base, joint_state=lambda *x, **k: 1 / 0)
        ctx = types.SimpleNamespace(
            args=a, dr_state={}, drp=types.SimpleNamespace(t_now_abs=T_NOW), **boom
        )
        got = deadreckon.joint_read(ctx)
        check(
            got == (None, TRK_STATE, "[trk]")
            and len(lines) == 1
            and "no joint state read (division by zero)" in lines[0],
            "a raising lookup: no state, one line (%s; %s)" % (got, lines),
        )
        del lines[:]
        a2 = types.SimpleNamespace(**vars(a))
        del a2.joint_trk_stale_s
        ctx = types.SimpleNamespace(
            args=a2, dr_state={}, drp=types.SimpleNamespace(t_now_abs=T_NOW), **base
        )
        got = deadreckon.joint_read(ctx)
        check(
            got[1] == JOINT_STATE and got[2].startswith("[trk->search: error:"),
            "an args without --joint-trk-stale-s reads JOINT_STATE (%s)" % (got[2],),
        )
    finally:
        deadreckon._log, deadreckon._log_rl = saved
    # dr_seed whole, TRK stale and the search state thin: the coast line formats with the
    # reason, with or without a stamped rate.
    ds = {
        "jclk_t": 30.0,
        "jclk_last": 150.8,
        "jclk_last_t": 30.0,
        "jclk_last_rate": None,
    }
    try:
        offs, clk_now, lines, _ = run_dr_seed(
            make_rx("fed", search="thin"), "trk", {3: 0.0, 9: 0.0}, ds=ds, t0=118.0
        )
        jc = _lines(lines, "JOINT-CLK")
        ok = clk_now == LEGACY_CLK or (
            jc and jc[0].startswith("JOINT-CLK[trk->search: stale 40 s]")
        )
    except Exception as e:
        ok, jc = False, e
    check(ok, "TRK stale, search thin, no stamped rate: no exception (%s)" % (jc,))
    offs, clk_now, lines, _ = run_dr_seed(
        make_rx("fed", search="thin"),
        "trk",
        {3: 0.0, 9: 0.0},
        ds=dict(ds, jclk_last_rate=0.001),
        t0=118.0,
    )
    jc = _lines(lines, "JOINT-CLK")
    check(
        len(jc) == 1
        and jc[0].startswith("JOINT-CLK[trk->search: stale 40 s]: legacy")
        and "COAST" in jc[0]
        and abs(clk_now - (150.8 + 0.088)) < 1e-9,
        "TRK stale, search thin, adopted 88 s ago: COAST (%s)" % jc,
    )


def test_rate_consumer_caller():
    """main's rate block, as written: trk reads TRK_STATE's clk_rate in its unit, and a
    refused state leaves the l-a EMA (None included) without raising."""
    rx = make_rx("fed")
    jt, js = rx._joint[TRK_STATE], rx._joint[JOINT_STATE]
    jt.x[1], js.x[1] = 0.02, -0.03  # chips/s: 0.00196 / -0.00293 ppm
    cb, lines = run_rate_consumer(rx, chain_args("trk"), {"rate"}, 1e-9)
    check(
        cb == 0.02 / REF and lines and "JOINT[trk] state" in lines[0],
        "trk: code rate from TRK_STATE's clk_rate (%r; %s)" % (cb, lines[:1]),
    )
    cb, lines = run_rate_consumer(rx, chain_args("search"), {"rate"}, 1e-9)
    check(
        cb == -0.03 / REF
        and lines
        and lines[0].startswith("code-rate clock from the JOINT state"),
        "search: from JOINT_STATE's, the line as it was (%r; %s)" % (cb, lines[:1]),
    )
    for kind in ("empty", "deaf", "absent"):
        rx = make_rx(kind)
        rx._joint[JOINT_STATE].x[1] = -0.03
        cb, lines = run_rate_consumer(rx, chain_args("trk"), {"rate"}, 1e-9)
        check(
            cb == -0.03 / REF
            and any(
                m.startswith("code-rate clock from the JOINT[trk->search: unfit")
                for m in lines
            ),
            "trk state %s, search state fit: code rate from JOINT_STATE's clk_rate (%r; %s)"
            % (kind, cb, lines),
        )
        for ema in (None, 2e-9):
            try:
                cb, lines = run_rate_consumer(
                    make_rx(kind, search="thin"), chain_args("trk"), {"rate"}, ema
                )
                ok = cb == ema and not _lines(lines, "code-rate")
            except Exception as e:
                ok, cb = False, e
            check(
                ok,
                "trk state %s, search state thin, l-a EMA %r: refused, EMA kept (%r)"
                % (kind, ema, cb),
            )


def test_startup_line():
    for src, consume, want in (
        (
            "trk",
            {"clk", "rate", "slew"},
            "clk,rate,slew on gal_e5a read the tracker-fed",
        ),
        ("search", {"clk", "rate"}, "clk,rate on gal_e5a read the search-fed"),
        ("trk", set(), "none on gal_e5a read the tracker-fed"),
    ):
        lines = run_startup_line(chain_args(src), consume)
        check(
            len(lines) == 1 and want in lines[0],
            "startup: %s %s -> %s" % (src, sorted(consume), lines),
        )
    check(
        "SatBiasFilter" in run_startup_line(chain_args("trk"), {"slew"})[0],
        "...a trk slew chain says the SatBiasFilter stops feeding its seeds",
    )
    check(
        not run_startup_line(chain_args("search"), set()), "search, no consumer: silent"
    )
    check(
        run_startup_line(chain_args("trk", signal=None), {"clk"})[0].count("this chain")
        == 1,
        "...and formats with no --signal",
    )


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            try:
                fn()
            except Exception as e:
                import traceback

                traceback.print_exc()
                check(False, "%s raised %s: %s" % (name, type(e).__name__, e))
    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: --joint-source (search unchanged, trk reads TRK_STATE)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
