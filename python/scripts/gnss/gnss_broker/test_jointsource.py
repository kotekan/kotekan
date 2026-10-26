"""--joint-source: which joint state a chain's consumers read (#191 step 3).

    python3 -m gnss_broker.test_jointsource     (from python/scripts/gnss)

docs/CHORD_JOINT_TRACKING.md 6.3 step 3. Two JointReceiverStates live in the Receiver:
JOINT_STATE (gps_l5's search detections feed it) and TRK_STATE (the tracker residuals,
jtrk.py). A chain with --joint-source trk has every consumer (clk, rate, slew, and the
JOINT-CLK / SEED-OFFSET lines) read TRK_STATE in TRK_STATE's unit; the SatBiasFilter no longer
moves the seed of a satellite the slew drives from it; a satellite TRK_STATE does not hold keeps
clk_now + b_sat; a state unfit to consume is refused and coasted on, never raised into the loop.

The broker's own code is exercised, not a copy of it: the `_joint_state` closure, the rate
consumer block and the startup line are lifted out of gps_distributed_broker.main by the AST
and run as written; deadreckon.dr_seed runs whole (clock consumer, SEED-OFFSET summary, the
per-satellite slew offset, the birth), with every log line formatted.
"""

import ast
import os
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


def run_rate_consumer(rx, args, joint_consume, code_ema):
    """main's `if 'rate' in joint_consume:` block as written: (cb_to_seed, log lines)."""
    lines = []
    ns = {
        "joint_consume": joint_consume,
        "_joint_state": broker_joint_state(),
        "rx": rx,
        "band_id": BAND,
        "args": args,
        "receiver": receiver,
        "_cb": types.SimpleNamespace(code_ema=code_ema),
        "cb_to_seed": code_ema,
        "_log_rl": lambda key, msg, every_s=10.0: lines.append(msg),
    }
    _run_nodes([_if_block("'rate' in joint_consume")], ns)
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


def make_rx(trk="fed"):
    """A Receiver with the search-fed state converged on SEARCH_*, and the tracker-fed one
    converged on TRK_* ("fed"), born and never fed ("empty"), deaf ("deaf"), or absent."""
    rx = Receiver(log=lambda m: None)
    _feed(rx.joint_receiver(BAND, L, REF, gauge_mode="prior"), SEARCH_CLK, SEARCH_B)
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
    t0=1000.0,
    argkw=None,
):
    """dr_seed once on a 10.23-Mcps chain with no search, every PRN of `prns` born from the
    model: returns ({prn: offset added to the model}, clk_now, log lines, ctx)."""
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
        receiver.joint_tag(a0) == "" and receiver.joint_tag(a1) == "[trk]",
        "the search lines carry no tag, the trk ones [trk]",
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
    offs, clk_now, lines, _ = run_dr_seed(
        rx, "trk", {3: 0.37, 9: 0.0}, consume=("slew",)
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


def test_trk_unfit_is_refused_and_coasts():
    """Right after a restart TRK_STATE is born empty (n 0, sigma 200), or deaf, or absent: the
    trk chain is refused, coasts on its last adoption when it has one (an absent state has no
    modulus, and cannot have been adopted from), and every satellite keeps clk_now + b_sat. No
    exception reaches the loop."""
    bsat = {3: 0.37, 9: -0.21}
    for kind in ("empty", "deaf", "absent"):
        rx = make_rx(kind)
        try:
            offs, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat)
            raised = None
        except Exception as e:  # the regression this test is for
            raised = e
        check(
            raised is None, "trk state %s: dr_seed does not raise (%s)" % (kind, raised)
        )
        if raised is not None:
            continue
        check(
            clk_now == LEGACY_CLK
            and all(
                offs[p] == (MODEL + LEGACY_CLK + bsat[p]) % MOD - MODEL for p in (3, 9)
            ),
            "trk state %s, no adoption behind: legacy clock, every seed clk_now + b_sat"
            % kind,
        )
        check(
            not _lines(lines, "SEED-OFFSET"),
            "...and no SEED-OFFSET line (nothing to compare)",
        )
        # With an adoption 60 s behind (from TRK_STATE: its stamps), the refusal coasts.
        ds = {
            "jclk_t": 940.0,
            "jclk_last": 150.8,
            "jclk_last_t": 940.0,
            "jclk_last_rate": 0.001,
        }
        rx = make_rx(kind)
        offs, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat, ds=ds)
        jc = _lines(lines, "JOINT-CLK")
        if kind == "absent":
            # No state, no modulus to coast in: legacy clock, and the consumer did not
            # create the state (its feeder builds it with escape_clock_min_sats).
            check(
                clk_now == LEGACY_CLK and not jc and TRK_STATE not in rx._joint,
                "trk state absent, stamps behind: legacy clock, no line, state not "
                "created by the consumer (%s)" % (sorted(rx._joint),),
            )
            continue
        check(
            abs(clk_now - (150.8 + 0.06)) < 1e-9
            and len(jc) == 1
            and jc[0].startswith("JOINT-CLK[trk]: legacy")
            and "not fit to consume" in jc[0]
            and "COAST" in jc[0],
            "trk state %s, adopted 60 s ago: COAST to %.3f (%s)"
            % (kind, clk_now, jc[:1]),
        )
        check(
            all(abs(offs[p] - (clk_now + bsat[p])) < 1e-9 for p in (3, 9)),
            "...the seeds ride the coasted clock + b_sat",
        )
    # A state born and fed once (4 satellites, still cold): the clk consumer's sigma bound.
    rx = make_rx("empty")
    jt = rx._joint[TRK_STATE]
    jt.cycle([(("G", p), TRK_CLK + TRK_B[p], 0.1) for p in (3, 5, 14, 21)], 0.0)
    _o, clk_now, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(9,))
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
    def escaped():
        rx = make_rx("fed")
        jt = rx._joint[TRK_STATE]
        _feed(jt, TRK_CLK, TRK_B, n=12, t_start=80.0, step={3: 2.0})
        t_last = 80.0 + 2.0 * 11
        e = jt.since_escape(k3, t_last)
        return rx, jt, (None if e is None else t_last - e)

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
    rx, jt, t_esc = escaped()
    offs, clk, lines, _ = run_dr_seed(rx, "trk", bsat, prns=(3, 5), t0=t_esc + 61.0)
    check(
        offs[3] == _trk_off(rx, clk, 3) and abs(offs[3] - clk - TRK_B[3] - 2.0) < 0.2,
        "61 s after: PRN 3 rides clk + b, the step included (%+.3f over clk)"
        % (offs[3] - clk),
    )
    # The helper itself, on inputs the caller should never hand it.
    check(
        deadreckon.trk_unsettled(jt, k3, None, chain_args("trk")) == "no epoch"
        and deadreckon.trk_unsettled(None, k3, 1.0, chain_args("trk")).startswith(
            "error:"
        ),
        "trk_unsettled never raises: no epoch, a broken state -> unsettled",
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
        for ema in (None, 2e-9):
            try:
                cb, lines = run_rate_consumer(
                    make_rx(kind), chain_args("trk"), {"rate"}, ema
                )
                ok = cb == ema and not lines
            except Exception as e:
                ok, cb = False, e
            check(
                ok, "trk state %s, l-a EMA %r: refused, EMA kept (%r)" % (kind, ema, cb)
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
