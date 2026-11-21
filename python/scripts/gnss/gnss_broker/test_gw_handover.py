"""gw_handover: when the table hands a GuoWang code to another satellite, the code's
per-satellite state (b_sat, the joint states' rows, the KCOH rate, the model-untrusted flag and
the search's escape count) is cleared, and nothing of another code or constellation is.
"""

from types import SimpleNamespace

from gnss_broker.deadreckon import gw_handover
from gnss_broker.receiver import Receiver
from gnss_broker.state_filter import SatBiasFilter


def _ctx():
    eph = {
        ("W", 831): [
            dict(t0_gpst=0.0, t1_gpst=600.0, name="DIGUI-20"),
            dict(t0_gpst=1800.0, t1_gpst=2400.0, name="DIGUI-22"),
        ],
        ("W", 1214): [dict(t0_gpst=0.0, t1_gpst=2400.0, name="DIGUI-21")],
        ("G", 5): [dict(toe_gpst=0.0)],
    }
    ctx = SimpleNamespace(
        dr_state={"eph": eph},
        dr_eph_mod=SimpleNamespace(gpst_of_utc=lambda t: t),
        drp=SimpleNamespace(t_now_abs=0.0),
        utc0_sample0=0.0,
        bsat=SatBiasFilter(),
        rx=Receiver(),
        kcoh_rates={},
        dr_untrusted={},
        dr_bad={},
        cpt=SimpleNamespace(escape={}, escape_sign={}),
    )
    js = ctx.rx.joint_receiver("L", 2046.0, 2.046e6)
    for t in range(0, 300, 2):
        for key in (("W", 831), ("W", 1214), ("G", 5)):
            js.update(key, 0.1, 0.05, float(t))
            js.update_rrate(key, 0.5, float(t), 1202.025e6)
    for prn in (831, 1214):
        for t in range(0, 300, 2):
            ctx.bsat.update(prn, 0.3, 6, float(t))
        ctx.kcoh_rates[prn] = 1.5
        ctx.dr_untrusted[prn] = "integrity residual"
        ctx.dr_bad[prn] = 3
        ctx.cpt.escape[prn] = 2
        ctx.cpt.escape_sign[prn] = 0.4
    return ctx, js


def _at(ctx, t):
    ctx.drp.t_now_abs = t
    gw_handover(ctx)


def test_handover_clears_the_code_only():
    ctx, js = _ctx()
    _at(ctx, 300.0)  # DIGUI-20 on 831: the first sighting clears nothing
    assert ctx.bsat.get(831, 300.0) != 0.0 and 831 in ctx.kcoh_rates
    _at(ctx, 1200.0)  # between passes: no record covers now
    assert ctx.bsat.get(831, 300.0) != 0.0 and ("W", 831) in js._idx
    _at(ctx, 1900.0)  # DIGUI-22 now flies 831
    assert ctx.bsat.get(831, 300.0) == 0.0 and 831 not in ctx.bsat._b
    assert ("W", 831) not in js._idx and ("W", 831) not in js._rr_idx
    for d in (ctx.kcoh_rates, ctx.dr_untrusted, ctx.dr_bad):
        assert 831 not in d
    assert 831 not in ctx.cpt.escape and 831 not in ctx.cpt.escape_sign
    # Another code and another constellation keep theirs.
    assert ctx.bsat.get(1214, 300.0) != 0.0 and ctx.kcoh_rates[1214] == 1.5
    assert ctx.dr_untrusted[1214] and ctx.cpt.escape[1214] == 2
    for key in (("W", 1214), ("G", 5)):
        assert key in js._idx and key in js._rr_idx
    assert ctx.dr_state["gw_name"] == {831: "DIGUI-22", 1214: "DIGUI-21"}


def test_same_satellite_again_clears_nothing():
    ctx, js = _ctx()
    _at(ctx, 100.0)
    _at(ctx, 500.0)
    assert ctx.bsat.get(831, 300.0) != 0.0 and ("W", 831) in js._idx
    assert ctx.dr_untrusted[831] and ctx.cpt.escape[831] == 2
