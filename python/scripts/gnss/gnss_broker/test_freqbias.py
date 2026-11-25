"""The clock-frequency bias of a chain that solved none this cycle (almanac.adopt_freq_bias).

--freq-bias-joint takes the joint state's clock rate, scaled to the chain's carrier as
dr_drift scales it, and refreshes meas_t, so the bias does not go stale while every visible
satellite is locked and the search detects nothing. These checks pin that, its gates (too
few satellites or a rate past joint_max_rate_ppm fall through to another chain's solve), and
that without the flag the path is unchanged.

    python3 -m gnss_broker.test_freqbias

@author Keith Vanderlinde
"""

import sys
import types

from gnss_broker import almanac, deadreckon
from gnss_broker.clockbias import ClockBias
from gnss_broker.receiver import JOINT_REF_RATE_HZ, Receiver

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


GW_HZ = 1202.025e6
L5_HZ = 1176.45e6


class _FakeJoint(object):
    def __init__(self, clk_rate, n):
        self.clk = 0.0
        self.clk_rate = clk_rate
        self.L = 10230.0
        self._idx = {("G", i): i for i in range(n)}


def _run(flag, clk_rate=0.0021, n=18, adopt_fb=True, drift_source="bias", donor=True):
    """One adopt_freq_bias call on a chain whose bias went stale at t=0; returns (joint?, cb)."""
    rx = Receiver(log=lambda m: None)
    js = _FakeJoint(clk_rate, n)
    rx._joint["__receiver__"] = js
    if donor:
        rx.contribute_freq_bias("gps_l5", -3.0 / L5_HZ, 5, 900.0)
    args = types.SimpleNamespace(
        freq_bias_joint=flag,
        dr_drift_source=drift_source,
        carrier_hz=GW_HZ,
        chip_rate_hz=2.046e6,
        seed_bias_source="zero",
        joint_min_sats=4,
        joint_max_rate_ppm=0.01,
    )
    ctx = types.SimpleNamespace(
        args=args,
        rx=rx,
        band_id="test",
        chain_id="gw_1202",
        t0=1000.0,
        cb=ClockBias(value=-1.0, ema=-1.0, meas_t=0.0),
        dr_state={},
        drp=types.SimpleNamespace(t_now_abs=None),
        joint_state=lambda _rx, _b, _a, which=None: js,
    )
    quiet = lambda key, msg, every_s=10.0: None
    _orig = (almanac._log_rl, deadreckon._log_rl)
    almanac._log_rl = deadreckon._log_rl = quiet
    try:
        joint = almanac.adopt_freq_bias(ctx, adopt_fb)
    finally:
        almanac._log_rl, deadreckon._log_rl = _orig
    return joint, ctx.cb


def test_joint():
    """The joint rate, scaled as dr_drift scales it, refreshes meas_t and beats a donor."""
    joint, cb = _run(True)
    want = 0.0021 / JOINT_REF_RATE_HZ * GW_HZ
    check(joint, "the joint rate is taken")
    check(abs(cb.ema - want) < 1e-9 and cb.value == cb.ema, "bias %+.4f Hz" % cb.ema)
    check(cb.meas_t == 1000.0, "meas_t refreshed, so the next cycle is not stale")
    check(cb.seed == 0.0, "seed-bias-source zero: the seeds ride 0 Hz")
    # dr_drift's two sources agree on this bias: cb.ema chip/carrier == chip (l-a).
    la = 0.0021 / JOINT_REF_RATE_HZ
    check(
        abs(cb.ema * 2.046e6 / GW_HZ - 2.046e6 * la) < 1e-12,
        "the bias path and the joint-rate path give dr_drift one drift",
    )


def test_gates():
    """Too few satellites or a rate past joint_max_rate_ppm fall through to the donor."""
    donor = -3.0 / L5_HZ * GW_HZ
    for what, kw in (
        ("under joint_min_sats", dict(n=3)),
        ("past joint_max_rate_ppm", dict(clk_rate=0.2)),
    ):
        joint, cb = _run(True, **kw)
        check(
            not joint and abs(cb.ema - donor) < 1e-9 and cb.meas_t == 1000.0,
            "%s: the donor's %+.2f Hz" % (what, cb.ema),
        )
    joint, cb = _run(True, n=3, donor=False)
    check(
        not joint and cb.ema == -1.0 and cb.meas_t == 0.0, "neither: held, stays stale"
    )


def test_default_unchanged():
    """Without the flag the donor is adopted as before, and a solve-drift chain with no
    --freq-bias-adopt adopts nothing."""
    donor = -3.0 / L5_HZ * GW_HZ
    joint, cb = _run(False)
    check(not joint and abs(cb.ema - donor) < 1e-9, "flag off: the donor's bias")
    joint, cb = _run(False, adopt_fb=False, drift_source="solve")
    check(not joint and cb.ema == -1.0 and cb.meas_t == 0.0, "flag off, solve: held")
    joint, cb = _run(True, adopt_fb=False, drift_source="solve")
    check(joint and cb.meas_t == 1000.0, "flag on, solve: the joint rate")


def main():
    for t in (test_joint, test_gates, test_default_unchanged):
        print(t.__name__)
        t()
    if _fails:
        print("FAILED: %d" % len(_fails))
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
