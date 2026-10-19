"""The INNOV and MINNOV histories hold the whole 10-minute window the publisher reads.

    /home/kvand/gnss/venv-ft/bin/python -m gnss_broker.test_innov_window     (from python/scripts/gnss)

fleetdll cuts innov_p95_10m / minnov_p95_10m (the model-primacy gate's statistic) from
ctx.innov_hist and ctx.minnov_hist by time, 600 s back from now. The writers append up to one
sample per broker pass (~2 s), so the histories must be pruned by age at 600 s, not by count: a
120-sample cap holds only ~240 s and the "10-minute" p95 is a 4-minute one. Checked on the
helper both writers share (fits.append_windowed), on dr_joint_shadow's MINNOV append driven
for 700 s, and on the INNOV write site in seeding.stage_detections_to_seeds, which appends a
detection the search re-serves only once (driven through test_escape_dedup's context).

@author Keith Vanderlinde
"""

import ast
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import deadreckon as dr  # noqa: E402
from gnss_broker import seeding  # noqa: E402
from gnss_broker import fits  # noqa: E402

_fails = []
WINDOW_S, PASS_S, RUN_S = 600.0, 2.0, 700.0


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _in_window(hist, now):
    """fleetdll's read-side cut."""
    return [(t, v) for t, v in hist if now - t <= WINDOW_S]


def test_helper():
    append = getattr(fits, "append_windowed", None)
    check(append is not None, "fits.append_windowed exists")
    if append is None:
        return
    hist, t = [], 0.0
    while t <= RUN_S:
        append(hist, t, t)
        t += PASS_S
    now = hist[-1][0]
    win = _in_window(hist, now)
    check(
        len(win) == 301,
        "one sample per 2 s for 700 s: the 600 s window holds %d samples (301)"
        % len(win),
    )
    check(
        all(now - tt <= WINDOW_S for tt, _ in hist),
        "nothing older than 600 s remains (oldest %.0f s)" % (now - hist[0][0]),
    )
    burst = []
    for i in range(2500):
        append(burst, 5.0, float(i))
    check(
        len(burst) == 1000 and burst[-1][1] == 2499.0,
        "the memory backstop keeps the newest 1000 of a same-second burst (%d)"
        % len(burst),
    )


class _JS(object):
    """The joint state as dr_joint_shadow reads it: every satellite established, predicted at
    zero, so MINNOV is the offset itself."""

    def __init__(self, keys):
        self._idx = {k: i for i, k in enumerate(keys)}
        self._n = {k: 99 for k in keys}

    def wrap(self, d):
        return ((d + 5115.0) % 10230.0) - 5115.0

    def predicted(self, key):
        return 0.0

    def cycle(self, meas, t):
        pass

    def drain_notes(self):
        return []


def test_minnov_append():
    logs = []
    dr._log = lambda msg: logs.append(msg)
    dr._log_rl = lambda key, msg, every_s=10.0: logs.append(msg)
    prns = (3, 5)
    js = _JS([("L5", p) for p in prns])
    c = NS()
    c.args = NS(
        joint_shadow=True,
        chip_rate_hz=10.23e6,
        joint_feed_warmup_s=0.0,
        joint_min_snr=10.0,
        joint_mask_after=5,
        joint_sigma=0.3,
    )
    c.band_id, c.code_len = "1176.45MHz", 10230.0
    c.rx = NS(joint_receiver=lambda *a, **k: js, joint_ambiguous=lambda: False)
    c.dr_state = {"clk": 150.0, "joint_log_next": 1e18}
    c.best = {p: (30.0,) for p in prns}
    c.uptime_s = 0.0
    c.minnov_hist = {}
    c.track_ok = lambda p: True
    c.p2c_hold = lambda js_, key: False
    c.p2c_tick = lambda js_, t: None
    t = 1000.0
    while t <= 1000.0 + RUN_S:
        c.t0 = t
        c.uptime_s = t - 1000.0
        c.drp = NS(offs=[(p, 0.5) for p in prns], tag="L5", t_now_abs=t, now_w=t)
        dr.dr_joint_shadow(c)
        t += PASS_S
    check(not [m for m in logs if "disabled" in m], "dr_joint_shadow ran every pass")
    for p in prns:
        h = c.minnov_hist.get(p, [])
        win = _in_window(h, c.t0)
        check(
            len(win) == 301,
            "PRN %d: MINNOV holds %d samples of the 600 s window (301)" % (p, len(win)),
        )


def test_innov_site():
    """The INNOV write in stage_detections_to_seeds goes through the windowed helper and keeps
    no count cap of its own."""
    with open(seeding.__file__) as f:
        tree = ast.parse(f.read(), seeding.__file__)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "stage_detections_to_seeds"
    ][0]
    calls = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "append_windowed"
        and "ctx.innov_hist" in ast.unparse(n.args[0])
    ]
    caps = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Delete)
        and any(
            isinstance(tg, ast.Subscript)
            and isinstance(tg.slice, ast.Slice)
            and ast.unparse(tg.value) in ("_ih", "ctx.innov_hist[prn]")
            for tg in n.targets
        )
    ]
    check(len(calls) == 1, "INNOV appends through append_windowed (%s)" % calls)
    check(not caps, "INNOV keeps no count cap of its own (%s)" % caps)


def test_innov_once_per_detection():
    """The search re-serves a detection, with its ref_hop, until its next pass; the INNOV
    history takes it once, so innov_n_10m counts detections and the p95 is theirs."""
    from gnss_broker import test_escape_dedup as td

    ctx = td.held_ctx()
    td.run(ctx, td.PRN, [td.H0 + td.STEP] * 10, td._off)
    n = len(ctx.innov_hist.get(td.PRN, []))
    check(n == 1, "one detection served for 10 cycles: %d INNOV sample(s) (1)" % n)
    ctx = td.held_ctx()
    td.run(ctx, td.PRN, [td.H0 + k * td.STEP for k in range(1, 9)], td._off)
    h = ctx.innov_hist.get(td.PRN, [])
    check(
        len(h) == 8 and all(abs(v - td.OFF) < 1e-6 for _, v in h),
        "8 detections: 8 samples, each the detection's +2 chips (%d)" % len(h),
    )


class _JSCount(_JS):
    """_JS that counts the feed: how many cycle() calls carried a measurement."""

    def __init__(self, keys):
        _JS.__init__(self, keys)
        self.fed = 0

    def cycle(self, meas, t):
        if meas:
            self.fed += 1


def test_warmup_cycle_clock():
    """B16/N01: the joint-feed warmup counts ctx.uptime_s, the cycle clock, never wall time.

    A replay: the recorded cycle clock sits far from this process's wall clock, and the run
    is driven for 600 s of recorded time in a fraction of a second of wall time. The feed must
    start exactly at joint_feed_warmup_s of recorded uptime.
    """
    logs = []
    dr._log = lambda msg: logs.append(msg)
    dr._log_rl = lambda key, msg, every_s=10.0: logs.append(msg)
    prns = (3, 5)
    js = _JSCount([("L5", p) for p in prns])
    c = NS()
    c.args = NS(
        joint_shadow=True,
        chip_rate_hz=10.23e6,
        joint_feed_warmup_s=240.0,
        joint_min_snr=10.0,
        joint_mask_after=5,
        joint_sigma=0.3,
    )
    c.band_id, c.code_len = "1176.45MHz", 10230.0
    c.rx = NS(joint_receiver=lambda *a, **k: js, joint_ambiguous=lambda: False)
    c.dr_state = {"clk": 150.0, "joint_log_next": 1e18}
    c.best = {p: (30.0,) for p in prns}
    c.minnov_hist = {}
    c.track_ok = lambda p: True
    c.p2c_hold = lambda js_, key: False
    c.p2c_tick = lambda js_, t: None
    t_first = 1.6e9  # a recorded clock, nowhere near time.time()
    fed_at = []
    t = t_first
    while t <= t_first + 600.0:
        c.t0 = t
        c.uptime_s = t - t_first
        c.drp = NS(offs=[(p, 0.5) for p in prns], tag="L5", t_now_abs=t, now_w=t)
        before = js.fed
        dr.dr_joint_shadow(c)
        if js.fed > before:
            fed_at.append(c.uptime_s)
        t += PASS_S
    check(
        bool(fed_at) and min(fed_at) >= 240.0,
        "nothing feeds the state before 240 s of cycle-clock uptime (first feed at %s)"
        % (fed_at[:1] or "never"),
    )
    check(
        bool(fed_at) and min(fed_at) < 240.0 + PASS_S,
        "the feed starts at the warmup on the recorded clock, not on wall time (first at %s)"
        % (fed_at[:1] or "never"),
    )
    check(
        any("JFEED WARMUP" in m for m in logs),
        "the warmup logs while it withholds",
    )
    check(not hasattr(c, "broker_t0"), "no broker_t0 on the context: the wall-clock stamp is gone")


def main():
    for fn in (
        test_helper,
        test_minnov_append,
        test_innov_site,
        test_innov_once_per_detection,
        test_warmup_cycle_clock,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
