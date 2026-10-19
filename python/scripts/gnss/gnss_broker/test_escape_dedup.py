"""The hold/escape referee counts each detection once, however long the search re-serves it.

    python3 -m gnss_broker.test_escape_dedup     (from python/scripts/gnss)

The search serves a PRN's latest detection, unchanged and with its ref_hop, until its next
pass, and seeding.stage_detections_to_seeds sees it every broker cycle. The escape referee
releases a held seed after 5 consecutive sign-consistent disagreements with the search whose
9-sample median also exceeds hold_max_cp_err; both rules assume independent detections. Fed
one re-served detection, they would release a hold on a single off-peak fix.

The checks drive stage_detections_to_seeds itself on a minimal context: one held PRN whose
frozen seed the detections disagree with by 2 chips (bar 1 chip), with a mature cp fit and
no veto. `seeding_ctx` and `run` are shared with test_innov_window and test_cprate.

@author Keith Vanderlinde
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import seeding  # noqa: E402
from gnss_broker.loopstate import CpTracking, HoldState  # noqa: E402
from gnss_broker.seed import Seed  # noqa: E402

HPS = 195312.5  # hops per second, CHORD
CHIP = 10.23e6
CARR = 1176.45e6
SGN = 1.0  # code_doppler_sign
L = 10230.0
DOP = 1000.0
H0 = 10_000_000  # the held seed's ref_hop (~51 s after sample 0)
STEP = int(10 * HPS)  # 10 s between the search's passes
CP0 = 4321.0
OFF = 2.0  # chips the detections sit off the held seed
PRN = 7

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def cp_to_seed_currency(pts, dop_seed, dop_rate=0.0):
    """gps_distributed_broker.main's conversion of search cp0 points into the seed's currency."""
    out = []
    h_anchor = pts[-1][0] if pts else 0
    for hh, cc, dd in pts:
        corr = (hh / HPS) * CHIP * (SGN * (dd - dop_seed) / CARR)
        dt = (hh - h_anchor) / HPS
        corr -= 0.5 * CHIP * SGN * dop_rate / CARR * dt * dt
        out.append((hh, (cc + corr) % L))
    return out


def seeding_ctx(**args):
    """A context stage_detections_to_seeds runs on: no almanac, no dead reckoning, no overlay,
    the code-rate cross-check off. `args` overrides the flags."""
    a = NS(
        dr_constellation="G",
        mask_deg=5.0,
        dll_deep_gate_from_search=0.0,
        almanac=False,
        dr_doppler_ignores_integrity=False,
        doppler_sign=1.0,
        seed_doppler="pred",
        fit_min_snr=10.0,
        dop_rate_min_pts=1000,
        dop_rate_min_span_s=600.0,
        dop_rate_max=2.0,
        dop_rate_model_tol=0.1,
        cp_rate_model_tol=0.0,
        fit_flush_on_reject=0,
        cp_rate_model_primary=0,
        hops_per_sec=HPS,
        chip_rate_hz=CHIP,
        carrier_hz=CARR,
        code_doppler_sign=SGN,
        acquire_snr=8.0,
        code_bias_max=3.0,
        nh_joint="off",
        nh_joint_min_snr=20.0,
        period_check_snr=30.0,
        nh_period_debounce=3,
        hold_max_cp_err=1.0,
        fit_maturity_span_s=30.0,
        escape_amp_veto=50.0,
        hold_snr=10.0,
        hold_on_present=0,
        dop_max_rate_hz=30.0,
        hold_max_dop_hz=100.0,
    )
    for k, v in args.items():
        setattr(a, k, v)
    return NS(
        args=a,
        best={},
        t0=0.0,
        dr_pd=None,
        dr_pd0=None,
        dr_pd2=None,
        dr_state=None,
        dr_untrusted={},
        up=None,
        pred={},
        cb=NS(seed=0.0, available=True, code_ema=None),
        dls=NS(deep_gate_seen={}, trim={}, readback={}, last={}),
        seeds={},
        cp_held=set(),
        cpt=CpTracking(),
        hold=HoldState(),
        max_gap_hops=120 * HPS,
        hist_len=40,
        cp_to_seed_currency=cp_to_seed_currency,
        dop_rate_rejected={},
        dop_rate_fitted={},
        cp_rate_rejected={},
        code_len=L,
        fitted=set(),
        la_samples=[],
        lc_seg=1,
        cl_report=[],
        status={},
        innov_hist={},
        mp_flipped=set(),
        mp_last_det={},
        sig_of_last=lambda rec: float((rec or {}).get("amp_snr", 0.0)),
        qpop=None,
        utc0_sample0=None,
        drp=NS(now_w=0.0),
    )


def run(ctx, prn, dets, cp_at, t0=1000.0, cycle_s=2.0):
    """One broker cycle per entry of `dets` (a detection's ref_hop, its cp0 cp_at(ref_hop));
    returns the log lines per cycle."""
    logs = []
    saved = seeding._log, seeding._log_rl
    seeding._log = lambda msg: logs[-1].append(msg)
    seeding._log_rl = lambda key, msg, every_s=10.0: logs[-1].append(msg)
    try:
        for i, h in enumerate(dets):
            logs.append([])
            ctx.fitted, ctx.la_samples, ctx.cl_report = set(), [], []  # per cycle
            ctx.t0 = t0 + i * cycle_s
            ctx.drp.now_w = ctx.t0
            ctx.best = {prn: (40.0, DOP, cp_at(h), h, -1, -1.0, -1.0)}
            seeding.stage_detections_to_seeds(ctx)
    finally:
        seeding._log, seeding._log_rl = saved
    return logs


def held_ctx():
    """PRN 7 held on a frozen seed at H0, with six earlier detections behind a mature fit."""
    ctx = seeding_ctx()
    ctx.seeds[PRN] = Seed.born(
        "test",
        epoch=H0,
        doppler_hz=DOP,
        code_phase_chips=CP0,
        code_phase_rate=0.0,
        ref_hop=H0,
    )
    ctx.cp_held.add(PRN)
    ctx.cpt.hist[PRN] = [(H0 - k * STEP, CP0 + OFF, DOP) for k in range(6, 0, -1)]
    ctx.status[PRN] = {"amp_snr": 20.0}
    return ctx


def _off(h):
    return CP0 + OFF


def _escapes(logs):
    """1-based cycles that logged an ESCAPE for PRN 7."""
    return [
        i + 1
        for i, lines in enumerate(logs)
        if any(m.startswith("ESCAPE PRN %d" % PRN) for m in lines)
    ]


def test_one_detection_never_escapes():
    ctx = held_ctx()
    logs = run(ctx, PRN, [H0 + STEP] * 12, _off)
    check(_escapes(logs) == [], "12 cycles of one re-served detection: no ESCAPE")
    check(
        len(ctx.cpt.err_hist.get(PRN, [])) == 1,
        "...it enters the median history once (%d)"
        % len(ctx.cpt.err_hist.get(PRN, [])),
    )
    check(PRN in ctx.cp_held, "...and the hold stands")


def test_distinct_detections_escape():
    """Nine detections: five fill the median's minimum, and the last five count."""
    ctx = held_ctx()
    logs = run(ctx, PRN, [H0 + k * STEP for k in range(1, 10)], _off)
    check(_escapes(logs) == [9], "9 distinct detections 2 chips off: ESCAPE on the 9th")


def test_reserved_detections_escape_on_detections():
    """Each detection seen on 3 cycles: the escape still takes 9 detections, 25 cycles."""
    ctx = held_ctx()
    dets = [H0 + k * STEP for k in range(1, 10) for _ in range(3)]
    logs = run(ctx, PRN, dets, _off)
    check(
        _escapes(logs) == [25],
        "9 detections, each served 3 cycles: ESCAPE on cycle 25 %s" % _escapes(logs),
    )


def main():
    for fn in (
        test_one_detection_never_escapes,
        test_distinct_detections_escape,
        test_reserved_detections_escape_on_detections,
    ):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
