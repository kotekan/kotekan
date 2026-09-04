"""--nh-joint: the joint overlay fit -- the verdict traps, not the plumbing.

Unlike test_nhdebounce this imports the REAL helpers from seeding.py (they are
standalone functions); only the inline apply/referee glue is proven on sky
(NH-JOINT / nhjapp / nhjref / nhjskip log lines), not here.

    python3 -m gnss_broker.test_nhjoint

@author Keith Vanderlinde
"""

import sys
import types

from gnss_broker.loopstate import CpTracking
from gnss_broker.seeding import (_nh_joint_consensus, _nh_joint_pred_chips, _nh_joint_snap,
                                 _nh_joint_vote)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


CODE_LEN = 10230.0
LC_SEG = 20
LLC = CODE_LEN * LC_SEG
CHIP_RATE = 10.23e6
T0 = 1788200000.0


def mk_ctx(delta_true_chips=0.0):
    a = types.SimpleNamespace(chip_rate_hz=CHIP_RATE, hops_per_sec=195312.5,
                              almanac=True, nh_joint="apply", nh_joint_min_prns=3,
                              nh_joint_min_snr=30.0, nh_joint_window_s=600.0,
                              nh_joint_tol_chips=1500.0, period_check_snr=60.0)
    ctx = types.SimpleNamespace(args=a, lc_seg=LC_SEG, lc_epoch=0.02, code_len=CODE_LEN,
                                utc0_sample0=T0, cpt=CpTracking(),
                                pred={}, drp=types.SimpleNamespace(now_w=T0 + 100.0))
    # 8 satellites with distinct delays and sat clocks
    for i, prn in enumerate((1, 3, 4, 6, 9, 23, 27, 32)):
        ctx.pred[prn] = (0.0, 0.0, 45.0, (0.067 + 0.0025 * i) * 3.0e8,
                         (-3e-4 + 8e-5 * i))
    ctx._delta = delta_true_chips
    return ctx


def true_ph(ctx, prn, ref_hop, err_chips=0.0):
    """A measured phase consistent with the common offset (plus per-sat noise)."""
    t_ref = ctx.utc0_sample0 + ref_hop / ctx.args.hops_per_sec
    return (_nh_joint_pred_chips(ctx, prn, t_ref) + ctx._delta + err_chips) % LLC


def dev(a, b):
    return ((a - b + LLC / 2) % LLC) - LLC / 2


def test_consensus_resolves():
    ctx = mk_ctx(delta_true_chips=74679.0)  # receiver clock +7.3 ms
    for j, prn in enumerate((1, 3, 4, 6, 9)):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000 + j, (-2, 1, 0, 2, -1)[j]),
                       1000 + j, 120.0, ctx.drp.now_w)
    c = _nh_joint_consensus(ctx, ctx.drp.now_w)
    check(c is not None, "five agreeing sats resolve")
    check(c and abs(dev(c[0], 74679.0)) < 5.0,
          "resolved offset within 5 chips of truth (got %s)" % (c and dev(c[0], 74679.0)))


def test_min_prns_falsifier():
    ctx = mk_ctx(delta_true_chips=50000.0)
    for prn in (1, 3):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000), 1000, 120.0, ctx.drp.now_w)
    check(_nh_joint_consensus(ctx, ctx.drp.now_w) is None,
          "two sats CANNOT resolve (min_prns)")


def test_split_votes_do_not_resolve():
    ctx = mk_ctx(delta_true_chips=30000.0)
    for prn in (1, 3, 4):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000), 1000, 100.0, ctx.drp.now_w)
    for prn in (6, 9, 23):  # a second population half an epoch away
        _nh_joint_vote(ctx, prn, (true_ph(ctx, prn, 1000) + LLC / 2) % LLC,
                       1000, 100.0, ctx.drp.now_w)
    check(_nh_joint_consensus(ctx, ctx.drp.now_w) is None,
          "a 50/50 split does NOT resolve (weight fraction)")


def test_boundary_is_not_special():
    # receiver clock exactly on a segment boundary: an INTEGER vote would split 50/50
    # between adjacent labels; the continuous consensus must still resolve one value.
    ctx = mk_ctx(delta_true_chips=0.5 * CODE_LEN)
    errs = (-30.0, 25.0, -20.0, 30.0, -25.0, 20.0)
    for j, prn in enumerate((1, 3, 4, 6, 9, 23)):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000 + j, errs[j]),
                       1000 + j, 100.0, ctx.drp.now_w)
    c = _nh_joint_consensus(ctx, ctx.drp.now_w)
    check(c is not None, "boundary-straddling clock still resolves")
    check(c and abs(dev(c[0], 0.5 * CODE_LEN)) < 35.0,
          "...to the true continuous offset")


def test_noisy_sat_is_carried():
    ctx = mk_ctx(delta_true_chips=74679.0)
    for j, prn in enumerate((1, 3, 4, 6, 9)):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000 + j), 1000 + j, 120.0,
                       ctx.drp.now_w)
    # G1-style: fine phase right, segment pure noise (7 periods off)
    prn, wrong = 27, 7
    ph_meas = (true_ph(ctx, prn, 2000) + wrong * CODE_LEN) % LLC
    _nh_joint_vote(ctx, prn, ph_meas, 2000, 40.0, ctx.drp.now_w)  # its vote is an outlier
    c = _nh_joint_consensus(ctx, ctx.drp.now_w)
    check(c is not None, "consensus survives one noise vote")
    snapped, k = _nh_joint_snap(ctx, prn, ph_meas, 2000, c[0])
    check(k == -wrong, "the noisy sat's segment is corrected by %+d (got %+d)"
          % (-wrong, k))
    check(abs(dev(snapped, true_ph(ctx, prn, 2000))) < 1.0,
          "...keeping its measured fine phase")


def test_consensus_kept_when_votes_expire():
    ctx = mk_ctx(delta_true_chips=12345.0)
    for j, prn in enumerate((1, 3, 4, 6)):
        _nh_joint_vote(ctx, prn, true_ph(ctx, prn, 1000), 1000, 100.0, ctx.drp.now_w)
    c1 = _nh_joint_consensus(ctx, ctx.drp.now_w)
    check(c1 is not None, "resolves")
    later = ctx.drp.now_w + 2 * ctx.args.nh_joint_window_s
    c2 = _nh_joint_consensus(ctx, later)
    check(c2 is not None and c2[0] == c1[0],
          "a resolved consensus is KEPT when the votes age out (run constant)")
    check(not ctx.cpt.nh_votes, "...while the stale votes themselves are pruned")


def main():
    for fn in (test_consensus_resolves, test_min_prns_falsifier,
               test_split_votes_do_not_resolve, test_boundary_is_not_special,
               test_noisy_sat_is_carried, test_consensus_kept_when_votes_expire):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
