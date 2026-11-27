"""The NH alignment hint (--nh-hint) ages its evidence from the detection, not from learning.

    python3 -m gnss_broker.test_searchhint

`stage_narrow_search` learns one receiver overlay offset from the search's (nh, ref_hop)
reports and drops it once fewer than nh_hint_min_samples are younger than nh_hint_max_age_s.
Nothing is learned while the clock solve is not established, so a detection made then is
learned when the solve recovers. This drives the stage with a stub ctx through such a gap and
checks that a detection older than the age bar is not adopted when learning resumes, while a
fresh one is.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import searchhint
from gnss_broker.loopstate import NhOverlay


_fails = []

U0 = 1.79e9  # utc0_sample0: UTC s of sample 0
HOPS = 25.0  # hops per second


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_ctx():
    return _Obj(
        args=_Obj(
            narrow_search=True,
            almanac=True,
            search_margin_hz=50.0,
            search_margin_wide_hz=500.0,
            nh_hint=True,
            code_length=10230,
            chip_rate_hz=10.23e6,
            nh_overlay_len=20,
            hops_per_sec=HOPS,
            nh_hint_max_age_s=120.0,
            nh_hint_min_samples=1,
            mask_deg=10.0,
        ),
        pred={5: (-1200.0, 0.4, 45.0, 2.2e7, 1e-5)},
        unhealthy=set(),
        up=None,
        unpredicted=set(),
        warm={"i": 0, "t": float("-inf"), "cur": []},
        capable=None,
        cb=_Obj(ema=None, stale=False, value=0.0),
        utc0_sample0=U0,
        nho=NhOverlay(),
        dr_state={"integ": {}},
        detectors=[],
        t0=U0,
    )


def cycle(ctx, t, healthy, det_t):
    """One stage call at t s after sample 0, the solve established or not, PRN 5's last
    detection made at det_t s; returns the lines logged."""
    lines = []
    ctx.t0 = U0 + t
    ctx.dr_state = {"integ": {1: (0.1, ctx.t0), 2: (0.2, ctx.t0)} if healthy else {}}
    ctx.nho.seen = {5: (7, int(round(det_t * HOPS)))}
    saved = (searchhint._log, searchhint._log_rl, searchhint.axis_now)
    saved_now = getattr(searchhint, "now_utc_s", None)
    searchhint._log = lambda msg: lines.append(msg)
    searchhint._log_rl = lambda key, msg, every_s=10.0: lines.append(msg)
    searchhint.axis_now = lambda c: c.t0 - c.utc0_sample0
    searchhint.now_utc_s = lambda c: c.t0
    try:
        searchhint.stage_narrow_search(ctx)
    finally:
        searchhint._log, searchhint._log_rl, searchhint.axis_now = saved
        if saved_now is None:
            del searchhint.now_utc_s
        else:
            searchhint.now_utc_s = saved_now
    return lines


def test_warm_search():
    """--warm-search-prns: the PRNs no orbit source predicts take turns, two at a time for
    WARM_DWELL_S each, over the whole MEO Doppler range about the clock bias; a turn's end
    clears its PRNs ahead of the new hints; a seeded one is hinted at its seed's Doppler; and
    the turns stop once nothing is unpredicted."""
    print("\nthe warm search for PRNs no orbit predicts")
    ctx = make_ctx()
    ctx.args.nh_hint = False
    ctx.args.warm_search_prns = 2
    ctx.args.carrier_hz = 1176.45e6
    ctx.detectors = ["d0"]
    ctx.seeds = {}
    ctx.cb.value = 100.0
    ctx.unpredicted = {3, 9, 12}
    posts = []
    saved = searchhint._post
    searchhint._post = lambda url, body: posts.append(list(body))

    def at(t):
        cycle(ctx, t, True, t)
        return {(h["prn"], h["margin_hz"] < 0): h for h in posts[-1]}, posts[-1]

    wide = 500.0 + searchhint.WARM_RANGE_RATE_MPS * 1176.45e6 / 299792458.0
    try:
        h, _ = at(0.0)
        check(
            sorted(p for p, c in h if not c) == [3, 5, 9]
            and h[(3, False)]["margin_hz"] == wide
            and h[(3, False)]["doppler_hz"] == 100.0,
            "first turn: PRNs 3 and 9 over +-%.0f Hz about the bias, beside PRN 5's hint"
            % wide,
        )
        h, _ = at(5.0)
        check(not any(c for _, c in h), "no clears inside a turn")
        h, body = at(10.0)
        first = [b["prn"] for b in body if b["margin_hz"] < 0]
        check(
            sorted(first) == [3, 9]
            and sorted(p for p, c in h if not c) == [3, 5, 12]
            and body.index(h[(3, True)]) < body.index(h[(3, False)]),
            "next turn: 3 and 9 cleared first, then 12 and 3 hinted (%s)" % body,
        )
        ctx.seeds = {9: {"doppler_hz": -2000.0}}
        h, _ = at(12.0)
        check(
            h.get((9, False), {}).get("doppler_hz") == -2000.0
            and h[(9, False)]["margin_hz"] == 500.0,
            "a seeded one is hinted at its seed's Doppler, the usual margin",
        )
        ctx.unpredicted = set()
        h, _ = at(20.0)
        check(
            sorted(p for p, c in h if c) == [3, 12]
            and sorted(p for p, c in h if not c) == [5]
            and not ctx.warm["cur"],
            "nothing unpredicted: the last turn is cleared and no turn follows",
        )
    finally:
        searchhint._post = saved


def main():
    test_warm_search()
    print("\nthe NH hint's sample ages")

    # 1. A detection at t=10 s, the solve not established until t=300 s: learned at 300 s,
    #    it is 290 s old against a 120 s bar and must not set the offset.
    ctx = make_ctx()
    lines = []
    for t in range(0, 300, 10):
        lines += cycle(ctx, float(t), False, 10.0)
    check(not ctx.nho.off_hist, "nothing learned while the solve is not established")
    lines += cycle(ctx, 300.0, True, 10.0)
    check(not any("nh hint failed" in m for m in lines), "the hint computation ran")
    check(len(ctx.nho.off_hist) == 1, "the held-back detection is learned at 300 s")
    check(
        ctx.nho.offset[0] is None,
        "a 290 s old detection is not adopted (offset %r)" % ctx.nho.offset[0],
    )

    # 2. A detection at t=295 s, learned at 300 s: fresh, adopted.
    ctx = make_ctx()
    lines = cycle(ctx, 300.0, True, 295.0)
    check(not any("nh hint failed" in m for m in lines), "the hint computation ran")
    check(
        ctx.nho.offset[0] is not None,
        "a 5 s old detection sets the offset (offset %r)" % ctx.nho.offset[0],
    )

    # 3. The same detection from a satellite with a failing health word: not learned.
    ctx = make_ctx()
    ctx.unhealthy = {5}
    lines = cycle(ctx, 300.0, True, 295.0)
    check(not any("nh hint failed" in m for m in lines), "the hint computation ran")
    check(
        not ctx.nho.off_hist and ctx.nho.offset[0] is None,
        "an unhealthy satellite's detection is not learned (%r)" % ctx.nho.off_hist,
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the NH hint's sample ages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
