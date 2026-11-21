"""Inside the boresight transit cone the seeding stage reads only the transiting satellite (#119).

    python3 -m gnss_broker.test_transit_dets     (from python/scripts/gnss)

A bright satellite near boresight leaks into every other PRN's search at the fit's SNR bar;
read, those detections poisoned the cp-fit history (the flush cascade) and re-seeded un-held
PRNs. The checks drive stage_detections_to_seeds on test_escape_dedup's minimal context with
G13 0.7 deg off boresight: PRN 7's detections leave no history and no seed, G13's own enter,
and with the satellite 30 deg away PRN 7's enter too.

@author Keith Vanderlinde
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import sky  # noqa: E402
from gnss_broker.test_escape_dedup import (
    STEP,
    check,
    run,
    seeding_ctx,
    _fails,
)  # noqa: E402

H = 10_000_000
BRIGHT = 13
VICTIM = 7


def _ctx(bright_el):
    ctx = seeding_ctx(dr_clock_transit_freeze_deg=5.0)
    ctx.transit_pd = {
        ("G", BRIGHT): {"el": bright_el, "az": 180.0},
        ("G", VICTIM): {"el": 30.0, "az": 90.0},
    }
    return ctx


def _dets():
    return [H + k * STEP for k in range(6)]


def test_victim_ignored_in_cone():
    ctx = _ctx(62.0)
    logs = run(ctx, VICTIM, _dets(), lambda h: 1234.0)
    check(
        not ctx.cpt.hist.get(VICTIM),
        "in the cone the victim's detections enter no history",
    )
    check(VICTIM not in ctx.seeds, "...and seed nothing")
    check(
        any("BORESIGHT TRANSIT G13" in m for lines in logs for m in lines),
        "...and the skip is logged with the transiting satellite",
    )


def test_transiting_satellite_read_in_cone():
    ctx = _ctx(62.0)
    run(ctx, BRIGHT, _dets(), lambda h: 1234.0)
    check(
        len(ctx.cpt.hist.get(BRIGHT, [])) == 6 and BRIGHT in ctx.seeds,
        "the transiting satellite's own detections are read (%d points)"
        % len(ctx.cpt.hist.get(BRIGHT, [])),
    )


def test_victim_read_outside_cone():
    ctx = _ctx(32.7)
    run(ctx, VICTIM, _dets(), lambda h: 1234.0)
    check(
        len(ctx.cpt.hist.get(VICTIM, [])) == 6 and VICTIM in ctx.seeds,
        "with the satellite 30 deg away the victim's detections are read (%d points)"
        % len(ctx.cpt.hist.get(VICTIM, [])),
    )


def main():
    sky.set_boresight(180.0, 62.7, "test")
    try:
        for t in (
            test_victim_ignored_in_cone,
            test_transiting_satellite_read_in_cone,
            test_victim_read_outside_cone,
        ):
            print(t.__name__)
            t()
    finally:
        sky.set_boresight(None, None, None)
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
