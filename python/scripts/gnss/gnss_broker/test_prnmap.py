"""Live PRN membership: the swap policy, exercised without a node or a sky.

    python3 -m gnss_broker.test_prnmap

WHY THIS EXISTS RATHER THAN A FIXTURE. The digest gate replays logged cycles; this stage
GETs and POSTs to node endpoints that a replay does not have, so its decision function would
never run and would never be compared to anything -- the blind spot that let the dead C_LIGHT
import ship. And unlike most broker policy, the cost of getting this one wrong is not a bad
number: it is a SLOT SWAP, which throws away a working satellite's acquisition. So the
hysteresis is asserted here directly.

WHAT IS BEING PINNED:
  * an incumbent that is merely DOWN is not evictable until down_hold has elapsed, and the
    clock RESETS if it comes back up (a satellite that flickers must never churn a slot);
  * an incumbent GONE from BRDC is evictable sooner, but not instantly;
  * a candidate below the admit mask claims nothing, however long a slot has been free;
  * `report` posts NOTHING, ever -- the property that makes it safe to arm first;
  * nodes that disagree stop the stage dead rather than being driven to a consensus nobody
    chose (nothing in this pipeline is per-node, so a split map is a fault, not a state).

@author Keith Vanderlinde
"""

import sys

from gnss_broker import prnmap


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class Args(object):
    def __init__(self, **kw):
        self.prn_reconfig = "apply"
        self.prn_reconfig_poll_s = 60.0
        self.prn_reconfig_interval_s = 900.0
        self.prn_reconfig_admit_deg = 10.0
        self.prn_reconfig_evict_deg = 0.0
        self.prn_reconfig_down_hold_s = 10800.0
        self.prn_reconfig_gone_hold_s = 7200.0
        self.prn_reconfig_timeout_s = 5.0
        self.signal = "GAL_E5A_Q_CS"
        self.__dict__.update(kw)


class Ctx(object):
    """Only what prnmap reads: args, trackers, t0, pred, and its own owner."""

    def __init__(self, slots, pred, t0=1000.0, **kw):
        self.args = Args(**kw)
        self.trackers = ["http://node1", "http://node2"]
        self.detectors = ["http://search1"]   # a follower: gets the map, is not polled
        self.t0 = t0
        self.pred = {p: (0.0, 0.0, el, 0.0, 0.0) for p, el in pred.items()}
        self.prnmap = prnmap.PrnMapState()
        # What the fake nodes HOLD. The stage clears its cache after every POST and will not
        # decide again until a full sweep has read the map back, so the test has to model the
        # nodes actually applying it -- which is the loop being tested, not scaffolding.
        self.node_state = list(slots)
        self.prnmap.maps = {ep: list(slots) for ep in self.trackers}
        self.prnmap.poll_t = t0  # pre-swept: no HTTP in this test


POSTS = []


def _no_get(url, timeout=5.0):
    raise AssertionError("the test pre-polls; no GET should happen: " + url)


CTX = [None]


def _rec_post(url, payload, timeout=5.0):
    POSTS.append((url, payload))
    if CTX[0] is not None:
        CTX[0].node_state = list(payload["prns"])  # the nodes apply it
    return {}


def run_cycle(ctx, t):
    """One broker cycle, with the read-back sweep already complete."""
    CTX[0] = ctx
    ctx.t0 = t
    ctx.prnmap.poll_t = t  # the sweep is stood in for below, so no HTTP happens
    ctx.prnmap.maps = {ep: list(ctx.node_state) for ep in ctx.trackers}
    prnmap.stage_prn_membership(ctx)


def main():
    prnmap._get = _no_get
    prnmap._post = _rec_post

    # ---- 1. a DOWN incumbent is held until the hysteresis expires ----------------------
    # Slot 0 holds PRN 1, which is below the horizon; PRN 36 is up at 83 deg with no slot --
    # the measured 2026-08-26 case, minus the dead slots.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: -20.0, 2: 40.0, 3: 30.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    check(not POSTS, "a down incumbent is NOT evicted on first sight")
    run_cycle(ctx, 1000.0 + 3600.0)
    check(not POSTS, "... nor an hour later, with down_hold at 3 h")
    run_cycle(ctx, 1000.0 + 10801.0)
    check(len([u for u, _ in POSTS if "node" in u]) == 2,
          "... but IS after down_hold, posted to both nodes")
    check(len([u for u, _ in POSTS if "search" in u]) == 1,
          "the SEARCH is driven with the same map (it holds its own copy and has no frame to "
          "learn it from, unlike the assembler)")
    if POSTS:
        check(POSTS[0][1]["prns"] == [36, 2, 3],
              "the swap replaces the down incumbent in ITS slot, leaving the rest alone")

    # ---- 2. a satellite that comes back up RESETS the clock ----------------------------
    # This is the whole reason for the hysteresis: BRDC visibility flickers, and a slot that
    # flickers is a satellite that never locks.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: -20.0, 2: 40.0, 3: 30.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 10000.0)
    ctx.pred[1] = (0.0, 0.0, 15.0, 0.0, 0.0)      # PRN 1 rises again
    run_cycle(ctx, 1000.0 + 10100.0)
    ctx.pred[1] = (0.0, 0.0, -20.0, 0.0, 0.0)     # and sets again
    run_cycle(ctx, 1000.0 + 10200.0)
    check(not POSTS, "a brief return above the mask restarts the down clock")
    run_cycle(ctx, 1000.0 + 10200.0 + 10801.0)
    check(len(POSTS) == 3, "... and the full hold must elapse again from the new start")

    # ---- 3. GONE from BRDC evicts sooner, but still not instantly ----------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0})   # PRN 1 absent entirely
    run_cycle(ctx, 1000.0)
    check(not POSTS, "a dead slot is not reclaimed on first sight either")
    run_cycle(ctx, 1000.0 + 7201.0)
    check(len(POSTS) == 3, "... but at gone_hold (2 h), sooner than a down slot's 3 h")

    # ---- 4. a candidate below the admit mask claims nothing ----------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 4.0})    # slot free, candidate grazing
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS,
          "a 4 deg satellite does not claim a slot however long one has been free "
          "(a swap costs a re-acquisition; buy it for a satellite we can USE)")

    # ---- 5. report mode posts nothing, ever -------------------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0}, prn_reconfig="report")
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS, "report mode NEVER posts -- the property that makes it safe to arm first")

    # ---- 6. off does nothing at all ---------------------------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0}, prn_reconfig="off")
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS, "off is off")

    # ---- 7. disagreeing nodes stop the stage ------------------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0})

    def split_cycle(t):
        CTX[0] = ctx
        ctx.t0 = t
        ctx.prnmap.poll_t = t
        ctx.prnmap.maps = {"http://node1": [1, 2, 3], "http://node2": [1, 2, 9]}
        prnmap.stage_prn_membership(ctx)

    split_cycle(1000.0)
    split_cycle(1000.0 + 100000.0)
    check(not POSTS,
          "nodes that disagree stop the stage -- nothing here is per-node, so a split map "
          "is a fault to fix and not a state to drive out of")

    # ---- 8. no prediction this cycle -> change nothing ---------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {})
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS, "an empty prediction evicts nobody (a BRDC outage is not a constellation "
                     "outage -- the 2026-08-19 stale-EOP lesson)")

    # ---- 9. one slot per interval ------------------------------------------------------
    POSTS[:] = []
    # PRNs 1 AND 2 both gone from BRDC, TWO satellites waiting: everything is in place for
    # two swaps, and exactly one may happen.
    ctx = Ctx([1, 2, 3], {3: 30.0, 30: 70.0, 36: 83.0})
    run_cycle(ctx, 1000.0)                      # start both gone-clocks
    run_cycle(ctx, 1000.0 + 7201.0)             # both now evictable
    check(len([u for u, _ in POSTS if "node" in u]) == 2,
          "with two evictable slots and two candidates, ONE swap posts")
    n_first = len(POSTS)
    run_cycle(ctx, 1000.0 + 7202.0)
    check(len(POSTS) == n_first, "a second swap inside the interval is refused")
    run_cycle(ctx, 1000.0 + 7202.0 + 901.0)
    check(len(POSTS) == n_first + 3, "... and allowed once the interval has passed")

    # ---- 10. the loop CONVERGES: once the nodes hold the map, nothing more is posted ----
    # The stage decides from the READ-BACK map, so a POST that took must stop being re-issued.
    # If it did not, this would swap one slot every interval forever.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {3: 30.0, 36: 83.0})   # PRN 1 and 2 gone, one candidate
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 7201.0)
    n_after = len(POSTS)
    for k in range(1, 8):
        run_cycle(ctx, 1000.0 + 7201.0 + k * 1000.0)
    check(len(POSTS) == n_after,
          "once the nodes hold the new map, no further swap is posted (the read-back loop "
          "converges instead of re-issuing)")
    check(36 in ctx.node_state, "... and the nodes ended up holding the new satellite")

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
