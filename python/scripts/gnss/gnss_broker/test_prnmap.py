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
        self.prn_reconfig_heartbeat_s = 1e18   # off by default in tests; check 11 arms it
        self.probe_require_slot = False
        self.noise_probes = 0
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
LOGS = []


def _rec_log(msg, *a):
    LOGS.append(msg % a if a else msg)


def _rec_log_rl(key, msg, every_s=0.0):
    LOGS.append(msg)


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
    prnmap._log = _rec_log
    prnmap._log_rl = _rec_log_rl

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

    # ---- 4. the admit mask governs EVICTIONS, not free slots ---------------------------
    # ⚠️ THIS TEST ASSERTED THE BUG until 2026-08-27. The up-now bar exists because an
    # eviction costs a re-acquisition -- only pay it for a satellite we can use immediately.
    # A FREE slot costs nothing, so applying the same bar to one left five Galileo slots
    # empty while E36 (active, and the satellite the whole mechanism exists for) was refused
    # for being below the horizon at that moment. A below-horizon satellite in a free slot is
    # not idle capacity: it is exactly what the noise PROBES need.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 4.0})   # PRN 1 GONE -> slot 0 is FREE
    run_cycle(ctx, 1000.0)
    check(not POSTS, "a free slot is not filled before the gone-hold has elapsed")
    run_cycle(ctx, 1000.0 + 7201.0)
    check(any("36" in str(pl) for _u, pl in POSTS),
          "a FREE slot IS filled by a 4 deg satellite -- nothing to re-acquire, and it is a "
          "probe today and a tracked satellite tomorrow")

    # ... but it must NOT buy an eviction. Every slot occupied by a satellite that is UP:
    # nothing is free, and a 4 deg candidate is not worth a re-acquisition.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: 20.0, 2: 40.0, 3: 30.0, 36: 4.0})
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS,
          "a 4 deg satellite does NOT evict anyone (the up-now bar still governs swaps)")

    # ---- 4b. the probe supply is never evicted -----------------------------------------
    # A satellite deep below the horizon for hours IS the chain's noise anchor. Trading it
    # for one more tracked satellite costs the presence gate its floor, which is worth far
    # more than the satellite the swap bought.
    POSTS[:] = []
    deep = {1: -70.0, 2: -65.0, 3: -60.0, 36: 83.0}      # all three held sats are deep probes
    ctx = Ctx([1, 2, 3], deep)
    ctx.args.noise_probes = 3
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS,
          "with noise_probes=3 and exactly 3 deep-below-horizon slots, NONE is evicted even "
          "for an 83 deg satellite -- the probe supply is not spare capacity")
    deep4 = dict(deep); deep4[4] = -55.0   # a 4th deep satellite exists, unslotted
    ctx2 = Ctx([1, 2, 3], deep4)
    ctx2.args.noise_probes = 2                            # only 2 must be held back
    POSTS[:] = []
    run_cycle(ctx2, 1000.0)
    run_cycle(ctx2, 1000.0 + 100000.0)
    check(any("36" in str(pl) for _u, pl in POSTS),
          "... but with only 2 needed, the shallowest deep slot IS free to be traded")

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

    # ---- 11. the heartbeat: armed-and-idle must be VISIBLE -----------------------------
    # Having nothing to propose is the normal state, so a stage that only speaks when it acts
    # is indistinguishable from one that is not running. Nothing here is posted, so the
    # heartbeat is asserted on the LOG rather than on POSTS.
    POSTS[:] = []
    LOGS[:] = []
    ctx = Ctx([1, 2, 3], {1: 40.0, 2: 40.0, 3: 30.0}, prn_reconfig="report")
    ctx.args.prn_reconfig_heartbeat_s = 900.0
    run_cycle(ctx, 1000.0)
    check(any("PRN MAP" in m and "REPORT" in m for m in LOGS),
          "an armed chain with NOTHING to propose still says so (the heartbeat)")
    check(not POSTS, "... and still posts nothing")
    n_beats = len([m for m in LOGS if "REPORT" in m])
    run_cycle(ctx, 1000.0 + 100.0)
    check(len([m for m in LOGS if "REPORT" in m]) == n_beats,
          "... rate-limited: not once per cycle")
    run_cycle(ctx, 1000.0 + 901.0)
    check(len([m for m in LOGS if "REPORT" in m]) == n_beats + 1,
          "... and it does beat again after the interval")

    # ---- 12. the map is PUBLISHED for read-only consumers, and fails OPEN ---------------
    # --probe-require-slot needs the live map but proposes no swaps, so "off" must still
    # poll -- and a consumer must never act on a partial or split map. This is what stops
    # the probe filter from silently emptying the probe set on a half-swept fleet.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0}, prn_reconfig="off")
    ctx.args.probe_require_slot = True
    run_cycle(ctx, 1000.0)
    check(ctx.prnmap.consensus == [1, 2, 3],
          "prn_reconfig=off + probe_require_slot: the map is polled and PUBLISHED")
    check(not POSTS, "... and still nothing is posted (off is off for SWAPS)")

    ctx2 = Ctx([1, 2, 3], {2: 40.0, 3: 30.0}, prn_reconfig="off")
    ctx2.args.probe_require_slot = True
    CTX[0] = ctx2
    ctx2.t0 = 1000.0
    ctx2.prnmap.poll_t = 1000.0
    ctx2.prnmap.maps = {"http://node1": [1, 2, 3]}          # HALF a sweep (2 endpoints)
    prnmap.stage_prn_membership(ctx2)
    check(ctx2.prnmap.consensus is None,
          "a HALF-SWEPT fleet publishes None -- consumers fail open, never on a partial map")

    ctx2.prnmap.maps = {"http://node1": [1, 2, 3], "http://node2": [1, 2, 9]}   # split
    prnmap.stage_prn_membership(ctx2)
    check(ctx2.prnmap.consensus is None, "... and so does a SPLIT fleet")

    ctx3 = Ctx([1, 2, 3], {2: 40.0}, prn_reconfig="off")   # flag off entirely
    run_cycle(ctx3, 1000.0)
    check(ctx3.prnmap.consensus is None,
          "with BOTH off the stage does nothing at all (no new GET -- replay stays exact)")

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
