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
        self.prn_reconfig_lead_s = 2.0
        self.prn_reconfig_axis_max_age_s = 30.0
        self.hops_per_sec = 125000.0
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
        # The F-engine hop the scheduler writes its deadline against, and the wall instant it
        # was sampled. None models "the axis is unknown", which must degrade to an UNSCHEDULED
        # post, not to no post. fe_hop_t defaults to t0 -- i.e. the poll happened at the top of
        # this cycle, which is where the real one happens.
        self.fe_hop_now = kw.pop("fe_hop_now", 1_000_000_000.0) \
            if "fe_hop_now" in kw else 1_000_000_000.0
        self.fe_hop_t = kw.pop("fe_hop_t", t0) if "fe_hop_t" in kw else t0


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

    # ---- 1. THE WHOLE MAP, EVERY CYCLE, WITH ELEVATION HYSTERESIS ---------------------
    # ⚠️ SUPERSEDES THE HOLD-TIMER TESTS (2026-08-27). The old policy held a down incumbent
    # for 3 h and a dead slot for 2 h before reusing either, with the clocks in this in-memory
    # state -- so every broker restart reset them and, measured after a day of restarts, the
    # eviction path had NEVER FIRED IN PRODUCTION: "7 dead, 15 below 0 deg" and "NO slot is
    # evictable" in one heartbeat, with E36 unslotted at 82 deg. KV: "3 below horizon + all
    # above", pushed regularly. The hysteresis those timers were standing in for is now
    # ELEVATION hysteresis, which no restart can lose.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: -20.0, 2: 40.0, 3: 30.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    check(len([u for u, _ in POSTS if "node" in u]) == 2,
          "a satellite up at 83 deg takes a below-horizon incumbent's slot IMMEDIATELY, on "
          "both nodes -- no hold timer, because a slot below the horizon is idle NOW")
    check(len([u for u, _ in POSTS if "search" in u]) == 1,
          "the SEARCH is driven with the same map (it holds its own copy and has no frame to "
          "learn it from, unlike the assembler)")
    if POSTS:
        check(POSTS[0][1]["prns"] == [36, 2, 3],
              "the swap replaces the down incumbent in ITS slot, leaving the rest alone")

    # ---- 2. HYSTERESIS: a held satellite is not dropped the instant it dips -------------
    # The lesson the down-clock encoded, and it still holds: BRDC visibility flickers, and a
    # slot that flickers is a satellite that never locks. It is now expressed in elevation --
    # admitted above admit_deg, KEPT until it falls below evict_deg -- so it survives a
    # restart, which the timer never did.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: 2.0, 2: 40.0, 3: 30.0, 9: 2.0})
    run_cycle(ctx, 1000.0)
    _held_kept = not POSTS or all(1 in pl["prns"] for _u, pl in POSTS)
    check(_held_kept, "a HELD satellite at +2 deg (below the 10 deg admit mask) is KEPT -- "
                      "that is the hysteresis band, and dropping it would be the flap")
    check(not any(9 in pl["prns"] for _u, pl in POSTS),
          "... while a NEW satellite at the same +2 deg is not admitted: the band has two "
          "edges or it is not hysteresis")

    # ---- 3. A DEAD SLOT IS REUSED AT ONCE ----------------------------------------------
    # A PRN with no ephemeris produces literally nothing. Waiting 2 h to reclaim its slot was
    # never justified by a re-acquisition cost, because there was nothing to re-acquire.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0})   # PRN 1 absent from BRDC entirely
    run_cycle(ctx, 1000.0)
    check(any(36 in pl["prns"] for _u, pl in POSTS),
          "a slot holding a PRN with NO ephemeris is reclaimed on sight")

    # ---- 4. A FREE SLOT GOES TO ANYONE, INCLUDING A LOW SATELLITE ----------------------
    # ⚠️ THIS TEST ASSERTED THE BUG until 2026-08-27 and then CAUGHT ITS REGRESSION the same
    # day. The admit mask exists to justify an EVICTION; a free slot is not an eviction. The
    # first version of the whole-map policy re-broke it: a satellite at +4 deg is neither "up"
    # (below the 10 deg admit mask) nor a probe (above -15 deg), so it fell in the gap and was
    # refused a slot that was standing empty.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 4.0})   # PRN 1 GONE -> slot 0 is FREE
    run_cycle(ctx, 1000.0)
    check(any(36 in pl["prns"] for _u, pl in POSTS),
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

    # ---- 9. MANY SLOTS, ONE POST, ONE DEADLINE -----------------------------------------
    # ⚠️ SUPERSEDES "one slot per interval" (2026-08-27). The old policy dribbled one slot out
    # per 15 min, so a sky that had moved took hours to follow and the fleet crossed a
    # discontinuity once per slot. Moving the whole map in ONE scheduled post is strictly
    # better: the nodes cross once, and the map is never in a half-applied state no node
    # agreed to. PRNs 1 and 2 are both gone from BRDC, two satellites are waiting.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {3: 30.0, 30: 70.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    _node_posts = [(u, pl) for u, pl in POSTS if "node" in u]
    check(len(_node_posts) == 2, "one swap, posted to both nodes")
    check(_node_posts and all(30 in pl["prns"] and 36 in pl["prns"] for _u, pl in _node_posts),
          "BOTH waiting satellites land in the SAME post -- not one per interval")
    check(_node_posts and len({pl.get("at_hop") for _u, pl in _node_posts}) == 1,
          "... on ONE deadline, so both slots move on the same frame fleet-wide")
    n_first = len(POSTS)
    run_cycle(ctx, 1000.0 + 1.0)
    check(len(POSTS) == n_first,
          "nothing further is posted: the map now matches the sky, and the read-back loop "
          "has to complete before another decision anyway")

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

    # ---- 12. SCHEDULED SWAPS: one frame, fleet-wide ------------------------------------
    # ⚠️ THE DEFECT THIS PREVENTS IS INVISIBLE DOWNSTREAM. A map posted "now" lands on
    # whatever frame each node happens to be building, so the combiner folds one window whose
    # instances disagree about which satellite slot p IS. Every row is individually
    # well-formed, so nothing downstream can catch it -- it is the accumulator-identity trap
    # with a network delay for a cause.
    print("scheduled swaps: every node crosses on the SAME frame")
    POSTS[:] = []
    _c = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0},
             prn_reconfig="apply", prn_reconfig_gone_hold_s=0.0,
             prn_reconfig_down_hold_s=0.0)
    # The axis was polled 3 s ago -- at the TOP of the cycle that is now posting. The
    # deadline must be advanced across that gap, not written against the stale hop.
    _c.fe_hop_now = 1_000_000_000.0
    _c.fe_hop_t = 2000.0 - 3.0
    run_cycle(_c, 2000.0)
    _tracker_posts = [(u, b) for u, b in POSTS if "search" not in u]
    _follow_posts = [(u, b) for u, b in POSTS if "search" in u]
    check(bool(_tracker_posts), "a swap is posted at all")
    check(all("at_hop" in b for _, b in _tracker_posts),
          "every TRACKER post carries an at_hop deadline")
    _seqs = {b["at_hop"] for _, b in _tracker_posts}
    check(len(_seqs) == 1,
          "and it is the SAME hop on every node -- that is the whole point (%s)" % _seqs)
    _want = 1_000_000_000 + int(round((3.0 + _c.args.prn_reconfig_lead_s)
                                      * _c.args.hops_per_sec))
    check(_seqs == {_want},
          "the deadline is HOPS: axis + (age + lead) x hops_per_sec, NOT samples "
          "(%d, got %s)" % (_want, _seqs))
    # ⚠️ THE REGRESSION: without the age term the deadline is 3 s of hops in the PAST, which
    # every node meets on its very next frame -- an unscheduled swap wearing a schedule.
    check(min(_seqs) > 1_000_000_000 + int(round(3.0 * _c.args.hops_per_sec)),
          "... and it is strictly AHEAD of the fleet's position at post time")

    # A STALE AXIS IS NOT EXTRAPOLATED. Past the age cap the swap still posts -- fail-open,
    # a slot stuck on a set satellite is worse -- but without a fabricated coordination point.
    POSTS[:] = []
    LOGS[:] = []
    _c4 = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0},
              prn_reconfig="apply", prn_reconfig_gone_hold_s=0.0,
              prn_reconfig_down_hold_s=0.0)
    _c4.fe_hop_now = 1_000_000_000.0
    _c4.fe_hop_t = 2000.0 - 120.0          # two minutes stale, cap is 30 s
    run_cycle(_c4, 2000.0)
    _sp = [(u, b) for u, b in POSTS if "search" not in u]
    check(bool(_sp), "a 120 s-stale axis still POSTS the map (fail-open)")
    check(all("at_hop" not in b for _, b in _sp),
          "... but names no deadline it cannot stand behind")
    check(any("FABRICATED deadline" in m for m in LOGS),
          "... and says so out loud")
    check(_follow_posts and all("at_hop" not in b for _, b in _follow_posts),
          "the SEARCH gets the map but NOT the deadline -- it has no frame boundary to test "
          "one against, and a deadline it cannot honour would wedge its map")

    # ⚠️ NO AXIS -> POST ANYWAY, UNSCHEDULED. An unsynchronised swap is worse than a
    # synchronised one and far better than none: a slot stuck on a set satellite produces
    # nothing at all. This is the fail-open direction, and it is deliberate.
    POSTS[:] = []
    _c2 = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0},
              prn_reconfig="apply", prn_reconfig_gone_hold_s=0.0,
              prn_reconfig_down_hold_s=0.0)
    _c2.fe_hop_now = None
    run_cycle(_c2, 2000.0)
    _tp = [(u, b) for u, b in POSTS if "search" not in u]
    check(bool(_tp), "with NO axis the swap is still posted (fail-open)")
    check(all("at_hop" not in b for _, b in _tp),
          "... and carries no deadline rather than a fabricated one")
    check(all(b.get("prns") for _, b in _tp), "... and still carries the map")

    # ---- 4b. THE HYSTERESIS BAND MUST BE A BAND ---------------------------------------
    # ⚠️ admit <= evict is the flap generator: the same satellite is admitted and evicted on
    # alternate cycles and each flap is a cold acquisition, which reads as "low satellites
    # cannot be tracked" -- a fault the stage would manufacture and then be believed about.
    print("\nthe hysteresis band: admit must be ABOVE evict")
    POSTS[:] = []
    LOGS[:] = []
    _c5 = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0},
              prn_reconfig="apply", prn_reconfig_gone_hold_s=0.0,
              prn_reconfig_down_hold_s=0.0,
              prn_reconfig_admit_deg=0.0, prn_reconfig_evict_deg=0.0)
    run_cycle(_c5, 2000.0)
    check(any("no hysteresis band" in m for m in LOGS),
          "admit == evict is REFUSED and named")
    check(not POSTS, "... and nothing is posted on a degenerate band")

    # The armed setting itself: admit AT the horizon, evict below it, is a valid band.
    POSTS[:] = []
    LOGS[:] = []
    _c6 = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +3.0},
              prn_reconfig="apply", prn_reconfig_gone_hold_s=0.0,
              prn_reconfig_down_hold_s=0.0,
              prn_reconfig_admit_deg=0.0, prn_reconfig_evict_deg=-2.0)
    _c6.fe_hop_t = 2000.0 - 1.0
    run_cycle(_c6, 2000.0)
    check(not any("no hysteresis band" in m for m in LOGS),
          "admit 0 / evict -2 is a valid band")
    _p6 = [b for u, b in POSTS if "search" not in u]
    check(_p6 and 9 in _p6[0]["prns"],
          "... and a satellite at +3 deg -- refused by the old 10 deg mask -- gets a slot")

    # ---- 5. THE DEADLINE CLOCK IS ALIVE ----------------------------------------------
    # ⚠️ THE REGRESSION THIS EXISTS FOR: a node that answers get_prns is a node whose frame
    # loop is turning, so last_hop < 0 means nothing is feeding note_frame_hop() and every
    # scheduled swap degrades to apply-immediately. That state ran unnoticed on twelve nodes
    # because nothing asked. Assert that it is now impossible to run silently.
    print("\nthe deadline clock: last_hop < 0 on a live node is an ALARM")
    _c3 = Ctx([1, 2, 3], {1: +40.0}, prn_reconfig="apply")
    _ep = _c3.trackers[0]

    def _get_dead(url, timeout=5.0):
        return {"prns": [1, 2, 3], "last_hop": -1, "pending_at_hop": -1}

    def _get_live(url, timeout=5.0):
        return {"prns": [1, 2, 3], "last_hop": 1_000_000_000, "pending_at_hop": -1}

    def _get_absent(url, timeout=5.0):
        return {"prns": [1, 2, 3]}          # an OLD node binary: no field at all

    for _fn, _want_alarm, _what in ((_get_dead, True, "last_hop = -1"),
                                    (_get_absent, True, "no last_hop field (old binary)"),
                                    (_get_live, False, "last_hop advancing")):
        prnmap._get = _fn
        LOGS[:] = []
        _c3.prnmap.maps = {}
        prnmap._poll(_c3, _c3.prnmap, [_ep])
        _hit = any("deadline clock is DEAD" in m for m in LOGS)
        check(_hit is _want_alarm,
              "%s -> %s" % (_what, "ALARM" if _want_alarm else "quiet"))
        check(_c3.prnmap.maps.get(_ep) == [1, 2, 3],
              "... and the map is still read back either way (%s)" % _what)
    prnmap._get = _no_get

    # ---- 6. A SATELLITE THAT CANNOT BROADCAST THE SIGNAL IS NOT A CANDIDATE ------------
    # ⚠️ 11 of gps_l5's 32 slots held Block IIR / IIR-M satellites, which predate L5 entirely.
    # G7 sat at 70 deg elevation reporting q 0.96 forever: a slot spent, and a noise row
    # folded into the presence population that every gate then has to survive.
    print("\nsignal capability: only satellites that can carry the signal get slots")
    import tempfile as _tf, os as _os
    from gnss_broker import prnmap as _pm
    _saved = _pm._TLE_CACHE
    try:
        with _tf.TemporaryDirectory() as _d:
            _f = _os.path.join(_d, "tle.txt")
            open(_f, "w").write(
                "GPS BIIR-5  (PRN 22)\n1 x\n2 x\n"
                "GPS BIIRM-3 (PRN 7)\n1 x\n2 x\n"
                "GPS BIIF-2  (PRN 1)\n1 x\n2 x\n"
                "GPS BIII-6  (PRN 4)\n1 x\n2 x\n")
            _pm._TLE_CACHE = _f
            _l5 = _pm.signal_incapable_prns("GPS_L5_Q_NH")
            check(_l5 == {22, 7},
                  "L5: Block IIR and IIR-M are excluded, IIF and III are not (got %s)" % sorted(_l5))
            check(_pm.signal_incapable_prns("GPS_L1CA") == set(),
                  "L1 C/A: every GPS satellite carries it, so nothing is excluded")
            check(_pm.signal_incapable_prns("GAL_E5A_Q_CS") == set(),
                  "Galileo: not modelled here, so nothing is excluded -- a filter that "
                  "pretended otherwise would be a second way for a constellation to go dark")
            # ⚠️ REFUSE ON DOUBT, IN THE SAFE DIRECTION.
            _pm._TLE_CACHE = _os.path.join(_d, "does-not-exist")
            check(_pm.signal_incapable_prns("GPS_L5_Q_NH") == set(),
                  "a MISSING capability source excludes NOTHING -- wrongly dropping a real "
                  "satellite is far worse than keeping a dead slot, and nothing would say why")
    finally:
        _pm._TLE_CACHE = _saved

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
