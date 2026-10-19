"""Live PRN membership: the swap policy, exercised without a node or a sky.

    python3 -m gnss_broker.test_prnmap

The replay gate (scripts/gnss/broker_equiv.py) does not exercise this stage: its recordings
run with --prn-reconfig at the default, off. And a wrong decision here is not a bad number but
a slot swap, which throws away a working satellite's acquisition. So the policy is asserted
directly.

What is pinned:
  * a candidate below the admit mask (not a noise probe) evicts nobody, however long it
    waits, while a dead slot still goes to any satellite;
  * two unwanted satellites never trade a slot (the map is a fixed point, not a flip-flop);
  * a node that left the fleet's map (a restart, a missed post) is put back on it, alone and
    once per interval; a split with no reference -- no majority, nothing this broker posted
    -- or one still mid-crossing changes nothing (nothing here is per-node).

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
        self.prn_reconfig_timeout_s = 5.0
        self.prn_reconfig_lead_s = 2.0
        self.prn_reconfig_axis_max_age_s = 30.0
        self.hops_per_sec = 125000.0
        self.prn_reconfig_heartbeat_s = (
            1e18  # no heartbeat unless a check sets one (the heartbeat check)
        )
        self.noise_probes = 0
        self.signal = "GAL_E5A_Q_CS"
        self.__dict__.update(kw)


class Ctx(object):
    """Only what prnmap reads: args, trackers, detectors, t0, pred, fe_hop_now / fe_hop_t
    and its PrnMapState."""

    def __init__(self, slots, pred, t0=1000.0, **kw):
        self.args = Args(**kw)
        self.trackers = ["http://node1", "http://node2"]
        self.detectors = ["http://search1"]  # a follower: gets the map, is not polled
        self.t0 = t0
        self.pred = {p: (0.0, 0.0, el, 0.0, 0.0) for p, el in pred.items()}
        self.prnmap = prnmap.PrnMapState()
        # What the fake nodes hold. The stage clears its cache after every POST and decides
        # again only after a full sweep has read the map back, so the test models the nodes
        # applying it: that read-back loop is part of what is tested.
        self.node_state = list(slots)
        self.prnmap.maps = {ep: list(slots) for ep in self.trackers}
        self.prnmap.poll_t = t0  # pre-swept: no HTTP in this test
        # The F-engine hop the deadline is built on, and when it was sampled. None models an
        # unknown axis, which must degrade to an unscheduled post, not to no post. fe_hop_t
        # defaults to t0: a status poll this cycle, stamped on the frozen cycle clock as the
        # broker does.
        self.fe_hop_now = (
            kw.pop("fe_hop_now", 1_000_000_000.0)
            if "fe_hop_now" in kw
            else 1_000_000_000.0
        )
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

    # ---- the whole map, every cycle, with elevation hysteresis -------------------------
    # No hold timers: a satellite up at 83 deg takes a below-horizon incumbent's slot on the
    # first cycle. The hysteresis is in elevation, which a broker restart cannot lose.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: -20.0, 2: 40.0, 3: 30.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    check(
        len([u for u, _ in POSTS if "node" in u]) == 2,
        "a satellite up at 83 deg takes a below-horizon incumbent's slot at once, on "
        "both nodes -- a slot below the horizon is idle",
    )
    check(
        len([u for u, _ in POSTS if "search" in u]) == 1,
        "the SEARCH is driven with the same map (it holds its own copy and has no frame to "
        "learn it from, unlike the assembler)",
    )
    if POSTS:
        check(
            POSTS[0][1]["prns"] == [36, 2, 3],
            "the swap replaces the down incumbent in ITS slot, leaving the rest alone",
        )

    # ---- hysteresis: a held satellite is not dropped the instant it dips --------------
    # Admitted above admit_deg, kept until it falls below evict_deg: BRDC visibility
    # flickers, and a slot that flickers is a satellite that never locks.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: 2.0, 2: 40.0, 3: 30.0, 9: 2.0})
    run_cycle(ctx, 1000.0)
    _held_kept = not POSTS or all(1 in pl["prns"] for _u, pl in POSTS)
    check(
        _held_kept,
        "a HELD satellite at +2 deg (below the 10 deg admit mask) is KEPT -- "
        "that is the hysteresis band, and dropping it would be the flap",
    )
    check(
        not any(9 in pl["prns"] for _u, pl in POSTS),
        "... while a new satellite at the same +2 deg is not admitted: the band has two "
        "edges or it is not hysteresis",
    )

    # ---- a dead slot is reused at once -------------------------------------------------
    # A PRN with no ephemeris produces nothing, so there is nothing to re-acquire.
    POSTS[:] = []
    ctx = Ctx(
        [1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0}
    )  # PRN 1 absent from BRDC entirely
    run_cycle(ctx, 1000.0)
    check(
        any(36 in pl["prns"] for _u, pl in POSTS),
        "a slot holding a PRN with no ephemeris is reclaimed on sight",
    )

    # ---- a free slot goes to anyone, including a low satellite -------------------------
    # The admit mask justifies an eviction, and a free slot is not one. A satellite at +4 deg
    # is neither up (below the 10 deg admit mask) nor a probe (above -15 deg), and must still
    # get a slot that stands empty.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 4.0})  # PRN 1 gone: slot 0 is free
    run_cycle(ctx, 1000.0)
    check(
        any(36 in pl["prns"] for _u, pl in POSTS),
        "a free slot is filled by a 4 deg satellite -- nothing to re-acquire, and it is a "
        "probe today and a tracked satellite tomorrow",
    )

    # ... but it must not buy an eviction. Every slot holds a satellite that is up: nothing
    # is free, and a 4 deg candidate is not worth a re-acquisition.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: 20.0, 2: 40.0, 3: 30.0, 36: 4.0})
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(
        not POSTS,
        "a 4 deg satellite does not evict anyone (the up-now bar still governs swaps)",
    )

    # ---- the probe supply is never evicted ---------------------------------------------
    # The deepest below-horizon satellites are the chain's noise probes. Trading one for
    # another tracked satellite costs the presence gate its noise floor.
    POSTS[:] = []
    deep = {
        1: -70.0,
        2: -65.0,
        3: -60.0,
        36: 83.0,
    }  # all three held sats are deep probes
    ctx = Ctx([1, 2, 3], deep)
    ctx.args.noise_probes = 3
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(
        not POSTS,
        "with noise_probes=3 and exactly 3 deep-below-horizon slots, none is evicted even "
        "for an 83 deg satellite -- the probe supply is not spare capacity",
    )
    deep4 = dict(deep)
    deep4[4] = -55.0  # a 4th deep satellite exists, unslotted
    ctx2 = Ctx([1, 2, 3], deep4)
    ctx2.args.noise_probes = 2  # only 2 must be held back
    POSTS[:] = []
    run_cycle(ctx2, 1000.0)
    run_cycle(ctx2, 1000.0 + 100000.0)
    check(
        any("36" in str(pl) for _u, pl in POSTS),
        "... but with only 2 needed, the shallowest deep slot is free to be traded",
    )

    # ---- dark probes (--probe-dark) -----------------------------------------------------
    # A satellite with ephemeris that cannot broadcast the signal (glo_l2oc: the GLO-M+ slots)
    # is signal-free at any elevation. probe_order puts those first, in PRN order, then the
    # deepest below -15 deg; the selector and desired_map share it.
    check(
        prnmap.probe_order({1: -20.0, 2: -40.0, 3: 10.0, 7: -16.0}, {12: (), 5: ()})
        == [5, 12, 2, 1, 7],
        "probe_order: the dark PRNs first in PRN order, then the deepest below -15 deg; a "
        "satellite above -15 deg is never a probe",
    )
    check(
        prnmap.probe_order({}, {1: (), 2: (), 4: (), 5: (), 12: ()}, held=[12, 5, 4, 8])
        == [4, 5, 12, 1, 2],
        "probe_order: the dark PRNs the nodes hold come first, so a plain GLO-M slot the map "
        "lacks never displaces a held M+ probe",
    )
    _wm2, _ = prnmap.desired_map([8, 9, 4, 5, 12], {8: 40.0, 9: 30.0}, 3, 10.0, 0.0,
                                 dark=[1, 2, 3, 4, 5, 12])
    check(
        _wm2 == [8, 9, 4, 5, 12],
        "desired_map: with every GLO-M and M+ slot dark, the held M+ probes stay and nothing "
        "is swapped (got %s)" % _wm2,
    )
    # Every slot is wanted once the dark probes count: a rising satellite finds no slot rather
    # than taking a dark probe's.
    _wm, _un = prnmap.desired_map([8, 9, 4, 5], {8: 40.0, 9: 30.0, 11: 50.0}, 2, 10.0, 0.0,
                                  dark=[4, 5, 12])
    check(
        _wm == [8, 9, 4, 5] and _un == [11],
        "desired_map: dark probes 4 and 5 keep their slots; the rising PRN 11 is reported "
        "unplaced (got %s, unplaced %s)" % (_wm, _un),
    )
    _wm0, _un0 = prnmap.desired_map([8, 9, 4, 5], {8: 40.0, 9: 30.0, 11: 50.0}, 2, 10.0, 0.0)
    check(
        11 in _wm0 and not _un0,
        "... and without --probe-dark the same slots read as dead and are reused (got %s)"
        % _wm0,
    )
    _wm1, _ = prnmap.desired_map([8, 9, 3, 6], {8: 40.0, 9: 30.0}, 2, 10.0, 0.0, dark=[4, 5])
    check(
        sorted(_wm1) == [4, 5, 8, 9],
        "desired_map: dark probes the nodes do not hold take the dead slots (got %s)" % _wm1,
    )
    # Through the stage: the map holds the dark probes and posts nothing.
    POSTS[:] = []
    ctx = Ctx([8, 9, 4, 5], {8: 40.0, 9: 30.0, 11: 50.0}, noise_probes=2, probe_dark=True)
    ctx.pred_dark = {4: (0.0, 0.0, 35.0, 0.0, 0.0), 5: (0.0, 0.0, -50.0, 0.0, 0.0)}
    run_cycle(ctx, 1000.0)
    check(
        not POSTS,
        "stage: with --probe-dark the dark probes' slots are not handed to a rising satellite",
    )

    # ---- an unwanted satellite does not displace another -------------------------------
    # Every slot holds a real satellite and one more has none, all below the horizon and none
    # a probe. The leftover-slot fill must not hand an unwanted occupant's slot to the
    # unslotted one: the next cycle would trade them back, a cold acquisition every interval
    # on every instance. Production thresholds.
    POSTS[:] = []
    sky = {1: 40.0, 2: 30.0, 3: -70.0, 4: -55.0, 5: -20.0, 6: -54.0}  # 6 has no slot
    ctx = Ctx(
        [1, 2, 3, 4, 5],
        sky,
        prn_reconfig_admit_deg=0.0,
        prn_reconfig_evict_deg=-2.0,
        noise_probes=1,
        prn_reconfig_interval_s=120.0,
    )
    for k in range(6):
        run_cycle(ctx, 1000.0 + k * 121.0)
    check(
        not POSTS,
        "two unwanted satellites never trade a slot: the map is a fixed point, not a swap "
        "every interval",
    )
    sky_dead = dict(sky)
    del sky_dead[5]  # ... but a dead slot still fills
    POSTS[:] = []
    ctx = Ctx(
        [1, 2, 3, 4, 5],
        sky_dead,
        prn_reconfig_admit_deg=0.0,
        prn_reconfig_evict_deg=-2.0,
        noise_probes=1,
        prn_reconfig_interval_s=120.0,
    )
    run_cycle(ctx, 1000.0)
    check(
        POSTS and POSTS[0][1]["prns"] == [1, 2, 3, 4, 6],
        "... while a slot whose PRN has no prediction is still handed to the unslotted one",
    )

    # ---- off does nothing at all -------------------------------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {2: 40.0, 3: 30.0, 36: 83.0}, prn_reconfig="off")
    run_cycle(ctx, 1000.0 + 100000.0)
    check(not POSTS, "off is off")

    # ---- disagreeing nodes with no reference stop the stage ----------------------------
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
    check(
        not POSTS,
        "a 1-1 split with no reference (no majority, nothing this broker posted) changes "
        "nothing -- nothing here is per-node, and neither side is the fleet's",
    )

    # ---- a node back on its boot list is put back on the fleet's map -------------------
    # A node restart reverts that node to its config's list.
    fleet, boot = [1, 2, 36], [1, 2, 3]
    nodes = ["http://node%d" % i for i in range(1, 5)]
    sky = {1: 40.0, 2: 30.0, 36: 83.0, 3: -5.0}
    odd = {n: list(fleet) for n in nodes}
    odd["http://node4"] = list(boot)

    def sweep(ctx, maps, t, pending=None):
        CTX[0] = None  # a resync reaches one node, not the shared node_state
        ctx.t0 = t
        ctx.prnmap.poll_t = t
        ctx.prnmap.maps = {k: list(v) for k, v in maps.items()}
        if pending is not None:
            ctx.prnmap.pending = dict(pending)
        prnmap.stage_prn_membership(ctx)

    POSTS[:] = []
    ctx = Ctx(fleet, sky)
    ctx.trackers = nodes
    sweep(ctx, odd, 1000.0)
    check(
        [u for u, _ in POSTS] == ["http://node4/set_prns"],
        "a 3-1 split (no broker history, as after a broker restart): the odd node, and "
        "only it, is sent the majority's map",
    )
    check(
        POSTS and POSTS[0][1] == {"prns": fleet},
        "... the whole map, unscheduled -- one node crosses, there is no fleet frame to share",
    )
    check(
        "http://node4" not in ctx.prnmap.maps,
        "... and its map is read back before anything else is decided",
    )
    n_posts = len(POSTS)
    sweep(ctx, {n: list(fleet) for n in nodes}, 1001.0)
    check(len(POSTS) == n_posts, "once it holds the map again, nothing more is posted")

    # ---- the map this broker last posted is the reference -----------------------------
    POSTS[:] = []
    ctx = Ctx(fleet, sky)
    ctx.prnmap.applied = list(fleet)
    sweep(ctx, {"http://node1": list(fleet), "http://node2": list(boot)}, 1000.0)
    check(
        [u for u, _ in POSTS] == ["http://node2/set_prns"],
        "a 1-1 split where one node holds the map this broker last posted: the other node "
        "missed it (a timeout, a restart) and is sent it",
    )

    # ---- mid-crossing is not a split --------------------------------------------------
    POSTS[:] = []
    LOGS[:] = []
    ctx = Ctx(fleet, sky)
    ctx.trackers = nodes
    sweep(ctx, odd, 1000.0, pending={"http://node1": True})
    check(
        not POSTS,
        "a swap still pending on any node: the fleet is mid-crossing, so nothing "
        "is resynced on a read-back that is not final",
    )
    check(
        not any("DISAGREE" in m for m in LOGS),
        "... and the crossing is not called a split",
    )

    # ---- a node that stays off the map is retried once per interval -------------------
    POSTS[:] = []
    ctx = Ctx(fleet, sky, prn_reconfig_interval_s=120.0)
    ctx.trackers = nodes
    sweep(ctx, odd, 1000.0)
    sweep(ctx, odd, 1010.0)  # still on its boot list: refused, or restarted again
    check(
        len(POSTS) == 1,
        "a node still off the map 10 s later is not re-posted every cycle",
    )
    sweep(ctx, odd, 1121.0)
    check(len(POSTS) == 2, "... and is retried once the interval has passed")

    # ---- no prediction this cycle: change nothing --------------------------------------
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {})
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 100000.0)
    check(
        not POSTS,
        "an empty prediction evicts nobody (a BRDC outage is not a constellation "
        "outage)",
    )

    # ---- many slots, one post, one deadline -------------------------------------------
    # Several slots move in one scheduled post: the nodes cross once, and the map is never
    # half-applied. PRNs 1 and 2 are both gone from BRDC; two satellites are waiting.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {3: 30.0, 30: 70.0, 36: 83.0})
    run_cycle(ctx, 1000.0)
    _node_posts = [(u, pl) for u, pl in POSTS if "node" in u]
    check(len(_node_posts) == 2, "one swap, posted to both nodes")
    check(
        _node_posts
        and all(30 in pl["prns"] and 36 in pl["prns"] for _u, pl in _node_posts),
        "both waiting satellites land in the same post -- not one per interval",
    )
    check(
        _node_posts and len({pl.get("at_hop") for _u, pl in _node_posts}) == 1,
        "... on one deadline, so both slots move on the same frame fleet-wide",
    )
    n_first = len(POSTS)
    run_cycle(ctx, 1000.0 + 1.0)
    check(
        len(POSTS) == n_first,
        "nothing further is posted: the map now matches the sky, and the read-back loop "
        "has to complete before another decision anyway",
    )

    # ---- the loop converges: once the nodes hold the map, nothing more is posted ------
    # The stage decides from the read-back map, so a POST that took is not re-issued;
    # otherwise it would swap a slot every interval forever.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {3: 30.0, 36: 83.0})  # PRN 1 and 2 gone, one candidate
    run_cycle(ctx, 1000.0)
    run_cycle(ctx, 1000.0 + 7201.0)
    n_after = len(POSTS)
    for k in range(1, 8):
        run_cycle(ctx, 1000.0 + 7201.0 + k * 1000.0)
    check(
        len(POSTS) == n_after,
        "once the nodes hold the new map, no further swap is posted (the read-back loop "
        "converges instead of re-issuing)",
    )
    check(36 in ctx.node_state, "... and the nodes ended up holding the new satellite")

    # ---- the heartbeat: armed and idle must be visible --------------------------------
    # Having nothing to propose is the normal state, so a stage that only speaks when it acts
    # is indistinguishable from one that is not running. Nothing is posted here, so the
    # heartbeat is asserted on the log rather than on POSTS.
    POSTS[:] = []
    LOGS[:] = []
    ctx = Ctx([1, 2, 3], {1: 40.0, 2: 40.0, 3: 30.0})
    ctx.args.prn_reconfig_heartbeat_s = 900.0
    run_cycle(ctx, 1000.0)
    check(
        any("PRN MAP" in m and "APPLY" in m for m in LOGS),
        "an armed chain with nothing to propose still says so (the heartbeat)",
    )
    check(not POSTS, "... and still posts nothing")
    n_beats = len([m for m in LOGS if "APPLY" in m])
    run_cycle(ctx, 1000.0 + 100.0)
    check(
        len([m for m in LOGS if "APPLY" in m]) == n_beats,
        "... rate-limited: not once per cycle",
    )
    run_cycle(ctx, 1000.0 + 901.0)
    check(
        len([m for m in LOGS if "APPLY" in m]) == n_beats + 1,
        "... and it does beat again after the interval",
    )

    # ---- the map is published for read-only consumers, and fails open ----------------
    # The probe selector (--probe-require-slot) reads the live map but proposes no swaps, and
    # must never act on a partial or split map: that would drop real probes from the probe
    # set on a half-swept fleet.
    POSTS[:] = []
    ctx = Ctx([1, 2, 3], {1: 40.0, 2: 40.0, 3: 30.0})
    run_cycle(ctx, 1000.0)
    check(
        ctx.prnmap.consensus == [1, 2, 3],
        "a fully swept, unanimous fleet: the map is PUBLISHED",
    )
    check(not POSTS, "... and nothing is posted while the map already matches the sky")

    ctx2 = Ctx([1, 2, 3], {2: 40.0, 3: 30.0})
    CTX[0] = ctx2
    ctx2.t0 = 1000.0
    ctx2.prnmap.poll_t = 1000.0
    ctx2.prnmap.maps = {"http://node1": [1, 2, 3]}  # half a sweep (2 endpoints)
    prnmap.stage_prn_membership(ctx2)
    check(
        ctx2.prnmap.consensus is None,
        "a half-swept fleet publishes None -- consumers fail open, never on a partial map",
    )

    ctx2.prnmap.maps = {"http://node1": [1, 2, 3], "http://node2": [1, 2, 9]}  # split
    prnmap.stage_prn_membership(ctx2)
    check(ctx2.prnmap.consensus is None, "... and so does a SPLIT fleet")

    ctx3 = Ctx([1, 2, 3], {2: 40.0}, prn_reconfig="off")
    run_cycle(ctx3, 1000.0)
    check(
        ctx3.prnmap.consensus is None,
        "--prn-reconfig off: the stage does nothing at all (no GET -- replay stays exact)",
    )

    # ---- scheduled swaps: one frame, fleet-wide ---------------------------------------
    # A map posted for "now" lands on whatever frame each node is building, so the combiner
    # folds a window whose instances disagree about which satellite slot p holds. Every row is
    # well-formed on its own, so nothing downstream can catch it.
    print("scheduled swaps: every node crosses on the same frame")
    POSTS[:] = []
    _c = Ctx([1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0}, prn_reconfig="apply",)
    # The axis sample is 3 s old (left by an earlier cycle; the broker stamps this cycle's
    # with t0). The deadline must advance it by that age, not be written against the stale
    # hop.
    _c.fe_hop_now = 1_000_000_000.0
    _c.fe_hop_t = 2000.0 - 3.0
    run_cycle(_c, 2000.0)
    _tracker_posts = [(u, b) for u, b in POSTS if "search" not in u]
    _follow_posts = [(u, b) for u, b in POSTS if "search" in u]
    check(bool(_tracker_posts), "a swap is posted at all")
    check(
        all("at_hop" in b for _, b in _tracker_posts),
        "every TRACKER post carries an at_hop deadline",
    )
    _seqs = {b["at_hop"] for _, b in _tracker_posts}
    check(
        len(_seqs) == 1,
        "and it is the same hop on every node -- that is the whole point (%s)" % _seqs,
    )
    _want = 1_000_000_000 + int(
        round((3.0 + _c.args.prn_reconfig_lead_s) * _c.args.hops_per_sec)
    )
    check(
        _seqs == {_want},
        "the deadline is in hops: axis + (age + lead) x hops_per_sec, not samples "
        "(%d, got %s)" % (_want, _seqs),
    )
    # Without the age term the deadline would be 1 s in the past (lead 2 s, age 3 s), met by
    # every node on its next frame: an unscheduled swap that claims a schedule.
    check(
        min(_seqs) > 1_000_000_000 + int(round(3.0 * _c.args.hops_per_sec)),
        "... and it is strictly AHEAD of the fleet's position at post time",
    )

    # A stale axis is not extrapolated. Past prn_reconfig_axis_max_age_s the swap still posts
    # (fail open: a slot stuck on a set satellite is worse), but without a deadline.
    POSTS[:] = []
    LOGS[:] = []
    _c4 = Ctx(
        [1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0}, prn_reconfig="apply",
    )
    _c4.fe_hop_now = 1_000_000_000.0
    _c4.fe_hop_t = 2000.0 - 120.0  # two minutes stale, cap is 30 s
    run_cycle(_c4, 2000.0)
    _sp = [(u, b) for u, b in POSTS if "search" not in u]
    check(bool(_sp), "a 120 s-stale axis still POSTS the map (fail-open)")
    check(
        all("at_hop" not in b for _, b in _sp),
        "... but names no deadline it cannot stand behind",
    )
    check(any("FABRICATED deadline" in m for m in LOGS), "... and says so out loud")
    check(
        _follow_posts and all("at_hop" not in b for _, b in _follow_posts),
        "the SEARCH gets the map but not the deadline -- it has no frame boundary to test "
        "one against, and a deadline it cannot honour would wedge its map",
    )

    # No axis: post anyway, unscheduled. An unsynchronised swap is worse than a synchronised
    # one and better than none, since a slot stuck on a set satellite produces nothing.
    POSTS[:] = []
    _c2 = Ctx(
        [1, 2, 3], {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0}, prn_reconfig="apply",
    )
    _c2.fe_hop_now = None
    run_cycle(_c2, 2000.0)
    _tp = [(u, b) for u, b in POSTS if "search" not in u]
    check(bool(_tp), "with no axis the swap is still posted (fail-open)")
    check(
        all("at_hop" not in b for _, b in _tp),
        "... and carries no deadline rather than a fabricated one",
    )
    check(all(b.get("prns") for _, b in _tp), "... and still carries the map")

    # ---- the hysteresis band must be a band -------------------------------------------
    # With admit <= evict the same satellite is admitted and evicted on alternate cycles,
    # each flap a cold acquisition that reads as "low satellites cannot be tracked".
    print("\nthe hysteresis band: admit must be ABOVE evict")
    POSTS[:] = []
    LOGS[:] = []
    _c5 = Ctx(
        [1, 2, 3],
        {1: +40.0, 2: -50.0, 3: -60.0, 9: +30.0},
        prn_reconfig="apply",
        prn_reconfig_admit_deg=0.0,
        prn_reconfig_evict_deg=0.0,
    )
    run_cycle(_c5, 2000.0)
    check(
        any("no hysteresis band" in m for m in LOGS),
        "admit == evict is REFUSED and named",
    )
    check(not POSTS, "... and nothing is posted on a degenerate band")

    # The production setting, admit at the horizon and evict below it, is a valid band.
    POSTS[:] = []
    LOGS[:] = []
    _c6 = Ctx(
        [1, 2, 3],
        {1: +40.0, 2: -50.0, 3: -60.0, 9: +3.0},
        prn_reconfig="apply",
        prn_reconfig_admit_deg=0.0,
        prn_reconfig_evict_deg=-2.0,
    )
    _c6.fe_hop_t = 2000.0 - 1.0
    run_cycle(_c6, 2000.0)
    check(
        not any("no hysteresis band" in m for m in LOGS),
        "admit 0 / evict -2 is a valid band",
    )
    _p6 = [b for u, b in POSTS if "search" not in u]
    check(
        _p6 and 9 in _p6[0]["prns"],
        "... and a satellite at +3 deg gets a slot",
    )

    # ---- the deadline clock is alive --------------------------------------------------
    # A node that answers get_prns has a turning frame loop, so last_hop < 0 means nothing
    # feeds note_frame_hop() and every scheduled swap there applies immediately. The stage
    # must say so.
    print("\nthe deadline clock: last_hop < 0 on a live node is an ALARM")
    _c3 = Ctx([1, 2, 3], {1: +40.0}, prn_reconfig="apply")
    _ep = _c3.trackers[0]

    def _get_dead(url, timeout=5.0):
        return {"prns": [1, 2, 3], "last_hop": -1}

    def _get_live(url, timeout=5.0):
        return {"prns": [1, 2, 3], "last_hop": 1_000_000_000}

    def _get_absent(url, timeout=5.0):
        return {"prns": [1, 2, 3]}  # a node binary without the field

    for _fn, _want_alarm, _what in (
        (_get_dead, True, "last_hop = -1"),
        (_get_absent, True, "no last_hop field (old binary)"),
        (_get_live, False, "last_hop advancing"),
    ):
        prnmap._get = _fn
        LOGS[:] = []
        _c3.prnmap.maps = {}
        prnmap._poll(_c3, _c3.prnmap, [_ep])
        _hit = any("deadline clock is DEAD" in m for m in LOGS)
        check(
            _hit is _want_alarm,
            "%s -> %s" % (_what, "ALARM" if _want_alarm else "quiet"),
        )
        check(
            _c3.prnmap.maps.get(_ep) == [1, 2, 3],
            "... and the map is still read back either way (%s)" % _what,
        )
    prnmap._get = _no_get

    # ---- a satellite that cannot broadcast the signal is not a candidate -------------
    # Block IIR / IIR-M satellites predate L5: in an L5 slot each is a wasted slot and a
    # noise row in the presence population.
    print("\nsignal capability: from the IGS metadata SINEX, not Celestrak")
    import tempfile as _tf, os as _os
    from gnss_broker import prnmap as _pm

    _saved = _pm._IGS_SNX
    try:
        with _tf.TemporaryDirectory() as _d:
            _f = _os.path.join(_d, "meta.snx")
            open(_f, "w").write(
                "+SATELLITE/IDENTIFIER\n"
                "*SVN_ COSPAR ID SatCat Block__________ Comment\n"
                " G044 2000-040A  26407 GPS-IIR-A       x\n"
                " G058 2005-038A  28874 GPS-IIR-M       x\n"
                " G062 2010-022A  36585 GPS-IIF         x\n"
                " G074 2018-074A  43873 GPS-IIIA        x\n"
                "-SATELLITE/IDENTIFIER\n"
                "+SATELLITE/PRN\n"
                " G044 2000:198:00000 0000:000:00000 G22\n"
                " G058 2005:257:00000 0000:000:00000 G07\n"
                " G062 2010:146:00000 0000:000:00000 G01\n"
                " G074 2018:357:00000 0000:000:00000 G04\n"
                "-SATELLITE/PRN\n"
            )
            _pm._IGS_SNX = _f
            check(
                _pm.signal_incapable_prns("GPS_L5_Q_NH") == {22, 7},
                "L5: IIR-A and IIR-M excluded, IIF and IIIA kept (got %s)"
                % sorted(_pm.signal_incapable_prns("GPS_L5_Q_NH")),
            )
            check(
                _pm.signal_incapable_prns("GPS_L1C_P") == {22, 7, 1},
                "L1C: Block III only -- IIF is excluded too, so the predicate is really "
                "reading the block rather than passing anything modern",
            )
            check(
                _pm.signal_incapable_prns("GPS_L1CA") == set(),
                "L1 C/A: every GPS satellite carries it",
            )
            check(
                _pm.signal_incapable_prns("GAL_E5A_Q_CS") == set(),
                "Galileo: not modelled, so nothing excluded -- a filter that pretended "
                "otherwise would be a second way for a constellation to go dark",
            )
            # GLONASS: L3OC rides the K generations and the modernised M+; a plain GLO-M
            # satellite is FDMA-only. The generations are interspersed across the slots, so
            # the block, not a PRN floor, must decide.
            open(_f, "a").write(
                "+SATELLITE/IDENTIFIER\n"
                " R730 2007-065B  32395 GLO-M           x\n"
                " R755 2014-075A  40315 GLO-M+          x\n"
                " R705 2011-009A  37372 GLO-K1          x\n"
                " R706 2023-002A  55260 GLO-K1B         x\n"
                " R707 2022-156A  54377 GLO-K1+         x\n"
                " R711 2023-173A  57517 GLO-K2          x\n"
                "-SATELLITE/IDENTIFIER\n"
                "+SATELLITE/PRN\n"
                " R730 2008:001:00000 0000:000:00000 R01\n"
                " R755 2015:001:00000 0000:000:00000 R04\n"
                " R705 2011:100:00000 0000:000:00000 R09\n"
                " R706 2023:100:00000 0000:000:00000 R11\n"
                " R707 2022:300:00000 0000:000:00000 R19\n"
                " R711 2023:300:00000 0000:000:00000 R06\n"
                "-SATELLITE/PRN\n"
            )
            check(
                _pm.signal_incapable_prns("GLO_L3OC_P_NH") == {1},
                "GLONASS L3OC: only the plain GLO-M slot excluded; K1/K1B/K1+/K2 and M+ "
                "kept (got %s)" % sorted(_pm.signal_incapable_prns("GLO_L3OC_P_NH")),
            )
            check(
                _pm.signal_incapable_prns("GLO_L2OF") == set(),
                "GLONASS L2OF: FDMA is on every satellite, nothing excluded",
            )
            check(
                _pm.signal_incapable_prns("GLO_L2OC_P") == {1, 4},
                "GLONASS L2OC: the K generations alone -- GLO-M+ carries L3OC, not L2OC "
                "(got %s)" % sorted(_pm.signal_incapable_prns("GLO_L2OC_P")),
            )
            check(
                _pm.signal_incapable_prns("GPS_L5_Q_NH") == {22, 7},
                "... and the GLONASS rows did not leak into the GPS answer",
            )
            _cap = _pm.registry_capable("glo_l3oc")
            check(
                _cap is not None and 1 not in _cap and {4, 6, 9, 11, 19} <= _cap,
                "registry_capable: the chain NAME resolves (glo_l3oc -> GLO), the GLO-M "
                "slot is out, K and M+ in (got %s)"
                % (sorted(_cap)[:6] if _cap else _cap),
            )
            check(
                _pm.registry_capable("GLO_L2OF") is None,
                "registry_capable: a signal every satellite carries proves nothing -> None",
            )
            check(
                _pm.ensure_registry() is not None and _pm.ensure_registry() < 60.0,
                "ensure_registry: a fresh copy is left alone and its age returned",
            )
            # A re-used PRN must resolve to today's satellite: an expired assignment (SVN
            # G099 flew G01 until 1995) must not count.
            open(_f, "a").write(
                "+SATELLITE/PRN\n"
                " G099 1990:001:00000 1995:001:00000 G01\n"
                "-SATELLITE/PRN\n"
            )
            check(
                _pm.signal_incapable_prns("GPS_L5_Q_NH") == {22, 7},
                "an expired PRN assignment is ignored -- current validity only",
            )
            # A replay pins the registry to its transcript's snapshot (GNSS_IGS_SNX). The
            # broker must read that file and never fetch over it, however old it is: a fetch
            # would replace the snapshot with today's registry.
            import importlib.util as _ilu
            import urllib.request as _ur

            _pin = _os.path.join(_d, "pinned.snx")
            open(_pin, "w").write(open(_f).read())
            _os.utime(_pin, (0, 0))
            _env = _os.environ.get("GNSS_IGS_SNX")
            _os.environ["GNSS_IGS_SNX"] = _pin
            try:
                _spec = _ilu.spec_from_file_location("prnmap_pinned", _pm.__file__)
                _pp = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_pp)
            finally:
                if _env is None:
                    del _os.environ["GNSS_IGS_SNX"]
                else:
                    _os.environ["GNSS_IGS_SNX"] = _env
            _fetches = []

            def _no_fetch(*a, **kw):
                _fetches.append(a)
                raise IOError("no network in a test")

            _urlopen = _ur.urlopen
            _ur.urlopen = _no_fetch
            try:
                _age = _pp.ensure_registry()
            finally:
                _ur.urlopen = _urlopen
            check(
                _pp._IGS_SNX == _pin
                and not _fetches
                and _age is not None
                and _age > _pp.REGISTRY_MAX_AGE_S,
                "a pinned registry (GNSS_IGS_SNX) is read as it is: an old one is not "
                "fetched over (fetches %d, age %s)" % (len(_fetches), _age),
            )
            check(
                _pp.signal_incapable_prns("GPS_L5_Q_NH") == {22, 7},
                "... and the capability gate reads that file (got %s)"
                % sorted(_pp.signal_incapable_prns("GPS_L5_Q_NH")),
            )
            _pm._IGS_SNX = _os.path.join(_d, "missing")
            check(
                _pm.signal_incapable_prns("GPS_L5_Q_NH") == set(),
                "a missing registry excludes nothing -- wrongly dropping a real satellite "
                "is far worse than keeping a dead slot, and nothing would say why",
            )
    finally:
        _pm._IGS_SNX = _saved

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
