"""LIVE PRN MEMBERSHIP: keep each node's slot->PRN map in step with the sky.

THE PROBLEM THIS EXISTS FOR. The node's PRN list was a hand-written comma string in
`config/gnss_fleet_chord.yaml` and the broker's view of the constellation comes from live BRDC
every cycle. Nothing reconciled the two, so they drifted apart in silence. Measured
2026-08-26: Galileo carried FIVE slots whose satellites no longer exist, while E36 -- which
transits at 83 deg elevation, essentially through the main beam -- had no slot at all. Slots
were not scarce; we simply had the wrong 32. It cost more than one satellite: the noise-probe
selector picks the DEEPEST below-horizon PRN, which is exactly what an excluded slot tends to
be, so it picked E36, the node could not represent it, the probe never reported, and both
Galileo chains silently fell from the q+p presence gate to BRIGHTNESS-ONLY.

⚠️ THE SLOT COUNT IS FIXED; ONLY MEMBERSHIP MOVES. n_prn sizes every buffer, GPU allocation
and wire frame in the pipeline, so changing it is a fleet-wide re-plumb and a node restart.
Swapping which satellite sits in a slot, at constant count, changes no size anywhere. That is
enough -- Galileo needs 28 of 32.

⚠️ A SWAP COSTS A FULL RE-ACQUISITION, so this is deliberately reluctant. The node resets the
slot COLD (code table, Phi cache, carrier NCO, seed, trim, power averages, element cal): that
is correct -- none of it describes the new satellite -- and it is also expensive. Recomputing
membership straight from BRDC every cycle would churn slots on every ephemeris flicker, which
is #92's disease one level up. Hence: an incumbent must be gone or DOWN for hours before its
slot is reclaimed, a candidate must be genuinely up, and at most one slot moves per interval.

⚠️ DEFAULT IS `off`, AND `report` IS THE INTERESTING MIDDLE. In `report` this posts nothing and
logs the swap it would make -- a live version of `gen_fleet.py --check-prns`, with no risk and
the same finding. Arm `apply` deliberately.

⚠️ WHY THE NODE IS ASKED RATHER THAN ASSUMED. The broker does not read the node's config; it
GETs the live map. So the map it diffs against is what the node actually holds, including
after a node restart reverted it to the config's list -- which it will, because this mechanism
is deliberately NOT persistent. The config remains the boot state; this is a runtime overlay,
and a restart is a clean slate rather than a silently-inherited history.

@author Keith Vanderlinde
"""

import os
import re

from gnss_broker.transport import _get, _post, _log, _log_rl, log_tag


class PrnMapState(object):
    """One chain's view of node membership, and the hysteresis that governs changes."""

    __slots__ = ("maps", "poll_t", "cursor", "last_swap_t", "down_since", "gone_since", "err",
                 "swaps", "refused", "beat_t", "consensus")

    def __init__(self):
        self.maps = {}          # endpoint -> [prn per slot]
        self.poll_t = 0.0       # last GET of the endpoint at `cursor`
        self.cursor = 0         # round-robin: ONE endpoint per cycle (see _poll)
        self.last_swap_t = 0.0  # rate limit
        self.down_since = {}    # prn -> t it was first seen below evict_deg (and never above)
        self.gone_since = {}    # prn -> t it first went missing from BRDC entirely
        self.err = ""
        self.swaps = 0
        self.refused = 0
        self.beat_t = 0.0    # last heartbeat (see the note in stage_prn_membership)
        # The unanimous live slot->PRN list, or None. READ by the probe selector;
        # None whenever the sweep is incomplete or the nodes disagree.
        self.consensus = None


def _endpoints(ctx):
    """The tracker endpoints -- the producer owns membership, so it is the one asked."""
    return list(ctx.trackers or [])


def _followers(ctx):
    """Stages that hold their OWN copy of the map and must be driven in step.

    Only the SEARCH: the record assembler learns the map from the frame itself
    (gnss_gpu::PrnCtl::prn), which is why it needs no endpoint and cannot drift. The search
    has no such channel -- it hunts the satellites its list names, so a stale list there is a
    satellite it can never detect -- and it is asked for the same map with the same payload.

    ⚠️ THEY ARE NOT POLLED FOR CONSENSUS. The producer is the authority; a follower that
    refuses a swap logs and is left behind rather than dragging the producer back, because a
    search that lags by one satellite is a degradation and a producer that lags is a lie about
    what the records contain.
    """
    return list(ctx.detectors or [])


def _poll(ctx, st, eps):
    """GET **one** endpoint's live map per call, round-robin.

    ⚠️ ONE, NOT ALL, AND THAT IS THE WHOLE DESIGN OF THIS FUNCTION. The stage runs inside the
    cycle loop, so a sweep of every endpoint costs (n_dead x timeout) on ONE cycle -- 12 nodes
    against a 5 s timeout is a full minute of broker stall the first time site work takes the
    fleet down. That is #81 exactly ("a dead feed cost a full timeout per frame", the failure
    that put the search 48.9 s behind sky). Polling one per cycle bounds the damage to a single
    timeout and still sweeps 12 endpoints in ~24 s at the live 2 s interval, comfortably inside
    the 60 s poll period.

    A failing endpoint is DROPPED from the map rather than left stale: a node that stopped
    answering has not agreed to anything, and letting its last answer stand would hide a split
    map behind a corpse.
    """
    if not eps:
        return
    st.cursor %= len(eps)
    ep = eps[st.cursor]
    st.cursor = (st.cursor + 1) % len(eps)
    try:
        r = _get("%s/get_prns" % ep, timeout=ctx.args.prn_reconfig_timeout_s)
        m = r.get("prns")
        if isinstance(m, list) and m:
            st.maps[ep] = [int(x) for x in m]
        else:
            st.maps.pop(ep, None)
        # ⚠️ IS THE DEADLINE EVEN TESTABLE? A node that answers this GET is running, so its
        # frame loop is turning; last_hop < 0 then means NO producer is feeding the deadline
        # clock, and every scheduled swap silently degrades to apply-immediately. That is
        # precisely how this mechanism shipped inert (2026-08-27: note_frame_hop() was wired
        # into cudaGnssChordTrack, the fleet runs cudaGnssInject), and nothing anywhere said
        # so -- the swaps posted, the nodes took them, the logs read healthy. Ask the question
        # the swap depends on, out loud, every time.
        _lh = r.get("last_hop")
        if _lh is None or int(_lh) < 0:
            _log_rl("prnmap-noclock",
                    "PRN MAP %s: %s reports last_hop=%s -- its deadline clock is DEAD, so "
                    "scheduled swaps there apply IMMEDIATELY and the fleet does not cross "
                    "together. Armed-but-inert: check that the stage owning the frame loop "
                    "calls note_frame_hop()."
                    % (log_tag() or ctx.args.signal, ep, _lh), every_s=600.0)
    except Exception as e:
        st.err = "%s: %s" % (ep, e)
        st.maps.pop(ep, None)
    # Endpoints that have left the configured list must not linger in the consensus.
    for gone in [k for k in st.maps if k not in eps]:
        st.maps.pop(gone, None)


def _consensus(maps):
    """The map every reporting node agrees on, or None.

    ⚠️ DISAGREEMENT IS NOT AVERAGED. Nothing here is per-node -- a per-node PRN list would be a
    bug on its own (chord-nothing-is-per-node) -- so if two nodes report different maps the
    right response is to say so and change nothing, not to pick one and drive the fleet toward
    it from a state nobody chose.
    """
    if not maps:
        return None
    first = None
    for m in maps.values():
        if first is None:
            first = m
        elif m != first:
            return None
    return first


# ---------------------------------------------------------------------------------------
# SIGNAL CAPABILITY: which satellites can carry this chain's signal at all
# ---------------------------------------------------------------------------------------
_TLE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps", "tle_gps-ops.txt")
_TLE_NAME_RE = re.compile(r"GPS\s+B([A-Z]+)-?\d.*\(PRN\s*(\d+)\)")


def signal_incapable_prns(signal):
    """PRNs that provably CANNOT broadcast `signal`, or an empty set if we cannot tell.

    ⚠️⚠️ WE WERE TRACKING SATELLITES THAT HAVE NEVER TRANSMITTED THE SIGNAL. Measured
    2026-08-28: 11 of gps_l5's 32 slots held Block IIR / IIR-M satellites, which predate L5
    entirely -- G7 sat at 70 deg elevation reporting q 0.96 forever. They cost a tracker slot
    each, and worse, they fold 11 noise rows into the presence population that the gates then
    have to be robust against.

    Block is METADATA -- no almanac or TLE element carries it -- but Celestrak encodes it in
    the satellite NAME ("GPS BIIR-5  (PRN 22)"), which is why this reads names rather than
    hardcoding a list that goes stale as satellites launch and retire.

    ⚠️ EXCLUDE ONLY WHAT WE CAN PROVE, AND KEEP EVERYTHING ELSE. A missing cache, an
    unparseable name, a PRN absent from the file, or a constellation this does not model all
    yield "not incapable" -- because the cost of wrongly excluding a real satellite (it can
    never be tracked, and nothing says why) is far worse than the cost of keeping a dead slot.
    Same polarity as _rec_is_fresh: reject the provable, never refuse on doubt.

    ⚠️ GPS ONLY, and deliberately. Every Galileo satellite carries E5a/E5b, and the BeiDou
    chains already exclude BDS-2 with alm_min_prn -- so there is nothing to model there, and a
    filter that pretended otherwise would be a second place for a constellation to go dark.
    """
    try:
        from gps_beamtrack import _signal_block_filter
        pred = _signal_block_filter(signal)
    except Exception:
        return set()
    if pred is None:                      # every satellite carries it (e.g. L1 C/A)
        return set()
    try:
        with open(_TLE_CACHE) as f:
            txt = f.read()
    except Exception:
        return set()
    out = set()
    for m in _TLE_NAME_RE.finditer(txt):
        block, prn = m.group(1), int(m.group(2))
        try:
            if not pred(block):
                out.add(prn)
        except Exception:
            continue
    return out


def stage_prn_membership(ctx):
    """Reconcile the nodes' slot->PRN map against live BRDC. Off unless --prn-reconfig.

    Produces at most ONE swap per --prn-reconfig-interval-s, and only when an incumbent has
    been evictable for hours and a candidate is genuinely up. See the module docstring for why
    it is this reluctant.
    """
    a = ctx.args
    # ⚠️ THE POLL AND THE SWAP ARE SEPARATE POWERS. --probe-require-slot needs the live map
    # but proposes no swaps, so "off" must still mean "poll if someone downstream needs the
    # map" -- while the MODE keeps governing whether anything is proposed or posted.
    if a.prn_reconfig == "off" and not a.probe_require_slot:
        return
    st = ctx.prnmap
    now = ctx.t0
    eps = _endpoints(ctx)

    # ---- 1. What the nodes hold ---------------------------------------------------------
    # The sweep is spread one endpoint per cycle, so `poll_s` divided by the number of
    # endpoints is the per-endpoint cadence -- not a burst every poll_s.
    if eps and now - st.poll_t >= a.prn_reconfig_poll_s / float(len(eps)):
        st.poll_t = now
        _poll(ctx, st, eps)
    # A full sweep must have completed before the map means anything: a partial one looks like
    # unanimity among however few nodes have answered so far.
    cur = _consensus(st.maps) if len(st.maps) >= len(eps) else None
    # PUBLISHED for consumers that only READ it (the probe selector). None whenever the sweep
    # is incomplete or the nodes disagree, so every consumer falls back to its previous
    # behaviour rather than acting on a partial truth.
    st.consensus = cur
    if cur is None:
        if st.maps and len(st.maps) >= len(eps):
            _log_rl("prnmap-split",
                    "PRN MAP: nodes DISAGREE about slot membership (%d reporting) -- changing "
                    "nothing. Nothing in this pipeline is per-node, so a split map is a fault "
                    "to fix, not a state to drive out of." % len(st.maps),
                    every_s=300.0)
        return
    if a.prn_reconfig == "off":
        return  # poll-only: --probe-require-slot wanted the map and nothing more

    # ---- 2. What the sky says -----------------------------------------------------------
    # ctx.pred is THIS chain's constellation already filtered by capability: brdc_predict is
    # called with alm_sys and alm_min_prn, which is the same "C1-C14 are BDS-2 and do not
    # broadcast B2a" rule the manifest states beside the list. So absence from pred means
    # "not a satellite this chain could ever track", not merely "not up".
    el = {p: v[2] for p, v in (ctx.pred or {}).items()}
    # ⚠️ A SATELLITE THAT CANNOT BROADCAST THIS SIGNAL IS NOT A CANDIDATE FOR A SLOT. Dropping
    # them here (rather than in desired_map) means they read as "not in the sky" throughout:
    # never wanted, and their slots therefore reusable.
    _incap = signal_incapable_prns(getattr(a, "signal", ""))
    if _incap:
        _drop = sorted(p for p in el if p in _incap)
        if _drop:
            el = {p: e for p, e in el.items() if p not in _incap}
            _log_rl("prnmap-incap",
                    "PRN MAP %s: %d satellite(s) excluded -- they do not broadcast this "
                    "signal at all (%s). They were holding tracker slots and folding noise "
                    "rows into the presence population."
                    % (log_tag() or a.signal, len(_drop),
                       ", ".join("PRN %d" % p for p in _drop[:10])), every_s=600.0)
    if not el:
        return  # no prediction this cycle: say nothing rather than evict the whole map

    held = set(cur)
    for p in list(st.down_since):
        if p not in el or el[p] >= a.prn_reconfig_evict_deg:
            st.down_since.pop(p, None)
    for p in held:
        if p in el and el[p] < a.prn_reconfig_evict_deg:
            st.down_since.setdefault(p, now)
    for p in list(st.gone_since):
        if p in el:
            st.gone_since.pop(p, None)
    for p in held:
        if p not in el:
            st.gone_since.setdefault(p, now)

    # ---- 3. Who wants a slot, and who can give one up -----------------------------------
    # ⚠️ THE UP-NOW BAR IS ABOUT PAYING FOR AN EVICTION, NOT ABOUT DESERVING A SLOT.
    # A candidate must be above the admit mask before it is worth EVICTING someone for --
    # a swap costs a re-acquisition, so only buy it for a satellite we can use immediately.
    # But a DEAD slot costs nothing to fill, and applying the up-now test to a free slot was
    # simply wrong: it left five Galileo slots empty while E36 -- active, and the satellite
    # this whole mechanism exists for -- was refused because it happened to be below the
    # horizon at that moment, and the heartbeat reported "0 satellites waiting for a slot"
    # beside five empty slots (KV, 2026-08-27).
    #
    # A below-horizon satellite in a FREE slot is not idle capacity, it is immediately useful:
    # it is exactly what the noise PROBES need. So free slots take any active satellite,
    # deepest-first (the best probes), and only evictions demand up-now.
    # The decision itself is desired_map() below; this is only for the heartbeat's count.
    want = [p for p in el if p not in held and el[p] >= a.prn_reconfig_admit_deg]

    # ⚠️ A HEARTBEAT, BECAUSE SILENCE HERE IS AMBIGUOUS. Every other branch of this stage only
    # speaks when it has something to propose, so an armed chain with nothing to do looks
    # EXACTLY like a stage that is not running -- and the normal state is nothing to do:
    # measured 2026-08-26, E36 is the one satellite this mechanism exists for and it sits below
    # the admit mask for ~13 h of every day. "Armed and healthy" has to be visible, or the
    # first question after every restart is unanswerable without a code read.
    tag = log_tag() or a.signal
    if now - st.beat_t >= a.prn_reconfig_heartbeat_s:
        st.beat_t = now
        n_dead = sum(1 for p in held if p in st.gone_since)
        n_down = sum(1 for p in held if p in st.down_since)
        _log("PRN MAP %s: %s, %d slots, %d nodes agree | %d dead, %d below %.0f deg, "
             "%d satellite(s) waiting for a slot | %d swap(s) so far%s"
             % (tag, a.prn_reconfig.upper(), len(cur), len(st.maps), n_dead, n_down,
                a.prn_reconfig_evict_deg, len(want), st.swaps,
                (" | last error: %s" % st.err) if st.err else ""))
    # ---- 3a. THE WHOLE MAP, EVERY CYCLE ---------------------------------------------------
    # KV, 2026-08-27: "the broker should push regular updates of available PRNs to all
    # trackers. 3 below horizon + all above." One statement of intent, re-evaluated each
    # cycle, replacing the old one-slot-per-15-minutes eviction with its multi-hour hold
    # timers -- a design that in production NEVER FIRED ONCE. Two reasons, and the second is
    # the one that mattered: `prn-reconfig` never left `report`, and `down_since`/`gone_since`
    # live in this in-memory state, so every broker restart reset them to now. Measured
    # 2026-08-27, after a day of restarts: the heartbeat reported "7 dead, 15 below 0 deg"
    # and "NO slot is evictable" in the same breath, while E36 sat unslotted at 82 deg -- the
    # best satellite in the sky, structurally invisible, in a chain with nine free slots.
    #
    # ⚠️ NO HOLD TIMERS IN THE DECISION AT ALL, DELIBERATELY. A PRN with no ephemeris is dead
    # NOW, not in two hours; a satellite below the horizon is below it NOW. The timers were
    # standing in for hysteresis, and hysteresis is what desired_map does properly (admit
    # above admit_deg, keep until below evict_deg) without any state that a restart can lose.
    n_probe = max(0, int(getattr(a, "noise_probes", 0) or 0))
    # ⚠️ THE HYSTERESIS BAND MUST BE A BAND. admit <= evict makes admission and eviction the
    # same test, so a satellite sitting on the threshold is admitted and evicted on alternate
    # cycles -- and every flap is a COLD acquisition, which looks exactly like a satellite
    # that cannot be tracked at low elevation. That is a fault this stage would otherwise
    # manufacture and then be believed about, so it is checked where the numbers are used,
    # not left to a docstring.
    if a.prn_reconfig_admit_deg <= a.prn_reconfig_evict_deg:
        _log_rl("prnmap-hyst",
                "PRN MAP %s: --prn-reconfig-admit-deg %.1f is not ABOVE "
                "--prn-reconfig-evict-deg %.1f, so there is no hysteresis band and a "
                "satellite at the threshold will flap in and out of its slot, paying a cold "
                "acquisition each time. Fix the pair; the map is left alone this cycle."
                % (tag, a.prn_reconfig_admit_deg, a.prn_reconfig_evict_deg), every_s=300.0)
        return
    want_map, unplaced = desired_map(cur, el, n_probe,
                                     a.prn_reconfig_admit_deg, a.prn_reconfig_evict_deg)
    moved = [i for i in range(len(cur)) if want_map[i] != cur[i]]
    if unplaced:
        # A REAL CAPACITY DECISION, and it must be said rather than silently absorbed.
        _log_rl("prnmap-full",
                "PRN MAP %s: %d satellite(s) want a slot and cannot have one (%s) -- every "
                "slot holds something we also want. Raise the slot count (node restart) or "
                "accept the loss."
                % (tag, len(unplaced), ", ".join("PRN %d el %+.0f" % (p, el.get(p, -99.0))
                                                 for p in unplaced[:6])),
                every_s=600.0)
    if not moved:
        return
    _why = "%d slot(s): %s" % (
        len(moved), ", ".join("s%d %d->%d" % (i, cur[i], want_map[i]) for i in moved[:6]))
    if a.prn_reconfig == "report":
        _log_rl("prnmap-report",
                "PRN MAP %s (REPORT ONLY, nothing posted): would move %s. Arm with "
                "--prn-reconfig apply." % (tag, _why), every_s=300.0)
        return
    # ONE POST, WHOLE MAP, ONE DEADLINE. Moving several slots in a single scheduled swap is
    # strictly better than dribbling them out one per interval: the nodes cross once instead
    # of N times, and the map is never in a half-applied state that no node agreed to.
    if now - st.last_swap_t < a.prn_reconfig_interval_s:
        return
    _apply_map(ctx, st, cur, want_map, moved, _why, el, now)
    return


def desired_map(cur, el, n_probe, admit_deg, evict_deg):
    """The map the chain SHOULD hold: every satellite worth tracking, plus the probes.

    KV's policy, 2026-08-27: "3 below horizon + all above." One statement of intent, evaluated
    every cycle, rather than the old one-slot-at-a-time eviction with hours-long hold timers --
    which never once fired in production, because the timers live in memory and every broker
    restart reset them.

    Returns (want_map, unplaced). `want_map` is `cur` with as much of the wanted set placed as
    there are slots; `unplaced` is whoever did not fit, best-first, which is a REAL capacity
    decision the caller must report rather than hide.

    ⚠️ STABILITY IS THE POINT, NOT MINIMALITY. Every slot that changes costs a re-acquisition,
    so a satellite already held KEEPS ITS SLOT and only genuinely unwanted slots are reused.
    Recomputing an "optimal" assignment each cycle would churn the whole map for no gain --
    the same reason the probe selector's deepest-N is sticky.

    ⚠️ HYSTERESIS, or a satellite at the horizon flaps in and out every cycle and each flap is
    a re-acquisition: admitted above `admit_deg`, kept until it falls below `evict_deg`. The
    two must differ; with admit == evict this degenerates to a flapper.
    """
    held = set(cur)
    # WANTED: everything up (with hysteresis), then the deepest below-horizon as probes.
    up = sorted((p for p, e in el.items() if e >= admit_deg or (p in held and e >= evict_deg)),
                key=lambda p: -el[p])
    probes = sorted((p for p, e in el.items() if e < -15.0), key=lambda p: el[p])[:n_probe]
    want = list(dict.fromkeys(up + probes))          # ordered, de-duplicated: up wins ties
    place = [p for p in want if p not in held]
    # Reusable slots, worst-first: PRNs the sky has nothing to say about at all (gone from
    # BRDC -- no ephemeris, so they produce literally nothing), then the ones furthest below
    # the horizon. Never a slot we still want.
    free = sorted((i for i, p in enumerate(cur) if p not in want),
                  key=lambda i: (cur[i] in el, el.get(cur[i], -91.0)))
    want_map = list(cur)
    for i, prn in zip(free, place):
        want_map[i] = prn
    unplaced = place[len(free):]
    # ⚠️ AND A SLOT LEFT OVER GOES TO WHOEVER IS LEFT. A slot the wanted set did not claim is
    # holding a PRN we do not want -- usually one with no ephemeris at all, producing nothing.
    # Handing it to ANY real satellite is pure gain, and there is no re-acquisition to pay for
    # because nothing was being tracked. This is the 2026-08-27 lesson restated: the admit
    # mask exists to justify an EVICTION, and a free slot is not an eviction. Without this a
    # satellite at +4 deg -- too low to admit, too high to be a probe -- falls in the gap and
    # is refused a slot that is standing empty.
    if len(place) < len(free):
        spare = [i for i in free[len(place):]]
        rest = sorted((p for p in el if p not in want and p not in want_map),
                      key=lambda p: -el[p])
        for i, prn in zip(spare, rest):
            want_map[i] = prn
    return want_map, unplaced


def _at_hop(ctx, a, now):
    """The absolute F-engine HOP every node should swap on, or None if we cannot pick one.

    ⚠️⚠️ THE POINT IS THAT TWELVE NODES CROSS THE DISCONTINUITY ON THE SAME FRAME. A map
    posted "now" lands on whatever frame each node happens to be building, so the combiner
    folds one window whose instances disagree about which satellite slot p IS -- an
    accumulator-identity error, and an invisible one, because every row is individually
    well-formed. Naming an absolute sample makes the swap simultaneous by construction.

    ⚠️ THE F-ENGINE COUNTER, NOT WALL TIME. It is the axis the records are indexed by and it
    is identical on every node; wall time is not (the AXIS INST spread runs to seconds), so a
    wall deadline would put the nodes back on twelve different frames.

    ⚠️⚠️ HOPS. `fe_hop_now` is the combiner's `pow_hop`, which is sample_seq/fft_len, and that
    is the only view of the axis the broker has. The first version of this posted the number
    under the key `at_seq` into a node that compared it against sample_seq -- 16384x larger --
    so the deadline was always already past and every swap took the apply-immediately degrade
    while logging as scheduled. The unit is in the wire key now (`at_hop`) so the two sides
    cannot drift apart again silently.

    Returns None when the axis is unknown -- and the caller then posts WITHOUT a deadline,
    i.e. the old ASAP behaviour. Fail-open is right here (a slot stuck on a set satellite
    produces nothing at all) but it is never SILENT: both degrade paths log.
    """
    fh = getattr(ctx, "fe_hop_now", None)
    ft = getattr(ctx, "fe_hop_t", None)
    if fh is None or ft is None:
        _log_rl("prnmap-noaxis",
                "PRN MAP %s: no F-engine axis this cycle -- a swap will post UNSCHEDULED, so "
                "the nodes will cross on different frames. Degraded, deliberately: an "
                "unsynchronised swap beats a slot stuck on a satellite that has set."
                % (log_tag() or a.signal), every_s=300.0)
        return None
    try:
        hps = float(a.hops_per_sec)
        # ⚠️ ADVANCE THE HOP TO NOW. `fh` was read during this cycle's status poll, which runs
        # near the top of the cycle; this stage runs near the bottom. On the live fleet that
        # gap is seconds and the lead is 2 s, so a deadline built on the raw `fh` lands in the
        # PAST and every node applies at its own next frame boundary -- the unsynchronised
        # swap this whole mechanism exists to avoid, arrived at silently because an
        # already-past deadline is indistinguishable from a met one.
        age = float(now) - float(ft)
        if not (0.0 <= age <= float(a.prn_reconfig_axis_max_age_s)):
            _log_rl("prnmap-axisage",
                    "PRN MAP %s: the F-engine axis sample is %.1f s old (limit %.0f) -- "
                    "extrapolating it that far would be a FABRICATED deadline, so this swap "
                    "posts UNSCHEDULED. Check the status poll and pow_hop."
                    % (log_tag() or a.signal, age, a.prn_reconfig_axis_max_age_s),
                    every_s=300.0)
            return None
        lead = float(a.prn_reconfig_lead_s)
        return int(round(float(fh) + (age + lead) * hps))
    except Exception as e:
        # ⚠️ SAY SO. A bare `return None` degrades every swap to unscheduled and looks exactly
        # like a healthy fleet with a quiet sky -- the silent-fallback shape purged on
        # 2026-08-27. A missing or unparseable knob is a CONFIG fault; it must be loud even
        # though the swap still goes out.
        _log_rl("prnmap-sched",
                "PRN MAP %s: cannot compute a swap deadline (%s) -- posting UNSCHEDULED. "
                "This is a configuration fault, not a sky condition."
                % (log_tag() or a.signal, e), every_s=300.0)
        return None


def _apply_map(ctx, st, cur, want_map, moved, why, el, now):
    """POST the WHOLE map to every producer, then to the followers, on one deadline.

    ⚠️ ONE IMPLEMENTATION AND ONE POST. The previous version moved a single slot per call and
    had two callers (free-slot fill, eviction); the whole map on one deadline is strictly
    better -- the nodes cross once instead of N times, and the map is never in a half-applied
    state that no node ever agreed to."""
    a = ctx.args
    tag = log_tag() or a.signal
    st.last_swap_t = now
    at_hop = _at_hop(ctx, a, now)
    body = {"prns": want_map}
    if at_hop is not None:
        body["at_hop"] = at_hop
    ok, bad = 0, ""
    for ep in _endpoints(ctx):
        try:
            _post("%s/set_prns" % ep, body, timeout=a.prn_reconfig_timeout_s)
            ok += 1
        except Exception as e:
            bad = "%s: %s" % (ep, e)
    n_follow = 0
    for ep in _followers(ctx):
        try:
            # ⚠️ THE SEARCH GETS THE MAP BUT NOT THE DEADLINE. It has no frame boundary to
            # test one against -- it applies between passes -- and a deadline it cannot
            # honour would just wedge its map. It is also not in the record path, so an
            # early swap there costs a re-scan, not a corrupted accumulator.
            _post("%s/set_prns" % ep, {"prns": want_map}, timeout=a.prn_reconfig_timeout_s)
            n_follow += 1
        except Exception as e:
            # A search whose list did not move searches for a satellite that has no slot and
            # misses one that does: a real degradation, and one worth naming, but not a reason
            # to leave the producer half-swapped.
            _log("PRN MAP: follower %s refused the map (%s) -- it now searches a DIFFERENT "
                 "set from the one the tracker holds." % (ep, e))
    if ok == 0:
        st.refused += 1
        st.err = bad
        _log("PRN MAP %s: %s REFUSED by every node (%s)" % (tag, why, bad))
        return
    st.swaps += 1
    # ⚠️ THE LOCAL MAP IS NOT UPDATED HERE. The next poll reads it back from the nodes, so
    # what this stage diffs against is always what the nodes actually hold -- a POST that
    # 200s but does not take (a slot the node refuses because the PRN has no code for that
    # signal) would otherwise be invisible for as long as the broker ran.
    st.maps.clear()   # force a FULL re-sweep before the next decision
    st.poll_t = 0.0
    _log("PRN MAP %s: %s -- posted to %d/%d node(s)%s%s, %s. Every moved slot acquires COLD: "
         "expect them dark for a minute or two. New satellites at el %s."
         % (tag, why, ok, len(_endpoints(ctx)),
            (" (%d failed: %s)" % (len(_endpoints(ctx)) - ok, bad)) if bad else "",
            (" + %d searcher(s)" % n_follow) if n_follow else "",
            ("all crossing together at hop %d" % at_hop) if at_hop is not None
            else "UNSCHEDULED (no axis) -- the nodes will cross on different frames",
            ", ".join("%+.0f" % el.get(want_map[i], -99.0) for i in moved[:6])))
