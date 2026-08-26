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

from gnss_broker.transport import _get, _post, _log, _log_rl, log_tag


class PrnMapState(object):
    """One chain's view of node membership, and the hysteresis that governs changes."""

    __slots__ = ("maps", "poll_t", "cursor", "last_swap_t", "down_since", "gone_since", "err",
                 "swaps", "refused")

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


def stage_prn_membership(ctx):
    """Reconcile the nodes' slot->PRN map against live BRDC. Off unless --prn-reconfig.

    Produces at most ONE swap per --prn-reconfig-interval-s, and only when an incumbent has
    been evictable for hours and a candidate is genuinely up. See the module docstring for why
    it is this reluctant.
    """
    a = ctx.args
    if a.prn_reconfig == "off":
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
    if len(st.maps) < len(eps):
        return
    cur = _consensus(st.maps)
    if cur is None:
        if st.maps:
            _log_rl("prnmap-split",
                    "PRN MAP: nodes DISAGREE about slot membership (%d reporting) -- changing "
                    "nothing. Nothing in this pipeline is per-node, so a split map is a fault "
                    "to fix, not a state to drive out of." % len(st.maps),
                    every_s=300.0)
        return

    # ---- 2. What the sky says -----------------------------------------------------------
    # ctx.pred is THIS chain's constellation already filtered by capability: brdc_predict is
    # called with alm_sys and alm_min_prn, which is the same "C1-C14 are BDS-2 and do not
    # broadcast B2a" rule the manifest states beside the list. So absence from pred means
    # "not a satellite this chain could ever track", not merely "not up".
    el = {p: v[2] for p, v in (ctx.pred or {}).items()}
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
    # A candidate must be ABOVE the admit mask right now. "Rises eventually" is not enough:
    # the cost of a swap is a re-acquisition, so it is only worth paying for a satellite we
    # can start using immediately -- and one that is up now will be up again tomorrow.
    want = sorted((p for p in el if p not in held and el[p] >= a.prn_reconfig_admit_deg),
                  key=lambda p: -el[p])
    if not want:
        return

    # Evictable, best first: a slot whose satellite is GONE from BRDC costs nothing at all,
    # then the ones that have been furthest below the horizon for longest.
    gone = sorted((p for p in held if p in st.gone_since
                   and now - st.gone_since[p] >= a.prn_reconfig_gone_hold_s),
                  key=lambda p: st.gone_since[p])
    down = sorted((p for p in held if p in st.down_since
                   and now - st.down_since[p] >= a.prn_reconfig_down_hold_s),
                  key=lambda p: (el.get(p, -90.0), st.down_since[p]))
    evictable = gone + [p for p in down if p not in gone]

    tag = log_tag() or a.signal
    if not evictable:
        _log_rl("prnmap-full",
                "PRN MAP %s: %d satellite(s) up with no slot (%s, peak el %.0f) and NO slot is "
                "evictable -- every one is either up or has not been down long enough. This is "
                "a real capacity decision, not a mistake: raise the count (node restart) or "
                "accept the loss."
                % (tag, len(want), ", ".join(str(p) for p in want[:6]), el[want[0]]),
                every_s=600.0)
        return

    new_prn, old_prn = want[0], evictable[0]
    slot = cur.index(old_prn)
    why = ("GONE from BRDC for %.1f h" % ((now - st.gone_since[old_prn]) / 3600.0)
           if old_prn in st.gone_since else
           "below %.0f deg for %.1f h (now %.0f)"
           % (a.prn_reconfig_evict_deg, (now - st.down_since[old_prn]) / 3600.0,
              el.get(old_prn, -90.0)))

    if a.prn_reconfig == "report":
        _log_rl("prnmap-report",
                "PRN MAP %s (REPORT ONLY, nothing posted): would swap slot %d, PRN %d -> %d. "
                "Incumbent %s; candidate is up at %.0f deg. %d slot(s) evictable, %d satellite(s) "
                "waiting. Arm with --prn-reconfig apply."
                % (tag, slot, old_prn, new_prn, why, el[new_prn], len(evictable), len(want)),
                every_s=300.0)
        return

    # ---- 4. Apply, one slot per interval ------------------------------------------------
    if now - st.last_swap_t < a.prn_reconfig_interval_s:
        return
    st.last_swap_t = now
    want_map = list(cur)
    want_map[slot] = new_prn
    ok, bad = 0, ""
    for ep in _endpoints(ctx):
        try:
            _post("%s/set_prns" % ep, {"prns": want_map},
                  timeout=ctx.args.prn_reconfig_timeout_s)
            ok += 1
        except Exception as e:
            bad = "%s: %s" % (ep, e)
    n_follow = 0
    for ep in _followers(ctx):
        try:
            _post("%s/set_prns" % ep, {"prns": want_map},
                  timeout=ctx.args.prn_reconfig_timeout_s)
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
        _log("PRN MAP %s: swap slot %d PRN %d -> %d REFUSED by every node (%s)"
             % (tag, slot, old_prn, new_prn, bad))
        return
    st.swaps += 1
    # ⚠️ THE LOCAL MAP IS NOT UPDATED HERE. The next poll reads it back from the nodes, so
    # what this stage diffs against is always what the nodes actually hold -- a POST that
    # 200s but does not take (a slot the node refuses because the PRN has no code for that
    # signal) would otherwise be invisible for as long as the broker ran.
    st.maps.clear()   # force a FULL re-sweep before the next decision
    st.poll_t = 0.0
    _log("PRN MAP %s: slot %d PRN %d -> %d posted to %d/%d node(s)%s%s. Incumbent %s; new "
         "satellite up at %.0f deg. That slot re-acquires COLD -- expect it dark for a "
         "minute or two."
         % (tag, slot, old_prn, new_prn, ok, len(_endpoints(ctx)),
            (" (%d failed: %s)" % (len(_endpoints(ctx)) - ok, bad)) if bad else "",
            (" + %d searcher(s)" % n_follow) if n_follow else "",
            why, el[new_prn]))
