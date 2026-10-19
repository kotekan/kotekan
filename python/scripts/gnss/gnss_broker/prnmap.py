"""Live PRN membership: keep each node's slot->PRN map in step with the sky.

The nodes boot on the PRN lists in config/gnss_fleet_chord.yaml, while the broker sees the
constellation through live BRDC every cycle. This stage reconciles the two: the slots hold
the satellites that are up, plus --noise-probes probes in probe_order (the dark satellites under
--probe-dark, then the deepest below -15 deg), the rule the probe selector picks noise probes by
(a probe the nodes hold no slot for is seeded but never reports).

The slot count is fixed; only membership moves. n_prn sizes every buffer, GPU allocation and
wire frame in the pipeline, so changing it is a fleet-wide re-plumb and a node restart.
Swapping which satellite sits in a slot, at constant count, changes no size anywhere.

A swap costs a full re-acquisition: the node resets the slot cold (code table, Phi cache,
carrier NCO, seed, trim, power averages, element cal), since none of it describes the new
satellite. Changes are therefore damped: elevation hysteresis (desired_map), a held satellite
keeps its slot, and at most one whole-map post per --prn-reconfig-interval-s.

--prn-reconfig defaults to off (the stage does nothing; the replay fixtures' argv predates the
flag); every live chain runs `apply`.

The broker does not read the node's config; it GETs the live map, so it diffs against what
each node actually holds. The map is a runtime overlay, not persisted: a node restart reverts
that node to its config list. A node back on its boot list while the rest of the fleet holds
the broker's map is put back on that map (_resync_split); until it is, the fleet is split and
nothing else changes.

@author Keith Vanderlinde
"""

import os
import re
import time

from gnss_broker.transport import _get, _post, _log, _log_rl, log_tag


class PrnMapState(object):
    """One chain's view of node membership, and the timers that pace changes."""

    __slots__ = (
        "maps",
        "poll_t",
        "cursor",
        "last_swap_t",
        "err",
        "swaps",
        "beat_t",
        "consensus",
        "pending",
        "applied",
        "resync_t",
    )

    def __init__(self):
        self.maps = {}  # endpoint -> [prn per slot]
        self.pending = {}  # endpoint -> a staged swap had not crossed when it was read
        self.applied = None  # the last map this broker posted and a node accepted
        self.resync_t = {}  # endpoint -> t of the last resync posted to it
        self.poll_t = 0.0  # when _poll last ran
        self.cursor = 0  # round-robin index of the next endpoint to GET (see _poll)
        self.last_swap_t = 0.0  # last whole-map post (--prn-reconfig-interval-s)
        self.err = ""
        self.swaps = 0
        self.beat_t = 0.0  # last heartbeat line (prn_reconfig_heartbeat_s)
        # The unanimous live slot->PRN list, or None while the sweep is incomplete or the
        # nodes disagree. Read by the probe selector (--probe-require-slot).
        self.consensus = None


def _endpoints(ctx):
    """The tracker endpoints: the producer owns membership, so it is the one asked."""
    return list(ctx.trackers or [])


def _followers(ctx):
    """Stages that hold their own copy of the map and must be driven in step: the search.

    The record assembler learns the map from the frame itself (gnss_gpu::PrnCtl::prn), so it
    needs no endpoint and cannot drift. The search has no such channel: it hunts the
    satellites its list names, so a stale list there is a satellite it can never detect. It is
    posted the same map, without the deadline (see _apply_map).

    Followers are not polled for consensus. The producer is the authority; a follower that
    refuses a swap is logged and left behind rather than holding the producer back, because a
    search one satellite behind is a degradation, while a producer behind misstates what the
    records contain.
    """
    return list(ctx.detectors or [])


def _poll(ctx, st, eps):
    """GET one endpoint's live map per call, round-robin.

    The stage runs inside the cycle loop, so GETting every endpoint in one call would cost
    n_dead x timeout on a single cycle whenever nodes are down. One GET per call bounds that
    to one prn_reconfig_timeout_s; stage_prn_membership spaces the calls at least
    poll_s / n_endpoints apart, so a full sweep takes at least prn_reconfig_poll_s.

    An endpoint that fails is dropped from the map rather than left stale: a node that stopped
    answering has not agreed to anything, and its last answer would hide a split map.
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
            st.pending[ep] = bool(r.get("pending"))
        else:
            st.maps.pop(ep, None)
            st.pending.pop(ep, None)
        # A node that answers this GET is running, so last_hop < 0 (or no last_hop at all)
        # means nothing feeds its deadline clock (note_frame_hop), and every scheduled swap
        # there applies at once with no error on either side. Nothing else would show it.
        _lh = r.get("last_hop")
        if _lh is None or int(_lh) < 0:
            _log_rl(
                "prnmap-noclock",
                "PRN MAP %s: %s reports last_hop=%s -- its deadline clock is DEAD, so "
                "scheduled swaps there apply IMMEDIATELY and the fleet does not cross "
                "together. Armed-but-inert: check that the stage owning the frame loop "
                "calls note_frame_hop()." % (log_tag() or ctx.args.signal, ep, _lh),
                every_s=600.0,
            )
    except Exception as e:
        st.err = "%s: %s" % (ep, e)
        st.maps.pop(ep, None)
        st.pending.pop(ep, None)
    # Endpoints that have left the configured list must not linger in the consensus.
    for gone in [k for k in st.maps if k not in eps]:
        st.maps.pop(gone, None)
        st.pending.pop(gone, None)


def _consensus(maps):
    """The map every reporting node agrees on, or None.

    Disagreement is not averaged. Nothing in the pipeline is per-node, so a split map never
    drives a decision: it is either a node that lost the fleet's map, which _resync_split puts
    back, or ambiguous, and then nothing changes.
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


def _resync_split(ctx, st, eps, now):
    """Put endpoints that left the fleet's map back on it. False if the split is ambiguous.

    A complete sweep splits for two ordinary reasons: a node restart reverts that node to its
    config's list, and a post that failed leaves a node on the previous map. In both cases the
    rest of the fleet still holds a map the broker chose, so the reference is the last map this
    broker posted, if any node holds it, else the map a strict majority holds (the case after a
    broker restart). With neither, nothing moves. Nor does it while any node has a swap
    pending: the fleet is mid-crossing and the read-back is not final.

    The resync is unscheduled. Only the odd node changes, and its moved slots acquire cold
    whenever they cross, so there is no fleet-wide frame to agree on.
    """
    a = ctx.args
    if any(st.pending.get(ep) for ep in eps):
        return True
    ref = None
    if st.applied is not None and any(st.maps.get(ep) == st.applied for ep in eps):
        ref = st.applied
    else:
        counts = {}
        for ep in eps:
            if ep in st.maps:
                k = tuple(st.maps[ep])
                counts[k] = counts.get(k, 0) + 1
        if counts:
            best, n = max(counts.items(), key=lambda kv: kv[1])
            if 2 * n > len(eps):
                ref = list(best)
    if ref is None:
        return False
    tag = log_tag() or a.signal
    odd = [ep for ep in eps if st.maps.get(ep) != ref]
    for ep in odd:
        # A node that keeps refusing is retried once per interval, not every cycle.
        if now - st.resync_t.get(ep, -1e18) < a.prn_reconfig_interval_s:
            continue
        st.resync_t[ep] = now
        n_moved = sum(1 for x, y in zip(st.maps[ep], ref) if x != y)
        try:
            _post("%s/set_prns" % ep, {"prns": ref}, timeout=a.prn_reconfig_timeout_s)
            _log(
                "PRN MAP %s: %s held a different map in %d slot(s) (a restart reverts a node "
                "to its config list) -- put back on the fleet's map. Those slots acquire COLD."
                % (tag, ep, n_moved)
            )
        except Exception as e:
            st.err = "%s: %s" % (ep, e)
            _log("PRN MAP %s: resync of %s REFUSED (%s)" % (tag, ep, e))
        # Read it back before believing it.
        st.maps.pop(ep, None)
        st.pending.pop(ep, None)
    return True


# ---------------------------------------------------------------------------------------
# Signal capability: which satellites can carry this chain's signal at all
# ---------------------------------------------------------------------------------------
# GNSS_IGS_SNX pins the registry to one file, read as it is and never re-fetched. A chain
# with a capability gate seeds only the satellites the registry calls capable, so
# broker_equiv points a replay at its transcript's snapshot. Live brokers never set it.
_IGS_PINNED = bool(os.environ.get("GNSS_IGS_SNX"))
_IGS_SNX = os.environ.get("GNSS_IGS_SNX") or os.path.join(
    os.path.expanduser("~"), ".cache", "kotekan_gps", "igs_satellite_metadata.snx"
)
_IGS_SNX_URL = "https://files.igs.org/pub/station/general/igs_satellite_metadata.snx"
# Re-fetch the registry past this age. GLONASS re-slots satellites a few times a year and the
# capability of a slot follows the satellite flying it, so a month-old copy can seed a slot
# that no longer carries the signal, or never seed one that now does. Every reader that
# starts a process (the broker, the viewer, gen_fleet --check-prns) calls ensure_registry()
# first; the broker then re-reads the file on CAPABLE_REFRESH_S without fetching, so a
# download never sits inside the cycle.
REGISTRY_MAX_AGE_S = 7 * 86400.0
CAPABLE_REFRESH_S = 6 * 3600.0


def ensure_registry(max_age_s=REGISTRY_MAX_AGE_S, timeout_s=20.0):
    """Fetch the IGS metadata SINEX when the cached copy is missing or older than max_age_s.

    Written beside the cache and renamed into place, so a reader never sees a partial file;
    a failed fetch keeps the old copy (fail open, as every registry reader does). A pinned
    registry (GNSS_IGS_SNX) is left as it is, whatever its age. Returns the file's age in
    seconds, or None when there is no copy."""
    try:
        age = time.time() - os.path.getmtime(_IGS_SNX)
    except OSError:
        age = None
    if _IGS_PINNED or (age is not None and age < max_age_s):
        return age
    try:
        import urllib.request

        os.makedirs(os.path.dirname(_IGS_SNX), exist_ok=True)
        tmp = _IGS_SNX + ".tmp.%d" % os.getpid()
        with urllib.request.urlopen(_IGS_SNX_URL, timeout=timeout_s) as r, open(
            tmp, "wb"
        ) as f:
            f.write(r.read())
        os.replace(tmp, _IGS_SNX)
        return 0.0
    except Exception as e:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        _log(
            "IGS satellite registry refresh FAILED (%s); %s"
            % (
                e,
                "keeping the %.0f-day-old copy" % (age / 86400.0) if age else "no copy",
            )
        )
        return age


def registry_age_s():
    """Seconds since the cached registry was fetched; None without a copy."""
    try:
        return time.time() - os.path.getmtime(_IGS_SNX)
    except OSError:
        return None


def registry_capable(signal):
    """PRNs the registry allows for `signal`, or None when it proves nothing (unreadable, or a
    signal every satellite carries): the capability set the broker filters seeds and hints on,
    recomputed on CAPABLE_REFRESH_S from the cached file."""
    incap = signal_incapable_prns(signal)
    return set(range(1, 64)) - incap if incap else None


# Which IGS metadata block strings carry each civil signal. Block names come from the SINEX
# SATELLITE/IDENTIFIER column ("GPS-IIA", "GPS-IIR-A", "GPS-IIR-M", "GPS-IIF", "GPS-IIIA").
# L1 C/A -> every satellite; L2C -> IIR-M and newer; L5 -> IIF and newer; L1C -> III only.
# gps_beamtrack._signal_block_filter holds the same table for the short block names in
# Celestrak TLE titles; gps_distributed_broker falls back to it when the registry excludes
# nothing.
def _block_carries(signal, block):
    s, b = (signal or "").upper(), (block or "").upper()
    if "L1CA" in s:
        return True
    if "L5" in s:
        return b.startswith("GPS-IIF") or b.startswith("GPS-III")
    if "L2C" in s or "L2_CM" in s:
        return (
            b.startswith("GPS-IIR-M")
            or b.startswith("GPS-IIF")
            or b.startswith("GPS-III")
        )
    if "L1C" in s:
        return b.startswith("GPS-III")
    if s.startswith("BDS"):
        # B1C, B2a and B2b are BDS-3 signals; B1I and B3I are on BDS-2 and BDS-3 alike. The
        # block type says which a satellite is, not its PRN: re-used BDS-2 PRNs below 19,
        # C11-C14 among them, fly BDS-3 satellites.
        if "B1C" in s or "B2A" in s or "B2B" in s:
            return b.startswith("BDS-3")
        return True
    if s.startswith("GLO"):
        # L3OC is on GLONASS-K (GLO-K1, GLO-K1B, GLO-K1+, GLO-K2) and on the modernised
        # GLO-M+; the other CDMA signals (L2OC, L1OC) on the K generations alone; the plain
        # GLO-M satellites broadcast only the FDMA L1OF/L2OF, which every satellite carries.
        # The generations are interspersed across the slots (10-17: 14 of the 28 assigned
        # slots carry L3OC), so no PRN floor can express it.
        if "L3OC" in s:
            return b.startswith("GLO-K") or b == "GLO-M+"
        if "OC" in s:
            return b.startswith("GLO-K")
        return True
    return True  # a signal we do not model: exclude nothing


def signal_incapable_prns(signal):
    """PRNs that cannot broadcast `signal`, or an empty set if that cannot be established.

    A satellite that predates the signal (Block IIR / IIR-M on L5, for example) would
    otherwise hold a tracker slot and add a noise row to the presence population.

    The source is the IGS metadata SINEX (prn_blocks), not Celestrak TLE titles, which carry
    identity errors, BeiDou worst of all.

    Only what can be proven is excluded. A missing file, an unparseable line, a PRN absent
    from the registry, or a constellation or signal not modelled all yield "not incapable": a
    real satellite wrongly excluded can never be tracked and nothing says why, which is worse
    than a dead slot. Same polarity as gnss_brdc_supply._rec_is_fresh.

    GPS, BeiDou and GLONASS are modelled (for BeiDou, the BDS-3-only B1C/B2a/B2b against the
    BDS-2 blocks; for GLONASS, the CDMA L3OC/L2OC against the plain GLO-M blocks); every other
    constellation returns an empty set. Galileo needs no model: every Galileo satellite carries
    the signals the chains track.
    """
    s = (signal or "").upper()
    sysc = (
        "G"
        if s.startswith("GPS")
        else "C"
        if s.startswith("BDS")
        else "R"
        if s.startswith("GLO")
        else None
    )
    if sysc is None:
        return set()
    return {p for p, b in prn_blocks(sysc).items() if not _block_carries(signal, b)}


def block_carries(signal, block):
    """Public face of _block_carries for consumers outside the broker (the viewer's
    capability marking, viscap_metadata.py), so one place owns which block carries each civil
    signal."""
    return _block_carries(signal, block)


def gps_prn_blocks():
    """{prn -> block} for every currently-assigned GPS PRN (the viewer's and viscap's view)."""
    return prn_blocks("G")


def prn_blocks(sysc):
    """{prn -> block} for every currently assigned PRN of constellation `sysc` ("G", "C", "R"),
    from the IGS metadata SINEX. For R the PRN is the orbital slot, the number BRDC and the
    L3OC code index both use.

    The registry half of signal_incapable_prns, exposed so the viewer marks not-transmitted
    cells from the same source the broker filters on. SVN -> block comes from
    SATELLITE/IDENTIFIER and SVN -> PRN from SATELLITE/PRN, keeping an assignment only while
    its Valid_To is open or not yet past today (Valid_From is not checked), so a re-used PRN
    resolves to the satellite flying it now. {} on any failure (fail open)."""
    try:
        svn_block, svn_prn = {}, {}
        section = None
        now = time.strftime("%Y:%j:00000", time.gmtime())
        for line in open(_IGS_SNX, errors="replace"):
            if line.startswith("+SATELLITE/"):
                section = line.strip().lstrip("+")
                continue
            if line.startswith("-SATELLITE/"):
                section = None
                continue
            if line.startswith("*") or not line.strip():
                continue
            f = line.split()
            if section == "SATELLITE/IDENTIFIER" and len(f) >= 4:
                svn_block[f[0]] = f[3]
            elif section == "SATELLITE/PRN" and len(f) >= 4:
                svn, t_to, prn = f[0], f[2], f[3]
                if (t_to == "0000:000:00000" or t_to >= now) and re.fullmatch(
                    r"[A-Z]\d{2}", prn
                ):
                    svn_prn[svn] = prn
    except Exception:
        return {}
    out = {}
    for svn, prn in svn_prn.items():
        if prn[0] != sysc:
            continue
        b = svn_block.get(svn)
        if b:
            out[int(prn[1:])] = b
    return out


def stage_prn_membership(ctx):
    """Reconcile the nodes' slot->PRN map against live BRDC. Does nothing unless
    --prn-reconfig apply.

    Each cycle computes the wanted map (desired_map); when it differs from the nodes' map it
    is posted whole, on one deadline, at most once per --prn-reconfig-interval-s. The module
    docstring says why changes are damped.
    """
    a = ctx.args
    if a.prn_reconfig == "off":
        return
    st = ctx.prnmap
    now = ctx.t0
    eps = _endpoints(ctx)

    # ---- 1. What the nodes hold ---------------------------------------------------------
    # One endpoint is read every poll_s / n_endpoints (at most one per cycle), so each is
    # re-read about once per poll_s rather than all of them in one burst.
    if eps and now - st.poll_t >= a.prn_reconfig_poll_s / float(len(eps)):
        st.poll_t = now
        _poll(ctx, st, eps)
    # A full sweep must have completed before the map means anything: a partial one looks like
    # unanimity among however few nodes have answered so far.
    cur = _consensus(st.maps) if len(st.maps) >= len(eps) else None
    # Published for read-only consumers (the probe selector). None while the sweep is
    # incomplete or the nodes disagree, and consumers then fall back to their unfiltered
    # behaviour rather than act on a partial map.
    st.consensus = cur
    if cur is None:
        if st.maps and len(st.maps) >= len(eps):
            if not _resync_split(ctx, st, eps, now):
                _log_rl(
                    "prnmap-split",
                    "PRN MAP: nodes DISAGREE about slot membership (%d reporting), and no "
                    "map is held by a majority or was last posted by this broker -- "
                    "changing nothing. Nothing in this pipeline is per-node, so a split "
                    "map is a fault to fix." % len(st.maps),
                    every_s=300.0,
                )
        return

    # ---- 2. What the sky says -----------------------------------------------------------
    # ctx.pred holds this chain's constellation, below-horizon satellites included. On the
    # BRDC path it is already filtered by ctx.capable (--signal-capability) or, on a BeiDou
    # chain without one, by the PRN floor alm_min_prn. Absence from pred therefore means no
    # ephemeris or not trackable on this signal, not merely "not up".
    el = {p: v[2] for p, v in (ctx.pred or {}).items()}
    # A satellite that cannot broadcast this signal is not a candidate for a slot. Dropping
    # it here rather than in desired_map makes it read as absent throughout: never wanted,
    # and its slot reusable.
    _incap = signal_incapable_prns(getattr(a, "signal", ""))
    if _incap:
        _drop = sorted(p for p in el if p in _incap)
        if _drop:
            el = {p: e for p, e in el.items() if p not in _incap}
            _log_rl(
                "prnmap-incap",
                "PRN MAP %s: %d satellite(s) excluded -- they do not broadcast this "
                "signal at all (%s). They were holding tracker slots and folding noise "
                "rows into the presence population."
                % (
                    log_tag() or a.signal,
                    len(_drop),
                    ", ".join("PRN %d" % p for p in _drop[:10]),
                ),
                every_s=600.0,
            )
    if not el:
        return  # no prediction this cycle: say nothing rather than evict the whole map
    # --probe-dark: satellites with ephemeris that cannot broadcast this signal, kept as noise
    # probes (probe_order). Not in `el`, so without this they would read as dead slots.
    dark = (
        sorted(getattr(ctx, "pred_dark", None) or {})
        if getattr(a, "probe_dark", False)
        else []
    )

    held = set(cur)

    # ---- 3. Who wants a slot, and who can give one up -----------------------------------
    # Unslotted satellites above the admit mask, for the heartbeat's "waiting for a slot"
    # count only; desired_map() decides. There a slot holding a real satellite is given up
    # only for one above the admit mask or a noise probe (a swap costs a re-acquisition),
    # while a dead slot costs nothing to fill and goes to any satellite.
    want = [p for p in el if p not in held and el[p] >= a.prn_reconfig_admit_deg]

    # Heartbeat: every other branch speaks only when it has something to propose, and
    # nothing to do is the normal state, so without it an armed, idle stage looks the same as
    # one that is not running.
    tag = log_tag() or a.signal
    if now - st.beat_t >= a.prn_reconfig_heartbeat_s:
        st.beat_t = now
        n_dead = sum(1 for p in held if p not in el and p not in dark)
        n_down = sum(1 for p in held if p in el and el[p] < a.prn_reconfig_evict_deg)
        _log(
            "PRN MAP %s: %s, %d slots, %d nodes agree | %d dead, %d below %.0f deg, "
            "%d satellite(s) waiting for a slot | %d swap(s) so far%s"
            % (
                tag,
                a.prn_reconfig.upper(),
                len(cur),
                len(st.maps),
                n_dead,
                n_down,
                a.prn_reconfig_evict_deg,
                len(want),
                st.swaps,
                (" | last error: %s" % st.err) if st.err else "",
            )
        )
    # ---- 3a. The whole map, every cycle ------------------------------------------------
    # One statement of intent (all satellites up plus the noise probes), re-evaluated each
    # cycle, with no hold timers: a PRN with no ephemeris is dead now, and the hysteresis is
    # in elevation (desired_map), so a broker restart loses no state the decision needs.
    n_probe = max(0, int(getattr(a, "noise_probes", 0) or 0))
    # The hysteresis band must be a band. With admit <= evict a satellite at the threshold
    # is admitted and evicted on alternate cycles, each flap a cold acquisition that looks
    # like a satellite that cannot be tracked low. Checked here, where the numbers are used.
    if a.prn_reconfig_admit_deg <= a.prn_reconfig_evict_deg:
        _log_rl(
            "prnmap-hyst",
            "PRN MAP %s: --prn-reconfig-admit-deg %.1f is not ABOVE "
            "--prn-reconfig-evict-deg %.1f, so there is no hysteresis band and a "
            "satellite at the threshold will flap in and out of its slot, paying a cold "
            "acquisition each time. Fix the pair; the map is left alone this cycle."
            % (tag, a.prn_reconfig_admit_deg, a.prn_reconfig_evict_deg),
            every_s=300.0,
        )
        return
    want_map, unplaced = desired_map(
        cur, el, n_probe, a.prn_reconfig_admit_deg, a.prn_reconfig_evict_deg, dark=dark
    )
    moved = [i for i in range(len(cur)) if want_map[i] != cur[i]]
    if unplaced:
        # A capacity decision: logged, not absorbed silently.
        _log_rl(
            "prnmap-full",
            "PRN MAP %s: %d satellite(s) want a slot and cannot have one (%s) -- every "
            "slot holds something we also want. Raise the slot count (node restart) or "
            "accept the loss."
            % (
                tag,
                len(unplaced),
                ", ".join(
                    "PRN %d el %+.0f" % (p, el.get(p, -99.0)) for p in unplaced[:6]
                ),
            ),
            every_s=600.0,
        )
    if not moved:
        return
    _why = "%d slot(s): %s" % (
        len(moved),
        ", ".join("s%d %d->%d" % (i, cur[i], want_map[i]) for i in moved[:6]),
    )
    # One post, the whole map, one deadline: the nodes cross once however many slots move,
    # and the map is never half-applied.
    if now - st.last_swap_t < a.prn_reconfig_interval_s:
        return
    _apply_map(ctx, st, cur, want_map, moved, _why, el, now)
    return


def probe_order(el, dark=(), held=()):
    """Noise-probe candidates, best first: the `dark` PRNs (those in `held` first, each group
    in PRN order), then every PRN of `el` ({prn: elevation deg}) below -15 deg, deepest first.

    Dark PRNs (--probe-dark) have ephemeris but cannot broadcast this signal, so they are
    signal-free at any elevation and never set. Every such satellite qualifies (on glo_l2oc the
    plain GLO-M slots as well as the M+ ones), so the ones the nodes already hold come first:
    the map then has nothing to swap. PRN order keeps the choice stable from cycle to cycle.
    The broker's selector (held = the nodes' map) and desired_map (held = the current map)
    both order probes by this, so the map holds the probes the selector picks. Callers take
    the first --noise-probes.
    """
    held = set(held or ())
    deep = sorted((p for p, e in el.items() if e < -15.0), key=lambda p: el[p])
    dark = sorted(dark, key=lambda p: (p not in held, p))
    return list(dict.fromkeys(dark + deep))


def desired_map(cur, el, n_probe, admit_deg, evict_deg, dark=()):
    """The map the chain should hold: every satellite worth tracking, plus the noise probes.

    Wanted: every satellite up (admitted at >= `admit_deg`, a held one kept until it falls
    below `evict_deg`), then `n_probe` noise probes in probe_order: the `dark` PRNs
    (--probe-dark; not in `el`), then the deepest below -15 deg.

    Returns (want_map, unplaced). `want_map` is `cur` with as much of the wanted set placed as
    there are slots; `unplaced` is whoever did not fit, best first, which the caller must
    report rather than hide.

    Stability over minimality: every changed slot costs a re-acquisition, so a satellite
    already held keeps its slot and only unwanted slots are reused. Recomputing an optimal
    assignment each cycle would churn the map for no gain.

    Hysteresis: without it a satellite at the horizon flaps in and out every cycle, each flap
    a re-acquisition. The caller refuses admit_deg <= evict_deg.
    """
    held = set(cur)
    # Wanted: everything up (with hysteresis), then the deepest below -15 deg as probes.
    up = sorted(
        (p for p, e in el.items() if e >= admit_deg or (p in held and e >= evict_deg)),
        key=lambda p: -el[p],
    )
    probes = probe_order(el, dark, held)[:n_probe]
    want = list(dict.fromkeys(up + probes))  # ordered, de-duplicated: up wins ties
    place = [p for p in want if p not in held]
    # Reusable slots, worst first: PRNs with no prediction at all (no ephemeris, or dropped
    # as incapable), which produce nothing, then the ones furthest below the horizon. Never a
    # slot we still want.
    free = sorted(
        (i for i, p in enumerate(cur) if p not in want),
        key=lambda i: (cur[i] in el, el.get(cur[i], -91.0)),
    )
    want_map = list(cur)
    for i, prn in zip(free, place):
        want_map[i] = prn
    unplaced = place[len(free) :]
    # A dead slot left over goes to the best remaining satellite. A slot whose PRN has no
    # prediction produces nothing, so filling it costs no re-acquisition; the admit mask
    # justifies evictions, and this is not one. Without it a satellite between -15 deg and the
    # admit mask (too high to be a probe, too low to admit) is refused a slot that stands
    # empty.
    # Only a dead slot: a slot holding a real satellite we merely do not want keeps it.
    # Trading it for another unwanted satellite gains nothing and costs a cold acquisition,
    # and with one satellite left unslotted the two would trade places every interval.
    if len(place) < len(free):
        spare = [i for i in free[len(place) :] if cur[i] not in el]
        rest = sorted(
            (p for p in el if p not in want and p not in want_map), key=lambda p: -el[p]
        )
        for i, prn in zip(spare, rest):
            want_map[i] = prn
    return want_map, unplaced


def _at_hop(ctx, a, now):
    """The absolute F-engine hop every node should swap on, or None if one cannot be picked.

    The point is that every node crosses the discontinuity on the same frame. A map posted for
    "now" lands on whatever frame each node is building, so the combiner folds a window whose
    instances disagree about which satellite slot p holds: an accumulator-identity error that
    no row shows, since each is well-formed on its own. Naming an absolute hop makes the swap
    simultaneous by construction.

    The F-engine counter, not wall time: it is the axis the records are indexed by and it is
    identical on every node, while the nodes run at different lags behind the sky (the AXIS
    INST line), so a wall-clock deadline would land on a different frame on each.

    Hops, not samples: `fe_hop_now` is the combiner's pow_hop (sample_seq / fft_len), and the
    node tests the deadline against its frame's first hop. The wire key `at_hop` names the
    unit.

    None when the axis is unknown, too old, or the deadline cannot be computed; the caller
    then posts without a deadline and each node applies at its next frame boundary. That
    fails open, because a slot stuck on a set satellite produces nothing, but every None path
    logs.
    """
    fh = getattr(ctx, "fe_hop_now", None)
    ft = getattr(ctx, "fe_hop_t", None)
    if fh is None or ft is None:
        _log_rl(
            "prnmap-noaxis",
            "PRN MAP %s: no F-engine axis this cycle -- a swap will post UNSCHEDULED, so "
            "the nodes will cross on different frames. Degraded, deliberately: an "
            "unsynchronised swap beats a slot stuck on a satellite that has set."
            % (log_tag() or a.signal),
            every_s=300.0,
        )
        return None
    try:
        hps = float(a.hops_per_sec)
        # Advance the sample to the cycle clock by its age, then add the lead. The status
        # poll stamps fe_hop_t on the frozen cycle clock, so this cycle's sample has age 0
        # and prn_reconfig_lead_s must cover the in-cycle delay to the POST; a positive age
        # means this cycle's status poll failed and the sample is left from an earlier one.
        # A deadline already past is applied at each node's next frame, indistinguishable
        # from a met one.
        age = float(now) - float(ft)
        if not (0.0 <= age <= float(a.prn_reconfig_axis_max_age_s)):
            _log_rl(
                "prnmap-axisage",
                "PRN MAP %s: the F-engine axis sample is %.1f s old (limit %.0f) -- "
                "extrapolating it that far would be a FABRICATED deadline, so this swap "
                "posts UNSCHEDULED. Check the status poll and pow_hop."
                % (log_tag() or a.signal, age, a.prn_reconfig_axis_max_age_s),
                every_s=300.0,
            )
            return None
        lead = float(a.prn_reconfig_lead_s)
        return int(round(float(fh) + (age + lead) * hps))
    except Exception as e:
        # Log it: a silent None would degrade every swap to unscheduled and look like a
        # healthy fleet with a quiet sky. A missing or unparseable setting is a configuration
        # fault, even though the swap still goes out.
        _log_rl(
            "prnmap-sched",
            "PRN MAP %s: cannot compute a swap deadline (%s) -- posting UNSCHEDULED. "
            "This is a configuration fault, not a sky condition."
            % (log_tag() or a.signal, e),
            every_s=300.0,
        )
        return None


def _apply_map(ctx, st, cur, want_map, moved, why, el, now):
    """POST the whole map to every producer with one deadline, then to the followers without
    it. One post moves any number of slots, so the nodes cross once and the map is never
    half-applied."""
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
            # The search gets the map but not the deadline: it applies a staged map between
            # passes, with no frame boundary to test a deadline against. It is not in the
            # record path, so an early swap there costs a re-scan, not a corrupted
            # accumulator.
            _post(
                "%s/set_prns" % ep, {"prns": want_map}, timeout=a.prn_reconfig_timeout_s
            )
            n_follow += 1
        except Exception as e:
            # A search whose list did not move hunts a satellite that has no slot and misses
            # one that has: worth logging, but no reason to leave the producer half-swapped.
            _log(
                "PRN MAP: follower %s refused the map (%s) -- it now searches a DIFFERENT "
                "set from the one the tracker holds." % (ep, e)
            )
    if ok == 0:
        st.err = bad
        _log("PRN MAP %s: %s REFUSED by every node (%s)" % (tag, why, bad))
        return
    st.swaps += 1
    st.applied = list(want_map)
    # The local map is not updated here: the next sweep reads it back, so the stage always
    # diffs against what the nodes hold, and a POST that is accepted but does not take stays
    # visible.
    st.maps.clear()  # a full re-sweep before the next decision
    st.poll_t = 0.0
    _log(
        "PRN MAP %s: %s -- posted to %d/%d node(s)%s%s, %s. Every moved slot acquires COLD: "
        "expect them dark for a minute or two. New satellites at el %s."
        % (
            tag,
            why,
            ok,
            len(_endpoints(ctx)),
            (" (%d failed: %s)" % (len(_endpoints(ctx)) - ok, bad)) if bad else "",
            (" + %d searcher(s)" % n_follow) if n_follow else "",
            ("all crossing together at hop %d" % at_hop)
            if at_hop is not None
            else "UNSCHEDULED (no axis) -- the nodes will cross on different frames",
            ", ".join("%+.0f" % el.get(want_map[i], -99.0) for i in moved[:6]),
        )
    )
