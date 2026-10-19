"""Per-satellite state tables, grouped by the loop that owns them.

Each class is one loop's memory: the tables that would be reset together if that loop were
restarted. main() builds one owner per class and hands it to the stages through the context
(ctx.dls, ctx.cpt, ...), so a stage in another module reaches a table through its owner.
An owner's name in main() must be used for nothing else there: rebinding it, even as a loop
variable or an unpack target, silently replaces the owner for every later line.

The per-PRN dicts and sets are mutated in place, never rebound: clear() rather than
`self.trim = {}`, so a table aliased anywhere (a local, a helper's argument) is never left
reading a dead copy. RateFeedState's per-cycle fields are the exception and say so.

@author Keith Vanderlinde
"""


class NhOverlay(object):
    """Neumann-Hoffman overlay alignment: which overlay chip the sky is on.

    `seen` holds each PRN's nh as the C++ search reported it. The search compares overlay
    candidates on the parabola vertex: an overlay-mismatch sideband lands on a Doppler grid
    bin while the true peak scallops between bins, so raw bin power would prefer the sideband.
    searchhint.py (--nh-hint) learns one receiver offset from these reports.
    """

    __slots__ = ("last_rh", "off_hist", "offset", "seen")

    def __init__(self):
        # prn -> the ref_hop last folded into off_hist: an unchanged detection is not counted
        # again, so nh_hint_min_samples counts evidence rather than uptime
        self.last_rh = {}
        # Pooled over PRNs: (t, (predicted - reported nh) mod nh_overlay_len), t the detection's
        # ref_hop as UTC s on the hop axis. The offset is one receiver-clock constant; the
        # timestamp lets a sample expire (nh_hint_max_age_s).
        self.off_hist = []
        # [offset in overlay chips]: the circular median of the fresh samples, None while
        # fewer than nh_hint_min_samples are fresh
        self.offset = [None]
        self.seen = {}  # prn -> (nh, ref_hop) last reported by the search


class DllLoopState(object):
    """The code delay-lock loop's memory, and the arming handshake with the C++ gather.

    One arm drives each PRN's trim. `armed_last` is the armed set last sent to the gather,
    whose C++ loop actuates it; the Python integrator (codeloop.py) stands down for those
    PRNs. It is recorded before the POST, so after a failed POST a PRN armed since the last
    successful one is driven by neither arm, and one released since then by both, until a POST
    succeeds (trimarm.stage_fleet_trim_arming).

    `hold` is presence with a memory: a PRN stays armed for fleet_trim_hold_s after it was
    last present. A released trim leaks away, so a satellite flickering across the presence
    gate would otherwise lose its standing trim every cycle.
    """

    __slots__ = (
        "trim",
        "last",
        "last_hop",
        "readback",
        "hold",
        "armed_last",
        "stat",
        "deep_gate_seen",
    )

    def __init__(self):
        self.trim = {}  # prn -> the Python integrator's standing trim, chips
        self.last = {}  # prn -> last discriminator seen (dedup: one integration/emit)
        self.last_hop = (
            {}
        )  # prn -> hop of the last window integrated (an exact integer test)
        # prn -> the gather's /get_dll row (trim_chips, armed). The tracker despreads at
        # seed + trim_chips; cleared when a read fails, so missing means unknown.
        self.readback = {}
        self.hold = {}  # prn -> cycle clock (ctx.t0) last present (the arming hold)
        self.armed_last = set()  # the armed set last sent to the gather
        # arming POSTs made / failed, and the last POST or readback error (the FLEET-TRIM line)
        self.stat = {"posts": 0, "fail": 0, "last_err": ""}
        self.deep_gate_seen = (
            {}
        )  # prn -> t0 of the last search detection at snr >= --dll-deep-gate-from-search


class HoldState(object):
    """Lock evidence carried between cycles, and the counters that hold or coast a seed.

    A hold is not a lock: `miss` and `low_hits` decide whether a seed is kept through a
    dropout, not whether the tracker is on the peak. Judge lock on q, never on hold state or
    on sig/deep/cn0_coh, which duty-cycle with the fold.
    """

    __slots__ = ("miss", "prev", "q", "low_hits")

    def __init__(self):
        self.miss = {}  # prn -> consecutive sub-gate reads of a held PRN (held below 3)
        self.prev = {}  # prn -> last cycle's fleet prompt ratio p_pow/p_med
        self.q = {}  # prn -> last cycle's fleet q
        self.low_hits = {}  # prn -> below-lock polls in a row (dropped at coast_polls)


class CpTracking(object):
    """Per-satellite code-phase history: fits, escapes, and the translated set.

    code_phase_chips is an argument, not a transportable quantity: it means something only
    against the epoch it was measured at (~5095 chips per Hz of Doppler), so it is never
    transported or differenced across epochs; only the at-ref phase is comparable.
    """

    __slots__ = (
        "err_hist",
        "escape",
        "escape_sign",
        "hist",
        "translated",
        "dop_hist",
        "ph_hist",
        "dop_clamped",
        "nh_pending",
        "rej_streak",
        "nh_votes",
        "nh_common",
    )

    def __init__(self):
        # prn -> a held PRN's last 9 innovations (detection - forecast - trim)
        self.err_hist = {}
        self.escape = {}  # prn -> consecutive same-sign escapes (re-anchor at 5)
        self.escape_sign = {}  # prn -> the last escape's cp_err (its sign)
        self.hist = {}  # prn -> recent (ref_hop, cp, dop) for the rate fit
        self.translated = set()  # prns whose TRANSLATE line has been logged
        self.dop_hist = {}  # prn -> recent (ref_hop, dop) for the Doppler-rate fit
        self.ph_hist = {}  # prn -> (ref_hop, ph, dop) for the period-continuity check
        self.dop_clamped = set()  # prns whose DOP-CLAMP line has been logged
        self.nh_pending = {}  # prn -> (m, count): a measured overlay period awaiting
        # --nh-period-debounce confirmation before it replaces the standing one
        self.nh_votes = (
            {}
        )  # prn -> (wall time, delta_chips, weight): each detection's measured
        # receiver-common overlay offset (chips mod the overlay epoch). The
        # offset is one receiver clock mod the overlay epoch that every
        # satellite votes on, not a per-satellite guess (--nh-joint)
        self.nh_common = (
            None  # (delta_chips, t_resolved, n_prns, weight_frac) once the votes
        )
        # resolve; kept until a newer resolution, never expired
        self.rej_streak = {}  # prn -> consecutive cp-rate rejections; at
        # --fit-flush-on-reject the wrap-poisoned history is flushed


class RateFeedState(object):
    """The carrier-rate observables fed to the joint receiver, and their commanded half.

    `cmd_applied` is the standing carrier command (Hz) shipped last poll, the reference the
    coarse and KCOH feeds add back to each measured residual under --rrate-feed-applied 1.
    That is right only for a tracker whose records still carry the actuation; a record
    assembler that folds the applied trim out already reports the whole residual, and adding
    the command back makes each row integrate its own actuation (see that flag).

    `full_ok` is how the command refuses a degraded feed: past its cap `deep_rate_hz` reports
    the best in-cap bin, which is noise, so no command is computed in a poll where no row
    carries the uncapped deep_rate_full_* fields.

    One residual computation per poll: `rate_residuals` advances its per-PRN continuity state,
    so asked twice in a cycle it reads the same window again and returns {} the second time.

    The per-cycle fields (`resid2`, `full_ok`) are reassigned every pass. They live here rather
    than as loop locals so the stages that read them can be module-level functions.
    """

    __slots__ = (
        "cmd_applied",
        "fine_t",
        "kcoh_t",
        "span_fed_t",
        "adr_prev",
        "adr_ring",
        "resid2",
        "full_ok",
        "railed",
        "released",
    )

    def __init__(self):
        self.cmd_applied = {}  # prn -> the standing carrier command, Hz (the reference)
        self.fine_t = {}  # prn -> last fine-feed time
        self.kcoh_t = {}  # prn -> last known-rate-fold feed time
        self.span_fed_t = {}  # prn -> last span feed time (the non-overlap throttle)
        self.adr_prev = {}  # prn -> previous ADR sample
        self.adr_ring = {}  # prn -> recent ADR ring buffer
        # ---- per-cycle, reassigned every pass ----
        self.resid2 = {}  # prn -> carrier-rate residual on the uncapped fields
        self.full_ok = False  # is the full-band rate field present this cycle?
        # How many commands hit the slew rail this pass, and how many were released (the
        # JRR-CMD line). `+=` counters incremented inside stage_push_seeds, so they cannot be
        # loop locals; the broker zeroes them before each call.
        self.railed = 0
        self.released = 0


class NavDecoders(object):
    """The broadcast navigation-message decoders, created lazily and kept across cycles.

    Eight decoders, one per message format. Each is built on the first row that needs it and
    must then persist: a rebuilt decoder has forgotten the subframes it had collected, so it
    can never complete an ephemeris.

    Idle on CHORD: --nav-bits defaults on but self-gates on nav_obs, which no CHORD combiner
    exports (bit_export), and no aux combiner (--cnav-combiner and the rest) is configured.
    Only test_navbits.py reaches the nav-bits stage, with fake decoders.
    """

    __slots__ = (
        "navbits",
        "cnav",
        "cnav2",
        "fnav",
        "inav",
        "bcnav1",
        "bcnav2",
        "bcnav3",
        "log_t",
        "cnav_log_t",
        "inav_log_t",
        "fnav_log_t",
        "bcnav1_log_t",
        "bcnav2_log_t",
        "bcnav3_log_t",
        "cnav2_log_t",
        "cnav_sig",
        "cnav_combiner",
        "cnav2_combiner",
        "fnav_combiner",
        "inav_combiner",
        "bcnav1_combiner",
        "bcnav2_combiner",
    )

    def __init__(self):
        self.navbits = None  # GPS LNAV
        self.cnav = None  # GPS CNAV (L2C-CM / L5-I)
        self.cnav2 = None  # GPS CNAV-2
        self.fnav = None  # Galileo F/NAV
        self.inav = None  # Galileo I/NAV
        self.bcnav1 = None  # BeiDou B-CNAV1
        self.bcnav2 = None  # BeiDou B-CNAV2
        self.bcnav3 = None  # BeiDou B-CNAV3
        self.log_t = 0.0  # last LNAV health-line time (rate limit)
        # One-cell lists [t]: each decoder's 60 s health-line stamp, written as x_log_t[0].
        self.cnav_log_t = [0.0]
        self.inav_log_t = [0.0]
        self.fnav_log_t = [0.0]
        self.bcnav1_log_t = [0.0]
        self.bcnav2_log_t = [0.0]
        self.bcnav3_log_t = [0.0]
        self.cnav2_log_t = [0.0]
        # Which CNAV variant this chain's carrier implies (L2C vs L5), resolved once.
        self.cnav_sig = None
        # Aux combiner endpoints (--cnav-combiner etc.), each None unless set: the data
        # symbols come from a sibling chain, not from this broker's own pilot combiner.
        self.cnav_combiner = None
        self.cnav2_combiner = None
        self.fnav_combiner = None
        self.inav_combiner = None
        self.bcnav1_combiner = None
        self.bcnav2_combiner = None
