"""Per-satellite state tables, grouped by the loop that owns them.

WHY GROUPED, AND WHY NOT ONE BAG. `main()` held roughly ninety long-lived tables as bare
locals -- `car_trim`, `wd_birth`, `nh_offset` and their neighbours -- with nothing saying which
loop owned which, or which had to move together. That is not merely untidy: it is why a stage
cannot leave `main()` (a bare local has no home to reference from another module), and it is
how the carrier loop was able to assign over the DLL's `fleet` dict for a whole cycle without
anything noticing.

Each class below is ONE loop's memory. The test for membership is not "is it about
satellites" -- everything here is -- but "would these be reset together if the loop were
restarted". A table that would survive its neighbours' reset belongs somewhere else.

⚠️ GIVE THE OWNER A NAME NOTHING ELSE USES. The first version of this called them `_car` and
`_wd` -- and `_car` is already a tuple element in the dead-reckon seeding loop, `_wd` already a
field of the seed-audit unpack. The second one is the worse case: it sits at `main()`'s own
level, so it REBOUND the owner every cycle, which is precisely the `fleet`-clobber class this
refactor exists to eliminate. Nothing broke only because no later line used it. The names are
`_carrier` and `_watchdog` now, and an owner name should be checked against the file before it
is chosen.

⚠️ THESE ARE PER-PRN DICTS, MUTATED IN PLACE. Nothing here should ever be REBOUND
(`self.trim = {}`); clear it instead. A rebind breaks every reference the context is holding,
silently, and the symptom appears in whichever stage reads it next rather than where it
happened.

@author Keith Vanderlinde
"""


class CarrierState(object):
    """The shared carrier loop's memory: per-PRN trim, lock, fade, and the bleed shadow.

    ⚠️ THE LOOP IS OFF IN PRODUCTION. `--carrier-gain` is 0.0 on CHORD, so every table here
    stays empty on the live instrument and no fixture reaches this code. That is a deliberate
    state -- the loop was measured to make tracking worse (#71) -- but it means nothing here
    is exercised by any gate. Treat changes as unverified until the loop is armed on sky.

    ⚠️ AND THAT IS NOT HARMLESS. Dead code is where the `fleet` clobber lived undisturbed:
    unreachable code cannot fail its way to your attention.
    """

    __slots__ = ("trim", "last", "locked", "fade", "step_hist", "step_t", "verify",
                 "bleed_hist", "bleed_log_t", "bleed_verify", "bleed_lock_t", "trim_force",
                 "repin_pending")

    def __init__(self):
        self.trim = {}          # prn -> commanded carrier trim, Hz
        self.last = {}          # prn -> last emitted value, for step detection
        self.locked = set()     # prns certified coherent since seed: BOOTSTRAP -> TRACK latch
        self.fade = {}          # prn -> consecutive faded emits (coast bookkeeping)
        self.step_hist = {}     # prn -> recent step sizes, for the gated/ungated agreement
        self.step_t = {}        # prn -> time of the last step
        self.verify = {}        # prn -> verification counters
        # f_ref TRIM-BLEED SHADOW: log-only. If the trim holds a STANDING value across the
        # stability window then f_ref is pinned somewhere it should not be, and the trim is
        # quietly absorbing a reference error rather than a satellite one.
        self.bleed_hist = {}
        self.bleed_log_t = {}
        self.bleed_verify = {}
        self.bleed_lock_t = {}
        # PRNs whose trim is forced by GNSS_TRIM_FORCE -- a diagnostic override, never a
        # control path.
        self.trim_force = {}
        # prn -> bleed amount (Hz) to carry as carrier_repin in the next seed
        self.repin_pending = {}


class WatchdogState(object):
    """How long each seeded satellite has gone without coherence.

    A satellite that is seeded but never locks would otherwise sit in the table forever,
    indistinguishable from one that is merely between passes. Probe PRNs are exempt: they are
    seeded deliberately so the combiner emits genuine noise records, and are never expected to
    lock.
    """

    __slots__ = ("birth", "coh_t", "strong_t", "weak_n")

    def __init__(self):
        self.birth = {}         # prn -> when this seed was born
        self.coh_t = {}         # prn -> last time coherence_s was above the floor
        self.strong_t = {}      # prn -> last time it was strong
        self.weak_n = {}        # prn -> consecutive weak polls


class NhOverlay(object):
    """Neumann-Hoffman overlay alignment: which overlay chip the sky is on.

    ⚠️ THE OVERLAY SIDEBANDS LAND ON THE SEARCH GRID while the true peak scallops between
    bins (#41). Comparing candidates on raw bin power therefore prefers a sideband; the
    comparison has to happen on the parabola VERTEX. The benches missed this for a while
    because their truth Doppler sat exactly on-grid, where the two agree.
    """

    __slots__ = ("last_rh", "off_hist", "offset", "seen")

    def __init__(self):
        self.last_rh = {}       # prn -> ref_hop already folded in, so a stale detection
                                #        cannot be re-counted and fake a full window of agreement
        # POOLED, not per-PRN: (t, predicted - reported) is ONE receiver-clock constant, and
        # it is timestamped so the offset can EXPIRE rather than harden.
        self.off_hist = []
        # A ONE-CELL LIST, not a scalar: the calibrated constant is read and rewritten from
        # several places, and the cell is what makes those the same object.
        self.offset = [None]
        self.seen = {}          # prn -> (nh, ref_hop) last REPORTED by the search


class DllLoopState(object):
    """The code delay-lock loop's memory, and the arming handshake with the C++ gather.

    ⚠️ AUTHORITY IS HELD BY EXACTLY ONE ARM AT A TIME. `armed_last` is what was last POSTED to
    the gather -- what the C++ loop is actuating RIGHT NOW -- not what this cycle is about to
    compute. The Python integrator stands down per-PRN against that set, so the trim is never
    driven by both and never by neither. It is recorded BEFORE the POST rather than after,
    because a failed POST must leave both sides believing the C++ side is still driving.

    ⚠️ `hold` IS PRESENCE WITH A MEMORY, not presence at an instant. A satellite flickering
    across the deep gate would otherwise be armed and released every cycle, and an arming
    change costs the standing trim.
    """

    __slots__ = ("trim", "last", "last_hop", "readback", "hold", "armed_last", "stat",
                 "deep_gate_seen", "reseed_prns")

    def __init__(self):
        self.trim = {}          # prn -> the Python integrator's standing trim, chips
        self.last = {}          # prn -> last discriminator seen (dedup: one integration/emit)
        self.last_hop = {}      # prn -> last window index integrated (an exact integer test)
        # prn -> the gather's reported standing trim. seed + THIS is where the tracker's tap
        # actually sits, which is the number every downstream judge of the seed was missing.
        self.readback = {}
        self.hold = {}          # prn -> last time PRESENT (the arming hold)
        self.armed_last = set() # what the C++ loop is actuating right now
        # ⚠️ THE KEY SET IS PART OF THE CONTRACT. `rb`/`rb_fail` count the READBACK poll
        # separately from the arming POST, and a consumer that finds them missing gets a
        # KeyError rather than a zero. Keep this dict exactly as its readers expect.
        self.stat = {"posts": 0, "fail": 0, "armed": 0, "last_err": "",
                     "rb": 0, "rb_fail": 0}
        self.deep_gate_seen = {}  # prn -> last time the SEARCH saw it above the deep bar (#79)
        self.reseed_prns = None   # the #50 armed set: None, True (all), or a set of PRNs


class HoldState(object):
    """Why a satellite is being held rather than dropped, and for how long.

    ⚠️ A HOLD IS NOT A LOCK. These tables decide whether to keep feeding a seed through a
    dropout; they say nothing about whether the tracker is actually on the peak. Judge that on
    q, never on hold state -- and never on sig/deep/cn0_coh, which duty-cycle with the fold.
    """

    __slots__ = ("miss", "prev", "q", "low_hits", "polls")

    def __init__(self):
        self.miss = {}          # prn -> consecutive polls with no usable record
        self.prev = {}          # prn -> the tracker's own last propagation (the slew anchor)
        self.q = {}             # prn -> last fleet q seen while held
        self.low_hits = {}      # prn -> consecutive below-threshold polls
        self.polls = {}         # prn -> coast poll count


class CpTracking(object):
    """Per-satellite code-phase history: fits, escapes, and the translated set.

    ⚠️ code_phase_chips IS AN ARGUMENT, NOT A TRANSPORTABLE QUANTITY. It is meaningful only
    against the epoch it was measured at (~5095 chips per Hz of Doppler), so it must never be
    transported or DIFFERENCED across epochs -- only `cp_at_ref` is comparable. Half the
    histories here exist to make that distinction checkable after the fact.
    """

    __slots__ = ("err_hist", "escape", "escape_sign", "fit_slope", "hist", "translated",
                 "dop_hist", "ph_hist", "dop_clamped")

    def __init__(self):
        self.err_hist = {}      # prn -> recent (predicted - observed) code phase
        self.escape = {}        # prn -> escape-detector state
        self.escape_sign = {}   # prn -> the sign it escaped in
        self.fit_slope = {}     # prn -> fitted cp rate, chips/s
        self.hist = {}          # prn -> recent (t, cp) for the rate fit
        self.translated = set() # prns whose cp has been translated this pass
        self.dop_hist = {}      # prn -> recent Doppler observations
        self.ph_hist = {}       # prn -> recent carrier-phase observations
        self.dop_clamped = set()  # prns whose Doppler hit a clamp this pass


class RateFeedState(object):
    """The carrier-rate observables fed to the joint receiver, and their commanded half.

    ⚠️ THE REFERENCE IS THE WHOLE POINT. `deep_rate` is measured on records the tracker has
    ALREADY derotated by the commanded trim, so what the search reports is only what REMAINS.
    The standing command (`cmd_applied`) is added back so the observable describes the SKY.
    Feed the residual raw and the estimator measures its own actuator -- that is #33 GAP 2's
    mirror, and it is invisible in the numbers because a mirror looks like a good fit.

    ⚠️ `full_ok` IS HOW THE COMMAND REFUSES A DEGRADED FEED. Past its cap, `deep_rate_hz`
    reports the best in-cap bin -- which is NOISE, and is what walked arm 1. The full-band
    field distinguishes them; without it the loop must not close.

    ⚠️ ONE RESIDUAL COMPUTATION, SHARED. `rate_residuals` consumes a window: asked twice in a
    cycle it reads "the same window again" and returns {} the second time. Consumers share one
    result; they must never each ask.

    The per-cycle fields (`resid`, `cons`, `resid2`, `full_ok`) are unconditionally reassigned
    at the top of every pass. They live here rather than as loop locals so that a stage which
    computes them can be a module-level function -- an attribute needs no `nonlocal`.
    """

    __slots__ = ("cmd_applied", "fine_t", "kcoh_t", "adr_span_now", "span_fed_t",
                 "adr_prev", "adr_ring", "resid", "cons", "resid2", "full_ok",
                 "railed", "released")

    def __init__(self):
        self.cmd_applied = {}   # prn -> the standing carrier command, Hz (the reference)
        self.fine_t = {}        # prn -> last fine-feed time
        self.kcoh_t = {}        # prn -> last known-rate-fold feed time
        # #93 shadow v2: prn -> (ADR span rate, t, n_rec), the DIRECT carrier observable the
        # rrate ROW is being judged against. The row measured ~97% carrier-only (GAP 3 F1).
        self.adr_span_now = {}
        self.span_fed_t = {}    # prn -> last span feed time (the non-overlap throttle)
        self.adr_prev = {}      # prn -> previous ADR sample
        self.adr_ring = {}      # prn -> recent ADR ring buffer
        # ---- per-cycle, reassigned every pass ----
        self.resid = {}         # prn -> carrier-rate residual (the shared computation)
        self.cons = None        # the consensus that came with it
        self.resid2 = {}        # the rrate-state variant of the same residual
        self.full_ok = False    # is the full-band rate field present this cycle?
        # How many commands hit the slew rail this pass, and how many were released. These
        # are `+=` counters, and an augmented assignment READS before it writes -- which is
        # what made them impossible to leave as loop locals once the stage that increments
        # them became a function. That cost gal_e5a a chain death on 2026-08-26.
        self.railed = 0
        self.released = 0


class NavDecoders(object):
    """The broadcast navigation-message decoders, created lazily and kept across cycles.

    Eleven objects, one per message format, plus the BRDC cross-check source and the
    agreement monitor. Each is built on the first row that needs it and must then PERSIST --
    a decoder that is rebuilt has forgotten the subframes it had collected, so it can never
    complete an ephemeris. Holding them as attributes is what let the nav-bit stage leave
    `main()`: eleven `nonlocal` declarations cannot cross a module boundary.

    ⚠️ OFF IN PRODUCTION. `--nav-bits` is not set in the CHORD yaml, so none of this runs on
    the live instrument and no fixture reaches it. Changes here are unverified until the flag
    is armed -- the same blind spot the carrier loop has.
    """

    __slots__ = ("navbits", "cnav", "cnav2", "fnav", "inav",
                 "bcnav1", "bcnav2", "bcnav3", "brdc", "health", "log_t",
                 "inav_log_t", "fnav_log_t", "bcnav1_log_t", "bcnav2_log_t",
                 "bcnav3_log_t", "cnav2_log_t", "cnav_sig",
                 "cnav_combiner", "cnav2_combiner", "fnav_combiner", "inav_combiner",
                 "bcnav1_combiner", "bcnav2_combiner")

    def __init__(self):
        self.navbits = None   # GPS LNAV
        self.cnav = None      # GPS CNAV (L2C-CM / L5-I)
        self.cnav2 = None     # GPS CNAV-2
        self.fnav = None      # Galileo F/NAV
        self.inav = None      # Galileo I/NAV
        self.bcnav1 = None    # BeiDou B-CNAV1
        self.bcnav2 = None    # BeiDou B-CNAV2
        self.bcnav3 = None    # BeiDou B-CNAV3
        self.brdc = None      # the BRDC source the decodes are cross-checked against
        self.health = None    # continuous predicted-vs-air agreement monitor
        self.log_t = 0.0      # last summary-line time (rate limit)
        # ⚠️ ONE-CELL LISTS, not floats. Each is a rate-limit stamp mutated in place by the
        # decoder that owns it; the cell is what makes the reader and the writer the same
        # object. Flattening them to scalars would silently give every writer its own copy.
        self.inav_log_t = [0.0]
        self.fnav_log_t = [0.0]
        self.bcnav1_log_t = [0.0]
        self.bcnav2_log_t = [0.0]
        self.bcnav3_log_t = [0.0]
        self.cnav2_log_t = [0.0]
        # Which CNAV variant this chain's carrier implies (L2C vs L5), resolved once.
        self.cnav_sig = None
        # Per-decoder combiner endpoints: a nav decode reads a DIFFERENT stream from the
        # tracking chain, so these are separate endpoints and may each be absent.
        self.cnav_combiner = None
        self.cnav2_combiner = None
        self.fnav_combiner = None
        self.inav_combiner = None
        self.bcnav1_combiner = None
        self.bcnav2_combiner = None
