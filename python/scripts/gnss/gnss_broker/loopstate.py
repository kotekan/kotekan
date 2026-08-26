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
