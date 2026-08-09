"""The Receiver: what every chain on this telescope shares (task #27 M3).

WHAT BELONGS HERE is decided by physics, not by convenience. CHORD is ONE receiver: one
F-engine digitising the whole band off one GPS-disciplined reference. GPS L5 and Galileo
E5a are not two receivers -- both launch scripts pass `--carrier-hz 1176.45e6`, because
they are the same 24 MHz of sky through the same feed, the same cable and the same clock.
Anything that is a property of the instrument rather than of a signal lives here and is
estimated ONCE:

    time anchor   utc0_sample0 -- the F-engine's frame 0. One number for the telescope.
    sky           the BRDC store. Multi-system by construction (keys are (sys, prn)) and
                  filtered per chain at predict time, so ONE parse serves every
                  constellation -- today each broker parses its own copy and throws the
                  other systems away.
    carrier clock RECEIVER scope. One reference, one frequency error.
    code clock    PER BAND. Cable and PFB group delay are per carrier, so a code bias
                  measured at 1176.45 MHz is not the one at 1207 MHz. This is exactly what
                  `--state-dongle l5` asserts by hand today, and it is load-bearing rather
                  than decoration.

CONTRIBUTE / CONSUME, NEVER OVERWRITE. A chain that has solved a quantity itself keeps its
own estimate and merely PUBLISHES it here; a chain that has not solved it (E5a runs
`--detectors` empty by design and can never solve a clock) CONSUMES a sibling's. That is
already the shape of `--dr-clock-adopt` and `--state-consume`; this class only replaces
their transport -- a JSON file written at flush cadence and re-read with a staleness gate
-- with the object the two chains could have shared all along.

⚠️ IT FOLLOWS THAT A SINGLE CHAIN IS UNAFFECTED, by construction: it always has its own
estimate, so `consume` never returns anything it did not itself contribute. That is what
keeps the M3 equivalence gate green, and it is a property of the contract, not a
coincidence to be re-checked at every call site.

WHAT IS DELIBERATELY NOT HERE: anything per-signal. Seeds, per-PRN track state, the DLL and
carrier loops, the CL machinery, nav decode -- all of that is the chain's, and the chain is
still `main()`'s closure (see docs/CHORD_BROKER_REFACTOR.md, the note above M3).
"""
import threading

from .transport import _log


class _Shared(object):
    """One contributed quantity: who said it, what, how well, and when."""

    __slots__ = ("value", "src", "weight", "t", "extra")

    def __init__(self, value, src, weight, t, extra=None):
        self.value, self.src, self.weight, self.t = value, src, weight, t
        self.extra = extra or {}


class Receiver(object):
    """Instrument-scope state, shared by every chain in the process."""

    def __init__(self, log=None):
        self._lock = threading.RLock()
        self._log = log or _log
        self._anchor = None            # utc0_sample0, latched once for the telescope
        self._anchor_src = None
        self._brdc = {}                # cache key -> the {"mod","eph","eph_t"} state dict
        self._carrier = {}             # chain -> _Shared (Hz, receiver scope)
        self._code = {}                # (band, chain) -> _Shared (dimensionless l-a)
        self._dr = {}                  # (band, chain) -> _Shared (chips, + drift in extra)
        self._joint = {}               # band -> JointReceiverState (P2a, shadow)

    # -- the joint receiver state (task #33 P2a) ------------------------------------------
    def joint(self, band, **kw):
        """The ONE [clk, clk_rate, b_sat[i]] solve for this band, shared by every chain.

        Receiver scope by the same physics as dr_clock: one F-engine, one reference, and a
        code clock that is per BAND because cable and PFB group delay do not survive a
        retune. On CHORD all three deployed chains (GPS L5, Galileo E5a, BeiDou B2a) sit at
        1176.45 MHz, so all three feed one state -- which is the point: the gauge's 1/N leak
        shrinks and the clock stops being one constellation's private median.

        Created on first use so a process that never enables the shadow never imports numpy.
        """
        from .state_filter import JointReceiverState
        with self._lock:
            if band not in self._joint:
                self._joint[band] = JointReceiverState(**kw)
            return self._joint[band]

    # -- time anchor --------------------------------------------------------------------
    def time_anchor(self, fetch, chain):
        """The F-engine's frame-0 UTC, fetched at most ONCE per process.

        `fetch` is called only if nobody has it yet, and may raise or return a falsy value
        (the endpoint being down is the normal outage case) -- in which case nothing is
        latched and the next chain tries again.

        ⚠️ Deliberately does NOT re-anchor. A changed frame 0 also invalidates the epoch
        every NODE cached at ITS startup, so the correct response is an operator restarting
        the fleet; silently re-anchoring here would fix this process's arithmetic and leave
        the nodes wrong. The loop's periodic re-read exists to SAY SO, loudly, not to heal.
        """
        with self._lock:
            if self._anchor:
                return self._anchor
        v = fetch()                     # outside the lock: this is a network call
        if not v:
            return None
        with self._lock:
            if not self._anchor:
                self._anchor, self._anchor_src = float(v), chain
                return self._anchor
            if abs(self._anchor - float(v)) > 1e-3:
                # Two chains disagreeing about frame 0 means the F-engine restarted between
                # their fetches. Neither is trustworthy; say which is in use and by whom.
                self._log("*** TIME ANCHOR DISAGREEMENT: %s has %.9f, %s fetched %.9f "
                          "(%+.3f days). The F-engine restarted mid-startup. Restart the "
                          "nodes AND this broker."
                          % (self._anchor_src, self._anchor, chain, float(v),
                             (float(v) - self._anchor) / 86400.0))
            return self._anchor

    def anchor(self):
        with self._lock:
            return self._anchor

    # -- sky ----------------------------------------------------------------------------
    def brdc(self, key, build):
        """The shared BRDC store for `key`, built once.

        The parse is multi-system -- entries are keyed (sys, prn) and every consumer filters
        at predict time -- so ONE store serves GPS, Galileo and BeiDou together. Today each
        broker parses its own copy of the same file and discards the other constellations.
        """
        with self._lock:
            if key in self._brdc:
                return self._brdc[key]
        st = build()                    # outside the lock: fetch + parse
        with self._lock:
            return self._brdc.setdefault(key, st)

    # -- clocks -------------------------------------------------------------------------
    def contribute_carrier_bias(self, chain, hz, n_sats, t):
        """Publish this chain's clock-frequency bias. Receiver scope: one reference."""
        if hz is None:
            return
        with self._lock:
            self._carrier[chain] = _Shared(float(hz), chain, int(n_sats or 0), t)

    def carrier_bias(self, exclude=None, max_age_s=120.0, t_now=None):
        """The best OTHER chain's carrier bias, or None. Weighted by satellite count --
        the only quality signal any of these estimators carries (receiver_state.py's
        standing complaint) -- and aged out, because a chain that stopped solving must
        stop being believed rather than fading silently."""
        return self._best(self._carrier, exclude, max_age_s, t_now, key2=None)

    def contribute_code_bias(self, chain, band, la, n_sats, t):
        """Publish the code-side (l-a) bias. PER BAND: group delay is per carrier."""
        if la is None:
            return
        with self._lock:
            self._code[(band, chain)] = _Shared(float(la), chain, int(n_sats or 0), t)

    def code_bias(self, band, exclude=None, max_age_s=120.0, t_now=None):
        return self._best(self._code, exclude, max_age_s, t_now, key2=band)

    def code_bias_any_band(self, exclude=None, max_age_s=120.0, t_now=None):
        """(l-a) from ANY band -- the clock RATE is receiver-wide even though the PHASE is not.

        ⚠️ READ THIS BEFORE USING IT FOR ANYTHING ELSE. `contribute_code_bias` is keyed per band
        because group delay is per carrier, and that is right for the code PHASE: a signal at
        1207.14 MHz arrives through a different filter chain than one at 1176.45 and sits at a
        different delay (`tau_band`). But the value stored here is a FRACTIONAL FREQUENCY --
        dimensionless, consumed as `code_bias * chip_hz / hops_per_sec` -- and tau_band is a
        CONSTANT offset whose time derivative is zero. A per-band delay therefore contributes
        NOTHING to the rate, so the rate is a property of the receiver, not of the band.

        WHY IT HAD TO EXIST (task #34, 2026-08-09). The per-band gate is a bootstrap trap for
        any band that cannot solve its own clock. Measured on sky: gal_e5a and bds_b2a adopted
        the clock 102 and 107 times from gps_l5 on the same 1176.45 MHz, while gal_e5b and
        bds_b2b -- the first chains at 1207.14 -- adopted it ZERO times, ever, and sat at the
        startup "PRIMED 0.00 chips" value. That left code_phase_rate = 0.0 on all 19 of their
        PRNs, so the fleet DLL logged "no live code_phase_rate, holding trim" and never closed,
        the code phase walked open-loop at ~0.003-0.01 chips/s (the #30 rate), the instances
        drifted apart from each other (fleet_coh_align 0.16 against 0.51 on the working bands),
        the fleet-coherent combine never cleared its floor (fleet_present 0/19), and every
        published number fell back to a single instance. The loop closes on itself: no clock ->
        no rate -> no lock -> no detections -> no measured (l-a) -> no clock.

        It is also self-defeating: `tau_band` is unobservable until the second band tracks, and
        the second band could not track because nothing supplied a band-offset clock.

        THE PHASE IS STILL PER BAND and this does not change that -- `dr_clock` keeps its band
        key, and adopting a code PHASE across carriers would inject exactly the tau_band offset
        we are trying to measure. This is the rate only.
        """
        return self._best(self._code, exclude, max_age_s, t_now, key2=None)

    def contribute_dr_clock(self, chain, band, chips, drift, t, code_length):
        """Publish the dead-reckon receiver clock (chips, mod the code period) + drift.

        ⚠️ THIS IS THE SEAM `--dr-clock-adopt` PAPERS OVER. `dr_state` straddles the
        receiver/chain boundary: clk and drift are the RECEIVER's, while seeded/pin/pd are
        the chain's per-PRN bookkeeping. Splitting them is the single highest-value
        structural change in this refactor -- it is why a detector-less chain needed a JSON
        file, a flush cadence and a two-read slew gate to learn a number the process next
        door already had.

        Carried WITH its code length: chips are modular, and a value mod 10230 means nothing
        to a chain whose code is 1023000 long. A consumer that ignores this would adopt a
        number that is numerically fine and physically wrong -- silently.
        """
        if chips is None:
            return
        with self._lock:
            self._dr[(band, chain)] = _Shared(float(chips), chain, 1, t,
                                              {"drift": drift, "code_length": code_length})

    def dr_clock(self, band, exclude=None, max_age_s=120.0, t_now=None):
        return self._best(self._dr, exclude, max_age_s, t_now, key2=band)

    # -- internals ----------------------------------------------------------------------
    def _best(self, store, exclude, max_age_s, t_now, key2):
        with self._lock:
            best = None
            for k, s in store.items():
                chain = k[1] if isinstance(k, tuple) else k
                if chain == exclude:
                    continue
                if key2 is not None and k[0] != key2:
                    continue
                if t_now is not None and max_age_s and (t_now - s.t) > max_age_s:
                    continue
                if best is None or s.weight > best.weight:
                    best = s
            return best

    def summary(self):
        """One line for the periodic log -- what the instrument currently believes."""
        with self._lock:
            return ("receiver: anchor=%s (%s), brdc=%d store(s), carrier from %s, "
                    "code from %s, dr from %s"
                    % ("%.9f" % self._anchor if self._anchor else "none",
                       self._anchor_src or "-", len(self._brdc),
                       sorted(self._carrier) or "-",
                       sorted(c for _, c in self._code) or "-",
                       sorted(c for _, c in self._dr) or "-"))
