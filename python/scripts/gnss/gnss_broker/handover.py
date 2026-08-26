"""#92: handing a standing C++ trim over to a re-based seed, in the same cycle.

THE DISEASE (E3's ~25-minute sawtooth). The broker's seed and the gather's standing trim are
two halves of one number: the tracker despreads at (seed + trim). When the dead-reckon seed
re-bases -- a birth, a slew hand-back, an axis re-anchor -- the seed jumps by `step` chips
while the C++ trim goes on carrying the SAME chips it accumulated against the old basis. The
tap moves by `step`, the satellite walks off the peak, q craters, and the loop rebuilds the
trim from scratch over the next ~25 minutes. Then it happens again.

THE HANDOVER. Post the compensating `-step` to the gather's `/adjust_trim` in the SAME cycle,
so the sum never leaves the sky. This is a transfer between two ledgers, not a new correction:
nothing about where the signal is has changed.

⚠️ THE BOUND IS THE WHOLE SAFETY ARGUMENT. A step the trim could not possibly have been
carrying is not a handover -- the shared-clock births move by HUNDREDS of chips, and posting
that would slam the C++ clamp (3.0) and destroy a good trim. Beyond `bound_chips` we skip
LOUDLY and let the trim rebuild the old way: one sawtooth, which is merely the old steady
state, versus a corrupted trim, which is new damage. Refuse the unknown, keep the known cost.

⚠️ ONLY FOR PRNs THE FLEET LOOP IS ACTUATING. A PRN with no standing trim has nothing to hand
over. Posting for the seeding-churn re-births (measured live 2026-08-25 18:53: PRNs 6/30
re-basing every ~10 s, refused every time) is spam on both logs, and it BURIES the one refusal
that would mean something: an armed satellite whose adjustment missed.

⚠️ A MISSED HANDOVER MUST NEVER TAKE SEEDING DOWN. Every failure path here is caught and
logged; the fallback is the pre-#92 behaviour.

⚠️ `adjcum` IS NOT BOOKKEEPING -- IT IS AN INSTRUMENT CORRECTION. #93's shadow estimates a
per-satellite drift from the standing trim's ramp. A handover moves that trim for reasons that
have nothing to do with drift, so the shadow subtracts the cumulative posted deltas. Without
it the handover masquerades as exactly the signal #93 is trying to measure.

STATUS 2026-08-26: mechanism PROVEN (hundreds of posts, zero failures, refusals gated away),
P2 (does it actually kill the sawtooth?) HOLD-OPEN -- no clean drift-then-rebase episode has
been captured yet; every >=0.3-chip event so far was restart or flicker churn.

REMAINDER OF #77: the birth site below is one of SEVERAL re-anchor points. The others still
wipe-and-rebuild. Wiring them here is a behaviour change and belongs in its own commit with
its own falsifier.

@author Keith Vanderlinde
"""


class TrimHandover:
    """Posts compensating trim adjustments to the gather, and remembers what it posted.

    The transport and the logger are injected rather than imported: this object is exercised
    in tests with neither, and the broker's `_post`/`_log` carry cycle context it must not
    duplicate.
    """

    def __init__(self, enabled=False, bound_chips=2.5, timeout_s=2.0):
        self.enabled = bool(enabled)
        self.bound_chips = float(bound_chips)
        self.timeout_s = float(timeout_s)
        # prn -> cumulative chips posted to the gather. Consumed by the #93 shadow.
        self.adjcum = {}
        self.posted = 0
        self.skipped = 0
        self.failed = 0

    def offer(self, prn, step_chips, armed, chain, url, post, log):
        """Offer a seed re-base step for handover. Returns True if a delta was posted.

        `step_chips` is how far the SEED is about to move; the trim adjustment posted is its
        negation. `armed` is whether the fleet loop is actuating this PRN. `post(url, body,
        timeout)` and `log(msg)` are the broker's own.
        """
        if not (self.enabled and url and armed):
            return False

        if abs(step_chips) > self.bound_chips:
            self.skipped += 1
            log("REBASE-ADJUST PRN %d skipped: step %+.3f beyond the %.1f-chip handover bound"
                % (prn, step_chips, self.bound_chips))
            return False

        try:
            # `post` returns the HTTP status; the per-PRN adjusted/refused echo is in the
            # GATHER's log and its get_stats, because only the gather knows which PRNs it
            # currently holds armed.
            rep = post("%s/adjust_trim" % url.rstrip("/"),
                       {"chains": {chain: {str(prn): -step_chips}}},
                       timeout=self.timeout_s)
            self.posted += 1
            self.adjcum[prn] = self.adjcum.get(prn, 0.0) - step_chips
            log("REBASE-ADJUST PRN %d: trim %+.3f posted to the gather (HTTP %s)"
                % (prn, -step_chips, rep))
            return True
        except Exception as e:
            self.failed += 1
            log("REBASE-ADJUST PRN %d FAILED (%s) -- trim rebuilds the old way" % (prn, e))
            return False

    def corrected(self, prn, trim_chips):
        """The standing trim with this PRN's handovers removed -- what #93's ramp fit must
        see, so a ledger transfer is never read as satellite drift."""
        return trim_chips - self.adjcum.get(prn, 0.0)
