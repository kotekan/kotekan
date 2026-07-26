#!/usr/bin/env python3
"""Does a nav-bit source's output actually MATCH THE AIR, per satellite, right now?

Every nav-bit source predicts signs that get handed to a SUBTRACTER. A source that is merely
wrong produces confidently wrong bits, the peel subtracts at the wrong sign on ~40% of records,
and the band gets DIRTIER -- the exact opposite of the point. Measured 2026-07-26, that failure
mode is not hypothetical: GPS went HURTING 0-1 -> 7-8 in one deploy.

Three separate ad-hoc gates were added the same day to catch three instances of it:
  * the combiner's dead-reckon anchor never expired -> published -180-chip tables at deep~0
  * select_eph resolved an ambiguous ephemeris by guessing -> G15 encoded from IODE 102 while
    the satellite broadcast IODE 41 (65.6% agreement, vs 93.7-100% on every matched satellite)
  * the peel's gain-SNR gate defaulted to INFINITY when it could not evaluate itself
Each was a different symptom of one question nobody was asking continuously: DO THE BITS WE
ARE ABOUT TO PUBLISH AGREE WITH THE ONES COMING OUT OF THE SKY? This asks it.

    h = BitAgreement()
    h.score(prn, row["nav_obs"], deep_snr)   # scores against what we PUBLISHED last cycle
    if h.veto(prn): drop this PRN's bits
    h.remember(prn, nb)                      # ...and remember what we are publishing now

SCORE AGAINST WHAT WAS PUBLISHED, NOT A FRESH PREDICTION. A table is generated starting at
"now" and running forward, so it does not cover observations that arrived a second ago.
Keeping the last table and scoring the next cycle's observations against it also asks the
RIGHT question -- did the bits we actually handed the subtracter match the sky? -- rather than
re-deriving a prediction that might differ from the one that shipped.

WHAT IT CAN AND CANNOT SEE. Scoring needs observed bits good enough that a disagreement means
OUR bits are wrong rather than the air reading being wrong, so only satellites above
MIN_SNR are scored. That is not the circular limitation it first looks like:
  * COMMON-MODE faults (bad clock offset, wrong week, a convention mismatch) hit every
    satellite, so they show up on the strong ones and condemn the whole source -- which is
    exactly the class that took down the GPS chain twice.
  * PER-SATELLITE faults (a wrong ephemeris record) show only on that satellite, and are
    caught wherever it is observable. G15 was strong, decoded, and still wrong.
  * A weak satellite that is unobservable is left to the elevation mask, the ephemeris
    ambiguity gate and the peel's own (now fail-closed) gain-SNR gate.

⚠️ POLARITY IS ARBITRARY PER EMIT. nav_obs signs come from a per-emit carrier rotation, so a
naive concatenation averages the polarities to noise and returns a confident ~50% for EVERY
satellite -- including ones that are provably 100% correct. That mistake was made twice on
2026-07-26 before being caught by satellites with known-good scores reading 56%. Each emit is
therefore scored at its OWN better polarity, which is what stitch() does for the decoder.
"""

MIN_SNR = 60.0       # below this the AIR is the unreliable party, not the prediction
MIN_BITS = 400       # samples before a verdict is allowed to condemn anything
BAD_RATE = 0.85      # sustained agreement below this = the source is wrong for this satellite
GOOD_RATE = 0.92     # ...and this much is needed to come back (hysteresis, not a knife edge)
ALPHA = 0.05         # EMA on the per-emit agreement rate


class BitAgreement:
    """Per-PRN agreement between predicted nav bits and the bits actually observed."""

    def __init__(self, log=None):
        self._log = log or (lambda m: None)
        self.rate = {}      # prn -> EMA agreement in [0,1]
        self.n = {}         # prn -> bits compared (cumulative)
        self.bad = set()    # prns currently vetoed
        self.n_veto = 0     # vetoes applied since the last report
        self._last = {}     # prn -> the table we last PUBLISHED (what to score against)

    def remember(self, prn, nb):
        """Stash the table being published for this PRN, to score next cycle's observations."""
        if nb and nb.get("bits"):
            self._last[prn] = nb

    # ---- ingest ------------------------------------------------------------------------
    def score(self, prn, nav_obs, deep_snr):
        """Fold one emit's observed bits into this PRN's agreement EMA, against the table we
        last PUBLISHED. No-op when the satellite is too weak for a disagreement to mean
        anything (see MIN_SNR), or when we have published nothing yet."""
        nb = self._last.get(prn)
        if not nb or not nav_obs or (deep_snr or 0.0) < MIN_SNR:
            return
        bits = nav_obs.get("bits")
        if not bits:
            return
        try:
            rec_dt = float(nav_obs["rec_dt"])
            phase = int(nav_obs["phase"])
            br = int(nav_obs["br"])
            utc_ref = float(nav_obs["utc_ref"])
        except (KeyError, TypeError, ValueError):
            return
        t0, bs, tab = nb["utc0"], nb["bit_s"], nb["bits"]
        n = agree = 0
        for bi, sgn in bits:
            t = utc_ref + (int(bi) * br - phase) * rec_dt
            # ROUND, not floor. nav_obs timestamps are BIT EDGES and the prediction grid is
            # bit-edge aligned too (measured sub-bit offset <= 0.44 ms of a 20 ms bit), so
            # (t - t0)/bs sits within a hair of an integer and truncation lands one bit EARLY
            # on whichever side of it noise falls -- which reads as a coin flip. The tracker's
            # bit_at() floors correctly because it samples at MID-bit; this samples at the
            # edge. Same grid-semantics trap as the pilot bit_pred bug, third instance in a
            # day: whether floor or round is right depends on where in the cell you sampled.
            k = int(round((t - t0) / bs))
            if k < 0 or k >= len(tab):
                continue
            b = tab[k]
            if b == 0:                      # unknown bit: not a disagreement
                continue
            n += 1
            agree += (b == int(sgn))
        if n < 10:                          # too few to resolve this emit's polarity
            return
        r = max(agree, n - agree) / float(n)    # THIS EMIT at its own better polarity
        self.rate[prn] = r if prn not in self.rate else (
            (1.0 - ALPHA) * self.rate[prn] + ALPHA * r)
        self.n[prn] = self.n.get(prn, 0) + n

    # ---- verdict -----------------------------------------------------------------------
    def verdict(self, prn):
        """'ok' | 'bad' | 'unknown'. Hysteretic: condemning is easier than exonerating, so a
        satellite does not flap in and out of the peel on noise."""
        r, n = self.rate.get(prn), self.n.get(prn, 0)
        if r is None or n < MIN_BITS:
            return "unknown"
        if prn in self.bad:
            if r >= GOOD_RATE:
                self.bad.discard(prn)
                self._log("navbit-health: PRN %d recovered (agreement %.1f%%)" % (prn, 100 * r))
                return "ok"
            return "bad"
        if r < BAD_RATE:
            self.bad.add(prn)
            self._log("navbit-health: PRN %d VETOED -- predicted bits agree with the air only "
                      "%.1f%% (n=%d); publishing them would subtract at the wrong sign"
                      % (prn, 100 * r, n))
            return "bad"
        return "ok"

    def veto(self, prn):
        """True if this PRN's bits must NOT be published."""
        v = self.verdict(prn) == "bad"
        if v:
            self.n_veto += 1
        return v

    def report(self):
        """One line for the broker log; empty when there is nothing scored yet."""
        if not self.rate:
            return ""
        parts = ["%d:%.0f%%%s" % (p, 100 * r, "*" if p in self.bad else "")
                 for p, r in sorted(self.rate.items())]
        return "navbit-health: %s (* = vetoed; %d veto(es) applied)" % (
            " ".join(parts), self.n_veto)
