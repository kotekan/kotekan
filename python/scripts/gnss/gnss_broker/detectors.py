"""Pathology detectors: make the already-understood failures label themselves.

Each detector turns a known failure shape into a logged, labelled event, so a case study
starts from a labelled event and an arm is never judged across an unnoticed one. They do not
replace looking at the data; they keep a recurring pathology from passing unseen.

No detector writes a seed, a trim or a command; each observes state another stage already
computed, because a detector that can change the instrument is no longer measuring it. Two
gates read their output: displaced admission is refused while BrownoutDetector.active()
(fleet.displaced_chain_event), and --hold-on-present admits a hold on a run of present
cycles in the QSeries (seeding._present_streak).

@author Keith Vanderlinde
"""


# ── D0: the population-honest q series ────────────────────────────────────────────────────

#: What the series knows about a satellite on a given cycle.
PRESENT = "present"  # in the fleet aggregate and passing the presence gate
ABSENT = "absent"  # seeded, but no fleet row this cycle or the gate said no
DROPPED = "dropped"  # in the series but no longer seeded


class QSeries(object):
    """Per-satellite q history that keeps the satellites that stopped reporting.

    The broker's `DLL:` log line lists only satellites that passed the presence gate, so a
    satellite whose q craters leaves the sample, and any statistic over that line measures
    survivors, which look healthy by construction. Judge an arm on this series instead.

    An absent satellite contributes a sample with `q=None` and `state=ABSENT` rather than
    nothing. `summary()` reports the q statistics over the present samples together with the
    present fraction, so how much of the window they describe is always visible.
    """

    def __init__(self, window_s=900.0):
        self.window_s = float(window_s)
        # prn -> [(t, q_or_None, state), ...]
        self.hist = {}

    def note_cycle(self, t, seeded, fleet):
        """Record one cycle for every seeded satellite, present or not.

        `seeded` is the set of PRNs the broker is currently seeding; `fleet` is the per-PRN
        aggregate (None when the DLL did not run this cycle, which records every seeded PRN
        ABSENT). A PRN in `hist` but no longer in `seeded` gets a DROPPED sample every cycle;
        that keeps its history non-empty, so it stays in the series at 0% present.
        """
        rows = fleet or {}
        for prn in set(seeded) | set(self.hist):
            if prn not in seeded:
                state, q = DROPPED, None
            else:
                fl = rows.get(prn)
                if isinstance(fl, dict) and fl.get("present"):
                    state, q = PRESENT, fl.get("q")
                else:
                    state, q = ABSENT, None
            h = self.hist.setdefault(prn, [])
            h.append((t, q, state))
            while h and t - h[0][0] > self.window_s:
                h.pop(0)
        for prn in [p for p, h in self.hist.items() if not h]:
            self.hist.pop(prn, None)

    def summary(self, prn):
        """(n_total, n_present, q_mean, q_sd, present_frac) over the window, or None.

        q_mean and q_sd are over the present samples only: an absent satellite has no q, and
        inventing one would be worse than omitting it. `present_frac` says how much of the
        window those numbers describe; a q_sd over 15% presence describes the 15% the
        satellite was well enough to be seen, not its stability.
        """
        h = self.hist.get(prn)
        if not h:
            return None
        qs = [q for _, q, s in h if s == PRESENT and q is not None]
        n = len(h)
        if not qs:
            return n, 0, None, None, 0.0
        mean = sum(qs) / len(qs)
        var = sum((x - mean) ** 2 for x in qs) / len(qs)
        return n, len(qs), mean, var ** 0.5, len(qs) / float(n)

    def line(self, chain, max_prns=12):
        """One compact log line (QPOP): the least-present satellites first, at most max_prns.

        Sorted by presence rather than by PRN: the interesting satellite is the one that is
        missing, and a PRN-ordered line buries it among the healthy ones.
        """
        rows = []
        for prn in self.hist:
            s = self.summary(prn)
            if s is None:
                continue
            n, npres, mean, sd, frac = s
            rows.append((frac, prn, npres, n, mean, sd))
        if not rows:
            return None
        rows.sort()
        parts = []
        for frac, prn, npres, n, mean, sd in rows[:max_prns]:
            parts.append(
                "%d:%d%%(%d/%d)%s"
                % (
                    prn,
                    int(round(frac * 100)),
                    npres,
                    n,
                    "" if mean is None else " q%.2f+-%.2f" % (mean, sd),
                )
            )
        return "QPOP %s (present%% over %ds, worst first): %s" % (
            chain,
            int(self.window_s),
            " ".join(parts),
        )


# ── D1: the brownout detector ─────────────────────────────────────────────────────────────


class BrownoutDetector(object):
    """A chain-wide collapse in how many satellites are present, as a labelled episode.

    The point is the label: an episode is logged (BROWNOUT open / closed) so a judge can
    exclude or stratify by it, and active() is read by displaced admission and by D3.

    A collapse is relative to this chain's own peak over window_s, never to a constant: fleet
    size, elevation mask and constellation all set the normal count. An episode opens when the
    present count falls below frac x that peak (peak >= min_base) and is announced once it
    has lasted min_len_s.

    It is a chain-level fault: when one band collapses while another holds, the fold or the
    band is at fault, not the satellites. This detector reports one chain; comparing chains is
    the judge's job.
    """

    def __init__(self, window_s=600.0, frac=0.6, min_base=4, min_len_s=60.0):
        self.window_s = float(window_s)
        self.frac = float(frac)
        self.min_base = int(min_base)
        self.min_len_s = float(min_len_s)
        self.pop = []  # [(t, n_present)] over the window
        self.open_ep = None  # [t_start, baseline, deepest] while in a brownout
        self.announced = False  # has the open episode been logged yet?

    def note_cycle(self, t, n_present):
        """Record the count. Returns a message when an episode opens or closes, else None."""
        if not self.pop or self.pop[-1][0] != t:
            self.pop.append((t, n_present))
            while self.pop and t - self.pop[0][0] > self.window_s:
                self.pop.pop(0)
        base = max(n for _, n in self.pop)
        low = base >= self.min_base and n_present < self.frac * base

        if low:
            if self.open_ep is None:
                self.open_ep = [t, base, n_present]
                return None  # do not announce until it has lasted
            self.open_ep[2] = min(self.open_ep[2], n_present)
            if t - self.open_ep[0] >= self.min_len_s and not self.announced:
                self.announced = True
                return (
                    "BROWNOUT open: %d present vs %d baseline (%d%%), %.0f s so far"
                    % (
                        n_present,
                        self.open_ep[1],
                        int(round(100.0 * n_present / max(self.open_ep[1], 1))),
                        t - self.open_ep[0],
                    )
                )
            return None

        if self.open_ep is not None:
            t0, b, deep = self.open_ep
            was = self.announced
            self.open_ep, self.announced = None, False
            if t - t0 >= self.min_len_s:
                return (
                    "BROWNOUT closed: %.0f s, %d present at worst vs %d baseline (%d%%)"
                    % (t - t0, deep, b, int(round(100.0 * deep / max(b, 1))))
                )
            if was:
                return "BROWNOUT closed: %.0f s (under the reporting length)" % (t - t0)
        return None

    def active(self):
        """Is a brownout in progress (from the first low cycle, before it is announced)? D3
        asks because a trim wiped during a chain-wide collapse is not a per-satellite fault,
        and displaced admission because a displacement then reports the chain's lost power."""
        return self.open_ep is not None


# ── D3: the handover sawtooth ─────────────────────────────────────────────────────────────


class SawtoothDetector(object):
    """A standing trim that ramps, then gets wiped.

    The seed and the C++ standing trim are two halves of one number: the tracker despreads at
    seed + trim. When the seed re-bases, the trim still carries chips accumulated against the
    old basis, the tap moves, the satellite walks off the peak, and the loop rebuilds the trim
    from scratch.

    The wipe is the event, not the ramp. A trim that keeps ramping is the loop tracking a real
    drift; a sawtooth is the discontinuity at the end, the accumulated correction discarded
    rather than handed over. Reporting on the ramp alone would flag every healthy satellite on
    a chain with clock drift. The detector collects qualifying episodes automatically rather
    than relying on the ones somebody noticed.

    A brownout (D1) wipes trims too, by taking the whole chain's presence away at once, so
    `browned_out` suppresses reporting; mixing the two classes would hide each one's structure.
    """

    def __init__(
        self,
        ramp_chips=0.5,
        window_s=1800.0,
        wipe_frac=0.5,
        cooldown_s=600.0,
        startup_hold_s=900.0,
        rebase_window_s=30.0,
    ):
        self.ramp_chips = float(ramp_chips)
        self.window_s = float(window_s)
        self.wipe_frac = float(wipe_frac)
        self.cooldown_s = float(cooldown_s)
        # No report within startup_hold_s of broker start: seeds churn while the clock
        # converges, and those wipes are not handover events.
        self.startup_hold_s = float(startup_hold_s)
        # A wipe within this many seconds after a birth step for the same PRN (ctx.birth_steps,
        # stamped at a dead-reckon re-birth) is a REBASE-WIPE: the seed re-based under the
        # trim. Otherwise it is a BARE-WIPE, typically a slew transfer: the seed slews onto a
        # stepped target and the trim's content moves into it, so the tap barely moves and the
        # cost is chopped ramp windows, not lost lock. The two are reported apart.
        self.rebase_window_s = float(rebase_window_s)
        self.hist = {}  # prn -> [(t, trim)]
        self.reported = {}  # prn -> t of last report

    def note(
        self,
        t,
        prn,
        trim,
        browned_out=False,
        uptime_s=None,
        rebase_age_s=None,
        present_frac=None,
        q_mean=None,
        in_transit=False,
    ):
        """Feed one satellite's standing trim. Returns a message on a wipe, else None.

        `uptime_s` gates the startup hold-off; `rebase_age_s` (seconds since this PRN's last
        birth step, None if never) classifies the wipe; `present_frac` and `q_mean` come from
        D0's window and disqualify a trim that was never doing sky work (present_frac < 0.5 or
        q_mean < 2.0): a ramp on a satellite that is absent or at the q floor is chasing churn.
        `browned_out` and `in_transit` suppress reporting. Suppressions never stamp the
        cooldown, so a suppressed report cannot swallow a later real one.
        """
        h = self.hist.setdefault(prn, [])
        h.append((t, trim))
        while h and t - h[0][0] > self.window_s:
            h.pop(0)
        if len(h) < 3 or browned_out:
            return None
        if in_transit:
            # A boresight transit rails the quantiser and knocks satellites off their peaks on
            # every chain at once; the wipes that follow are the transit's, not a handover's.
            return None

        prev_t, prev = h[-2]
        peak = max(abs(v) for _, v in h[:-1])
        if peak < self.ramp_chips:
            return None
        # A wipe: the magnitude collapses in a single step, from a value that had ramped.
        if abs(prev) >= self.ramp_chips and abs(trim) <= self.wipe_frac * abs(prev):
            if uptime_s is not None and uptime_s < self.startup_hold_s:
                return None
            if (present_frac is not None and present_frac < 0.5) or (
                q_mean is not None and q_mean < 2.0
            ):
                return None
            if t - self.reported.get(prn, -1e9) < self.cooldown_s:
                return None
            self.reported[prn] = t
            kind = (
                "REBASE-WIPE"
                if (
                    rebase_age_s is not None
                    and 0.0 <= rebase_age_s <= self.rebase_window_s
                )
                else "BARE-WIPE"
            )
            # The span is censored by the detector's own window: `hist` is trimmed to window_s,
            # so a ramp older than that reports a span just under it, and the reports cluster
            # near window_s as if the plant had that period. A span within 5% of the window is
            # a lower bound and is printed as one (">=").
            span = t - h[0][0]
            censored = span >= 0.95 * self.window_s
            return (
                "SAWTOOTH PRN %d: standing trim %+.2f -> %+.2f chips in one cycle "
                "(peak %+.2f over %s%.0f s%s) -- ramp discarded | %s%s"
                % (
                    prn,
                    prev,
                    trim,
                    peak,
                    ">=" if censored else "",
                    span,
                    ", CENSORED at my %.0f s window -- a LOWER BOUND, not a period"
                    % self.window_s
                    if censored
                    else "",
                    kind,
                    (" (birth-step %.0f s before)" % rebase_age_s)
                    if kind == "REBASE-WIPE"
                    else " (no birth-step in window: slew/other)",
                )
            )
        return None

    def drop(self, prn):
        """Forget a satellite: a released trim decays by the leak, which is not a wipe."""
        self.hist.pop(prn, None)
