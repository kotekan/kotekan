"""Automatic pathology detectors: make the already-understood failures announce themselves.

WHY THESE EXIST. Every mechanism found in the week to 2026-08-26 came from a human noticing a
shape in the viewer. That works and should continue. But the same pathologies now RECUR, and
on 2026-08-25 the recurrence actively corrupted an experiment: four #90 admission fires rode
an unnoticed band brownout, and a "no harm" verdict was computed on a population that silently
dropped the sick satellites.

So the goal is NOT to replace case studies. It is to make the understood pathologies label
themselves, so a case study starts from a labelled event and an arm is never judged across an
unnoticed one.

Everything here is READ-ONLY. No detector writes a seed, a trim, or a command; each one
observes state another stage already computed. That is deliberate: a detector that can change
the instrument is no longer measuring it.

@author Keith Vanderlinde
"""


# ── D0: THE POPULATION-HONEST q SERIES ────────────────────────────────────────────────────

#: What the series knows about a satellite on a given cycle.
PRESENT = "present"        # in the fleet aggregate and passing the presence gate
ABSENT = "absent"          # SEEDED, and the model says it is up -- but the gate says no
DROPPED = "dropped"        # no longer seeded at all (set, or retired)


class QSeries(object):
    """Per-satellite q history that KEEPS the satellites that stopped reporting.

    ⚠️ THIS IS THE WHOLE POINT, AND IT IS EASY TO MISS. The broker's `DLL:` log line lists
    only satellites that passed the presence gate, so a satellite whose q craters LEAVES THE
    SAMPLE. Any statistic computed over that line measures SURVIVORS -- and survivors always
    look healthy, because that is what being a survivor means.

    Measured cost, 2026-08-25: E4 was absent from the e5b line for 85 minutes, which was
    exactly its sick interval, while a q-SD comparison over that line reported "no harm" for
    an armed change. The verdict was withdrawn. Any future arm judges on THIS series.

    A satellite that is absent contributes a sample with `q=None` and `state=ABSENT` rather
    than contributing nothing. `summary()` reports both the survivor statistic and the honest
    one, side by side, so the difference between them is visible rather than a matter of
    which source somebody happened to use.
    """

    def __init__(self, window_s=900.0):
        self.window_s = float(window_s)
        # prn -> [(t, q_or_None, state), ...]
        self.hist = {}

    def note_cycle(self, t, seeded, fleet):
        """Record one cycle for EVERY seeded satellite, present or not.

        `seeded` is the set of PRNs the broker is currently seeding; `fleet` is the per-PRN
        aggregate (may be None when the DLL did not run this cycle). A PRN in `hist` but no
        longer in `seeded` is recorded as DROPPED once, then ages out of the window.
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

        ⚠️ q_mean/q_sd ARE OVER THE PRESENT SAMPLES ONLY -- there is no q when a satellite is
        absent, and inventing one would be worse than omitting it. `present_frac` is what
        makes the statistic honest: it says how much of the window those numbers describe. A
        q_sd computed over 15% presence is not a measurement of that satellite's stability,
        it is a measurement of the 15% it was well enough to be seen.
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

    def episodes(self, prn, min_absence_s=300.0):
        """Contiguous runs of ABSENT longer than `min_absence_s`, as (t_start, t_end).

        This is the shape a "sick interval" actually has, and the thing a judge script needs
        to exclude or stratify by. It is also exactly D2's targeting signal.
        """
        h = self.hist.get(prn) or []
        out, start, last = [], None, None
        for t, _q, s in h:
            if s == ABSENT:
                if start is None:
                    start = t
                last = t
            else:
                if start is not None and last - start >= min_absence_s:
                    out.append((start, last))
                start, last = None, None
        if start is not None and last - start >= min_absence_s:
            out.append((start, last))
        return out

    def line(self, chain, max_prns=12):
        """One compact log line: the satellites whose window is LEAST present come first.

        Sorted by presence rather than by PRN deliberately -- the interesting satellite is the
        one that is missing, and a PRN-ordered line buries it among the healthy ones.
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
            parts.append("%d:%d%%(%d/%d)%s" % (
                prn, int(round(frac * 100)), npres, n,
                "" if mean is None else " q%.2f+-%.2f" % (mean, sd)))
        return "QPOP %s (present%% over %ds, worst first): %s" % (
            chain, int(self.window_s), " ".join(parts))


# ── D1: THE BROWNOUT DETECTOR (#91) ───────────────────────────────────────────────────────

class BrownoutDetector(object):
    """A chain-wide collapse in how many satellites are present, as a labelled EPISODE.

    ⚠️ THE POINT IS THE LABEL, NOT THE SUPPRESSION. This rule already existed inside #90's
    admission gate, where it silently held fires back. That is the right action but the wrong
    place to leave it: nothing downstream could tell that a window contained a brownout, so on
    2026-08-25 four admission fires rode an unnoticed one and the arm judging them was
    confounded before it started. Promoted here, an episode is a thing a judge script can
    exclude or stratify by.

    ⚠️ A COLLAPSE IS RELATIVE TO THIS CHAIN'S OWN RECENT PEAK, never to a constant. Fleet
    size, elevation mask and constellation all set the normal count, so any fixed threshold is
    right for exactly one chain at one time of day.

    ⚠️ IT IS A CHAIN-LEVEL FAULT, WHICH IS WHY IT DESERVES A NAME. When e5b collapses while
    e5a holds steady, the fold or the band is the patient -- and per-satellite corrections are
    then treating the wrong organ. That band-vs-band asymmetry is what identified the
    2026-08-25 23:00 event; this detector reports one chain, and comparing chains is the
    judge's job.
    """

    def __init__(self, window_s=600.0, frac=0.6, min_base=4, min_len_s=60.0, dark_at=1):
        self.window_s = float(window_s)
        self.frac = float(frac)
        self.min_base = int(min_base)
        self.min_len_s = float(min_len_s)
        # A chain holding <= dark_at satellites is not tracking at all -- band-wide loss,
        # not the ordinary dip a brownout describes.
        self.dark_at = int(dark_at)
        self.last_dark_t = None
        self.pop = []            # [(t, n_present)] over the window
        self.open_ep = None      # [t_start, baseline, deepest] while in a brownout
        self.announced = False   # has the open episode been logged yet?
        self.episodes = []       # closed: (t_start, t_end, baseline, deepest)

    def note_cycle(self, t, n_present):
        """Record the count. Returns a message when an episode opens or closes, else None."""
        # ⚠️ DARKNESS IS NOT A BROWNOUT, AND THAT IS WHY IT NEEDS ITS OWN CLOCK. A brownout is
        # measured against the chain's OWN 600 s peak, so a chain that has been dark long
        # enough for that peak to decay CANNOT register one -- `base >= min_base` fails when
        # every sample in the window is zero. The suppression that D2/D3 rely on therefore
        # switches itself off exactly when the disturbance is largest, which is the shape of
        # a gate that cannot fail. Recording the last dark cycle separately gives the
        # recovery its own, decay-proof clock.
        if n_present <= self.dark_at:
            self.last_dark_t = t
        if not self.pop or self.pop[-1][0] != t:
            self.pop.append((t, n_present))
            while self.pop and t - self.pop[0][0] > self.window_s:
                self.pop.pop(0)
        base = max(n for _, n in self.pop)
        low = base >= self.min_base and n_present < self.frac * base

        if low:
            if self.open_ep is None:
                self.open_ep = [t, base, n_present]
                return None                   # do not announce until it has lasted
            self.open_ep[2] = min(self.open_ep[2], n_present)
            if t - self.open_ep[0] >= self.min_len_s and not self.announced:
                self.announced = True
                return ("BROWNOUT open: %d present vs %d baseline (%d%%), %.0f s so far"
                        % (n_present, self.open_ep[1],
                           int(round(100.0 * n_present / max(self.open_ep[1], 1))),
                           t - self.open_ep[0]))
            return None

        if self.open_ep is not None:
            t0, b, deep = self.open_ep
            was = self.announced
            self.open_ep, self.announced = None, False
            if t - t0 >= self.min_len_s:
                self.episodes.append((t0, t, b, deep))
                return ("BROWNOUT closed: %.0f s, %d present at worst vs %d baseline (%d%%)"
                        % (t - t0, deep, b, int(round(100.0 * deep / max(b, 1)))))
            if was:
                return "BROWNOUT closed: %.0f s (under the reporting length)" % (t - t0)
        return None

    def recovering(self, t, hold_s):
        """Has the chain seen signal again only recently, after going dark?

        THE SAME CONVERGENCE, ON A DIFFERENT CLOCK. `LatchDetector`'s startup hold-off exists
        because presence flaps while the clock and seeds converge -- and that is a property of
        the PLANT restarting, not of the process restarting. When the analog front-ends came
        back at 17:16 on 2026-08-26 the broker had been up 88 minutes, so every uptime-based
        guard was long expired, and D2 immediately reported PRN 3 and PRN 15 as latched: both
        had locked at q 3.5 for THIRTY SECONDS during the scramble and then dropped. That is
        #90 flight 3's startup-convergence population arriving through a door the startup
        hold-off does not cover.
        """
        return (hold_s > 0.0 and self.last_dark_t is not None
                and t - self.last_dark_t < hold_s)

    def established(self):
        """Open AND past `min_len_s` -- the trigger for POLICY, as opposed to suppression.

        ⚠️ NOT THE SAME AS `active()`, AND THE DIFFERENCE IS DELIBERATE. `active()` is eager:
        D2/D3 should stop reporting the moment presence starts collapsing, because a report
        made mid-collapse is wrong even if the collapse turns out to be one cycle long. A
        policy that ACTS on the chain -- #91's trim freeze -- must NOT fire on a flicker:
        presence flickers constantly (that is the entire reason `--fleet-trim-hold-s`
        exists), and freezing on every flicker would gut the loop's duty cycle, which is
        the opposite of what #91 is for. So this waits out the same `min_len_s` the
        announcement waits out, and a policy consumer should ask for THIS one.
        """
        return self.open_ep is not None and self.announced

    def active(self):
        """Is a brownout in progress? D2 asks, because a satellite missing during a chain-wide
        collapse is not a per-satellite fault."""
        return self.open_ep is not None


# ── D2: THE DEEP-LATCH DETECTOR (#90 v3's targeting, running UNARMED) ─────────────────────

class LatchDetector(object):
    """A satellite that was healthy, went absent, and STAYED absent -- with no chain-wide cause.

    ⚠️ THIS RUNS UNARMED ON PURPOSE, and that is the whole design. #90 flew four armed flights
    in one evening and produced ZERO genuine latch targets: every fire was startup convergence,
    a band brownout, or threshold flicker. So the missing number is the BASE RATE -- how often
    does this actually happen? -- and a detector measures it at zero risk, where an armed gate
    measures it by intervening.

    The rule below is exactly #90 v3's proposed admission rule, which is the point: if this
    reports nothing over a week, v3 should not be armed, and that is a far cheaper answer than
    another flight.

    ⚠️ RECENT-LOCK IS WHAT SEPARATES A LATCH FROM A SET. A satellite below the horizon is
    absent for entirely good reasons. Requiring it to have been HEALTHY shortly before the
    absence began is what makes this a fault report rather than an ephemeris report.

    ⚠️ AND THE STARTUP SOLVE IS NOT A LATCH -- this is flight 3a's lesson, and the detector
    needs it as much as the gate did. Presence FLAPS while the clock converges (PRN 9,
    2026-08-25: present at q 2.32 at t+28 s, absent by t+2 min), so a satellite can be
    genuinely healthy-then-absent within the first few minutes without anything being wrong.
    Counting those would inflate the base rate this exists to measure, which is the one number
    it must not get wrong. Reports are held until the process is `startup_hold_s` old.
    """

    def __init__(self, min_absence_s=300.0, lookback_s=900.0, healthy_q=2.0, cooldown_s=1800.0,
                 startup_hold_s=900.0):
        self.min_absence_s = float(min_absence_s)
        self.lookback_s = float(lookback_s)
        self.healthy_q = float(healthy_q)
        self.cooldown_s = float(cooldown_s)
        self.startup_hold_s = float(startup_hold_s)
        self.reported = {}       # prn -> t of last report (one per episode, not per cycle)
        self.suppressed_startup = 0   # how many reports the hold-off swallowed, for honesty

    def scan(self, t, qseries, browned_out, uptime_s=None, recovering=False):
        """[(prn, absent_s, q_before)] for satellites that look latched right now.

        `browned_out` suppresses everything: during a chain-wide collapse a missing satellite
        is a symptom of the chain, and reporting each one would bury the signal that matters
        under a constellation of noise.

        `recovering` is the same suppression on the PLANT's clock rather than the process's --
        see BrownoutDetector.recovering(). Counted in `suppressed_startup` alongside the
        process case: they are one population and separating the counters would only invite
        reading either as a rate.
        """
        if browned_out:
            return []
        startup = ((uptime_s is not None and uptime_s < self.startup_hold_s)
                   or bool(recovering))
        out = []
        for prn, h in qseries.hist.items():
            if not h or h[-1][2] != ABSENT:
                continue
            start = None
            for tt, _q, s in reversed(h):
                if s != ABSENT:
                    break
                start = tt
            if start is None or t - start < self.min_absence_s:
                continue
            qs = [q for tt, q, s in h
                  if s == PRESENT and q is not None and start - self.lookback_s <= tt < start]
            if not qs or max(qs) < self.healthy_q:
                continue
            if t - self.reported.get(prn, -1e9) < self.cooldown_s:
                continue
            if startup:
                # Do NOT stamp `reported` here: the satellite may still be genuinely latched
                # once the solve settles, and swallowing the report must not also swallow the
                # later, real one.
                self.suppressed_startup += 1
                continue
            self.reported[prn] = t
            out.append((prn, t - start, max(qs)))
        return out


# ── D3: THE HANDOVER SAWTOOTH (#92) ───────────────────────────────────────────────────────

class SawtoothDetector(object):
    """A standing trim that ramps, then gets wiped -- the shape #92 exists to remove.

    THE MECHANISM. The seed and the C++ standing trim are two halves of one number: the
    tracker despreads at (seed + trim). When the seed re-bases, the trim goes on carrying
    chips accumulated against the old basis, the tap moves, the satellite walks off the peak,
    and the loop rebuilds the trim from scratch over ~25 minutes. Then it happens again.

    ⚠️ THE WIPE IS THE EVENT, NOT THE RAMP. A trim that ramps and keeps ramping is the loop
    working: it is tracking a real drift. What makes it a sawtooth is the DISCONTINUITY at the
    end -- the accumulated correction being discarded rather than handed over. Reporting on
    ramp alone would flag every healthy satellite on a chain with clock drift.

    ⚠️ AND THIS IS WHY #92's P2 HAS STAYED OPEN. Every >=0.3-chip event examined by eye so far
    was restart or flicker churn, not a drift-then-rebase. The detector's job is to produce
    the population automatically, so P2 is judged on episodes that qualify rather than on
    whichever one somebody happened to notice.

    ⚠️ IT MUST DISTINGUISH ITSELF FROM D1. A brownout wipes trims too, by taking the whole
    chain's presence away at once. `browned_out` suppresses reporting for the same reason it
    does in D2 -- superposing the two is what made E3's per-satellite structure look
    heterogeneous in the first place.
    """

    def __init__(self, ramp_chips=0.5, window_s=1800.0, wipe_frac=0.5, cooldown_s=600.0,
                 startup_hold_s=900.0, rebase_window_s=30.0):
        self.ramp_chips = float(ramp_chips)
        self.window_s = float(window_s)
        self.wipe_frac = float(wipe_frac)
        self.cooldown_s = float(cooldown_s)
        # FIRST FLIGHT (2026-08-26 14:54): 6 of 12 reports landed inside the broker's first
        # two minutes -- seed churn while the clock converges, the same population that gave
        # D2 four false reports. Same cure: a hold-off, with the suppression COUNTED.
        self.startup_hold_s = float(startup_hold_s)
        # A wipe within this many seconds of a BIRTH-STEP for the same PRN is the class the
        # #92 handover addresses; a wipe with no rebase in the window is the SLEW-TRANSFER
        # class (gps_l5, ~600 s churn cadence: the seed slews onto a stepped target and the
        # trim's content transfers into it -- the tap barely moves and the cost is chopped
        # ramp windows, not lost lock). Superposing them is E3's heterogeneity mistake.
        self.rebase_window_s = float(rebase_window_s)
        self.hist = {}          # prn -> [(t, trim)]
        self.reported = {}      # prn -> t of last report
        self.episodes = []      # (prn, t, peak_trim, after_trim, kind)
        self.suppressed_startup = 0
        self.suppressed_weak = 0

    def note(self, t, prn, trim, browned_out=False, uptime_s=None, rebase_age_s=None,
             present_frac=None, q_mean=None):
        """Feed one satellite's standing trim. Returns a message on a wipe, else None.

        `trim` should already have #92's own handover deltas removed (see
        `TrimHandover.corrected`) -- otherwise a successful handover, which is the CURE, reads
        as the disease it was applied to.

        `uptime_s` gates the startup hold-off; `rebase_age_s` (seconds since this PRN's last
        BIRTH-STEP, None if never) classifies the wipe; `present_frac`/`q_mean` come from D0's
        window and disqualify a trim that was never doing sky work -- a ramp on a satellite
        that is absent or at the q floor is churn-chasing, and counting it (PRN 34 on e5a,
        14:54:16) pollutes both sides of the P2 comparison. Suppressions are counted and never
        stamp the cooldown, so a suppressed report cannot swallow a later real one.
        """
        h = self.hist.setdefault(prn, [])
        h.append((t, trim))
        while h and t - h[0][0] > self.window_s:
            h.pop(0)
        if len(h) < 3 or browned_out:
            return None

        prev_t, prev = h[-2]
        peak = max(abs(v) for _, v in h[:-1])
        if peak < self.ramp_chips:
            return None
        # A WIPE: the magnitude collapses in a single step, from a value that had ramped.
        if abs(prev) >= self.ramp_chips and abs(trim) <= self.wipe_frac * abs(prev):
            if uptime_s is not None and uptime_s < self.startup_hold_s:
                self.suppressed_startup += 1
                return None
            if ((present_frac is not None and present_frac < 0.5)
                    or (q_mean is not None and q_mean < 2.0)):
                self.suppressed_weak += 1
                return None
            if t - self.reported.get(prn, -1e9) < self.cooldown_s:
                return None
            self.reported[prn] = t
            kind = ("REBASE-WIPE" if (rebase_age_s is not None
                                      and 0.0 <= rebase_age_s <= self.rebase_window_s)
                    else "BARE-WIPE")
            self.episodes.append((prn, t, prev, trim, kind))
            return ("SAWTOOTH PRN %d: standing trim %+.2f -> %+.2f chips in one cycle "
                    "(peak %+.2f over %.0f s) -- ramp discarded, not handed over | %s%s"
                    % (prn, prev, trim, peak, t - h[0][0], kind,
                       (" (birth-step %.0f s before)" % rebase_age_s)
                       if kind == "REBASE-WIPE" else " (no birth-step in window: slew/other)"))
        return None

    def drop(self, prn):
        """Forget a satellite: a released trim decays by the LEAK, which is not a wipe."""
        self.hist.pop(prn, None)
