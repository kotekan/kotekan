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
