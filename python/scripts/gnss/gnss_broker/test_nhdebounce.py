"""#97: the overlay-period debounce -- the verdict traps, not the plumbing.

The state machine under test mirrors the inline block in seeding.py (nh_pending /
_nh_deferred); reachability of that block is proven on sky (phdeb/ADOPTED log lines),
not here.

    python3 -m gnss_broker.test_nhdebounce

@author Keith Vanderlinde
"""

import sys

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


CODE_LEN = 10230
LC_SEG = 20
LLC = CODE_LEN * LC_SEG


class Machine(object):
    """The debounce state machine, as spliced into seeding.py."""

    def __init__(self, n):
        self.n = n
        self.pending = {}     # prn -> (m, count)
        self.ph_hist = {}     # prn -> ph  (anchor only; hop/dop elided)

    def det(self, prn, ph, m):
        """One detection: measured phase `ph`, disagreement `m` vs history.
        Returns (seeded_ph, deferred)."""
        deferred = False
        if m and self.n > 0:
            pm, pc = self.pending.get(prn, (0, 0))
            pc = pc + 1 if pm == m else 1
            self.pending[prn] = (m, pc)
            if pc < self.n:
                ph = (ph + m * CODE_LEN) % LLC
                deferred = True
            else:
                self.pending.pop(prn, None)
        elif not m:
            self.pending.pop(prn, None)
        if not deferred:
            self.ph_hist[prn] = ph
        return ph, deferred


def test_single_flip_is_suppressed():
    q = Machine(2)
    q.det(7, 5000.0, 0)
    ph, d = q.det(7, 5000.0 - CODE_LEN, +1)          # source flips one period low
    check(d, "a single flipped detection is deferred")
    check(abs(ph - 5000.0) < 1e-9,
          "...and the seeded phase carries the STANDING period, measured fine phase")
    check(q.ph_hist[7] == 5000.0,
          "...and history did not move (neither correction nor unconfirmed measurement)")
    ph, d = q.det(7, 5000.0, 0)                      # source back to normal
    check(not d and 7 not in q.pending,
          "the flip-back clears the pending state; no step ever reached the seed")


def test_real_change_is_adopted():
    q = Machine(2)
    q.det(7, 5000.0, 0)
    _, d1 = q.det(7, 5000.0 - CODE_LEN, +1)
    ph, d2 = q.det(7, 5000.0 - CODE_LEN, +1)         # confirmed by a second detection
    check(d1 and not d2, "a real period change is adopted on the Nth consecutive detection")
    check(abs(ph - (5000.0 - CODE_LEN)) < 1e-9, "...at the MEASURED phase, no correction")
    check(q.ph_hist[7] == ph, "...and history now anchors at the adopted phase")


def test_alternating_flips_never_adopt():
    q = Machine(2)
    q.det(7, 5000.0, 0)
    for m in (+1, -1, +1, -1, +1, -1):
        _, d = q.det(7, 5000.0 - m * CODE_LEN, m)
        check(d, "alternating flip m=%+d stays deferred (count resets on sign change)" % m)
    check(q.ph_hist[7] == 5000.0, "history never moved through six alternating flips")


def test_fine_phase_is_never_invented():
    """THE 2026-08-02 TRAP. The deferral may only move the period, never the phase."""
    q = Machine(2)
    q.det(7, 5000.0, 0)
    ph, _ = q.det(7, 5003.7 - CODE_LEN, +1)          # flipped AND drifted 3.7 chips
    check(abs(ph - 5003.7) < 1e-9,
          "deferred seed keeps the measured 3.7-chip drift -- only the period is held")


def test_zero_disables():
    q = Machine(0)
    ph, d = q.det(7, 5000.0 - CODE_LEN, +1)
    check(not d and abs(ph - (5000.0 - CODE_LEN)) < 1e-9,
          "N=0 is the pre-#97 behaviour: the flip goes straight to the seed")


def test_source_matches_seeding_py():
    """The machine above must stay a faithful copy of the inline block."""
    import os
    src = open(os.path.join(os.path.dirname(__file__), "seeding.py")).read()
    for frag in ("_pc = _pc + 1 if _pm == m else 1",
                 "ph = (ph + m * ctx.code_len) % LLc",
                 "_nh_deferred = True",
                 "ctx.cpt.nh_pending.pop(prn, None)",
                 "if not _nh_deferred and (snr >= ctx.args.period_check_snr"):
        check(frag in src, "seeding.py still contains: %s" % frag)


def main():
    for fn in (test_single_flip_is_suppressed, test_real_change_is_adopted,
               test_alternating_flips_never_adopt, test_fine_phase_is_never_invented,
               test_zero_disables, test_source_matches_seeding_py):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
