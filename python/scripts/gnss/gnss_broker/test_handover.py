"""#92 trim handover: the bound, the gating, and the instrument correction.

    python3 -m gnss_broker.test_handover

The handover posts a compensating delta to a LIVE gather that is actuating a real satellite.
Its bound is the only thing standing between a 400-chip shared-clock birth and a slammed C++
clamp, and no fixture can exercise it -- broker_equiv replays a transcript whose gather never
answers. So the bound gets a test.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.handover import TrimHandover


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class Spy:
    """Records posts; can be told to fail like a gather that is down."""

    def __init__(self, boom=False):
        self.calls = []
        self.logs = []
        self.boom = boom

    def post(self, url, body, timeout=None):
        if self.boom:
            raise OSError("connection refused")
        self.calls.append((url, body, timeout))
        return 200

    def log(self, msg):
        self.logs.append(msg)


def test_posts_the_negation():
    print("the transfer itself")

    h = TrimHandover(enabled=True)
    s = Spy()
    ok = h.offer(7, +0.42, True, "gal_e5a", "http://127.0.0.1:12051/fleet_trim/", s.post, s.log)
    check(ok, "an armed PRN inside the bound posts")
    url, body, _ = s.calls[0]
    check(url == "http://127.0.0.1:12051/fleet_trim/adjust_trim", "to /adjust_trim, one slash")
    check(body == {"chains": {"gal_e5a": {"7": -0.42}}},
          "carrying the NEGATED step, keyed by string PRN under its chain")
    check(abs(h.adjcum[7] + 0.42) < 1e-12, "and remembers what it posted")


def test_the_bound_is_the_safety_argument():
    print("the bound (a shared-clock birth is not a handover)")

    h = TrimHandover(enabled=True)
    s = Spy()
    # The births this must refuse move by hundreds of chips: the trim cannot have been
    # carrying that, and posting it would slam the 3.0-chip C++ clamp.
    ok = h.offer(7, -412.7, True, "gal_e5a", "u", s.post, s.log)
    check(not ok and not s.calls, "a 412-chip step posts NOTHING")
    check(any("skipped" in m for m in s.logs), "and says so loudly rather than silently")
    check(7 not in h.adjcum, "a skipped step leaves no trace in the instrument correction")
    check(h.skipped == 1, "it is counted")

    # Just inside and just outside, since the bound is the whole argument.
    check(h.offer(7, +2.5, True, "c", "u", s.post, s.log), "exactly at the bound posts")
    check(not h.offer(7, +2.51, True, "c", "u", s.post, s.log), "a hair past it does not")


def test_gating():
    print("gating")

    s = Spy()
    h = TrimHandover(enabled=False)
    check(not h.offer(7, +0.4, True, "c", "u", s.post, s.log), "disabled: never posts")

    h = TrimHandover(enabled=True)
    check(not h.offer(7, +0.4, False, "c", "u", s.post, s.log),
          "an UNARMED PRN has no standing trim to hand over")
    check(not h.offer(7, +0.4, True, "c", "", s.post, s.log), "no fleet-trim url: nothing to post to")
    check(not s.calls, "none of those touched the transport")


def test_failure_never_takes_seeding_down():
    print("failure handling")

    h = TrimHandover(enabled=True)
    s = Spy(boom=True)
    ok = h.offer(7, +0.4, True, "c", "u", s.post, s.log)          # must not raise
    check(not ok and h.failed == 1, "a dead gather is a False, not an exception")
    check(any("FAILED" in m for m in s.logs), "logged")
    check(7 not in h.adjcum,
          "and NOT counted into adjcum -- crediting an unposted delta would corrupt #93")


def test_instrument_correction():
    print("the #93 instrument correction")

    h = TrimHandover(enabled=True)
    s = Spy()
    check(h.corrected(7, 1.50) == 1.50, "with no handovers, the trim passes through")

    h.offer(7, +0.30, True, "c", "u", s.post, s.log)               # posts -0.30
    h.offer(7, +0.20, True, "c", "u", s.post, s.log)               # posts -0.20
    check(abs(h.adjcum[7] + 0.50) < 1e-12, "deltas accumulate across re-bases")
    # The gather has moved the trim by -0.50; removing that leaves the series continuous, so
    # what remains is drift rather than a ledger transfer.
    check(abs(h.corrected(7, 1.00) - 1.50) < 1e-12,
          "the corrected series adds the handovers back out")
    check(h.corrected(9, 1.00) == 1.00, "another PRN is untouched")


if __name__ == "__main__":
    print("#92 trim handover\n")
    for fn in (test_posts_the_negation, test_the_bound_is_the_safety_argument, test_gating,
               test_failure_never_takes_seeding_down, test_instrument_correction):
        fn()
    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    sys.exit(1 if _fails else 0)
