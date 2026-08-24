#!/usr/bin/env python3
"""Concurrency stress for the state the GIL has been protecting by accident.

docs/CHORD_FREE_THREADING.md section 6: the four broker_equiv digests and the eight
fleetdll_gate legs are good gates and not ONE of them exercises concurrency. A race passes
all twelve. This is the missing gate, and it is deliberately NOT a longer broker run -- a
soak over a racy broker returns a plausible number (see the no-burn-in rule).

Three lanes, each an invariant that a lock protects and nothing else does:

  RECEIVER    N chain threads contribute and consume clocks/biases at once. Every value read
              back must be one that was actually written, carrying ITS OWN owner and satellite
              count. A torn _Shared -- chain A's value with chain B's n_sats -- is a clock
              adopted from a chain that never published it, and the consumer cannot tell.

  JOINT       N threads drive one JointReceiverState, as the three chains on 1176.45 MHz
              really do. Membership churns while other threads read, which is the shape that
              killed the gps_l5 chain on 2026-08-15: a diagnostic indexed P[0, 57] on a 57x57
              P mid-resize and the IndexError took the thread down. Asserts the state stays
              square and self-consistent, and that no reader raises.

  NOTES       drain_notes against concurrent writers. Notes drained + notes still queued must
              equal notes written. This lane FAILS on the pre-fix code -- `--selftest` puts
              the unlocked read-then-rebind back and requires it to -- which is what makes it
              a gate rather than decoration.

              ⚠️ AND IT FAILS ONLY FREE-THREADED. Under the GIL the selftest survives 20
              attempts undetected; free-threaded it trips on the first, at +24 notes on
              1600. That is not a weakness of the lane, it is the entire thesis of section 6
              stated as a measurement: the accidental atomicity is real, code depends on it,
              and removing the GIL is what makes the dependence visible.

⚠️ IT IS A RACE DETECTOR, SO A PASS IS EVIDENCE, NOT PROOF. It runs a fixed number of
iterations, not a duration -- run it again rather than longer, and run it on BOTH
interpreters: the GIL arm is the control that says the lanes are wired up at all.

    venv/bin/python    scripts/gnss/ft_stress.py
    venv-ft/bin/python scripts/gnss/ft_stress.py
"""
import argparse
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "python", "scripts", "gnss"))

from gnss_broker import receiver as rx_mod            # noqa: E402
from gnss_broker import state_filter                  # noqa: E402

CHAINS = ["gps_l5", "gal_e5a", "bds_b2a", "gal_e5b", "bds_b2b"]


def _run(fns, name):
    """Start every thread, join them all, and surface each one's own traceback.

    A thread that dies silently here would read as a PASS -- the loop it was supposed to run
    simply would not have run. That is the failure mode this whole file exists to catch, so
    it must not be the failure mode of the harness itself.
    """
    err = {}

    def wrap(i, fn):
        try:
            fn(i)
        except BaseException:
            import traceback
            err[i] = traceback.format_exc()

    ts = [threading.Thread(target=wrap, args=(i, f), name="%s%d" % (name, i))
          for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return err


def lane_receiver(iters, nthread):
    """Contribute and consume against one Receiver from every chain at once."""
    rx = rx_mod.Receiver(log=lambda *a, **k: None)
    # n_sats is the chain's index+1, forever. A torn read shows up as a value whose owner
    # and satellite count disagree -- which is the whole point of checking the pair.
    bad = []

    def worker(i):
        chain = CHAINS[i % len(CHAINS)]
        n = (i % len(CHAINS)) + 1
        rng = random.Random(1000 + i)
        for k in range(iters):
            t = 1_000_000.0 + k * 0.1
            hz = float(i * 1000 + k)
            rx.contribute_carrier_bias(chain, hz, n, t)
            rx.contribute_code_bias(chain, "1176.45", hz * 1e-9, n, t)
            rx.contribute_dr_clock(chain, "1176.45", float(k % 10230), 0.001, t, 10230.0)
            for got in (rx.carrier_bias(t_now=t), rx.code_bias("1176.45", t_now=t),
                        rx.dr_clock("1176.45", t_now=t),
                        rx.code_bias_any_band(t_now=t), rx.dr_clock_any_band(t_now=t)):
                if got is None:
                    continue
                owner = getattr(got, "chain", None)
                nsat = getattr(got, "n_sats", None)
                if owner in CHAINS and nsat is not None:
                    if nsat != CHAINS.index(owner) + 1:
                        bad.append("torn: owner %s carries n_sats %r (its own is %d)"
                                   % (owner, nsat, CHAINS.index(owner) + 1))
            if rng.random() < 0.05:
                rx.summary()

    err = _run([worker] * nthread, "rx")
    return err, bad, None


def lane_joint(iters, nthread):
    """One JointReceiverState, many chains -- the real 1176.45 MHz arrangement."""
    js = state_filter.JointReceiverState(code_len=10230.0)
    bad = []

    def worker(i):
        rng = random.Random(2000 + i)
        # Overlapping PRN sets on purpose: disjoint sets would never make two threads touch
        # the same row, which is where the interesting collisions are.
        prns = [(i * 3 + j) % 32 + 1 for j in range(8)]
        for k in range(iters):
            t = 2_000_000.0 + k * 0.5
            for prn in prns:
                key = ("gps", prn)
                try:
                    js.update(key, rng.gauss(0.0, 0.4), 0.3, t, band="1176.45")
                except Exception as e:
                    bad.append("update raised %r" % (e,))
            try:
                js.predict(t)
                js.gauge()
                # The read accessors that are deliberately unlocked, plus locked ones that
                # index by membership -- this is the 2026-08-15 shape.
                js.clk; js.clk_rate
                for prn in prns:
                    js.bias(("gps", prn)); js.sigma(("gps", prn)); js.age(("gps", prn), t)
                js.summary(t)
                n = len(js._idx)
                P = js.P
                if P.shape[0] != P.shape[1]:
                    bad.append("P not square: %r" % (P.shape,))
                if P.shape[0] < n:
                    bad.append("P is %dx%d but membership is %d" % (P.shape + (n,)))
            except Exception as e:
                bad.append("reader raised %r" % (e,))
            if rng.random() < 0.02:
                js.shorten_modulus(10230.0)

    err = _run([worker] * nthread, "joint")
    return err, bad, None


def lane_notes(iters, nthread):
    """drain_notes against concurrent writers: not one note may be lost."""
    js = state_filter.JointReceiverState(code_len=10230.0)
    drained = []
    lock = threading.Lock()
    written = iters * nthread

    def worker(i):
        local = []
        for k in range(iters):
            # UNDER THE LOCK, because that is the production contract: `_note` is private and
            # all thirteen of its call sites sit inside @_locked methods (update, predict,
            # gauge, cycle, update_carrier, update_rrate). Poking it bare -- which is how this
            # lane was first written -- races the drainer's rebind against an append no
            # production path can make, and reports lost notes that the broker cannot lose.
            # A harness that violates the contract it is testing manufactures its own bugs.
            with js._lk:
                js._note("t%d-%d" % (i, k))
            if k % 4 == 3:
                local += js.drain_notes()
        local += js.drain_notes()
        with lock:
            drained.extend(local)

    err = _run([worker] * nthread, "notes")
    # `_note` TRIMS THE QUEUE TO 200, so notes legitimately disappear once it overflows and
    # the count invariant stops meaning anything. What matters is not how many notes were
    # written in total -- they are drained as they go -- but how many can pile up BETWEEN
    # drains: every thread drains each 4th iteration, so the queue holds at most about
    # 4 per thread, plus slack. Comparing the TOTAL against the bound (which is how this was
    # first written) declares a perfectly conclusive 16-thread run inconclusive.
    left = len(js.drain_notes())
    bad, inconclusive = [], None
    in_flight = nthread * 5
    if in_flight >= 200:
        inconclusive = ("%d threads can queue ~%d notes between drains, against the 200-deep "
                        "trim in _note -- drops would be legitimate. Use fewer threads."
                        % (nthread, in_flight))
    elif len(drained) + left != written:
        # BOTH DIRECTIONS ARE THE SAME DEFECT. The unlocked swap is read-then-rebind, so two
        # drainers take the SAME list object and both return its contents: the dominant
        # failure is DUPLICATION (drained > written), not loss. An operator reading the log
        # sees a note twice and concludes the filter did something twice.
        d = len(drained) + left - written
        bad.append("MISCOUNT %+d note(s) (%s): wrote %d, drained %d, queued %d"
                   % (d, "duplicated" if d > 0 else "lost", written, len(drained), left))
    return err, bad, inconclusive


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lane", action="append", default=[],
                    choices=["receiver", "joint", "notes"],
                    help="run only these lanes (repeatable)")
    ap.add_argument("--selftest", action="store_true",
                    help="put the PRE-FIX unlocked drain_notes back and require the NOTES "
                         "lane to FAIL. A race detector that has never been seen detecting "
                         "is indistinguishable from a no-op, and this one is fast enough "
                         "and green enough to be exactly that by accident.")
    a = ap.parse_args()

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()

    if a.selftest:
        # The defect as it stood before the audit: read-then-rebind with no lock, so two
        # chains draining at once can both take the list and one loses its notes.
        def unlocked_drain(self):
            out, self.notes = self.notes, []
            return out
        state_filter.JointReceiverState.drain_notes = unlocked_drain
        print("SELFTEST: unlocked drain_notes installed; the NOTES lane must now FAIL")
        for attempt in range(1, 21):
            err, bad, _inc = lane_notes(a.iters, a.threads)
            if err or bad:
                print("  detected on attempt %d: %s" % (attempt, (bad or ["thread died"])[0]))
                print("SELFTEST PASS -- the lane can fail, so a pass from it means something")
                return 0
        print("  20 attempts, nothing detected")
        if gil:
            # NOT a broken harness. The GIL makes the read-then-rebind effectively atomic,
            # so the defect is INVISIBLE here -- which is the finding, not a shortfall: it
            # is section 6's argument ("a race would pass every gate") reproduced on demand.
            # Run this arm under the free-threaded interpreter to see the lane bite.
            print("SELFTEST INCONCLUSIVE UNDER THE GIL -- expected. The GIL hides this "
                  "defect; that is WHY the audit had to be an audit. Re-run free-threaded.")
            return 0
        print("SELFTEST FAIL -- free-threaded and still blind: the NOTES lane cannot see "
              "the defect it exists to catch, so a pass from it means nothing")
        return 1

    print("%s  GIL %s  threads=%d iters=%d"
          % (sys.version.split()[0], "ON" if gil else "OFF", a.threads, a.iters))

    lanes = [("RECEIVER", lane_receiver), ("JOINT", lane_joint), ("NOTES", lane_notes)]
    if a.lane:
        lanes = [l for l in lanes if l[0].lower() in a.lane]

    fail = skipped = 0
    for name, fn in lanes:
        t0 = time.perf_counter()
        err, bad, inconclusive = fn(a.iters, a.threads)
        dt = time.perf_counter() - t0
        if inconclusive and not (err or bad):
            # NOT a failure. A gate that cries FAIL when it merely could not tell teaches
            # everyone to ignore its red, which costs more than the lane is worth.
            skipped += 1
            print("  %-9s SKIP  (%.2f s) -- %s" % (name, dt, inconclusive))
        elif err or bad:
            fail += 1
            print("  %-9s FAIL  (%.2f s)" % (name, dt))
            for i, tb in sorted(err.items()):
                print("      thread %d died:\n%s" % (i, tb))
            for m in bad[:6]:
                print("      %s" % m)
            if len(bad) > 6:
                print("      ... and %d more" % (len(bad) - 6))
        else:
            print("  %-9s PASS  (%.2f s)" % (name, dt))
    tail = "" if not skipped else " (%d lane(s) SKIPPED, see above)" % skipped
    print("STRESS %s%s" % ("PASS" if not fail else "FAIL on %d lane(s)" % fail, tail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
