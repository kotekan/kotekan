#!/usr/bin/env python3
"""Concurrency stress for the broker's shared state that the GIL protects by accident.

The broker_equiv digests and the fleetdll_gate legs exercise no concurrency, so a race passes
all of them. This is the gate for that, and it is deliberately not a longer broker run: a long
run over a racy broker still returns a plausible number.

Three lanes, each an invariant that a lock protects and nothing else does:

  RECEIVER    N chain threads contribute and read clocks and biases at once. Each value read
              back must be whole: a code bias carries its writing thread in its value, and its
              src and weight (n_sats, the chain's index + 1) must name that thread's chain; a
              dead-reckon clock must carry weight 1, the 10230-chip code length and chips
              inside it. `--selftest` installs a code-bias writer that stores a wrong n_sats
              and requires this lane to fail.

  JOINT       N threads drive one JointReceiverState, as the broker's chains do
              (Receiver.joint_receiver shares one state across the receiver). Membership
              churns while other threads read: a diagnostic that indexed P while another
              thread resized it would raise IndexError and kill the chain thread. Asserts the
              state stays square and self-consistent, and that no reader raises.

  NOTES       drain_notes against concurrent writers. Notes drained + notes still queued must
              equal notes written. `--selftest` installs an unlocked read-then-rebind
              drain_notes and requires this lane to fail, which shows the lane can detect the
              race.

              Only the free-threaded interpreter shows it: under the GIL the read-then-rebind
              is effectively atomic and the selftest reports inconclusive.

It is a race detector, so a pass is evidence, not proof. It runs a fixed number of iterations,
not a duration: run it again rather than longer, and run it on both interpreters; the GIL run
is the control that shows the lanes are wired up at all.

    venv/bin/python    scripts/gnss/ft_stress.py
    venv-ft/bin/python scripts/gnss/ft_stress.py
"""
import argparse
import os
import random
import sys
import sysconfig
import threading
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
        "scripts",
        "gnss",
    ),
)

from gnss_broker import receiver as rx_mod  # noqa: E402
from gnss_broker import state_filter  # noqa: E402

CHAINS = ["gps_l5", "gal_e5a", "bds_b2a", "gal_e5b", "bds_b2b"]


def _run(fns, name):
    """Start every thread, join them all, and return {thread index: traceback} for each one
    that raised: a thread that died silently would read as a pass."""
    err = {}

    def wrap(i, fn):
        try:
            fn(i)
        except BaseException:
            import traceback

            err[i] = traceback.format_exc()

    ts = [
        threading.Thread(target=wrap, args=(i, f), name="%s%d" % (name, i))
        for i, f in enumerate(fns)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return err


def lane_receiver(iters, nthread):
    """Contribute and consume against one Receiver from every chain at once."""
    rx = rx_mod.Receiver(log=lambda *a, **k: None)
    # Every code-bias write passes n_sats = the chain's index + 1 and a value of
    # (k * nthread + i) * 1e-9, so the value names the writing thread i and a read whose src
    # or weight (n_sats) names another chain is torn. dr_clock values are checked per store:
    # their weight is always 1, so the code-bias rule would misfire on them.
    bad = []

    def worker(i):
        chain = CHAINS[i % len(CHAINS)]
        n = (i % len(CHAINS)) + 1
        rng = random.Random(1000 + i)
        for k in range(iters):
            t = 1_000_000.0 + k * 0.1
            hz = float(k * nthread + i)
            rx.contribute_code_bias(chain, "1176.45", hz * 1e-9, n, t)
            rx.contribute_dr_clock(
                chain, "1176.45", float(k % 10230), 0.001, t, 10230.0
            )
            for got in (
                rx.code_bias("1176.45", t_now=t),
                rx.code_bias_any_band(t_now=t),
            ):
                if got is None:
                    continue
                writer = CHAINS[round(got.value * 1e9) % nthread % len(CHAINS)]
                if got.src != writer or got.weight != CHAINS.index(writer) + 1:
                    bad.append(
                        "torn code bias: written by %s, src %r, n_sats %r (its own is %d)"
                        % (writer, got.src, got.weight, CHAINS.index(writer) + 1)
                    )
            for got in (
                rx.dr_clock("1176.45", t_now=t),
                rx.dr_clock_any_band(t_now=t),
            ):
                if got is None:
                    continue
                cl = got.extra.get("code_length")
                if (
                    got.src not in CHAINS
                    or got.weight != 1
                    or cl != 10230.0
                    or not 0 <= got.value < 10230
                ):
                    bad.append(
                        "torn dr clock: src %r, weight %r, code_length %r, chips %r"
                        % (got.src, got.weight, cl, got.value)
                    )
            if rng.random() < 0.05:
                rx.summary()

    err = _run([worker] * nthread, "rx")
    return err, bad, None


def lane_joint(iters, nthread):
    """One JointReceiverState, many chains, as Receiver.joint_receiver shares it."""
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
                    js.update(key, rng.gauss(0.0, 0.4), 0.3, t)
                except Exception as e:
                    bad.append("update raised %r" % (e,))
            try:
                js.predict(t)
                js.gauge()
                # clk and clk_rate are unlocked properties; bias, sigma and age index by
                # membership while other threads change it.
                js.clk
                js.clk_rate
                for prn in prns:
                    js.bias(("gps", prn))
                    js.sigma(("gps", prn))
                    js.age(("gps", prn), t)
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
    """drain_notes against concurrent writers: no note may be lost or duplicated."""
    js = state_filter.JointReceiverState(code_len=10230.0)
    drained = []
    lock = threading.Lock()
    written = iters * nthread

    def worker(i):
        local = []
        for k in range(iters):
            # Under js._lk, as in production: `_note` is private and every call site runs
            # inside an @_locked method. An unlocked append would race the drainer in a way no
            # production path can, and report lost notes the broker cannot lose.
            with js._lk:
                js._note("t%d-%d" % (i, k))
            if k % 4 == 3:
                local += js.drain_notes()
        local += js.drain_notes()
        with lock:
            drained.extend(local)

    err = _run([worker] * nthread, "notes")
    # `_note` keeps only the newest 200 notes, so the count invariant holds only while the
    # queue cannot overflow. What bounds the queue is the pile-up between drains, not the
    # total written: every thread drains each 4th iteration, so the queue holds about 4 notes
    # per thread; in_flight allows 5.
    left = len(js.drain_notes())
    bad, inconclusive = [], None
    in_flight = nthread * 5
    if in_flight >= 200:
        inconclusive = (
            "%d threads can queue ~%d notes between drains, against the 200-deep "
            "trim in _note -- drops would be legitimate. Use fewer threads."
            % (nthread, in_flight)
        )
    elif len(drained) + left != written:
        # Either sign is the same defect. An unlocked read-then-rebind lets two drainers take
        # the same list object and both return it, so the usual symptom is duplication
        # (drained > written): an operator reading the log sees a note twice.
        d = len(drained) + left - written
        bad.append(
            "MISCOUNT %+d note(s) (%s): wrote %d, drained %d, queued %d"
            % (d, "duplicated" if d > 0 else "lost", written, len(drained), left)
        )
    return err, bad, inconclusive


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument(
        "--lane",
        action="append",
        default=[],
        choices=["receiver", "joint", "notes"],
        help="run only this lane; repeat for several (default: all three)",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="install a code-bias writer that stores a wrong n_sats and require the "
        "RECEIVER lane to fail, then an unlocked read-then-rebind drain_notes and "
        "require the NOTES lane to fail: a race detector that has never been seen "
        "detecting is indistinguishable from a no-op. Only the free-threaded "
        "interpreter can show the NOTES race; under the GIL that part reports "
        "inconclusive.",
    )
    a = ap.parse_args()

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()

    if a.selftest:
        # A code-bias writer that stores a wrong n_sats. Not a race, so the RECEIVER lane
        # must flag it on either interpreter.
        contribute = rx_mod.Receiver.contribute_code_bias

        def wrong_n_sats(self, chain, band, la, n_sats, t):
            contribute(self, chain, band, la, n_sats + 1, t)

        rx_mod.Receiver.contribute_code_bias = wrong_n_sats
        _err, bad, _inc = lane_receiver(a.iters, a.threads)
        rx_mod.Receiver.contribute_code_bias = contribute
        if not bad:
            print(
                "SELFTEST FAIL -- the RECEIVER lane passed a code bias carrying a wrong "
                "n_sats, so a pass from it means nothing"
            )
            return 1
        print("SELFTEST: wrong n_sats detected by the RECEIVER lane: %s" % bad[0])

        # drain_notes without its lock: read-then-rebind, so two chains draining at once can
        # take the same list.
        def unlocked_drain(self):
            out, self.notes = self.notes, []
            return out

        state_filter.JointReceiverState.drain_notes = unlocked_drain
        print("SELFTEST: unlocked drain_notes installed; the NOTES lane must now FAIL")
        for attempt in range(1, 21):
            err, bad, _inc = lane_notes(a.iters, a.threads)
            if err or bad:
                print(
                    "  detected on attempt %d: %s"
                    % (attempt, (bad or ["thread died"])[0])
                )
                print(
                    "SELFTEST PASS -- the lane can fail, so a pass from it means something"
                )
                return 0
        print("  20 attempts, nothing detected")
        if gil:
            # Not a harness fault: the GIL makes the read-then-rebind effectively atomic, so
            # the race cannot show here. The selftest means something only free-threaded.
            # Inconclusive passes only on a GIL build. On a free-threaded build the GIL was
            # re-enabled (PYTHON_GIL=1, or an extension that does not declare support), and
            # GNSS_REQUIRE_FT (the test runner sets it) means this run had to be the real one.
            if sysconfig.get_config_var("Py_GIL_DISABLED"):
                print(
                    "SELFTEST INCONCLUSIVE -- FAIL: a free-threaded build running with the "
                    "GIL enabled, so the race cannot show"
                )
                return 1
            if os.environ.get("GNSS_REQUIRE_FT"):
                print(
                    "SELFTEST INCONCLUSIVE -- FAIL: GNSS_REQUIRE_FT is set, and this "
                    "interpreter has a GIL"
                )
                return 1
            print(
                "SELFTEST INCONCLUSIVE UNDER THE GIL -- expected. The GIL hides this "
                "defect. Re-run free-threaded."
            )
            return 0
        print(
            "SELFTEST FAIL -- free-threaded and still blind: the NOTES lane cannot see "
            "the defect it exists to catch, so a pass from it means nothing"
        )
        return 1

    print(
        "%s  GIL %s  threads=%d iters=%d"
        % (sys.version.split()[0], "ON" if gil else "OFF", a.threads, a.iters)
    )

    lanes = [("RECEIVER", lane_receiver), ("JOINT", lane_joint), ("NOTES", lane_notes)]
    if a.lane:
        lanes = [l for l in lanes if l[0].lower() in a.lane]

    fail = skipped = 0
    for name, fn in lanes:
        t0 = time.perf_counter()
        err, bad, inconclusive = fn(a.iters, a.threads)
        dt = time.perf_counter() - t0
        if inconclusive and not (err or bad):
            # Reported as SKIP, not FAIL: a lane that could not tell must not turn the gate
            # red, or its red stops meaning a detected defect.
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
