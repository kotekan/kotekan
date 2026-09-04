"""The 07:05 incident's regression test: five threads share one JointReceiverState.

2026-08-18 07:05:17 on sky: a thread switch inside a structure mutation (x and P are
resized in adjacent statements) desynced the shared filter by one row during a C-sat
rejection storm; every subsequent birth appended into inconsistent geometry (112 vs 111
growing to 771) and the joint state was silently dead for three hours behind per-consumer
guards. The fix is the instance lock (_locked on every public entry point).

Both arms run here, per [[chord-a-gate-that-cannot-fail]]:
  - LOCKED (the fix): geometry must stay consistent under a hostile switch interval.
  - UNLOCKED (the control): the same workload with the lock stubbed out must reproduce
    the desync -- if it cannot, this test is not exercising the race and a green LOCKED
    arm proves nothing.

Run: python3 -m unittest test_joint_thread_safety
"""
import contextlib
import sys
import threading
import unittest

from gnss_broker.state_filter import JointReceiverState

F_L5 = 1176.45e6


class _NoLock:
    """Stands in for the RLock in the control arm."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _hammer(js, n_iters, stop_on_desync):
    """The five-chain workload, compressed: concurrent code births/updates (with a tiny
    max_age so _drop fires constantly), carrier rows, gauges, predicts, readouts."""
    desynced = threading.Event()

    def code(tid):
        t = 0.0
        for i in range(n_iters):
            if stop_on_desync and desynced.is_set():
                return
            prn = (i * 7 + tid) % 40
            try:
                js.update(("G", prn), 150.0 + 0.1 * prn, 0.3, t)
                js.gauge()
            except Exception:
                desynced.set()
                return
            t += 0.01

    def carrier(tid):
        t = 0.0
        for i in range(n_iters):
            if stop_on_desync and desynced.is_set():
                return
            prn = (i * 5 + tid) % 60
            try:
                js.update_rrate(("C", prn), 0.1 * ((i + tid) % 7 - 3), t, F_L5,
                                sigma_hz=0.2)
                js.gauge_rrate()
            except Exception:
                desynced.set()
                return
            t += 0.01

    def reader():
        for i in range(n_iters):
            if stop_on_desync and desynced.is_set():
                return
            try:
                js.predict(0.01 * i)
                js.summary(0.01 * i)
                js.rrate(("C", i % 60))
            except Exception:
                desynced.set()
                return

    threads = ([threading.Thread(target=code, args=(k,)) for k in range(2)]
               + [threading.Thread(target=carrier, args=(k,)) for k in range(2)]
               + [threading.Thread(target=reader)])
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return desynced.is_set() or js.x.size != js.P.shape[0]


class TestSharedFilterThreadSafety(unittest.TestCase):
    def setUp(self):
        self._old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)   # hostile scheduler: maximize interleavings

    def tearDown(self):
        sys.setswitchinterval(self._old_interval)

    def _mk(self):
        # tiny max_age keeps _drop() firing, which is the raciest mutation
        return JointReceiverState(code_len=204600.0, ref_band="L5", clk0=150.0,
                                  max_age_s=0.05)

    def test_locked_filter_survives_concurrent_mutation(self):
        js = self._mk()
        broken = _hammer(js, n_iters=400, stop_on_desync=False)
        self.assertFalse(broken, "x=%d P=%s desynced UNDER THE LOCK"
                                 % (js.x.size, js.P.shape))
        self.assertEqual(js.x.size, js.P.shape[0])

    def test_control_unlocked_filter_reproduces_the_incident(self):
        """The gate's teeth: with the lock stubbed out, the same workload must desync.
        Several rounds because a race is a probability, not a schedule."""
        for attempt in range(6):
            js = self._mk()
            js._lk = _NoLock()
            if _hammer(js, n_iters=400, stop_on_desync=True):
                return
        self.fail("unlocked control never desynced in 6 rounds -- the workload no "
                  "longer exercises the race and the locked arm's pass is decorative")


if __name__ == "__main__":
    unittest.main()
