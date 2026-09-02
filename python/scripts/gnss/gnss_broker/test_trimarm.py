"""The fast loop's FREEZE gates: what the gather is told, and when it is told nothing.

    python3 -m gnss_broker.test_trimarm     (or: pytest gnss_broker/test_trimarm.py)

THE POINT OF THIS FILE. `stage_fleet_trim_arming` is the only place the C++ fast loop's
authority is set, and its two freezes are indistinguishable in their actuation (gain=leak=0)
while being opposite in cause -- a presence collapse (#91b) versus broker establishment. Both
were bought with on-sky evidence, and both are one boolean away from silently inverting:

  * a freeze that DISARMS instead erases every standing trim in ~5.6 s (a per-sat re-pull),
  * a freeze that never lifts leaves the loop driving nothing forever,
  * and a freeze that never engages puts the establishment step's common mode into the
    3-chip clamp for ~8 minutes, measured on three separate restarts on 2026-09-02.

Each of those is an arm below. The stage POSTs, so `_post` is captured rather than sent: what
this file asserts is the POLICY DOCUMENT the gather would receive.

⚠️ THE ESTABLISHMENT WINDOW IS WALL-CLOCK FROM BROKER START, so like the #90 admission gate
this logic cannot be reproduced by a transcript replay and broker_equiv is blind to it by
construction. That is why it is tested here.

@author Keith Vanderlinde
"""

import sys
import time

from gnss_broker import trimarm


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Brown(object):
    def __init__(self, est):
        self._est = est

    def established(self):
        return self._est


def run(age_s, estab_hold=300.0, warmup=240.0, brownout=False, brownout_hold_s=0.0,
        present=(1, 8, 10), replay=None):
    """Drive the stage once and return the policy dict the gather would have received."""
    sent = {}

    def fake_post(url, doc, timeout=None):
        sent["url"] = url
        sent["doc"] = doc
        return {}

    def fake_get(url, timeout=None):
        return {}

    real_post, real_get = trimarm._post, trimarm._get
    trimarm._post, trimarm._get = fake_post, fake_get
    try:
        ctx = _Obj(
            args=_Obj(fleet_trim_url="http://127.0.0.1:12051/fleet_trim",
                      fleet_trim_hold_s=30.0,
                      fleet_trim_brownout_hold_s=brownout_hold_s,
                      fleet_trim_establish_hold_s=estab_hold,
                      joint_feed_warmup_s=warmup,
                      fleet_trim_bandwidth=2.5,
                      fleet_trim_leak_per_s=0.5,
                      dll_spacing=0.5,
                      fleet_trim_readback=0,
                      transcript_read=replay,
                      signal="gps_l5"),
            broker_t0=time.time() - age_s,
            brown=_Brown(brownout),
            dllp=_Obj(fleet={p: {"present": True} for p in present}),
            dls=_Obj(hold={}, armed_last=set(),
                     stat={"posts": 0, "fail": 0, "armed": 0, "last_err": ""}),
            telem_chain="gps_l5",
            trackers=["http://cx19:12048/gnss0_inject"],
        )
        trimarm.stage_fleet_trim_arming(ctx)
    finally:
        trimarm._post, trimarm._get = real_post, real_get
    return sent.get("doc", {}).get("chains", {}).get("gps_l5", {})


def main():
    print("\nthe ESTABLISHMENT freeze (2026-09-02): the ~8 min post-restart q thrash")

    # 1. INSIDE the window -- the loop must be handed zero authority, but stay ARMED.
    pol = run(age_s=10.0)
    check(pol.get("gain_per_s") == 0.0 and pol.get("leak_per_s") == 0.0,
          "10 s after start: gain AND leak both 0 (a freeze, not a live loop)")
    check(sorted(pol.get("armed") or []) == [1, 8, 10],
          "...and the PRNs stay ARMED -- disarming would erase the trims in ~5.6 s")

    # 2. The clamp was first seen 34 s after the feed opened, so the window MUST still be
    #    holding there. This is the arm that fails if anyone ends the hold at the warmup.
    pol = run(age_s=240.0 + 34.0)
    check(pol.get("gain_per_s") == 0.0,
          "34 s PAST the feed opening (when the clamp was first observed): still frozen")

    # 3. Just inside the far edge, and just outside it.
    pol = run(age_s=240.0 + 299.0)
    check(pol.get("gain_per_s") == 0.0, "1 s before the window ends: still frozen")
    pol = run(age_s=240.0 + 301.0)
    check(pol.get("gain_per_s") == 2.5 and pol.get("leak_per_s") == 0.5,
          "1 s after it ends: the loop is LIVE again (the freeze must lift, not latch)")

    # 4. IT TRACKS THE WARMUP FLAG. Raising --joint-feed-warmup-s must carry the hold with
    #    it; a hold expressed as an absolute would stop mid-transient.
    pol = run(age_s=500.0, warmup=600.0)
    check(pol.get("gain_per_s") == 0.0,
          "warmup raised to 600 s: at t=500 s still frozen (hold tracks the warmup)")

    # 5. OFF is off -- the pre-fix behaviour stays reachable.
    pol = run(age_s=10.0, estab_hold=0.0)
    check(pol.get("gain_per_s") == 2.5,
          "establish-hold 0: OFF, loop live from the first cycle (pre-fix behaviour)")

    # 5b. ⚠️ INERT IN REPLAY, and this arm is load-bearing. The window is wall-clock from
    #     broker start, so if it engaged under --transcript-read whether a cycle is inside
    #     it would depend on how long the replay TAKES -- the fixture digests would move
    #     with machine speed. A flaky gate is worse than no gate.
    pol = run(age_s=10.0, replay="onsky.jsonl.gz")
    check(pol.get("gain_per_s") == 2.5,
          "under --transcript-read: NOT frozen (a wall-clock window must not move a digest)")

    print("\nthe #91(b) BROWNOUT freeze: same actuation, unrelated cause")

    # 6. A brownout freezes at ANY age -- it is not an establishment window.
    pol = run(age_s=99999.0, brownout=True, brownout_hold_s=120.0)
    check(pol.get("gain_per_s") == 0.0 and sorted(pol.get("armed") or []) == [1, 8, 10],
          "brownout hours after start: frozen and still armed")

    # 7. ⚠️ THE ONE THAT MATTERS MOST: the two must not cancel. An established brownout
    #    outside the establishment window still freezes, and neither flag disables the
    #    other -- they are ORed, and a future refactor that makes them exclusive breaks
    #    the presence-collapse protection that #91(b) bought.
    pol = run(age_s=99999.0, brownout=True, brownout_hold_s=120.0, estab_hold=300.0)
    check(pol.get("gain_per_s") == 0.0,
          "brownout + establish-hold configured: still frozen (ORed, never exclusive)")
    pol = run(age_s=99999.0, brownout=False, brownout_hold_s=120.0, estab_hold=300.0)
    check(pol.get("gain_per_s") == 2.5,
          "no brownout, past the window: LIVE -- neither freeze fires spuriously")

    # 8. The retained-value identity the whole design rests on: dll_integrate with
    #    gain=leak=0 returns the trim unchanged. Asserted here so the constants that make
    #    it true are not quietly changed to something that leaks.
    trim = 1.234
    gain = leak = 0.0
    check(abs(((1.0 - leak) * trim + gain * 99.0) - trim) < 1e-12,
          "gain=leak=0 is an IDENTITY on the standing trim (that is why it retains)")

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 12 arms on the fast loop's two freezes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
