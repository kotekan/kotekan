#!/usr/bin/env python3
"""The instance liveness verdict (fits.instance_stall_verdict) against wedged instances.

    python3 python/scripts/gnss/test_instance_stall.py

The fixture: four of twelve instances (cx42/gnss0, cx43/gnss0, cx44/gnss1, cx51/gnss0) with
frozen DPDK capture windows, each dropping its whole stream while serving plausible rows to
every poll. Healthy instances advance ~5.9M hops per 30 s; a wedged one advances exactly 0.
Both answer 200.

The third arm is why a per-instance check exists: fe_axis_stale_s watches one hop, the
--combiner instance's, and a healthy instance's hop keeps climbing while four of twelve are
frozen. That guard answers a different question; it cannot see this.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker.fits import instance_stall_verdict  # noqa: E402

HEALTHY_PER_S = 5_900_000.0 / 30.0  # ~one hop per 5.12 us
WEDGED = ["cx42:gnss0", "cx43:gnss0", "cx44:gnss1", "cx51:gnss0"]
ALL = ["cx%d:gnss%d" % (n, g) for n in (19, 27, 42, 43, 44, 51) for g in (0, 1)]
CYCLE = 13.0  # s between verdicts
BAR = 90.0  # instance_stall_s (gnss_broker/cli.py _FROZEN)


def run(n_cycles, wedged=WEDGED, t_start=1000.0):
    """Walk the fleet forward, returning (accusations_per_cycle, hop_state)."""
    prev, hist = {}, []
    hops = {u: 600_000_000.0 for u in ALL}
    t = t_start
    for _ in range(n_cycles):
        for u in ALL:
            if u not in wedged:
                hops[u] += HEALTHY_PER_S * CYCLE
        prev, stalled = instance_stall_verdict(prev, dict(hops), t, BAR)
        hist.append(sorted(u for u, _, _ in stalled))
        t += CYCLE
    return hist, hops


def main():
    fails = []

    # 1. Four wedged, eight healthy.
    hist, _ = run(12)
    final = hist[-1]
    if final != sorted(WEDGED):
        fails.append("expected exactly %s stalled, got %s" % (sorted(WEDGED), final))
    else:
        print("ok  4 of 12 wedged -> named exactly, and only, those four")

    # ...and not before the bar: 90 s is ~7 cycles, and no cycle inside it may accuse.
    early = [i for i, h in enumerate(hist) if h and (i + 1) * CYCLE < BAR]
    if early:
        fails.append("accused before the %.0f s bar, at cycles %s" % (BAR, early))
    else:
        print("ok  silent until the bar (no accusation inside %.0f s)" % BAR)

    # 2. The control: a healthy fleet must produce no accusation, or the guard means nothing.
    hist_ok, _ = run(12, wedged=[])
    if any(hist_ok):
        fails.append("accused a healthy fleet: %s" % [h for h in hist_ok if h])
    else:
        print("ok  healthy fleet -> zero accusations (the guard can be silent)")

    # 3. Why fe_axis_stale_s cannot do this: it watches one hop (the --combiner instance's).
    #    In the same wedged fleet a healthy instance's hop, here the fleet maximum, climbs at
    #    the full rate throughout.
    _, hops = run(12)
    hi = max(hops.values())
    if hi <= 600_000_000.0:
        fails.append("fixture is wrong -- the fleet maximum did not advance")
    else:
        print(
            "ok  the fleet MAXIMUM advanced %.1fM hops while 4 instances sat frozen"
            % ((hi - 600_000_000.0) / 1e6)
        )
        print(
            "    -> fe_axis_stale_s watches one hop (the --combiner's), which climbs "
            "like this one, so it is silent here;"
        )
        print("       the per-instance axis is the one it cannot resolve.")

    # 4. The control clause: if most of the fleet stops, the cause is global (paused
    #    F-engine, replay, clock step) and blaming instances would misdirect the diagnosis.
    hist_all, _ = run(12, wedged=ALL)
    if any(hist_all):
        fails.append(
            "accused instances during a FLEET-WIDE stop: %s"
            % [h for h in hist_all if h]
        )
    else:
        print("ok  whole fleet frozen -> says nothing (global, not per-instance)")

    # 5. Unreachable is not stalled. An instance that drops out of the poll is a different
    #    fault, already visible as n_src falling. It must never be accused of stalling.
    prev = {}
    t = 1000.0
    hops = {u: 600_000_000.0 for u in ALL}
    for i in range(12):
        for u in ALL:
            hops[u] += HEALTHY_PER_S * CYCLE
        cur = {
            u: h for u, h in hops.items() if u != "cx27:gnss1"
        }  # one vanishes entirely
        prev, stalled = instance_stall_verdict(prev, cur, t, BAR)
        t += CYCLE
    if stalled:
        fails.append("accused an UNREACHABLE instance: %s" % stalled)
    elif "cx27:gnss1" in prev:
        fails.append(
            "kept an unreachable instance in the state (it will look stuck forever)"
        )
    else:
        print("ok  unreachable instance -> dropped from state, never accused")

    # 6. Recovery: once a wedged instance starts advancing, the accusation must clear.
    prev, t = {}, 1000.0
    hops = {u: 600_000_000.0 for u in ALL}
    for i in range(20):
        for u in ALL:
            if u not in WEDGED or i >= 12:  # the node is restarted at cycle 12
                hops[u] += HEALTHY_PER_S * CYCLE
        prev, stalled = instance_stall_verdict(prev, dict(hops), t, BAR)
        t += CYCLE
    if stalled:
        fails.append("still accusing after recovery: %s" % stalled)
    else:
        print("ok  recovery clears the accusation (it is a state, not a latch)")

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 6 arms, including the one-hop blind spot and the global control")
    return 0


if __name__ == "__main__":
    sys.exit(main())
