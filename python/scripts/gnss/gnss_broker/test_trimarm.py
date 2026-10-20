"""The fast loop's arming policy: what the gather is told it may actuate.

    python3 -m gnss_broker.test_trimarm     (or: pytest gnss_broker/test_trimarm.py)

`stage_fleet_trim_arming` is the only place the C++ fast loop's authority is set. The stage
POSTs, so `_post` is captured rather than sent, and this file asserts on the policy document
the gather would receive: which PRNs are armed (presence with a hold), the loop constants, and
the probe-anchored floor.

@author Keith Vanderlinde
"""

import sys
import time

from gnss_broker import combdll, trimarm


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def new_dls(hold=None):
    return _Obj(
        hold=dict(hold or {}),
        armed_last=set(),
        readback={},
        stat={"posts": 0, "fail": 0, "last_err": ""},
        trim={},
        held_off=set(),
        readback_ok=False,
        lobe_since={},
        lobe_release={},
        lobe_done={},
    )


def run(
    present=(1, 8, 10),
    hold=None,
    fleet=None,
    t0=None,
    dls=None,
    dr_state=None,
    chip_rate_hz=10.23e6,
):
    """Drive the stage once at cycle clock t0 (default: now); return (policy dict the gather
    would receive, ctx). Pass `dls` to carry the loop state across cycles."""
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
            args=_Obj(
                fleet_trim_url="http://127.0.0.1:12051/fleet_trim",
                fleet_trim_hold_s=30.0,
                fleet_trim_bandwidth=2.5,
                fleet_trim_leak_per_s=0.5,
                dll_spacing=0.5,
                fleet_trim_readback=0,
                signal="gps_l5",
                chip_rate_hz=chip_rate_hz,
            ),
            dllp=_Obj(
                fleet=fleet
                if fleet is not None
                else {p: {"present": True} for p in present}
            ),
            dls=dls if dls is not None else new_dls(hold),
            dr_state=dr_state,
            telem_chain="gps_l5",
            trackers=["http://cx19:12048/gnss0_inject"],
            t0=time.time() if t0 is None else t0,
        )
        trimarm.stage_fleet_trim_arming(ctx)
    finally:
        trimarm._post, trimarm._get = real_post, real_get
    return sent.get("doc", {}).get("chains", {}).get("gps_l5", {}), ctx


def main():
    print("\nthe arming policy")

    # 1. Present PRNs are armed, with the loop's own constants, and the armed set is
    #    recorded for the Python integrator to stand down against.
    pol, ctx = run()
    check(sorted(pol.get("armed") or []) == [1, 8, 10], "present PRNs 1, 8, 10 armed")
    check(
        pol.get("gain_per_s") == 2.5 and pol.get("leak_per_s") == 0.5,
        "gain/leak are the fleet-trim bandwidth and leak, not the slow DLL's constants",
    )
    check(
        pol.get("targets") == ["http://cx19:12048/gnss0_inject/set_trim"],
        "the targets are the trackers' /set_trim",
    )
    check(ctx.dls.armed_last == {1, 8, 10}, "armed_last records what was handed over")
    check(ctx.dls.stat["posts"] == 1, "one policy POST counted")

    # 2. Presence with a hold: a PRN absent this cycle stays armed inside the hold and is
    #    released once it has been absent longer than fleet_trim_hold_s.
    now = time.time()
    pol, _ = run(present=(1,), hold={8: now - 10.0, 10: now - 100.0}, t0=now)
    check(
        sorted(pol.get("armed") or []) == [1, 8],
        "absent 10 s: still armed; absent 100 s (> 30 s hold): released",
    )

    # 2b. The hold runs on the cycle clock, not the wall clock: with the wall clock stopped
    #     (a replay running fast) and away from the cycle clock, PRN 8 absent for 100 s of
    #     cycle time is released. The stamp and the age must both be on the cycle clock.
    real_time = time.time
    time.time = lambda: 9000.0
    try:
        pol, ctx = run(present=(1, 8), t0=5000.0)
        armed_then = sorted(pol.get("armed") or [])
        pol, _ = run(present=(1,), hold=ctx.dls.hold, t0=5100.0)
    finally:
        time.time = real_time
    check(
        armed_then == [1, 8] and sorted(pol.get("armed") or []) == [1],
        "absent 100 s of cycle time, 0 s of wall time: released (armed %r -> %r)"
        % (armed_then, sorted(pol.get("armed") or [])),
    )

    # 3. Only a probe-anchored floor ships. A peer-bar floor would rebuild the competition
    #    the absolute floor removes, so anything else ships 0 (the C++ loop then steps none).
    pol, _ = run(
        fleet={1: {"present": True, "p_floor_src": "probes:3", "p_floor": 0.02}}
    )
    check(pol.get("p_floor_abs") == 0.02, "probe-anchored floor ships as p_floor_abs")
    pol, _ = run(fleet={1: {"present": True, "p_floor_src": "peers", "p_floor": 0.02}})
    check(pol.get("p_floor_abs") == 0.0, "a peer-bar floor ships 0")

    # 4. The clamp stays under half the tracker comb's grating lobe (320 ns): 1.47 chips at
    #    10.23 Mcps, 0.74 at E6's 5.115. L2C's lobe (0.16 chip) lies inside its code's own peak,
    #    so the clamp keeps the full 3 chips.
    clamps = [run(chip_rate_hz=r)[0].get("clamp") for r in (10.23e6, 5.115e6, 511.5e3)]
    check(clamps == [1.473, 0.737, 3.0], "clamp by chip rate: %r" % clamps)

    # 5. A slewing seed is not trimmed: PRN 8 is 12 chips from its model, PRN 10 is 0.3 (an
    #    ordinary step), PRN 1's 9-chip gap was stamped 100 s ago and has gone stale (it left
    #    dead reckoning). 8 is left out of the armed set and its Python trim is dropped.
    d = new_dls()
    d.trim = {8: 1.1, 10: 0.2}
    now = 5000.0
    gaps = {8: (now, 12.0), 10: (now, 0.3), 1: (now - 100.0, 9.0)}
    pol, ctx = run(dls=d, dr_state={"slew_gap": gaps}, t0=now)
    check(
        sorted(pol.get("armed") or []) == [1, 10] and ctx.dls.held_off == {8},
        "slewing PRN 8 held off, stale gap ignored (armed %r)"
        % sorted(pol.get("armed") or []),
    )
    check(ctx.dls.trim == {10: 0.2}, "the held-off PRN's Python trim is dropped")

    # 6. The lobe referee. PRN 4 reads xcoh 0.02 over 128 records with its fast trim at 3.0
    #    (past a quarter lobe), PRN 7 is on its peak (0.97), PRN 9 reads low on 3 records only,
    #    PRN 5 reads low with a 0.3-chip trim: a low xcoh the trim did not make. 4 is released
    #    after 30 s, stays released while its trim reads 2.0, is re-armed once it reads ~0, and
    #    is not released again inside the 120-s holdoff.
    fl = {
        4: {"present": True, "xcoh": 0.02, "n_xcoh": 128},
        5: {"present": True, "xcoh": 0.02, "n_xcoh": 128},
        7: {"present": True, "xcoh": 0.97, "n_xcoh": 128},
        9: {"present": True, "xcoh": 0.05, "n_xcoh": 3},
    }
    d = new_dls()
    d.readback_ok = True
    d.readback = {
        4: {"trim_chips": 3.0},
        5: {"trim_chips": 0.3},
        7: {"trim_chips": 0.2},
    }
    seq = []
    for t, rb4 in ((1000.0, 3.0), (1031.0, 3.0), (1033.0, 2.0), (1035.0, 0.01)):
        d.readback[4] = {"trim_chips": rb4}
        pol, ctx = run(fleet=fl, dls=d, t0=t)
        seq.append(4 in (pol.get("armed") or []))
    check(
        seq == [True, False, False, True],
        "PRN 4 armed/released/released/re-armed: %r" % seq,
    )
    check(9 in (pol.get("armed") or []), "PRN 9 (3 records) is never judged")
    check(7 in (pol.get("armed") or []), "PRN 7 on its peak stays armed throughout")
    check(
        5 not in d.lobe_done and 5 in (pol.get("armed") or []),
        "PRN 5 (small trim) kept",
    )
    d.readback[4] = {"trim_chips": 3.0}  # walked back onto the lobe
    held = []
    for t in (1080.0, 1120.0, 1160.0):
        pol, ctx = run(fleet=fl, dls=d, t0=t)
        held.append(4 in ctx.dls.held_off)
    check(
        held == [False, False, True],
        "holdoff: not released again until 120 s: %r" % held,
    )

    # 7. A slewing PRN is not judged by the referee: its xcoh is low because of the slew.
    d = new_dls()
    d.readback = {4: {"trim_chips": 3.0}}
    for t in (2000.0, 2040.0):
        pol, ctx = run(fleet=fl, dls=d, t0=t, dr_state={"slew_gap": {4: (t, 5.0)}})
    check(4 not in ctx.dls.lobe_release, "slewing PRN 4: no lobe release")

    # 8. The release ends on a read that succeeded and no longer lists the PRN (the gather
    #    erases a decayed trim's row), never on a failed read.
    def release_then(rb_ok):
        d = new_dls()
        d.readback_ok = True
        d.readback = {4: {"trim_chips": 3.0}}
        for t in (3000.0, 3031.0):
            run(fleet=fl, dls=d, t0=t)
        d.readback, d.readback_ok = {}, rb_ok
        pol, _ = run(fleet=fl, dls=d, t0=3033.0)
        return 4 in (pol.get("armed") or [])

    check(release_then(True), "empty successful read: trim erased, PRN 4 re-armed")
    check(not release_then(False), "failed read: still released")

    # 9. Presence flicker does not reset the timer: PRN 4 absent this cycle but held armed
    #    (present 16 s ago) is still judged and released at 30 s.
    d = new_dls()
    d.readback = {4: {"trim_chips": 3.0}}
    run(fleet=fl, dls=d, t0=4000.0)
    run(fleet=fl, dls=d, t0=4015.0)
    flick = dict(fl)
    flick[4] = {"present": False, "xcoh": 0.02, "n_xcoh": 128}
    run(fleet=flick, dls=d, t0=4020.0)
    pol, ctx = run(fleet=flick, dls=d, t0=4031.0)
    check(4 in ctx.dls.held_off, "absent but held: still judged, released at 30 s")

    # 10. No referee where the lobe lies inside the code's own peak (L2C, 0.16 chip).
    d = new_dls()
    d.readback = {4: {"trim_chips": 3.0}}
    for t in (6000.0, 6040.0):
        pol, ctx = run(fleet=fl, dls=d, t0=t, chip_rate_hz=511.5e3)
    check(not ctx.dls.lobe_release, "L2C: no lobe release")

    # 11. Production rows: fleet_dll_comb carries n_xcoh from the gather's taps, so the referee
    #     sees the record count it gates on.
    taps = {
        4: {
            "e": 1.0,
            "p": 4.0,
            "l": 1.0,
            "hop": 1,
            "n_inst": 10,
            "n_chan": 72.0,
            "n_rec": 128,
            "xcoh": 0.02,
            "n_xcoh": 128,
        }
    }
    rows = combdll.fleet_dll_comb(None, "gps_l5", taps_src=lambda ch, pr: taps)
    check(
        rows.get(4, {}).get("n_xcoh") == 128,
        "fleet_dll_comb row carries n_xcoh (got %r)" % rows.get(4, {}).get("n_xcoh"),
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the fast loop's arming policy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
