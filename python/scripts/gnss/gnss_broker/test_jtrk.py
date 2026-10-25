"""The shadow joint state's tracker feed (gnss_broker/jtrk.py, #191 step 2).

    python3 -m gnss_broker.test_jtrk

Each per-satellite gate refuses on its own, a passing residual goes in converted to reference
chips, nothing falls back, the per-cycle holds (warm-up, transit, thin set) feed nothing, a
record already fed is not fed again, and the shadow is a state of its own beside the live one.
"""

import argparse
import sys
import types

from gnss_broker import jtrk, sky
from gnss_broker.receiver import JOINT_STATE, TRK_STATE, Receiver

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def row(raw, n=20, hop=1000):
    return {"raw": raw, "chips": raw - 150.0, "n": n, "hop": hop, "sd": 0.01}


def fl(q, floor=1.1):
    return {"q": q, "q_floor": floor, "present": True}


def test_gates():
    trk = {
        1: row(150.2),
        2: row(150.3),
        3: row(150.4, n=2),  # too few records
        4: row(150.5),  # q under the floor
        5: row(150.6),  # spectrum fit off the peak
        6: row(150.7, hop=900),  # not newer than the last fed
        7: row(150.8),  # the chain's track gate refuses it
        8: row(150.9),  # no fleet row: no q, so no feed
    }
    fleet = {p: fl(3.0) for p in (1, 2, 3, 5, 6, 7)}
    fleet[4] = fl(1.0)
    spec = {1: {"tau_chips": 0.05}, 5: {"tau_chips": -0.4}}
    rows, failed, hops = jtrk.feed_rows(
        trk, fleet, spec, lambda p: p != 7, {6: 900}, "E", 2.0, 0.1, 0.3, 4,
    )
    keys = [k for k, _y, _s in rows]
    check(
        keys == [("E", 1), ("E", 2)],
        "only the satellites passing every gate (%s)" % keys,
    )
    check(
        rows[0][1] == 300.4 and rows[0][2] == 0.1,
        "the residual goes in times k, at the reference sigma (%s)" % (rows[0],),
    )
    check(
        failed == {"track": 1, "q": 2, "tau": 1, "nrec": 1, "stale": 1},
        "each refusal counted once, under its first failing gate (%s)" % failed,
    )
    check(hops == {1: 1000, 2: 1000}, "the fed rows' hops are returned to record")
    rows2, failed2, _ = jtrk.feed_rows(
        {1: row(150.2)}, {1: fl(3.0)}, {}, lambda p: True, {}, "E", 1.0, 0.1, 0.3, 4
    )
    check(len(rows2) == 1, "no spectrum fit this cycle is not a refusal")


def test_two_states():
    rx = Receiver(log=lambda m: None)
    check(rx.joint_trk_snapshot(0.0) is None, "no snapshot before any chain feeds")
    live = rx.joint_receiver("1176", 10230, 10.23e6, gauge_mode="prior")
    trk = rx.joint_receiver("1176", 10230, 10.23e6, which=TRK_STATE, gauge_mode="prior")
    check(live is not trk, "the shadow is a state of its own")
    check(
        rx.joint_receiver("1176", 10230, 10.23e6) is live
        and rx.joint_receiver("1176", 10230, 10.23e6, which=JOINT_STATE) is live,
        "the default is still the live state",
    )
    e6 = rx.joint_receiver("1278", 5115, 5.115e6, which=TRK_STATE)
    check(
        e6 is trk and trk.L == 10230.0,
        "E6's 1-ms code shares the 10230-reference modulus",
    )
    for t in range(40):
        trk.cycle([(("E", p), 151.0 + 0.01 * p, 0.1) for p in (1, 2, 3, 4)], float(t))
        live.cycle([(("G", p), 150.8 + 0.01 * p, 0.3) for p in (1, 2, 3, 4)], float(t))
    snap = rx.joint_trk_snapshot(39.0)
    check(
        snap["n_sats"] == 4 and set(snap["sats"]) == {"E01", "E02", "E03", "E04"},
        "the snapshot names its satellites (%s)" % sorted(snap["sats"]),
    )
    check(
        abs(snap["clk_delta"] - trk.wrap(trk.clk - live.clk)) < 1e-3
        and abs(snap["clk_delta"]) < 1.0,
        "clk_delta is the shadow clock minus the live one (%+.3f)" % snap["clk_delta"],
    )


class _Pub(object):
    def __init__(self):
        self.calls = []

    def set_joint(self, snap, feed, t):
        self.calls.append((snap, feed, t))


def ctx_for(rx, trk, uptime=1000.0, t=100.0):
    a = argparse.Namespace(
        joint_feed_warmup_s=240.0,
        dr_clock_transit_freeze_deg=2.0,
        chip_rate_hz=10.23e6,
        joint_trk_sigma=0.1,
        joint_trk_spec_tau_max=0.3,
        joint_trk_min_records=4,
        joint_trk_min_sats=3,
        signal="gal_e5a",
    )
    return types.SimpleNamespace(
        args=a,
        uptime_s=uptime,
        dr_state={},
        dllp=types.SimpleNamespace(
            trk=trk, fleet={p: fl(3.0) for p in trk}, spec_fit={}
        ),
        track_ok=lambda p: True,
        drp=types.SimpleNamespace(tag="E", t_now_abs=t, pd=None, transit_pd=None),
        band_id="1176",
        code_len=10230,
        rx=rx,
        publisher=_Pub(),
        t0=t,
    )


def test_stage():
    real = sky.nearest_boresight
    try:
        sky.nearest_boresight = lambda pd: None
        rx = Receiver(log=lambda m: None)
        three = {p: row(151.0, hop=2000) for p in (1, 2, 3)}

        c = ctx_for(rx, three, uptime=100.0)
        jtrk.stage_joint_trk(c)
        held = c.dr_state["jtrk"]["last"]
        check(
            held["held"].startswith("warm-up") and held["accepted"] == 0,
            "nothing is fed during the warm-up (%s)" % held["held"],
        )

        c = ctx_for(rx, {1: row(151.0), 2: row(151.0)})
        jtrk.stage_joint_trk(c)
        check(
            c.dr_state["jtrk"]["last"]["held"].startswith("thin set")
            and c.dr_state["jtrk"]["thin"] == 1,
            "two satellites are a thin set: nothing fed",
        )

        c = ctx_for(rx, three)
        jtrk.stage_joint_trk(c)
        last = c.dr_state["jtrk"]["last"]
        check(
            last["held"] is None and last["accepted"] == 3,
            "three passing satellites are fed (%s)" % last,
        )
        check(
            c.publisher.calls and c.publisher.calls[-1][0]["n_sats"] == 3,
            "the snapshot and the feed are published",
        )
        n0 = rx.joint_receiver("1176", 10230, 10.23e6, which=TRK_STATE).n_updates
        jtrk.stage_joint_trk(c)
        check(
            c.dr_state["jtrk"]["last"]["failed"]["stale"] == 3
            and rx.joint_receiver("1176", 10230, 10.23e6, which=TRK_STATE).n_updates
            == n0,
            "the same records are not fed twice",
        )
        check(
            rx._joint.get(JOINT_STATE) is None,
            "the live state is untouched by the shadow feed",
        )

        sky.nearest_boresight = lambda pd: (0.8, ("E", 7))
        c = ctx_for(rx, {p: row(151.0, hop=3000) for p in (1, 2, 3)})
        jtrk.stage_joint_trk(c)
        check(
            c.dr_state["jtrk"]["last"]["held"].startswith("transit"),
            "a boresight transit holds the feed",
        )
    finally:
        sky.nearest_boresight = real


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            try:
                fn()
            except Exception as e:
                import traceback

                traceback.print_exc()
                check(False, "%s raised %s: %s" % (name, type(e).__name__, e))
    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the shadow joint state's tracker feed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
