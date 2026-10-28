"""The array code group-delay map (gdvmap) and its dead-reckon step (deadreckon.dr_gdv): the
smoothed lookup, no correction off the fitted sky, the boresight gate, the rows it fills, and the
shipped map against the chains yaml.

    python3 -m gnss_broker.test_gdvmap

@author Keith Vanderlinde
"""

import os
import types

import yaml

from gnss_broker import deadreckon, sky
from gnss_broker.gdvmap import GdvMap, resolve

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _doc(cells):
    return {
        "pointing": "ptest",
        "boresight": {"az": 180.0, "el": 62.7},
        "bands": {"b": {"M": cells, "sigma": {}}},
    }


def test_lookup():
    print("lookup")
    # a flat patch: the kernel mean returns the patch value, on and between cell centres
    flat = {
        "%d,%s" % (az, el): 6.0 for az in range(20, 60, 5) for el in (5.0, 7.5, 10.0)
    }
    m = GdvMap(_doc(flat), "b")
    check(
        abs(m.delay_m(37.5, 8.75) - 6.0) < 1e-9,
        "flat patch: the patch value at a centre",
    )
    check(
        abs(m.delay_m(40.0, 10.0) - 6.0) < 1e-9,
        "flat patch: the patch value on a cell edge",
    )
    check(
        m.delay_m(200.0, 30.0) == 0.0,
        "no fitted cell within reach: 0.0, not an extrapolation",
    )
    # a step between two cells is smoothed: monotone and bounded across the edge
    step = {"30,5.0": 0.0, "35,5.0": 10.0}
    s = GdvMap(_doc(step), "b")
    v = [s.delay_m(az, 6.25) for az in (32.5, 34.0, 35.0, 36.0, 37.5)]
    check(
        all(a <= b for a, b in zip(v, v[1:])),
        "a step is smoothed monotonically: %s" % v,
    )
    check(0.0 < v[0] < 5.0 < v[-1] < 10.0, "and stays inside the two cell values")
    # the two grid resolutions join: a high cell's centre is az+5, el+2.5
    hi = GdvMap(_doc({"100,40": 3.0}), "b")
    check(
        abs(hi.delay_m(105.0, 42.5) - 3.0) < 1e-9,
        "high-sky cell centred at (az+5, el+2.5)",
    )
    try:
        GdvMap(_doc(flat), "nope")
        check(False, "an absent band refuses to load")
    except ValueError:
        check(True, "an absent band refuses to load")


def test_applies():
    print("boresight gate")
    m = GdvMap(_doc({"30,5.0": 1.0}), "b")
    check(m.applies(180.0, 62.7), "the map's own boresight applies")
    check(m.applies(180.0, 63.1), "0.4 deg off applies")
    check(not m.applies(180.0, 81.4), "another pointing does not")
    check(not m.applies(None, None), "no installed boresight does not")


def _ctx(path, band):
    logs = []
    deadreckon._log = lambda s: logs.append(s)
    args = types.SimpleNamespace(gdv_map=path, gdv_band=band)
    return types.SimpleNamespace(args=args, dr_state={}), logs


def test_dr_gdv():
    print("dead-reckon step")
    p = "/tmp/test_gdvmap_%d.json" % os.getpid()
    import json

    with open(p, "w") as f:
        json.dump(
            _doc(
                {
                    "%d,%s" % (az, el): 8.0
                    for az in range(20, 60, 5)
                    for el in (5.0, 7.5)
                }
            ),
            f,
        )
    try:
        ctx, logs = _ctx(p, "b")
        row = lambda: {"az": 38.0, "el": 8.0}
        pd, pd2, pd0 = (
            {("E", 2): row(), ("E", 9): {"az": 180.0, "el": 70.0}},
            {("E", 2): row()},
            {},
        )
        sky.set_boresight(180.0, 62.7, "test")
        deadreckon.dr_gdv(ctx, pd, pd2, pd0)
        check(
            abs(pd[("E", 2)]["gdv_m"] - 8.0) < 1e-9, "north-low row gets the map value"
        )
        check(
            pd2[("E", 2)]["gdv_m"] == pd[("E", 2)]["gdv_m"],
            "the +2 s row takes the same value",
        )
        check(pd[("E", 9)]["gdv_m"] == 0.0, "a direction off the fitted sky gets 0.0")
        check(any(" ON " in s for s in logs), "logs ON once")
        n = len(logs)
        deadreckon.dr_gdv(ctx, pd, pd2, pd0)
        check(len(logs) == n, "no log while the state holds")
        sky.set_boresight(180.0, 81.41, "test")
        deadreckon.dr_gdv(ctx, pd, pd2, pd0)
        check(pd[("E", 2)]["gdv_m"] == 0.0, "another pointing: 0.0")
        check(any(" OFF " in s for s in logs[n:]), "and logs OFF")
        ctx2, logs2 = _ctx("/nonexistent/map.json", "b")
        deadreckon.dr_gdv(ctx2, pd, pd2, pd0)
        check(
            ctx2.dr_state["gdv"] is False and any("NOT loaded" in s for s in logs2),
            "an unreadable map logs and stays off",
        )
        ctx3 = types.SimpleNamespace(
            args=types.SimpleNamespace(gdv_map=""), dr_state={}
        )
        q = {("E", 2): row()}
        deadreckon.dr_gdv(ctx3, q, {}, {})
        check("gdv_m" not in q[("E", 2)], "--gdv-map empty: rows untouched")
    finally:
        os.remove(p)


def test_shipped_map():
    print("shipped map vs the chains yaml")
    here = os.path.dirname(__file__)
    chains = yaml.safe_load(
        open(
            os.path.join(
                here, "..", "..", "..", "..", "config", "gnss_chains_chord.yaml"
            )
        )
    )["chains"]
    used = {k: v for k, v in chains.items() if v and v.get("gdv-map")}
    check(len(used) >= 1, "%d chain(s) name a map" % len(used))
    for k, v in sorted(used.items()):
        try:
            m = GdvMap.load(v["gdv-map"], v.get("gdv-band", ""))
        except (OSError, ValueError) as e:
            check(False, "%s: %s" % (k, e))
            continue
        nl = m.delay_m(38.0, 8.0)  # E02's direction, 10-28
        high = m.delay_m(200.0, 75.0)
        check(
            2.0 < nl < 14.0 and abs(high) < 1.5,
            "%s (%s): north-low %+.1f m late, high sky %+.1f m" % (k, m.band, nl, high),
        )
        check(m.applies(180.0, 62.7), "%s: the map is the Tau A pointing's" % k)
    check(
        os.path.exists(resolve("config/gdvmap_taua_20261024.json")),
        "a relative path resolves against the repository root",
    )


if __name__ == "__main__":
    test_lookup()
    test_applies()
    test_dr_gdv()
    test_shipped_map()
    print("\n%s (%d check(s) failed)" % ("PASS" if not _fails else "FAIL", len(_fails)))
    raise SystemExit(1 if _fails else 0)
