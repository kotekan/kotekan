"""The DEEP GATE (auto) line follows the search-admitted set through empty and back.

    python3 -m gnss_broker.test_deep_gate_auto

fleetdll.deep_gate_auto logs the search-admitted deep-gate set when it changes. An emptied set
must be logged and must reset the last-logged set, so the same PRNs coming back print again.
"""

import sys
from types import SimpleNamespace

from gnss_broker import fleetdll

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def stub_ctx(deep_gate=None):
    return SimpleNamespace(
        args=SimpleNamespace(
            dll_deep_gate_from_search=100.0, dll_deep_gate_search_hold_s=30.0
        ),
        deep_gate=deep_gate,
        dls=SimpleNamespace(deep_gate_seen={}),
        dllp=SimpleNamespace(deep_gate_eff=None),
        dg_auto_last=[set()],
        t0=1000.0,
    )


def test_empty_and_back():
    print("deep_gate_auto: {4}, then empty, then {4} again")
    lines = []
    fleetdll._log = lines.append
    ctx = stub_ctx()
    ctx.dls.deep_gate_seen = {4: 995.0}
    fleetdll.deep_gate_auto(ctx)
    check(ctx.dllp.deep_gate_eff == {4}, "PRN 4 admitted (%s)" % ctx.dllp.deep_gate_eff)
    check(len(lines) == 1 and "PRN 4" in lines[0], "the admission is logged")
    fleetdll.deep_gate_auto(ctx)
    check(len(lines) == 1, "an unchanged set is not logged again")

    ctx.t0 = 1100.0  # past the 30 s hold
    fleetdll.deep_gate_auto(ctx)
    check(ctx.dllp.deep_gate_eff is None, "the hold expired: no deep gate")
    check(len(lines) == 2 and "empty" in lines[-1], "the emptied set is logged")
    fleetdll.deep_gate_auto(ctx)
    check(len(lines) == 2, "a still-empty set is not logged again")

    ctx.dls.deep_gate_seen = {4: 1095.0}
    fleetdll.deep_gate_auto(ctx)
    check(
        len(lines) == 3 and "PRN 4" in lines[-1],
        "the same set coming back is logged again (%d line(s))" % len(lines),
    )


def test_hand_listed_stays():
    print("deep_gate_auto: the hand-listed set stands when the search set empties")
    lines = []
    fleetdll._log = lines.append
    ctx = stub_ctx(deep_gate={7})
    ctx.dls.deep_gate_seen = {4: 995.0}
    fleetdll.deep_gate_auto(ctx)
    check(ctx.dllp.deep_gate_eff == {4, 7}, "search + hand-listed")
    ctx.t0 = 1100.0
    fleetdll.deep_gate_auto(ctx)
    check(ctx.dllp.deep_gate_eff == {7}, "hand-listed only after the hold")


if __name__ == "__main__":
    test_empty_and_back()
    test_hand_listed_stays()
    if _fails:
        print("FAIL: %d check(s)" % len(_fails))
        sys.exit(1)
    print("PASS")
