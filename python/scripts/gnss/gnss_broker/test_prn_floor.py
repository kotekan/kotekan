"""The PRN floor a chain predicts, maps and seeds from while no capability set decides
(signals.prn_floor): 19 for the BDS-3-only BeiDou signals, none for B3I, which BDS-2 and BDS-3
both broadcast, and the broker takes both of its floors from it.

    python3 -m gnss_broker.test_prn_floor

@author Keith Vanderlinde
"""

import ast
import os
import re
import sys

import yaml

from gnss_broker import signals

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_values():
    print("\nprn_floor by signal")
    for sysc, key, want in (
        ("C", "bds_b3i", 1),
        ("C", "bds_b2a", 19),
        ("C", "bds_b2b", 19),
        ("C", "bds_b1c", 19),
        ("C", None, 19),
        ("G", "gps_l5", 1),
        ("E", "gal_e6", 1),
        ("R", "glo_l3oc", 1),
        ("G", None, 1),
    ):
        got = signals.prn_floor(sysc, key)
        check(got == want, "%s %s -> %d (got %r)" % (sysc, key, want, got))


def test_broker_uses_it():
    """Both of the broker's floors come from prn_floor, not from the constellation alone."""
    print("\nthe broker's almanac and dead-reckon floors")
    tree = ast.parse(open(BROKER).read())
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in ("alm_min_prn", "dr_min_prn"):
                calls.setdefault(t.id, False)
                for sub in ast.walk(node.value):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "prn_floor"
                    ):
                        calls[t.id] = True
    for name in ("alm_min_prn", "dr_min_prn"):
        check(calls.get(name) is True, "%s is set from signals.prn_floor" % name)


def test_fleet_slots_above_floor():
    """A chain with no capability gate seeds only PRNs at or above its floor, so every PRN the
    fleet gives its trackers a slot for must clear it: a slot below the floor holds a PRN the
    broker never predicts, maps or seeds."""
    print("\nfleet slots of the ungated chains clear their floor")
    chains = yaml.safe_load(
        open(os.path.join(REPO, "config", "gnss_chains_chord.yaml"))
    )
    fleet = open(os.path.join(REPO, "config", "gnss_fleet_chord.yaml")).read()
    lists = {}
    for m in re.finditer(r"^\s*- ([A-Z0-9_]+):([0-9,]+)\s*$", fleet, re.M):
        lists[m.group(1)] = [int(p) for p in m.group(2).split(",")]
    n = 0
    for name, block in chains["chains"].items():
        key = block.get("signal")
        if not key or block.get("signal-capability") or block.get("dr-min-prn"):
            continue
        sig = signals.get(key)
        if sig.min_prn is not None:
            continue  # the signal implies the registry gate
        prns = lists.get(sig.long) or lists.get(sig.primary)
        if not prns:
            continue
        n += 1
        floor = signals.prn_floor(sig.constellation, key)
        below = [p for p in prns if p < floor]
        check(
            not below,
            "%s: every fleet slot >= the floor %d (below: %s)"
            % (name, floor, below or "none"),
        )
    check(n > 0, "at least one ungated chain has a fleet slot list (%d)" % n)


def main():
    print("PRN floor checks")
    test_values()
    test_broker_uses_it()
    test_fleet_slots_above_floor()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
