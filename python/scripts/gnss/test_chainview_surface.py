#!/usr/bin/env python3
"""The broker only ever holds a _ChainView -- so every publisher method it calls must exist
on the PROXY, not just on FleetPublisher.

    python3 python/scripts/gnss/test_chainview_surface.py

⚠️ WHY THIS EXISTS. On 2026-08-19 set_rf was added to FleetPublisher and not to _ChainView.
register() hands the broker a view, so the first `publisher.set_rf(...)` raised
AttributeError and killed the chain -- and because rf-stats-endpoints is armed on gps_l5
alone, the casualty was the ONLY chain with a search, whose clock the other four adopt. The
other four ran on happily, so the process looked alive.

Four fixtures replay the broker end to end and NONE of them caught it, because none arms
rf-stats-endpoints. A digest gate vouches for what its fixture runs and nothing else. This
test asks a different question -- does the proxy cover the surface? -- and needs no fixture.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "gnss_broker", "publish.py")
BROKER = os.path.join(HERE, "gps_distributed_broker.py")


def main():
    tree = ast.parse(open(PUB).read())
    cls = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for need in ("FleetPublisher", "_ChainView"):
        if need not in cls:
            print("FAIL: %s not found in publish.py" % need)
            return 1
    pub_m = {n.name for n in cls["FleetPublisher"].body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    view_m = {n.name for n in cls["_ChainView"].body
              if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}

    # What does the broker actually call on its publisher handle?
    src = open(BROKER).read()
    called = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "publisher"):
            called.add(node.func.attr)

    print("FleetPublisher public methods : %s" % ", ".join(sorted(pub_m)))
    print("_ChainView proxies            : %s" % ", ".join(sorted(view_m)))
    print("broker calls on `publisher.`  : %s" % ", ".join(sorted(called)))
    print("-" * 72)

    # register() is the FACTORY -- it is called on the publisher to OBTAIN a view, so it
    # legitimately does not appear on the view. Anything else that the broker calls and the
    # view lacks is a live AttributeError waiting for the chain that arms it.
    FACTORY = {"register"}
    missing = sorted(m for m in called if m in pub_m and m not in view_m and m not in FACTORY)
    unknown = sorted(m for m in called if m not in pub_m and m not in view_m)
    if missing:
        for m in missing:
            print("FAIL: broker calls publisher.%s(), FleetPublisher has it, "
                  "_ChainView does NOT -> AttributeError at runtime, per chain." % m)
        return 1
    if unknown:
        # Not fatal: could be a builtin or a differently-named handle. Say so rather than
        # failing on something this crude cannot resolve.
        print("note: %s called but not found on either class -- check by hand"
              % ", ".join(unknown))
    print("GATE GOOD: every publisher method the broker calls is reachable through the view")
    return 0


if __name__ == "__main__":
    sys.exit(main())
