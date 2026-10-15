#!/usr/bin/env python3
"""The broker only ever holds a _ChainView, so every publisher method it calls must exist
on the proxy, not just on FleetPublisher.

    python3 python/scripts/gnss/test_chainview_surface.py

register() hands the broker a view, so a method added to FleetPublisher but not to
_ChainView raises AttributeError at its first call and kills that chain while the others run
on, and the process looks alive. A broker replay fixture vouches only for what it runs: a
call behind a flag no fixture arms (--rf-stats-endpoints, say) is never reached. This test
asks a different question -- does the proxy cover the surface? -- and needs no fixture.

It collects calls on a bare `publisher` name in gps_distributed_broker.py only; calls made
through ctx.publisher in the gnss_broker modules are not seen.
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
    pub_m = {
        n.name
        for n in cls["FleetPublisher"].body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }
    view_m = {
        n.name
        for n in cls["_ChainView"].body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }

    # The calls on a bare `publisher` name in gps_distributed_broker.py.
    src = open(BROKER).read()
    called = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "publisher"
        ):
            called.add(node.func.attr)

    print("FleetPublisher public methods : %s" % ", ".join(sorted(pub_m)))
    print("_ChainView proxies            : %s" % ", ".join(sorted(view_m)))
    print("broker calls on `publisher.`  : %s" % ", ".join(sorted(called)))
    print("-" * 72)

    # register() is the factory: it is called on the publisher to obtain a view, so it does
    # not appear on the view. Anything else the broker calls that the view lacks is an
    # AttributeError waiting for the chain that arms it.
    FACTORY = {"register"}
    missing = sorted(
        m for m in called if m in pub_m and m not in view_m and m not in FACTORY
    )
    unknown = sorted(m for m in called if m not in pub_m and m not in view_m)
    if missing:
        for m in missing:
            print(
                "FAIL: broker calls publisher.%s(), FleetPublisher has it, "
                "_ChainView does NOT -> AttributeError at runtime, per chain." % m
            )
        return 1
    if unknown:
        # Not fatal: could be a builtin or a differently-named handle. Say so rather than
        # failing on something this crude cannot resolve.
        print(
            "note: %s called but not found on either class -- check by hand"
            % ", ".join(unknown)
        )
    print(
        "GATE GOOD: every publisher method the broker calls is reachable through the view"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
