"""The seed-disposition chain in seeding.py: escape-if -> hold-elif -> release-else.

The defect this pins (shipped 2026-07-20 -> 2026-08-28): the TRACK-vs-MODEL monitor,
documented "log-only", was inserted BETWEEN the escape `if` and the hold `elif`. That
re-chained the elif to the monitor's own condition, so whenever a held sat had a fresh
detection while dr integrity was populated, freeze/translate/release were ALL skipped and
the seed silently became a per-detection re-anchor. On sky: every deep-gate GPS sat
re-anchored every ~2 s once integrity warmed (~10 min after a restart), the DLL railed
chasing the stepping command, and q cratered every few minutes (G28, 08-28 evening).
The cold-start judge missed it because integrity was still empty then.

An `elif` has no name for what it chains to -- only structure pins it. Hence ast, not grep.

    python3 -m gnss_broker.test_holdchain

@author Keith Vanderlinde
"""

import ast
import os
import sys

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _seeding_tree():
    path = os.path.join(os.path.dirname(__file__), "seeding.py")
    with open(path) as f:
        return ast.parse(f.read())


def test_hold_elif_chains_to_the_escape_if():
    """The hold branch must be the escape-if's orelse, and release its else."""
    found = []

    class V(ast.NodeVisitor):
        def visit_If(self, node):
            t = ast.unparse(node.test)
            if "cpt.escape.get(prn, 0) >= 5" in t:
                found.append(node)
            self.generic_visit(node)

    V().visit(_seeding_tree())
    check(len(found) == 1, "exactly one escape-if (found %d)" % len(found))
    if len(found) != 1:
        return
    o = found[0].orelse
    is_elif = len(o) == 1 and isinstance(o[0], ast.If)
    check(is_elif, "escape-if has an elif")
    if not is_elif:
        return
    hold = o[0]
    check("hold_on_present" in ast.unparse(hold.test),
          "the elif is the hold branch (mentions hold_on_present)")
    check(bool(hold.orelse) and not (len(hold.orelse) == 1
                                     and isinstance(hold.orelse[0], ast.If)),
          "the hold elif has a plain else (the release)")


def test_monitor_is_not_a_chain_member():
    """The log-only monitor must be a standalone if: no orelse at all."""
    found = []

    class V(ast.NodeVisitor):
        def visit_If(self, node):
            t = ast.unparse(node.test)
            if "cp_err is not None" in t and "dr_state" in t and "integ" in t:
                found.append(node)
            self.generic_visit(node)

    V().visit(_seeding_tree())
    check(len(found) >= 1, "the monitor if exists")
    for n in found:
        check(not n.orelse,
              "monitor if at line %d carries no elif/else (log-only means log-only)"
              % n.lineno)


if __name__ == "__main__":
    test_hold_elif_chains_to_the_escape_if()
    test_monitor_is_not_a_chain_member()
    if _fails:
        print("FAILED: %d" % len(_fails))
        sys.exit(1)
    print("all pass")
