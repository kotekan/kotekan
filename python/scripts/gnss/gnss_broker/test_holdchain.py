"""The seed-disposition chain in seeding.py: escape-if -> hold-elif -> release-else.

The log-only TRACK-vs-MODEL monitor must stay a standalone statement above the escape `if`.
Placed between the escape `if` and the hold `elif`, it re-chains the `elif` to its own
condition: whenever a held satellite has a fresh detection while the dead-reckon integrity
residuals are populated, freeze/translate/release are all skipped and the seed silently becomes
a per-detection re-anchor, which the DLL then chases. A check at cold start cannot see it,
because the integrity residuals are still empty then.

An `elif` has no name for what it chains to; only structure pins it. Hence ast, not grep.

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
    check(
        "hold_on_present" in ast.unparse(hold.test),
        "the elif is the hold branch (mentions hold_on_present)",
    )
    check(
        bool(hold.orelse)
        and not (len(hold.orelse) == 1 and isinstance(hold.orelse[0], ast.If)),
        "the hold elif has a plain else (the release)",
    )


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
        check(
            not n.orelse,
            "monitor if at line %d carries no elif/else (log-only means log-only)"
            % n.lineno,
        )


def test_err_hist_takes_one_sample_per_detection_and_never_none():
    """Every append to the median gate's history is guarded by `cp_err is not None`.

    The gate medians the last 9 samples: a second, unguarded append halves its window
    and lets a None in, on which statistics.median raises.
    """
    found = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.guards = []

        def visit_If(self, node):
            self.guards.append(ast.unparse(node.test))
            for n in node.body:
                self.visit(n)
            self.guards.pop()
            for n in node.orelse:
                self.visit(n)

        def visit_Call(self, node):
            if "err_hist" in ast.unparse(node) and ast.unparse(node).endswith(
                ".append(cp_err)"
            ):
                found.append((node.lineno, list(self.guards)))
            self.generic_visit(node)

    V().visit(_seeding_tree())
    check(len(found) == 1, "exactly one err_hist append (found %d)" % len(found))
    for lineno, guards in found:
        check(
            any(g == "cp_err is not None" for g in guards),
            "err_hist append at line %d is inside `if cp_err is not None`" % lineno,
        )


if __name__ == "__main__":
    test_hold_elif_chains_to_the_escape_if()
    test_monitor_is_not_a_chain_member()
    test_err_hist_takes_one_sample_per_detection_and_never_none()
    if _fails:
        print("FAILED: %d" % len(_fails))
        sys.exit(1)
    print("all pass")
