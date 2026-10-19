"""Every ctx.dllp / ctx.drp attribute the stages read is a slot of its products class.

    python3 -m gnss_broker.test_dllp_slots

_DllProducts and _DrProducts (gps_distributed_broker.py) declare __slots__, so a misspelt or
misplaced plain read raises at runtime, but getattr(ctx.dllp, "x", default) does not: it
returns the default every cycle and hides the wrong name. This scans the gnss_broker modules'
source for both forms and checks each name against the slots parsed from the broker's source.
"""

import ast
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")
OWNERS = {"dllp": "_DllProducts", "drp": "_DrProducts"}

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def slots():
    """{owner attribute: set of slot names}, from the class bodies in the broker's source."""
    tree = ast.parse(open(BROKER).read())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in OWNERS.values():
            for st in node.body:
                if (
                    isinstance(st, ast.Assign)
                    and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id == "__slots__"
                ):
                    found[node.name] = set(ast.literal_eval(st.value))
    return {k: found.get(cls) for k, cls in OWNERS.items()}


def _owner(node):
    """'dllp' / 'drp' when `node` is ctx.dllp / ctx.drp, else None."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in OWNERS
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
    ):
        return node.attr
    return None


def reads(path):
    """[(owner, name, line)] for ctx.dllp.X / ctx.drp.X and getattr(ctx.dllp, "X", ...)."""
    out = []
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Attribute) and _owner(node.value):
            out.append((_owner(node.value), node.attr, node.lineno))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr", "setattr")
            and len(node.args) >= 2
            and _owner(node.args[0])
            and isinstance(node.args[1], ast.Constant)
        ):
            out.append((_owner(node.args[0]), node.args[1].value, node.lineno))
    return out


def test_every_read_is_a_slot():
    print("ctx.dllp / ctx.drp names against the products classes' __slots__")
    sl = slots()
    for k, v in sl.items():
        check(bool(v), "%s.__slots__ parsed (%d names)" % (OWNERS[k], len(v or ())))
    if not all(sl.values()):
        return
    n = 0
    for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        for owner, name, line in reads(path):
            n += 1
            if name not in sl[owner]:
                check(
                    False,
                    "%s:%d ctx.%s.%s is not a slot of %s"
                    % (os.path.basename(path), line, owner, name, OWNERS[owner]),
                )
    check(n > 0, "%d reads checked" % n)


if __name__ == "__main__":
    test_every_read_is_a_slot()
    if _fails:
        print("FAIL: %d check(s)" % len(_fails))
        sys.exit(1)
    print("PASS")
