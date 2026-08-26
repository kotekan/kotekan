#!/usr/bin/env python3
"""What does a block of the broker's cycle loop actually read and write?

    broker_iface.py map                      # every large block, with its interface
    broker_iface.py iface <lo> <hi>          # one line range, in detail
    broker_iface.py promote <lo> <hi> <name> <docfile> [nonlocal,names]

THE PROBLEM THIS SOLVES. `main()` in gps_distributed_broker.py is one function whose cycle loop
shares a single flat namespace across thousands of lines. Nothing in that shape makes it
visible which names are cycle-spanning STATE and which are throwaway temporaries -- so
extracting any block is a guess, and shadowing is invisible. On 2026-08-26 that cost a latent
bug where the carrier loop assigned a sorted LIST over the DLL's per-PRN state DICT (`fleet`)
for the rest of the cycle; `map` finds that class in one pass.

THE THREE CATEGORIES, and why each matters when a block becomes a function:

  INPUTS       read before being written in the block. A closure supplies these free, so they
               cost nothing -- which is why promotion to a NESTED function is so much cheaper
               than promotion to a module-level one.

  CARRY-OVER   read before written AND written. This is state that spans CYCLES. Wrapping the
               block in a function silently turns it into a per-call local: nothing raises,
               the state just resets every cycle, and the symptom appears somewhere else
               entirely. Must be declared `nonlocal`.

  LIVE OUTPUTS written here, read by a LATER stage of the same cycle. Same failure from the
               other side: without `nonlocal` the later reader sees the previous cycle's
               value. Stale, not absent -- so still silent. Must be declared `nonlocal`.

⚠️ COMPREHENSION AND except-AS NAMES ARE EXCLUDED. Both are their own scope in Python 3 (and
`except ... as e` additionally DELETES the name at handler exit, which is why a block using it
cannot simply declare `nonlocal e`). Counting them produces false carry-over and sends you
chasing state that does not exist -- and, worse, would have you declare `nonlocal` on a name
Python then deletes.

⚠️ THE READ-BEFORE-WRITE TEST IS LEXICAL, NOT A DATA-FLOW ANALYSIS. A name whose only read sits
textually above its only write is reported as carry-over even when execution always assigns
first. That is the safe direction to be wrong in -- it refuses a promotion rather than
permitting a bad one -- but it means a refusal is a prompt to go and look, not a verdict.
`clk_now` in the dead-reckon stage is the standing example: genuinely cycle-spanning, with no
binding in `main` at all, so it cannot be promoted without first making its state explicit.

@author Keith Vanderlinde
"""

import ast
import os
import subprocess
import sys

BROKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "python", "scripts", "gnss", "gps_distributed_broker.py")
BROKER = os.path.normpath(BROKER)


def _load(path):
    tree = ast.parse(open(path).read())
    main = [n for n in tree.body if getattr(n, "name", "") == "main"][0]
    loop = [s for s in main.body if isinstance(s, ast.While)][0]
    return tree, main, loop


def _scoped_names(main):
    """Names that are NOT main locals: comprehension targets, lambda args, except-as."""
    out = set()
    for n in ast.walk(main):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        out.add(t.id)
        elif isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            # PARAMETERS ARE THEIR OWN SCOPE TOO, and forgetting that reads exactly like
            # shared state: `_dh_obs(sig, prn, h, eph_obj, xc)` loads `xc` with no preceding
            # assignment, so a first-use test called it a read of the enclosing `xc` and
            # demanded a nonlocal for a name that is merely an argument.
            a = n.args
            for g in (a.args, a.posonlyargs, a.kwonlyargs):
                for x in g:
                    out.add(x.arg)
            for x in (a.vararg, a.kwarg):
                if x is not None:
                    out.add(x.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def _import_aliases(node):
    """Names bound by `import x as y` / `from m import x`. These are Stores that never appear
    as an ast.Name, so a first-use test sees only the later Loads and calls a locally-imported
    module a read-before-write. `import receiver_state as _rs` inside the clock-adopt stage is
    the standing example."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
    return out


def _augmented(node):
    """Names targeted by an augmented assignment (`x += 1`).

    ⚠️ AN AUGMENTED ASSIGNMENT READS BEFORE IT WRITES, but its target carries ctx=Store and
    NOTHING ELSE -- there is no Load node to find. A first-use scan therefore records the
    write and misses the read, so a counter incremented in a block but INITIALISED outside it
    reads as purely local. That is not theoretical: `_rr_railed += 1` cleared this analysis,
    the seed-push stage was promoted without a nonlocal, and gal_e5a died with an
    UnboundLocalError 20 seconds into the live swap on 2026-08-26.
    """
    return {n.target.id for n in ast.walk(node)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}


def analyze(main, lo, hi):
    skip = _scoped_names(main) | _import_aliases(main)
    uses = sorted((n.lineno, n.id, isinstance(n.ctx, ast.Load))
                  for n in ast.walk(main) if isinstance(n, ast.Name))
    # An augmented target reads at the SAME line it writes, and at the same line a Store sorts
    # first -- so injecting a synthetic Load into `uses` does not work. Handle it directly
    # below via `aug_lines` instead.
    aug_lines = {}
    for n in ast.walk(main):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            aug_lines.setdefault(n.target.id, []).append(n.lineno)
    inside = [u for u in uses if lo <= u[0] <= hi]
    first = {}
    for ln, nm, ld in inside:
        if nm not in first:
            first[nm] = ld
    writes = {nm for _, nm, ld in inside if not ld}
    inputs = sorted(nm for nm, ld in first.items() if ld and nm not in skip)
    # A name is carry-over if its first use in the block is a read, OR if the block AUGMENTS it
    # without a plain assignment first -- the augment's read half needs a value from outside.
    plain_store = {}
    for n in ast.walk(main):
        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                and lo <= n.lineno <= hi and n.lineno not in aug_lines.get(n.id, ())):
            plain_store.setdefault(n.id, n.lineno)
    aug_first = {nm for nm, lns in aug_lines.items()
                 if any(lo <= l <= hi for l in lns)
                 and (nm not in plain_store
                      or plain_store[nm] > min(l for l in lns if lo <= l <= hi))}
    carry = sorted(nm for nm in writes
                   if (first.get(nm) is True or nm in aug_first) and nm not in skip)
    # ⚠️ LINE ORDER IS NOT EXECUTION ORDER, AND ASSUMING IT WAS COST A RED GATE.
    # Once a stage is promoted, its body sits BEFORE the cycle loop in the file while running
    # LATER in the cycle. So "read at a line after this block" stops meaning "read after this
    # block runs": `up` is written by the almanac stage at line 5859 and read by the coast/drop
    # stage at line 3166 -- earlier in the file, later in the pass. Judging by line number
    # declared it dead, promotion made it a local, and three fixtures moved.
    #
    # The rule that is actually safe does not depend on order at all: ANY name this block
    # writes that is used ANYWHERE ELSE in main() must stay shared. Declaring `nonlocal` on a
    # name that turns out to be dead is harmless -- it preserves exactly today's behaviour,
    # including today's accidental sharing. Making a shared name local is what is not.
    # A name is genuinely SHARED only if some OTHER scope READS it without first assigning it
    # -- that scope was relying on somebody else's value. A scope that assigns before reading
    # is an independent temporary that merely reuses the name, and there are many of those:
    # `hints` and `period` are each written-then-read inside three different scopes and shared
    # by none of them. Requiring nonlocal for those is not merely noisy, it is IMPOSSIBLE --
    # there is no main-level binding to attach it to.
    outs = sorted(nm for nm in writes
                  if nm not in skip and nm not in carry and _read_before_write_elsewhere(main, nm, lo, hi))
    return inputs, carry, outs


def _other_scopes(main, lo, hi):
    """Every scope that could read a name the block writes: each nested routine outside the
    block, plus main's own body with all nested routines removed."""
    fns = [n for n in ast.walk(main)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not main]
    out = [f for f in fns if not (lo <= f.lineno and f.end_lineno <= hi)]
    return out, fns


def _read_before_write_elsewhere(main, name, lo, hi):
    others, allfns = _other_scopes(main, lo, hi)
    for f in others:
        sk = _scoped_names(f) | _import_aliases(f)
        if name in sk:
            continue
        inner = [g for g in allfns if g is not f and f.lineno <= g.lineno and g.end_lineno <= f.end_lineno]
        us = sorted((n.lineno, n.col_offset, isinstance(n.ctx, ast.Load))
                    for n in ast.walk(f) if isinstance(n, ast.Name) and n.id == name
                    and not any(g.lineno <= n.lineno <= g.end_lineno for g in inner))
        if us and us[0][2]:
            return True
    # main's own body, outside every routine and outside the block
    sk = _scoped_names(main) | _import_aliases(main)
    if name in sk:
        return False
    us = sorted((n.lineno, n.col_offset, isinstance(n.ctx, ast.Load))
                for n in ast.walk(main) if isinstance(n, ast.Name) and n.id == name
                and not (lo <= n.lineno <= hi)
                and not any(g.lineno <= n.lineno <= g.end_lineno for g in allfns))
    return bool(us) and us[0][2]


def cmd_map(argv):
    minlines = int(argv[0]) if argv else 60
    _, main, loop = _load(BROKER)
    print("%-6s %-6s %6s  %-26s %-26s %s"
          % ("start", "end", "lines", "carry-over (nonlocal)", "live outputs (nonlocal)", "block"))
    total_safe = 0
    for s in loop.body:
        n = s.end_lineno - s.lineno + 1
        if n < minlines:
            continue
        _, carry, outs = analyze(main, s.lineno, s.end_lineno)
        try:
            lbl = ast.unparse(s)[:40].replace("\n", " ")
        except Exception:
            lbl = type(s).__name__
        safe = not carry and not outs
        total_safe += n if safe else 0
        print("%-6d %-6d %6d  %-26s %-26s %s%s"
              % (s.lineno, s.end_lineno, n, ",".join(carry)[:24] or "-",
                 ",".join(outs)[:24] or "-", lbl, "   <= promotable as-is" if safe else ""))
    print("\n%d lines sit in blocks promotable with no nonlocal at all." % total_safe)


def cmd_iface(argv):
    lo, hi = int(argv[0]), int(argv[1])
    _, main, _ = _load(BROKER)
    inputs, carry, outs = analyze(main, lo, hi)
    print("lines %d-%d (%d)" % (lo, hi, hi - lo + 1))
    print("  INPUTS       (%d): %s" % (len(inputs), ", ".join(inputs)))
    print("  CARRY-OVER   (%d): %s" % (len(carry), ", ".join(carry) or "-"))
    print("  LIVE OUTPUTS (%d): %s" % (len(outs), ", ".join(outs) or "-"))
    if carry or outs:
        print("\n  promote with: nonlocal %s" % ", ".join(sorted(set(carry) | set(outs))))
    else:
        print("\n  promotable with no nonlocal.")


def cmd_promote(argv):
    """Move a whole statement of the cycle loop into a nested routine defined before the loop.

    Nested, not module-level, deliberately: the routine keeps closing over every per-cycle
    local, so there is no interface to get wrong and no argument list to drift. Promotion to a
    real module comes later, once a block's state has been made explicit.
    """
    lo, hi, name, docfile = int(argv[0]), int(argv[1]), argv[2], argv[3]
    declared = [x for x in (argv[4].split(",") if len(argv) > 4 and argv[4] else []) if x]

    src = open(BROKER).read().splitlines(True)
    _, main, loop = _load(BROKER)
    # The range must be whole statements of SOME block -- the cycle loop's own body, or a
    # block nested inside it (the DLL stage's instrument polls live one level down).
    ok = False
    for parent in ast.walk(main):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        st = [s for s in body if getattr(s, "lineno", None) and s.lineno >= lo and s.end_lineno <= hi]
        if st and st[0].lineno == lo and st[-1].end_lineno == hi:
            ok = True
            break
    assert ok, "range must be whole statements of one block"

    _, carry, outs = analyze(main, lo, hi)
    missing = [c for c in sorted(set(carry) | set(outs)) if c not in declared]
    assert not missing, ("REFUSING: undeclared carry-over/output state: %s\n"
                         "  Declaring these nonlocal is mandatory -- without it the state "
                         "silently resets or goes stale, and nothing raises." % ", ".join(missing))
    # ⚠️ THE BINDING MUST BE AT main()'s OWN LEVEL. A Store inside an already-promoted stage
    # binds THAT function's local, not main's, so counting it makes `nonlocal` a SyntaxError
    # at import time. (`hints` is written by the almanac stage and read by narrow-search --
    # both now routines -- so once almanac moved, nothing bound it in main at all.)
    nested = [n for n in ast.walk(main)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
              and n is not main]
    def _binds_at_main_level(d):
        for n in ast.walk(main):
            if not (isinstance(n, ast.Name) and n.id == d and isinstance(n.ctx, ast.Store)):
                continue
            if lo <= n.lineno <= hi:
                continue
            if any(f.lineno <= n.lineno <= f.end_lineno for f in nested):
                continue
            return True
        return False
    for d in declared:
        assert _binds_at_main_level(d), \
            ("REFUSING: `%s` has no binding at main()'s own level outside the block, so "
             "`nonlocal %s` is a SyntaxError. Give the state an explicit home first." % (d, d))

    block = src[lo - 1:hi]
    ind = min(len(l) - len(l.lstrip()) for l in block if l.strip())
    assert ind >= 8 and ind % 4 == 0, "unexpected block indent %d" % ind
    if ind > 8:                      # a block nested inside another: dedent to routine body
        cut = ind - 8
        block = [(l[cut:] if l.strip() else l) for l in block]
    doc = open(docfile).read().rstrip("\n")
    head = ["    def %s():\n" % name, '        """%s"""\n' % doc.replace("\n", "\n        ")]
    if declared:
        head.append("        nonlocal %s\n" % ", ".join(declared))
    # The CALL keeps the block's original indent -- a poll nested inside the DLL stage sits at
    # 12, not at the loop body's 8. (Getting this wrong is an IndentationError, not a silent
    # bug, but it wastes a gate run.)
    new = (src[:loop.lineno - 1] + head + block + ["\n"]
           + src[loop.lineno - 1:lo - 1] + [" " * ind + "%s()\n" % name] + src[hi:])
    open(BROKER, "w").writelines(new)
    print("promoted %s: %d lines -> 1 call" % (name, hi - lo + 1))


if __name__ == "__main__":
    cmds = {"map": cmd_map, "iface": cmd_iface, "promote": cmd_promote}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(2)
    cmds[sys.argv[1]](sys.argv[2:])
