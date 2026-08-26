#!/usr/bin/env python3
"""Move a nested cycle stage out of `main()` into a module, rewriting its closure into `ctx`.

    broker_extract.py <routine> <module> [--apply]

WHAT IT DOES. Takes a `_stage_*` / `_instr_*` / `_dr_*` routine, computes the free names it
reads out of `main()`, rewrites each to the matching `ctx.<attr>` (AST-precise -- never text
inside strings), appends it to `gnss_broker/<module>.py` as a module-level function, and
replaces the call in `main()`.

⚠️ IT REFUSES ANYTHING IT CANNOT MAP. A free name with no slot on ChainContext is a stage
reaching for state that has no home yet -- the honest answer is to give that state an owner
first, not to smuggle it through as another attribute. Refusing is the whole point: the
argument for this refactor is that the interface should be impossible to leave unnamed.

⚠️ IT REFUSES A ROUTINE THAT WRITES ANY SHARED NAME. `nonlocal` cannot cross a module
boundary, so a stage carrying cycle state must have that state moved onto an owner object
first (as `_dllp`/`_drp` were). Checked via broker_iface's analyzer, including its AugAssign
rule -- the one that cost gal_e5a a chain death on 2026-08-26.

@author Keith Vanderlinde
"""

import ast
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.normpath(os.path.join(HERE, "..", "..", "python", "scripts", "gnss",
                                       "gps_distributed_broker.py"))
PKG = os.path.join(os.path.dirname(BROKER), "gnss_broker")

_spec = importlib.util.spec_from_file_location("bi", os.path.join(HERE, "broker_iface.py"))
bi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bi)

# free name in main()  ->  attribute on ChainContext
NAME_MAP = {
    "args": "args", "band_id": "band_id", "chain_id": "chain_id", "CODE_LEN": "code_len",
    "telem_chain": "telem_chain", "base": "base", "alm_sys": "alm_sys",
    "alm_min_prn": "alm_min_prn", "LC_SEG": "lc_seg", "LC_EPOCH": "lc_epoch",
    "rx": "rx", "publisher": "publisher", "telem_client": "telem_client",
    "detectors": "detectors", "dll_combiners": "dll_combiners",
    "spectrum_endpoints": "spectrum_endpoints", "_spec_writer": "spec_writer",
    "_state_dir": "state_dir", "_xb_read_dir": "xb_read_dir", "sig_of": "sig_of",
    "_dllp": "dllp", "_drp": "drp", "_handover": "handover", "_adm_gate": "adm_gate",
    "_g3_ramp": "g3_ramp",
    "seeds": "seeds", "dr_state": "dr_state", "bsat": "bsat", "cp_held": "cp_held",
    "dr_untrusted": "dr_untrusted",
    "_est_last": "est_last", "_kcoh_rates": "kcoh_rates", "_rf_last": "rf_last",
    "_elem_arch_t": "elem_arch_t", "_elem_poll_t": "elem_poll_t",
    "mp_cooldown": "mp_cooldown", "mp_flipped": "mp_flipped", "mp_last_det": "mp_last_det",
    "t0": "t0", "best": "best", "status": "status", "pred": "pred", "up": "up",
    "probe_set": "probe_set",
}


def _module_level(tree):
    out = set()
    for s in tree.body:
        if isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            out.add(s.name)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                out.add(a.asname or a.name.split(".")[0])
        else:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
    return out


def analyze_free(tree, main, fn):
    """The free names `fn` reads out of main() -- its closure interface."""
    import builtins
    mod = _module_level(tree)
    local = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local |= bi._scoped_names(fn) | bi._import_aliases(fn)
    local |= {g.name for g in ast.walk(fn) if isinstance(g, ast.FunctionDef) and g is not fn}
    reads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(reads - local - set(dir(builtins)) - mod)


def main_():
    routine, module = sys.argv[1], sys.argv[2]
    apply_ = "--apply" in sys.argv

    src = open(BROKER).read().splitlines(True)
    tree = ast.parse("".join(src))
    main = [n for n in tree.body if getattr(n, "name", "") == "main"][0]
    fn = [n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == routine][0]

    _, carry, outs = bi.analyze(main, fn.lineno, fn.end_lineno)
    shared = sorted(set(carry) | set(outs))
    assert not shared, ("REFUSING: `%s` writes shared state %s. `nonlocal` cannot cross a "
                        "module boundary -- give that state an owner object first."
                        % (routine, ", ".join(shared)))

    free = analyze_free(tree, main, fn)
    unmapped = [f for f in free if f not in NAME_MAP]
    assert not unmapped, ("REFUSING: no ChainContext slot for %s. Give that state a home "
                          "before moving `%s` out." % (", ".join(unmapped), routine))

    # ---- rewrite the body: free names -> ctx.attr, AST-precise -------------------------
    body = src[fn.lineno - 1:fn.end_lineno]
    ind = min(len(l) - len(l.lstrip()) for l in body if l.strip())
    hits = [n for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in free]
    by_line = {}
    for n in hits:
        by_line.setdefault(n.lineno, []).append(n)
    for ln, nodes in by_line.items():
        i = ln - fn.lineno
        line = body[i]
        for n in sorted(nodes, key=lambda x: -x.col_offset):
            c = n.col_offset
            assert line[c:c + len(n.id)] == n.id, "mismatch at %d:%d" % (ln, c)
            line = line[:c] + "ctx." + NAME_MAP[n.id] + line[c + len(n.id):]
        body[i] = line

    body = [(l[ind:] if l.strip() else l) for l in body]
    newname = routine.lstrip("_")
    body[0] = body[0].replace("def %s():" % routine, "def %s(ctx):" % newname, 1)
    assert body[0].startswith("def %s(ctx):" % newname), body[0][:60]

    if not apply_:
        print("%s -> %s.%s(ctx)\n  free (%d): %s"
              % (routine, module, newname, len(free), ", ".join(free)))
        return

    target = os.path.join(PKG, module + ".py")
    with open(target, "a") as f:
        f.write("\n\n" + "".join(body).rstrip("\n") + "\n")
    rest = src[:fn.lineno - 1] + src[fn.end_lineno:]
    txt = "".join(rest).replace("%s()\n" % routine, "%s.%s(_ctx)\n" % (module, newname))
    open(BROKER, "w").write(txt)
    print("moved %s -> %s.%s(ctx)  [%d free names]" % (routine, module, newname, len(free)))


if __name__ == "__main__":
    main_()
