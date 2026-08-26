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
import builtins as builtins_mod
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
    "utc0_sample0": "utc0_sample0", "_xb_pred": "xb_pred", "coast_polls": "coast_polls",
    "have_sig": "have_sig", "la_samples": "la_samples", "fitted": "fitted",
    "cl_report": "cl_report", "dr_pd": "dr_pd", "dr_pd0": "dr_pd0", "dr_pd2": "dr_pd2",
    "innov_hist": "innov_hist", "minnov_hist": "minnov_hist", "p2c": "p2c",
    "dop_rate_fitted": "dop_rate_fitted", "dop_rate_rejected": "dop_rate_rejected",
    "dll_hop_window": "dll_hop_window", "_deep_gate": "deep_gate",
    "_dg_auto_last": "dg_auto_last", "_est_next": "est_next",
    "combiner": "combiner", "gating": "gating", "_capable": "capable",
    "receiver_state": "receiver_state", "_alm_now": "alm_now", "_cb": "cb",
    "almanac_sats": "almanac_sats", "brdc_alm": "brdc_alm", "det_fresh": "det_fresh",
    "state_w": "state_w", "_clk_persist_t": "clk_persist_t",
    "_carrier": "car", "_watchdog": "wd", "_nho": "nho",
    "_dls": "dls", "_hold": "hold", "_cpt": "cpt", "_rf": "rf", "_nav": "nav", "_cls": "cls", "payload": "payload",
    "HIST_LEN": "hist_len", "MAX_GAP_HOPS": "max_gap_hops", "Q_ALIAS_HZ": "q_alias_hz",
    "CARRIER_EXPLAIN_HZ": "carrier_explain_hz", "CARRIER_VERIFY_EMITS": "carrier_verify_emits",
    "trackers": "trackers", "joint_consume": "joint_consume", "broker_t0": "broker_t0",
    "_fuse_cached": "fuse_cached", "cp_to_seed_currency": "cp_to_seed_currency",
    "_dh_obs": "dh_obs", "cp_predicted": "cp_predicted", "_joint_state": "joint_state",
    "_track_ok": "track_ok", "_p2c_tick": "p2c_tick", "_p2c_hold": "p2c_hold",
    "_decoded_entries": "decoded_entries",
    "sig_of_last": "sig_of_last", "dr_eph_mod": "dr_eph_mod", "dr_min_prn": "dr_min_prn",
}

# `_ctx` is the context itself, not a slot on it: a stage that already writes through the
# context (`_ctx.pred = ...`) must address it as its own parameter once it moves out.
SELF_NAME = "_ctx"


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

    # ⚠️ A MODULE-LEVEL NAME IS FREE ONLY IF THE TARGET MODULE HAS IT TOO. `analyze_free`
    # discounts everything bound at the broker's module level -- imports and constants -- on
    # the grounds that a module-level function can see them. That is true of the BROKER's
    # module level, not the destination's. `C_LIGHT` is the standing example: a bare NameError
    # deep inside the moved stage, on whichever path happens to read it first.
    target_src = ""
    tpath = os.path.join(PKG, module + ".py")
    if os.path.exists(tpath):
        target_src = open(tpath).read()
    ttree = ast.parse(target_src) if target_src else ast.Module(body=[], type_ignores=[])
    have = _module_level(ttree) | set(dir(builtins_mod))
    mod_used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id in _module_level(tree)}
    absent = sorted(mod_used - have)
    assert not absent, ("REFUSING: %s uses module-level %s, which `%s` does not import. Add "
                        "the import (or give the constant a shared home) first."
                        % (routine, ", ".join(absent), module))

    free = analyze_free(tree, main, fn)
    unmapped = [f for f in free if f not in NAME_MAP and f != SELF_NAME]
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
    # ⚠️ col_offset IS A BYTE OFFSET, NOT A CHARACTER OFFSET. This file is full of `⚠️` in
    # comments, and every one of those is 3-6 bytes for a single character -- so slicing the
    # str by col_offset silently lands in the wrong place on any line preceded by one. Work in
    # BYTES and decode once at the end. The assert below caught this rather than a corrupted
    # rewrite, which is the only reason it cost minutes instead of a debugging session.
    for ln, nodes in by_line.items():
        i = ln - fn.lineno
        raw = body[i].encode("utf-8")
        for n in sorted(nodes, key=lambda x: -x.col_offset):
            c = n.col_offset
            tok = n.id.encode("utf-8")
            assert raw[c:c + len(tok)] == tok, "mismatch at %d:%d" % (ln, c)
            repl = b"ctx" if n.id == SELF_NAME else b"ctx." + NAME_MAP[n.id].encode("utf-8")
            raw = raw[:c] + repl + raw[c + len(tok):]
        body[i] = raw.decode("utf-8")

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
