"""Every ChainContext slot must actually be supplied.

    python3 -m gnss_broker.test_context

WHY THIS EXISTS. A slot the broker declares but never passes is silently None, and the symptom
appears far away and much later -- `ctx.lc_seg > 1` raising a TypeError deep inside the seeding
stage, on the one code path that reads it. That happened on 2026-08-26 with `lc_seg`/`lc_epoch`,
and nothing but a replay caught it.

This is a STRUCTURAL check, not a value check: several slots are legitimately None at
construction (`receiver_state` without --state-file, `dr_eph_mod` without --dead-reckon,
`brdc_alm` before the first fetch), so asserting non-None would be wrong. What can be asserted
is that every slot is accounted for -- passed by the constructor call, given a default, or
declared per-cycle and written by the stage that computes it.

@author Keith Vanderlinde
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")

# Written by whichever stage computes them, once the cycle is under way -- never passed in.
PER_CYCLE = {
    "t0", "best", "status", "pred", "up", "probe_set", "utc0_sample0",
    "xb_pred", "coast_polls", "have_sig", "la_samples", "fitted", "cl_report",
    "dr_pd", "dr_pd0", "dr_pd2", "payload",
    "jrc", "rr_cmd_new", "bit_known", "bit_src",
    # Re-stamped from the status poll every cycle, None whenever the axis is unknown.
    "fe_hop_now", "fe_hop_t",
}

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_chainview_covers_the_publisher():
    """⚠️ EVERY PUBLIC FleetPublisher METHOD MUST EXIST ON THE PER-CHAIN VIEW.

    The broker only ever holds a _ChainView, so a method added to FleetPublisher and forgotten
    here is an AttributeError at the first call -- on whichever chain happens to use it, while
    the others run on. That is exactly what killed gps_l5 on 2026-08-19: set_rf was added to
    the publisher and not the view, and the one chain with rf-stats-endpoints armed died at its
    first poll. The chain that carries the only search, and whose clock the other four adopt,
    is the worst one to lose.

    This is the check that would have caught it, and it is cheap.
    """
    from gnss_broker.publish import FleetPublisher, _ChainView
    print("\n_ChainView covers FleetPublisher's public surface")
    pub = {n for n in dir(FleetPublisher)
           if not n.startswith("_") and callable(getattr(FleetPublisher, n, None))}
    view = {n for n in dir(_ChainView) if not n.startswith("_")}
    # `register` makes a view and is not itself a view operation.
    missing = sorted(pub - view - {"register", "start", "stop"})
    check(not missing,
          "no FleetPublisher method is missing from _ChainView (missing: %s)" % (missing or "none"))


def main():
    print("ChainContext slot coverage\n")
    ctx = ast.parse(open(os.path.join(HERE, "context.py")).read())
    cls = [n for n in ctx.body if isinstance(n, ast.ClassDef) and n.name == "ChainContext"][0]
    slots, defaults = [], set()
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if getattr(t, "id", "") == "__slots__":
                    slots = [e.value for e in n.value.elts]
                elif getattr(t, "id", "") == "DEFAULTS":
                    defaults = {k.value for k in n.value.keys}

    broker = ast.parse(open(BROKER).read())
    main_fn = [n for n in broker.body if getattr(n, "name", "") == "main"][0]
    calls = [n for n in ast.walk(main_fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ChainContext"]
    check(len(calls) == 1, "the broker constructs exactly one ChainContext")
    passed = {k.arg for k in calls[0].keywords} if calls else set()

    unaccounted = [s for s in slots if s not in passed and s not in defaults and s not in PER_CYCLE]
    check(not unaccounted, "every slot is passed, defaulted, or per-cycle"
          + ("" if not unaccounted else " -- MISSING: " + ", ".join(unaccounted)))

    stale = sorted(p for p in passed if p not in slots)
    check(not stale, "no keyword is passed that has no slot"
          + ("" if not stale else " -- EXTRA: " + ", ".join(stale)))

    # A mutable default on the class would be shared by every chain in the process.
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEFAULTS" for t in n.targets):
            bad = [k.value for k, v in zip(n.value.keys, n.value.values)
                   if not isinstance(v, ast.Name)]
            check(not bad, "DEFAULTS holds factories, not values (5 chains share one process)"
                  + ("" if not bad else " -- LITERAL: " + ", ".join(map(str, bad))))

    test_chainview_covers_the_publisher()

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
