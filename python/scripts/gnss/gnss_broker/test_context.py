"""Every ChainContext slot must actually be supplied.

    python3 -m gnss_broker.test_context

A slot the broker declares but never passes is silently None, and the symptom appears far away
and much later: a comparison like `ctx.lc_seg > 1` raising a TypeError deep inside a stage, on
the one code path that reads it.

This is a structural check, not a value check: several slots are legitimately None at
construction (`dr_eph_mod` without --dead-reckon, `brdc_alm` before the first fetch), so
asserting non-None would be wrong. What can be asserted is that every slot is accounted for:
passed by the constructor call, given a default, or declared per-cycle and written by the stage
that computes it.

@author Keith Vanderlinde
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BROKER = os.path.join(os.path.dirname(HERE), "gps_distributed_broker.py")

# Written by whichever stage computes them, once the cycle is under way; never passed in.
PER_CYCLE = {
    "t0",
    "best",
    "status",
    "pred",
    "up",
    "probe_set",
    "utc0_sample0",
    "coast_polls",
    "have_sig",
    "la_samples",
    "fitted",
    "cl_report",
    "dr_pd",
    "dr_pd0",
    "dr_pd2",
    "transit_pd",
    "payload",
    "jrc",
    "rr_cmd_new",
    # Re-stamped from the status poll every cycle, None whenever the axis is unknown.
    "fe_hop_now",
    "fe_hop_t",
}

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_chainview_covers_the_publisher():
    """Every public FleetPublisher method must exist on the per-chain view.

    The broker only ever holds a _ChainView, so a method added to FleetPublisher and missing
    from the view is an AttributeError at the first call, killing whichever chain uses it while
    the others run on.
    """
    from gnss_broker.publish import FleetPublisher, _ChainView

    print("\n_ChainView covers FleetPublisher's public surface")
    pub = {
        n
        for n in dir(FleetPublisher)
        if not n.startswith("_") and callable(getattr(FleetPublisher, n, None))
    }
    view = {n for n in dir(_ChainView) if not n.startswith("_")}
    # `register` makes a view and is not itself a view operation.
    missing = sorted(pub - view - {"register", "start", "stop"})
    check(
        not missing,
        "no FleetPublisher method is missing from _ChainView (missing: %s)"
        % (missing or "none"),
    )


def main():
    print("ChainContext slot coverage\n")
    ctx = ast.parse(open(os.path.join(HERE, "context.py")).read())
    cls = [
        n for n in ctx.body if isinstance(n, ast.ClassDef) and n.name == "ChainContext"
    ][0]
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
    calls = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ChainContext"
    ]
    check(len(calls) == 1, "the broker constructs exactly one ChainContext")
    passed = {k.arg for k in calls[0].keywords} if calls else set()

    unaccounted = [
        s for s in slots if s not in passed and s not in defaults and s not in PER_CYCLE
    ]
    check(
        not unaccounted,
        "every slot is passed, defaulted, or per-cycle"
        + ("" if not unaccounted else " -- MISSING: " + ", ".join(unaccounted)),
    )

    stale = sorted(p for p in passed if p not in slots)
    check(
        not stale,
        "no keyword is passed that has no slot"
        + ("" if not stale else " -- EXTRA: " + ", ".join(stale)),
    )

    # A mutable default on the class would be shared by every chain in the process.
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", "") == "DEFAULTS" for t in n.targets
        ):
            bad = [
                k.value
                for k, v in zip(n.value.keys, n.value.values)
                if not isinstance(v, ast.Name)
            ]
            check(
                not bad,
                "DEFAULTS holds factories, not values (5 chains share one process)"
                + ("" if not bad else " -- LITERAL: " + ", ".join(map(str, bad))),
            )

    test_chainview_covers_the_publisher()

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
