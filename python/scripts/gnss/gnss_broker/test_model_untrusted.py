"""The MODEL-UNTRUSTED line names the seed Doppler source seeding actually uses.

    /home/kvand/gnss/venv-ft/bin/python -m gnss_broker.test_model_untrusted     (from python/scripts/gnss)

seeding.stage_detections_to_seeds picks a demoted satellite's seed Doppler by four rules: the
search's under --seed-doppler det; the BRDC model's, kept, for an integrity demotion under
--dr-doppler-ignores-integrity (gps_l5 sets it); the almanac predictor's when it holds the
PRN; else the detection's grid Doppler. The dead-reckon stage's MODEL-UNTRUSTED line must
name the same one.

@author Keith Vanderlinde
"""

import ast
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import deadreckon as dr  # noqa: E402

_fails = []
INTEG = "integrity residual +1.40 chips"
EPH = "ephemeris 4.2 h old"


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _args(seed_doppler="pred", ignores=True, almanac=True):
    return NS(
        seed_doppler=seed_doppler,
        dr_doppler_ignores_integrity=ignores,
        almanac=almanac,
    )


def test_label():
    label = getattr(dr, "untrusted_doppler_label", None)
    check(label is not None, "deadreckon.untrusted_doppler_label exists")
    if label is None:
        return
    pred = {7: (1234.0, 0.1, 40.0)}
    cases = (
        (_args(), 7, INTEG, "BRDC kept"),
        (_args(), 7, EPH, "pred (almanac)"),
        (_args(), 9, EPH, "DET(grid)"),
        (_args(ignores=False), 7, INTEG, "pred (almanac)"),
        (_args(ignores=False), 9, INTEG, "DET(grid)"),
        (_args(almanac=False), 7, EPH, "DET(grid)"),
        (_args(seed_doppler="det"), 7, INTEG, "det (--seed-doppler det)"),
    )
    for a, prn, why, want in cases:
        got = label(a, prn, why, pred)
        check(
            want in got,
            "seed_doppler %s, ignores-integrity %s, almanac %s, PRN %d %sin pred, %s: %s"
            % (
                a.seed_doppler,
                a.dr_doppler_ignores_integrity,
                a.almanac,
                prn,
                "" if prn in pred else "not ",
                why.split()[0],
                got,
            ),
        )


def test_log_site():
    """stage_dead_reckon's MODEL-UNTRUSTED line takes its source from the label."""
    with open(dr.__file__) as f:
        tree = ast.parse(f.read(), dr.__file__)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "stage_dead_reckon"
    ][0]
    sites = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "_log"
        and "MODEL-UNTRUSTED" in ast.unparse(n)
    ]
    check(
        len(sites) == 1 and "untrusted_doppler_label(" in sites[0],
        "the MODEL-UNTRUSTED log names the source through untrusted_doppler_label",
    )


def main():
    for fn in (test_label, test_log_site):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
