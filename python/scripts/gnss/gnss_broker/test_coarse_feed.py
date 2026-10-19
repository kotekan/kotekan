"""The coarse rate feed admits cohering satellites and never a noise probe.

    python3 -m gnss_broker.test_coarse_feed     (from python/scripts/gnss)

The coarse feed's rows are receiver-wide: each satellite's deep-rate residual lands in its
joint rrate row, which every chain's carrier command reads. A noise probe's residual is noise,
and a probe that happens to clear the coherence test (coh_frac >= 0.3 on a noise fold) would
feed it into that row. Dark probes have no prediction on the chain, below-horizon probes do;
both are excluded. Checked on ratefeed.coarse_feed_rows and on the broker's call to it.

@author Keith Vanderlinde
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import ratefeed  # noqa: E402

BROKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gps_distributed_broker.py",
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_rows():
    rows = getattr(ratefeed, "coarse_feed_rows", None)
    check(rows is not None, "ratefeed.coarse_feed_rows exists")
    if rows is None:
        return
    resid2 = {3: 0.4, 5: -0.2, 7: 1.5, 30: 2.0, 31: -3.0}
    status = {
        3: {"coherence_s": 0.8, "coh_frac": 0.9},  # tracked, cohering
        5: {"coherence_s": 0.0, "coh_frac": 0.35},  # tracked, coh_frac alone
        7: {"coherence_s": 0.0, "coh_frac": 0.1},  # tracked, decohered
        30: {"coherence_s": 0.0, "coh_frac": 0.4},  # probe below the horizon (in pred)
        31: {"coherence_s": 0.2, "coh_frac": 0.5},  # dark probe (no prediction)
    }
    got = rows(resid2, status, {30, 31})
    check(got == {3: 0.4, 5: -0.2}, "only the cohering tracked satellites (%s)" % got)
    check(30 not in got, "a below-horizon probe at coh_frac 0.4 is not fed")
    check(31 not in got, "a dark probe is not fed")
    check(rows(resid2, {}, set()) == {}, "a satellite with no status row is not fed")


def test_call_site():
    """The broker filters resid2 through coarse_feed_rows with the cycle's probe_set."""
    with open(BROKER) as f:
        tree = ast.parse(f.read(), BROKER)
    calls = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "ratefeed.coarse_feed_rows"
    ]
    check(
        calls == ["ratefeed.coarse_feed_rows(_rf.resid2, _ctx.status, _ctx.probe_set)"],
        "the broker's coarse feed filter is coarse_feed_rows (%s)" % calls,
    )


def main():
    for fn in (test_rows, test_call_site):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
