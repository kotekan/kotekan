"""The detections the publisher re-serves are every detector's, not the last one's.

    python3 -m gnss_broker.test_last_dets     (from python/scripts/gnss)

With several --detectors the broker polls each one per cycle, and the publisher's
/get_detections must carry all of their lists. Checked on fits.merge_detections and on the
broker's detection poll, which must assign last_dets once per cycle from it, after the loop
over detectors.

@author Keith Vanderlinde
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import fits  # noqa: E402

BROKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gps_distributed_broker.py",
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_merge():
    merge = getattr(fits, "merge_detections", None)
    check(merge is not None, "fits.merge_detections exists")
    if merge is None:
        return
    a, b, c = {"prn": 3}, {"prn": 5}, {"prn": 9}
    check(
        merge([[a], [b, c], []]) == [a, b, c], "three detectors -> all of their lists"
    )
    check(merge([[], None, [b]]) == [b], "an empty or missing list adds nothing")
    check(
        merge([[], []]) == [],
        "no detections anywhere -> empty (the caller keeps the last)",
    )


def test_poll_site():
    """In main(), last_dets is assigned outside the `for d_ep in detectors` loop, from
    merge_detections."""
    with open(BROKER) as f:
        tree = ast.parse(f.read(), BROKER)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and ast.unparse(n.iter) == "detectors"
    ]
    check(len(loops) == 1, "one loop over the detectors (%d)" % len(loops))
    in_loop = [
        n
        for lp in loops
        for n in ast.walk(lp)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t).startswith("last_dets") for t in n.targets)
    ]
    check(not in_loop, "nothing inside it assigns last_dets (%d)" % len(in_loop))
    merged = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "merge_detections"
    ]
    check(len(merged) == 1, "the poll merges the lists with merge_detections")


def main():
    for fn in (test_merge, test_poll_site):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
