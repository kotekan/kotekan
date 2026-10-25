"""The comb check: each tap's frames against the node table's gpu_offsets_mod16.

    python3 -m gnss_broker.test_comb_check

poll_rf_stats keeps the comb (freq_id mod 16) the tap's frames carry, gnss_arraymap.node_combs
reads the table, and comb_mismatches names a tap whose frames moved off its line (#194: a
rewiring at the F-engine that left four GPUs despreading the wrong channels for an hour).
"""

import os
import sys
import tempfile

import gnss_arraymap
from gnss_broker import fleet

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def tap(freq_ids, enabled=True):
    return {
        "enabled": enabled,
        "chans": list(range(len(freq_ids))),
        "power": [1.0] * len(freq_ids),
        "clip_lo": [0.0] * len(freq_ids),
        "clip_hi": [0.0] * len(freq_ids),
        "freq_ids": freq_ids,
    }


def main():
    served = {
        "http://cx19:12048/gnss0_srch_tap": tap([5972, 5988, 6004]),
        "http://cx19:12048/gnss1_srch_tap": tap([5980, 5996]),  # comb 12, table 14
        "http://cx27:12048/gnss0_srch_tap": tap([]),  # no frame measured yet
        "http://cx44:12048/gnss0_srch_tap": tap([5986], enabled=False),
    }
    real = fleet._get
    fleet._get = lambda url: served[url.rsplit("/rf_stats", 1)[0]]
    try:
        rf = fleet.poll_rf_stats(list(served), lambda *a, **k: [])
    finally:
        fleet._get = real
    check(
        rf["http://cx19:12048/gnss0_srch_tap"]["comb"] == [4],
        "poll_rf_stats keeps the frames' comb",
    )
    check(
        rf["http://cx27:12048/gnss0_srch_tap"]["comb"] == [],
        "a tap with no freq_ids yet has no comb",
    )

    table = {"cx19": [4, 14], "cx27": [0, 8], "cx44": [2, 10]}
    got = fleet.comb_mismatches(rf, table)
    check(
        list(got) == ["http://cx19:12048/gnss1_srch_tap"]
        and got["http://cx19:12048/gnss1_srch_tap"]
        == {"node": "cx19", "gpu": 1, "table": 14, "frames": [12]},
        "the moved GPU is named with both combs, and nothing else (%s)" % got,
    )
    check(
        fleet.comb_mismatches(rf, {}) == {},
        "a node the table does not hold has no verdict",
    )
    rf2 = dict(rf)
    rf2["http://cx19:12048/gnss0_srch_tap"] = dict(
        rf["http://cx19:12048/gnss0_srch_tap"], comb=[4, 5]
    )
    check(
        "http://cx19:12048/gnss0_srch_tap" in fleet.comb_mismatches(rf2, table),
        "a tap carrying two combs is a mismatch",
    )

    repo = gnss_arraymap.node_combs()
    check(
        repo.get("cx19") == [4, 14] and repo.get("cx47") == [3, 11] and len(repo) >= 8,
        "node_combs reads the repo's table (%d hosts)" % len(repo),
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "n.yaml")
        with open(p, "w") as fh:
            fh.write(
                "nodes:\n    # cx99: {gpu_offsets_mod16: [9, 9]}\n"
                "    cx01: {gpu_offsets_mod16: [1, 2]}\n"
                "    bringup_order: [cx01]\n"
            )
        check(
            gnss_arraymap.node_combs(p) == {"cx01": [1, 2]},
            "node_combs skips comments and other keys",
        )
        check(
            gnss_arraymap.node_combs(os.path.join(d, "missing.yaml")) == {},
            "an unreadable table reads as empty",
        )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the comb check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
