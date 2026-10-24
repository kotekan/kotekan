"""#200: nothing in a chain hangs off one node.

    python3 -m gnss_broker.test_failover

The frame-0 anchor comes from --rest-url or, failing that, any tracker host (node_bases,
time0_from_any), and the status source fails over along the chain's own instance combiners
(Failover). On 10-18 cx19, the rest-url, was down at a broker start: every model-primary chain
logged "time anchor unavailable" and only the search chain tracked.

@author Keith Vanderlinde
"""

import sys

from gnss_broker.transport import Failover, node_bases, time0_from_any

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_node_bases():
    b = node_bases(
        "http://cx19:12048",
        [
            "http://cx19:12048/gnss0_inject",
            "http://cx19:12048/gnss1_inject",
            "http://cx27:12048/gnss0_inject",
            "http://cx43:12048/gnss1_inject",
        ],
    )
    check(
        b == ["http://cx19:12048", "http://cx27:12048", "http://cx43:12048"],
        "rest-url first, then each tracker host once, in order (got %s)" % b,
    )
    check(
        node_bases("", ["bare_name"]) == [], "bare names and an empty base give no host"
    )


def test_time0_from_any():
    calls = []

    def get(url):
        calls.append(url)
        if url.startswith("http://cx19"):
            raise OSError("[Errno 111] Connection refused")
        if url.startswith("http://cx27"):
            return {"time0_ns": 0}
        return {"time0_ns": 1792201957000003815}

    v, src = time0_from_any(
        ["http://cx19:12048", "http://cx27:12048", "http://cx43:12048"],
        "/telescope/time0_ns",
        get,
    )
    check(
        abs(v - 1792201957.000003815) < 1e-6 and src == "http://cx43:12048",
        "cx19 down and cx27 answering 0: the anchor comes from cx43 (got %r from %s)"
        % (v, src),
    )
    check(
        calls[0] == "http://cx19:12048/telescope/time0_ns",
        "the path is joined without a doubled slash (%s)" % calls[0],
    )
    try:
        time0_from_any(["http://a:1"], "telescope/time0_ns", lambda u: {"time0_ns": 0})
        ok = False
    except RuntimeError as e:
        ok = "a:1" in str(e)
    check(ok, "no node answering raises, naming each failure")


def test_failover():
    down = {"http://cx19:12048/gnss0_l2oc_n2combine"}
    switches = []

    def get(url):
        if any(url.startswith(d) for d in down):
            raise OSError("refused")
        return [{"prn": 19, "src": url}]

    fo = Failover(
        [
            "http://cx19:12048/gnss0_l2oc_n2combine",
            "http://cx19:12048/gnss0_l2oc_n2combine",  # duplicates collapse
            "http://cx27:12048/gnss0_l2oc_n2combine",
            "http://cx43:12048/gnss1_l2oc_n2combine",
        ],
        on_switch=lambda o, n: switches.append((o, n)),
    )
    check(
        len(fo.candidates) == 3,
        "duplicate candidates collapse (got %d)" % len(fo.candidates),
    )
    r = fo.get("get_status", get)
    check(
        r[0]["src"].startswith("http://cx27") and fo.current.startswith("http://cx27"),
        "the first candidate down: the next one answers and becomes current",
    )
    check(
        len(switches) == 1, "... and the switch is reported once (%d)" % len(switches)
    )
    down.clear()
    fo.get("get_status", get)
    check(
        fo.current.startswith("http://cx27") and len(switches) == 1,
        "sticky: the old one coming back does not move it (no flapping)",
    )
    down.update({"http://cx27", "http://cx43", "http://cx19"})
    try:
        fo.get("get_status", get)
        ok = False
    except OSError:
        ok = True
    check(
        ok, "every candidate down: the last error is raised, as a single endpoint did"
    )


def main():
    for fn in (test_node_bases, test_time0_from_any, test_failover):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
