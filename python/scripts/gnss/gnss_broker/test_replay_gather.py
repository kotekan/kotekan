"""A replay (--transcript-read) never reads a live telemetry gather.

    python3 -m gnss_broker.test_replay_gather     (from python/scripts/gnss)

A transcript does not hold the telemetry stream, so a broker replaying one with the recorded
--telem-gather would fold whatever a live gather serves into the run (one listens on the gnss
VM). broker_equiv points its own replays at a closed port; a bare replay must not depend on
that. The broker resolves the endpoint through cli.telem_gather_endpoint, which these checks
pin, together with the call site that builds the client from it.

@author Keith Vanderlinde
"""

import ast
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import cli  # noqa: E402

BROKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gps_distributed_broker.py",
)
LIVE = "127.0.0.1:11061"

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _refused(endpoint):
    host, _, port = endpoint.rpartition(":")
    try:
        socket.create_connection((host or "127.0.0.1", int(port)), timeout=1.0).close()
    except OSError:
        return True
    return False


def test_endpoint():
    resolve = getattr(cli, "telem_gather_endpoint", None)
    check(resolve is not None, "cli.telem_gather_endpoint exists")
    if resolve is None:
        return
    ap = cli.build_parser("test")
    base = ["--trackers", "trk"]
    live = ap.parse_args(base + ["--telem-gather", LIVE])
    replay = ap.parse_args(
        base + ["--telem-gather", LIVE, "--transcript-read", "t.jsonl"]
    )
    bare = ap.parse_args(base + ["--transcript-read", "t.jsonl"])
    check(
        resolve(live.telem_gather, live.transcript_read) == LIVE,
        "a live run connects to the gather it names",
    )
    got = resolve(replay.telem_gather, replay.transcript_read)
    check(
        got != LIVE and _refused(got),
        "a replay that names %s connects to %s, where every connect is refused"
        % (LIVE, got),
    )
    check(
        not resolve(bare.telem_gather, bare.transcript_read),
        "a replay that names no gather still has none",
    )


def test_call_site():
    """main() builds the telemetry client from the resolved endpoint, never the raw flag."""
    with open(BROKER) as f:
        tree = ast.parse(f.read(), BROKER)
    calls = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_telem.parse_endpoint"
    ]
    check(
        calls
        == [
            "_telem.parse_endpoint(telem_gather_endpoint(args.telem_gather, "
            "args.transcript_read))"
        ],
        "the gather endpoint is parsed only through telem_gather_endpoint (%s)" % calls,
    )


def main():
    for fn in (test_endpoint, test_call_site):
        print(fn.__name__)
        fn()
    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
