#!/usr/bin/env python3
"""broker_multi.py's slack-column interval, offline: no broker is started.

    python3 scripts/gnss/test_broker_multi.py

Each chain runs its yaml flags plus the command line's extra args, and the broker keeps the
last --interval, so the cycle report must divide by that one. The production chains yaml,
resolved as broker_multi resolves it, must keep the carrier feeds off the command on every chain
that closes the carrier loop.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import broker_multi  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_slack_interval():
    si = broker_multi._slack_interval
    got = si([("a", ["--interval", "2"])], ["--interval", "0.5"])
    check(got == 0.5, "an --interval in extra overrides the yaml's (got %r)" % got)
    got = si([("a", ["--interval", "2"]), ("b", ["--interval", "2"])], [])
    check(got == 2.0, "chains that agree give their interval (got %r)" % got)
    got = si([("a", ["--interval", "2"]), ("b", ["--interval", "1"])], [])
    check(got is None, "chains that disagree give None (got %r)" % got)
    got = si([("a", ["--interval=1.5"]), ("b", [])], [])
    check(got == 1.5, "--interval=X is read; a chain without one abstains (%r)" % got)
    got = si([("a", ["--interval", "2"]), ("b", ["--interval", "1"])], ["--interval=3"])
    check(got == 3.0, "extra makes disagreeing chains agree (got %r)" % got)
    got = si([("a", ["--min-elev", "10"])], [])
    check(got is None, "no --interval anywhere gives None (got %r)" % got)


def test_commanded_chains_feed_no_command():
    """Every chain's record assembler folds the applied carrier trim out of its records (one
    generator function builds them all), so a chain that commands the carrier
    (--rrate-command) must not add the command back to its coarse and KCOH rates."""
    from gnss_broker.cli import build_parser

    ap = build_parser("test")
    path = os.path.join(HERE, "..", "..", "config", "gnss_chains_chord.yaml")
    n = 0
    for name, argv in broker_multi.load(path):
        a = ap.parse_args(argv)
        if not a.rrate_command:
            continue
        n += 1
        check(
            a.rrate_feed_applied == 0,
            "%s: rrate-command runs with rrate-feed-applied 0 (got %d)"
            % (name, a.rrate_feed_applied),
        )
    check(n > 0, "the chains yaml has commanded chains (%d)" % n)


def main():
    print("broker_multi.py slack interval checks")
    test_slack_interval()
    print("commanded chains in config/gnss_chains_chord.yaml")
    test_commanded_chains_feed_no_command()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
