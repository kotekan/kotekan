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
import tempfile

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


# Each chain's broker --dll-spacing: 0.5 (the broker's default) unless named. A row's spacing
# moves the node taps and the trim law together, so changing one changes this pin too.
PINNED_DLL_SPACING = {
    "gps_l2c": 0.33,
    "glo_l2oc": 0.4,
    "glo_l2of": 0.33,
    "bds_b1i": 0.5,
    "gps_l1ca": 0.5,
    "gal_e1c": 0.2,
    "sbas_l1": 0.5,
}


def test_dll_spacing_from_the_chains_table():
    """The broker's dll_spacing per chain, parsed from broker_multi's argv as the chain's
    thread parses it, is its chains-table row's (the node taps' value) and the pinned one."""
    from gnss_broker.cli import build_parser
    import gnss_chain_table

    ap = build_parser("test")
    path = os.path.join(HERE, "..", "..", "config", "gnss_chains_chord.yaml")
    got = {
        name: ap.parse_args(argv).dll_spacing for name, argv in broker_multi.load(path)
    }
    table = gnss_chain_table.load()
    check(
        gnss_chain_table.dll_spacing_drift(table, got, "broker") == [],
        "every chain's dll_spacing is its row's (%s)"
        % gnss_chain_table.dll_spacing_drift(table, got, "broker"),
    )
    want = {c: PINNED_DLL_SPACING.get(c, 0.5) for c in got}
    check(
        got == want and set(PINNED_DLL_SPACING) <= set(got),
        "and the pinned value, chain for chain (%s)"
        % {c: (got[c], want[c]) for c in got if got[c] != want[c]},
    )


def test_iface_is_not_a_chain_flag():
    """common `iface:` configures the interface watcher (gnss_broker/iface_watch.py); passed on
    as --iface, every chain would refuse to start."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as fh:
        fh.write(
            "common:\n  iface: enforce\n  interval: 2\nchains:\n  a: {signal: gps_l5}\n"
        )
        fh.flush()
        got = broker_multi.load(fh.name)
    check(
        got
        == [("a", ["--interval", "2", "--signal", "gps_l5", "--dll-spacing", "0.5"])],
        "common iface: is dropped from the chains' flags (got %r)" % got,
    )


def main():
    print("broker_multi.py slack interval checks")
    test_slack_interval()
    print("commanded chains in config/gnss_chains_chord.yaml")
    test_commanded_chains_feed_no_command()
    print("the interface watcher's settings")
    test_iface_is_not_a_chain_flag()
    print("dll-spacing from the chains table")
    test_dll_spacing_from_the_chains_table()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
