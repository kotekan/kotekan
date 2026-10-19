#!/usr/bin/env python3
"""ft_stress.py's RECEIVER lane flags a value that does not match its writer.

    python3 scripts/gnss/test_ft_stress.py

The lane runs against Receiver writers that store a wrong field, one store at a time, and must
report them; against the shipped writers it must report nothing.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_stress  # noqa: E402

Receiver = ft_stress.rx_mod.Receiver

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_shipped_writers_pass():
    err, bad, _ = ft_stress.lane_receiver(50, 4)
    check(
        not err and not bad, "shipped writers: no error, nothing flagged (%s)" % bad[:1]
    )


def test_wrong_n_sats_is_flagged():
    orig = Receiver.contribute_code_bias

    def wrong_n_sats(self, chain, band, la, n_sats, t):
        orig(self, chain, band, la, n_sats + 1, t)

    Receiver.contribute_code_bias = wrong_n_sats
    try:
        _err, bad, _ = ft_stress.lane_receiver(50, 4)
    finally:
        Receiver.contribute_code_bias = orig
    check(
        any(b.startswith("torn code bias") for b in bad),
        "a code bias stored with another n_sats is flagged",
    )


def test_wrong_code_length_is_flagged():
    orig = Receiver.contribute_dr_clock

    def wrong_length(self, chain, band, chips, drift, t, code_length, **kw):
        orig(self, chain, band, chips, drift, t, code_length * 2, **kw)

    Receiver.contribute_dr_clock = wrong_length
    try:
        _err, bad, _ = ft_stress.lane_receiver(50, 4)
    finally:
        Receiver.contribute_dr_clock = orig
    check(
        any(b.startswith("torn dr clock") for b in bad),
        "a dr clock stored with another code length is flagged",
    )


def main():
    print("ft_stress.py RECEIVER lane checks")
    test_shipped_writers_pass()
    test_wrong_n_sats_is_flagged()
    test_wrong_code_length_is_flagged()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
