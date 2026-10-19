#!/usr/bin/env python3
"""trim_sign_gate.py's verdict on a synthetic e2e: no C++ build needed.

    python3 scripts/gnss/test_trim_sign_gate.py

The e2e run is replaced by a model with a known comb_mult m: err = 0.8 + trim/m, an E/L
discriminator that rails beyond the pull-in region, and q peaked at err 0. A correct actuator
must pass for every m, an inverted one must fail, and a slope that is not 1/m must be refused.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trim_sign_gate  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def model(per_trim, disc_sign=1.0):
    """A synthetic e2e: (err, q, disc) for one commanded trim. The discriminator measures the
    error in combined chips, m * err."""
    m = round(1.0 / abs(per_trim))

    def run(exe, trim, extra):
        err = 0.8 + trim * per_trim
        disc = disc_sign * 4.0 * max(-1.0, min(1.0, m * err))
        return err, max(0.0, 1.0 - abs(err)), disc

    return run


def gate(run):
    """(exit status, output) of the gate's main() with e2e replaced by `run`."""
    orig, argv = trim_sign_gate.run, sys.argv
    trim_sign_gate.run = run
    sys.argv = ["trim_sign_gate.py", "--exe", sys.executable]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            try:
                rc = trim_sign_gate.main()
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
                print(e.code)
    finally:
        trim_sign_gate.run, sys.argv = orig, argv
    return rc, out.getvalue()


def test_comb_mult_1_passes():
    rc, out = gate(model(1.0))
    check(rc == 0 and "null at trim -0.8000" in out, "comb_mult 1 passes, null -0.8")


def test_comb_mult_2_passes():
    rc, out = gate(model(0.5))
    check(rc == 0, "comb_mult 2 passes (rc %r)" % rc)
    check("null at trim -1.6000" in out, "comb_mult 2: null located at -1.6")
    check("tau/err = -1.00 .. -1.00" in out, "comb_mult 2: tau/err reads -1")


def test_inverted_actuator_fails():
    rc, _ = gate(model(-1.0))
    check(rc == 1, "a trim that moves err the wrong way fails (rc %r)" % rc)


def test_inverted_discriminator_fails():
    rc, _ = gate(model(0.5, disc_sign=-1.0))
    check(rc == 1, "comb_mult 2 with an inverted discriminator fails (rc %r)" % rc)


def test_slope_not_one_over_m_is_refused():
    rc, out = gate(model(0.7))
    check(rc == 1 and "refusing" in out, "slope 0.7 is refused (rc %r)" % rc)


def main():
    print("trim_sign_gate.py checks")
    test_comb_mult_1_passes()
    test_comb_mult_2_passes()
    test_inverted_actuator_fails()
    test_inverted_discriminator_fails()
    test_slope_not_one_over_m_is_refused()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
