#!/usr/bin/env python3
"""Does the beam-cube page refuse to sum two days that are not the same currency?

    PYTHON_GIL=0 /home/kvand/gnss/venv-ft/bin/python test_beamcube_currency.py

The page co-adds the days picked from index.json, and a day that differs from the first one
in units, pointing or array epoch must be skipped: summing across the array epochs adds two
different arrays and still draws a plausible beam. sameCurrency is extracted from the shipped
beamcube.js and run with dukpy, so this checks what the browser runs, not a re-implementation.

@author Keith Vanderlinde
"""
import re
import sys

import dukpy

HERE = __file__.rsplit("/", 1)[0]
PAGE = HERE + "/beamcube.js"

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def same_currency_js():
    """The sameCurrency declaration, from `const sameCurrency =` to its closing `;`."""
    m = re.search(r"^const sameCurrency = [\s\S]*?;\n", open(PAGE).read(), re.M)
    if not m:
        raise SystemExit(
            "cannot find sameCurrency in %s -- the page was restructured" % PAGE
        )
    return m.group(0)


def same(js, a, b):
    return dukpy.evaljs(js + "\nsameCurrency(dukpy['a'], dukpy['b']);", a=a, b=b)


def main():
    js = same_currency_js()
    base = {
        "units": "pedestal",
        "pointing": "p0_dec40p73",
        "array_epoch_key": "2026-09-14-polswap:9db46e1f",
    }
    other_epoch = dict(base, array_epoch_key="2026-08-29-baseline:471481b1")
    unstamped = dict(base, array_epoch_key=None)
    print("sameCurrency extracted from beamcube.js:\n%s" % js)
    check(
        same(js, base, dict(base)),
        "two days of one epoch, units and pointing are summed",
    )
    check(
        not same(js, base, other_epoch),
        "two days of different array epochs (same units, pointing) are refused",
    )
    check(
        not same(js, base, unstamped), "a stamped day and an unstamped day are refused"
    )
    check(not same(js, base, dict(base, units="power")), "different units are refused")
    check(
        not same(js, base, dict(base, pointing="p1")), "different pointings are refused"
    )
    print("%d failure(s)" % len(_fails))
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
