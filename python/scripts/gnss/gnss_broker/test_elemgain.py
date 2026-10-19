"""The element gain table's significance column stays aligned with its elements.

    python3 -m gnss_broker.test_elemgain

`gain_table` serves, per (PRN, instance), amp[e], ph[e] and sig[e] = |u_e| / n_floor_e, the
floor being the median of the noise probes' |u_e|. The viewer reads sig[e] as element e's. An
element whose probes all read exactly zero (an input delivered as zeros) has a zero floor and
no significance; this checks that it reads None there and the later elements keep their own
values.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import elemgain


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def row(u, keff=20.0):
    return {"u": u, "p2": [1.0] * len(u), "q": [1.0] * len(u), "keff": keff, "hop": 100}


def main():
    print("\nthe element gain table")

    # Probes 31-33: element 1 reads exactly zero on every probe (floor 0); elements 0 and 2
    # have floors 0.5 and 0.25. Satellite 5: |u| = 2.0, 3.0, 4.0.
    inst = {
        31: row([(0.4, 0.0), (0.0, 0.0), (0.2, 0.0)]),
        32: row([(0.5, 0.0), (0.0, 0.0), (0.25, 0.0)]),
        33: row([(0.6, 0.0), (0.0, 0.0), (0.3, 0.0)]),
        5: row([(2.0, 0.0), (0.0, 3.0), (-4.0, 0.0)]),
    }
    tab = elemgain.gain_table({"cx19/0": inst}, [31, 32, 33])
    ir = tab[5]["inst"]["cx19/0"]
    sig = ir.get("sig") or []
    check(
        len(sig) == len(ir["amp"]) == 3, "sig has one entry per element (got %r)" % sig
    )
    check(
        len(sig) == 3 and sig[1] is None,
        "zero-floor element 1 reads None (got %r)" % sig,
    )
    check(
        len(sig) == 3 and sig[0] == 4.0 and sig[2] == 16.0,
        "elements 0 and 2 keep their own sig 2.0/0.5 and 4.0/0.25 (got %r)" % sig,
    )

    # No probes: no anchor anywhere, so no sig column.
    tab = elemgain.gain_table({"cx19/0": {5: inst[5]}}, [])
    check(
        "sig" not in tab[5]["inst"]["cx19/0"], "without probes the sig column is absent"
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the element gain table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
