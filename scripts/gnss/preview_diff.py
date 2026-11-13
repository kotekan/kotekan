#!/usr/bin/env python3
"""Compare two directories of generated configs (gen_fleet.py --preview-dir output) file by
file, as gen_fleet --check compares: the live EOP table, the bad-input block and the interface
stamp left out, everything else byte for byte.

    preview_diff.py DIR_A DIR_B

Prints one line per *.yaml file, naming the first top-level key that differs, and exits 1 if
any file differs or is on one side only.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_fleet  # noqa: E402


def normalise(text):
    return gen_fleet._drop_stamp(gen_fleet._drop_live(text))


def first_key(a, b):
    """The first top-level key (sorted) whose value differs, or '(header)'."""
    ca, cb = (
        yaml.load(gen_fleet._split_header(t)[1], Loader=yaml.CSafeLoader) or {}
        for t in (a, b)
    )
    for k in sorted(set(ca) | set(cb), key=str):
        if k not in ca or k not in cb or ca[k] != cb[k]:
            return str(k)
    return "(header)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args()

    names = sorted(
        {f for d in (a.a, a.b) for f in os.listdir(d) if f.endswith(".yaml")}
    )
    bad = 0
    for name in names:
        pa, pb = os.path.join(a.a, name), os.path.join(a.b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("ONLY     %s in %s" % (name, a.a if os.path.exists(pa) else a.b))
            bad += 1
            continue
        with open(pa) as fa, open(pb) as fb:
            ta, tb = normalise(fa.read()), normalise(fb.read())
        if ta == tb:
            print("same     %s" % name)
        else:
            print("DIFFERS  %s  first at %s" % (name, first_key(ta, tb)))
            bad += 1
    print("%d of %d differ" % (bad, len(names)) if bad else "identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
