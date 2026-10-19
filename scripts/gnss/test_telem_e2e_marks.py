#!/usr/bin/env python3
"""telem_e2e.py's address marks, offline: no kotekan build needed.

    python3 scripts/gnss/test_telem_e2e_marks.py

The gate catches a mis-addressed row or comb column because the value written there encodes its
own (instance, record, PRN, slot). Two addresses that share a mark hide a mis-addressing between
them, so every mark the fixture writes must be distinct and exact in float32.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import telem_e2e  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_marks_are_distinct():
    dup = telem_e2e.duplicate_marks()
    check(
        not dup,
        "every fixture mark is distinct (%d shared, e.g. %s)" % (len(dup), dup[:1]),
    )


def test_marks_are_exact_in_float32():
    inst, start = telem_e2e.INSTANCES[-1]
    w = start + telem_e2e.N_WINDOWS - 1
    wstart = (w * telem_e2e.REC_PER_FRAME + telem_e2e.REC_PER_FRAME - 1) * (
        telem_e2e.HOPS_PER_RECORD * telem_e2e.FFT_LEN
    )
    m = telem_e2e._mark(inst, wstart, telem_e2e.PRNS[-1], 400 + telem_e2e.N_CHAN - 1)
    f32 = struct.unpack("<f", struct.pack("<f", m))[0]
    check(m < 2 ** 24 and f32 == m, "the largest mark %.0f is exact in float32" % m)


def test_check_sees_a_collision():
    orig = telem_e2e._mark
    telem_e2e._mark = lambda inst, wstart, prn, slot: orig(inst, wstart, 3, slot)
    try:
        dup = telem_e2e.duplicate_marks()
    finally:
        telem_e2e._mark = orig
    check(bool(dup), "a mark that ignores the PRN is reported (%d shared)" % len(dup))


def main():
    print("telem_e2e.py mark checks")
    test_marks_are_distinct()
    test_marks_are_exact_in_float32()
    test_check_sees_a_collision()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
