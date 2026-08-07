#!/usr/bin/env python3
"""RECORD_FLOATS / ELEM_FLOATS, read from the C++ header that defines them.

WHY THIS FILE EXISTS. The record width is a C++ constant (gnssRecord.hpp) but the frame is
sized in yaml (`n_prn * record_floats * sizeof_float`), and until 2026-08-07 nothing linked
the two. RECORD_FLOATS went 24 -> 26 on the CHORD branch and config/gnss_node.yaml kept
saying 24, so 34 airspy stages FATAL'd at construction on a frame 256 B short per PRN. The
guard that caught it is good (it prints the number to use), but the drift should not be
possible in the first place -- so both generators now READ the header instead of restating
it. The CHORD generator had the same latent bug in a different disguise: a Python literal
`26 + n_elem * 12` whose comment claimed the value was "kept in one place".

Parsed, not imported: these are plain `constexpr int` one-liners, and a regex over them is
cheaper and more robust than any build-time codegen. If the header stops matching, this
raises rather than silently returning a stale default.
"""

import os
import re

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "lib", "stages", "gnss", "gnssRecord.hpp")


def _read(name, header=None):
    path = header or HEADER
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"^\s*constexpr\s+int\s+%s\s*=\s*(\d+)\s*;" % name, text, re.M)
    if not m:
        raise SystemExit("%s: could not parse `constexpr int %s` -- the header's shape "
                         "changed and config/gnss_record_layout.py needs updating" % (path, name))
    return int(m.group(1))


def record_floats(header=None):
    """Per-PRN record header width, floats (gnss::RECORD_FLOATS)."""
    return _read("RECORD_FLOATS", header)


def elem_floats(header=None):
    """Per-element block width, floats (gnss::ELEM_FLOATS)."""
    return _read("ELEM_FLOATS", header)


def record_stride(n_elem, header=None):
    """Full per-PRN stride: RECORD_FLOATS + n_elem * ELEM_FLOATS."""
    return record_floats(header) + n_elem * elem_floats(header)


if __name__ == "__main__":
    print("RECORD_FLOATS %d  ELEM_FLOATS %d" % (record_floats(), elem_floats()))
